from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether frozen LaneRowNet queries attend to missing GT lanes and "
            "causally test small additive GT-corridor biases. This is an oracle "
            "diagnostic, never a benchmark result."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--corridor-radius-px", type=float, default=16.0)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--bias-strengths",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=3,
        help="One-based decoder layer receiving the soft intervention.",
    )
    parser.add_argument(
        "--bias-modes",
        nargs="+",
        choices=("single", "cascade"),
        default=["single", "cascade"],
        help="'single' biases only target-layer; 'cascade' biases target-layer through final.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _best_iou_by_gt(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> dict[int, float]:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    result: dict[int, float] = {}
    for gt_index in range(int(gt_x.shape[0])):
        if int(valid[gt_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(
            candidates,
            gt_x[gt_index],
            valid[gt_index],
            line_width=float(line_width),
        )
        result[gt_index] = float(ious.max()) if ious.numel() else 0.0
    return result


def _group_zero_assignments(
    match: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for pred_index, gt_index in zip(
        match["pred_indices"].tolist(),
        match["gt_indices"].tolist(),
    ):
        if int(pred_index) < int(group_size):
            result[int(gt_index)] = int(pred_index)
    return result


def _build_corridor_indicator(
    *,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    batch: int,
    instances: int,
    rows: int,
    x_bins: int,
    group_size: int,
    input_w: float,
    radius_px: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return 1 inside each assigned query's GT corridor and 0 elsewhere."""

    indicator = torch.zeros(
        (batch, rows, instances, x_bins),
        device=device,
        dtype=dtype,
    )
    bin_centers = (
        torch.arange(x_bins, device=device, dtype=torch.float32) + 0.5
    ) * float(input_w) / float(x_bins)
    for batch_index, (target, match) in enumerate(zip(targets, matches)):
        assignments = _group_zero_assignments(match, group_size=group_size)
        x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
        valid = target["valid_mask"].to(device=device).bool()
        if int(x_rows.shape[-1]) != int(rows):
            raise ValueError(
                f"GT/head row mismatch: target={x_rows.shape[-1]} head={rows}"
            )
        for gt_index, pred_index in assignments.items():
            valid_rows = valid[gt_index].nonzero(as_tuple=False).flatten()
            for row_index in valid_rows.tolist():
                allowed = (
                    bin_centers - x_rows[gt_index, int(row_index)]
                ).abs() <= float(radius_px)
                if not bool(allowed.any()):
                    nearest = int(
                        (
                            bin_centers - x_rows[gt_index, int(row_index)]
                        ).abs().argmin()
                    )
                    allowed[nearest] = True
                indicator[
                    batch_index,
                    int(row_index),
                    int(pred_index),
                    allowed,
                ] = 1
    return indicator


def _layer_forward_with_attention(
    layer: nn.Module,
    row_tokens: torch.Tensor,
    row_value_features: torch.Tensor,
    row_key_features: torch.Tensor,
    *,
    num_groups: int,
    cross_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror one production decoder block and expose per-head cross attention."""

    batch, instances, rows, channels = row_tokens.shape
    _, value_rows, x_bins, _ = row_value_features.shape
    _, key_rows, key_x_bins, _ = row_key_features.shape
    if value_rows != rows or key_rows != rows or key_x_bins != x_bins:
        raise ValueError("row key/value dimensions do not match row tokens")

    query = row_tokens.permute(0, 2, 1, 3).reshape(
        batch * rows, instances, channels
    )
    query_norm = layer.norm_cross(query)
    key = row_key_features.reshape(batch * rows, x_bins, channels)
    value = row_value_features.reshape(batch * rows, x_bins, channels)

    attention_mask = None
    if cross_bias is not None:
        if tuple(cross_bias.shape) != (batch, rows, instances, x_bins):
            raise ValueError(
                "cross bias shape mismatch: "
                f"{tuple(cross_bias.shape)} vs {(batch, rows, instances, x_bins)}"
            )
        heads = int(layer.cross_attn.num_heads)
        attention_mask = (
            cross_bias.to(device=query.device, dtype=query.dtype)
            .reshape(batch * rows, instances, x_bins)
            .unsqueeze(1)
            .expand(-1, heads, -1, -1)
            .reshape(batch * rows * heads, instances, x_bins)
        )

    cross_delta, attention = layer.cross_attn(
        query_norm,
        key,
        value,
        attn_mask=attention_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    query = query + layer.drop(cross_delta)

    query_norm = layer.norm_inter(query)
    query = query + layer.drop(
        layer._grouped_inter_attention(
            query_norm,
            batch_rows=batch * rows,
            num_instances=instances,
            num_groups=num_groups,
        )
    )
    query = query.view(batch, rows, instances, channels)
    query = query.permute(0, 2, 1, 3).contiguous()

    lane_rows = query.reshape(batch * instances, rows, channels)
    lane_rows_norm = layer.norm_intra(lane_rows)
    lane_rows = lane_rows + layer.drop(
        layer.intra_attn(
            lane_rows_norm,
            lane_rows_norm,
            lane_rows_norm,
            need_weights=False,
        )[0]
    )
    lane_rows_norm = layer.norm_ffn(lane_rows)
    lane_rows = lane_rows + layer.drop(layer.ffn(lane_rows_norm))

    heads = int(attention.shape[1])
    attention = attention.view(batch, rows, heads, instances, x_bins)
    return (
        lane_rows.view(batch, instances, rows, channels).contiguous(),
        attention,
    )


def _structured_forward_with_attention(
    head: nn.Module,
    features: torch.Tensor,
    *,
    cross_bias_by_layer: dict[int, torch.Tensor] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    list[dict[str, torch.Tensor]],
    list[torch.Tensor],
]:
    """Run the production head math while retaining every layer's attention."""

    batch = int(features.shape[0])
    instance = head.instance_tokens.weight.to(
        device=features.device,
        dtype=features.dtype,
    )
    row = head.row_tokens.weight.to(device=features.device, dtype=features.dtype)
    row_tokens = instance[:, None, :] + row[None, :, :]
    row_tokens = row_tokens.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    row_values, row_keys = head._row_features(features)

    stages: list[dict[str, torch.Tensor]] = []
    attention_by_layer: list[torch.Tensor] = []
    for layer_index, layer in enumerate(head.layers):
        row_tokens, attention = _layer_forward_with_attention(
            layer,
            row_tokens,
            row_values,
            row_keys,
            num_groups=head.num_groups,
            cross_bias=(
                None
                if cross_bias_by_layer is None
                else cross_bias_by_layer.get(layer_index)
            ),
        )
        attention_by_layer.append(attention)
        stages.append(
            head._predict_from_row_tokens(
                row_tokens,
                instance,
                include_quality=layer_index == len(head.layers) - 1,
            )
        )
    return stages[-1], stages, attention_by_layer


@dataclass(frozen=True)
class LaneRecord:
    image_index: int
    gt_index: int
    pred_index: int
    stage_ious: tuple[float, ...]

    @property
    def final_iou(self) -> float:
        return self.stage_ious[-1]


def _record_buckets(record: LaneRecord) -> tuple[str, ...]:
    buckets = ["all"]
    buckets.append("final_hit_050" if record.final_iou >= 0.5 else "final_miss_050")
    if len(record.stage_ious) >= 3:
        l2_hit = record.stage_ious[1] >= 0.5
        l3_hit = record.stage_ious[2] >= 0.5
        if l2_hit:
            buckets.append("l2_hit_050")
        elif l3_hit:
            buckets.append("l2_miss_l3_recovered_050")
        else:
            buckets.append("l2_miss_l3_unresolved_050")
    return tuple(buckets)


class AttentionStats:
    def __init__(self) -> None:
        self.lanes = 0
        self.rows = 0
        self.mass_sum = 0.0
        self.uniform_mass_sum = 0.0
        self.top1_inside = 0
        self.centroid_error_sum = 0.0
        self.peak_error_sum = 0.0
        self.centroid_iou_sum = 0.0
        self.peak_iou_sum = 0.0
        self.centroid_hits_050 = 0
        self.peak_hits_050 = 0

    def update(
        self,
        attention: torch.Tensor,
        target: dict[str, torch.Tensor],
        record: LaneRecord,
        *,
        input_w: float,
        radius_px: float,
        line_width: float,
    ) -> None:
        # attention: [B, R, H, N, X]. Heads are averaged before spatial metrics.
        weights = attention[
            record.image_index, :, :, record.pred_index, :
        ].float().mean(dim=1)
        rows, x_bins = int(weights.shape[0]), int(weights.shape[1])
        gt_x = target["x_rows"][record.gt_index].to(
            device=weights.device,
            dtype=torch.float32,
        )
        valid = target["valid_mask"][record.gt_index].to(
            device=weights.device
        ).bool()
        if int(gt_x.numel()) != rows:
            raise ValueError(f"GT/head row mismatch: {gt_x.numel()} vs {rows}")
        if not bool(valid.any()):
            return
        weights = weights[valid]
        gt_x = gt_x[valid]
        centers = (
            torch.arange(x_bins, device=weights.device, dtype=torch.float32) + 0.5
        ) * float(input_w) / float(x_bins)
        allowed = (centers[None, :] - gt_x[:, None]).abs() <= float(radius_px)
        nearest = (centers[None, :] - gt_x[:, None]).abs().argmin(dim=1)
        no_allowed = ~allowed.any(dim=1)
        if bool(no_allowed.any()):
            allowed[no_allowed, nearest[no_allowed]] = True

        attention_mass = (weights * allowed.float()).sum(dim=1)
        uniform_mass = allowed.float().mean(dim=1)
        peak_index = weights.argmax(dim=1)
        centroid_x = (weights * centers[None, :]).sum(dim=1)
        peak_x = centers[peak_index]
        compact_valid = torch.ones_like(peak_x, dtype=torch.bool)
        centroid_iou = line_iou_against_gt(
            centroid_x.unsqueeze(0),
            gt_x,
            compact_valid,
            line_width=float(line_width),
        )
        peak_iou = line_iou_against_gt(
            peak_x.unsqueeze(0),
            gt_x,
            compact_valid,
            line_width=float(line_width),
        )
        centroid_iou_value = (
            float(centroid_iou[0]) if centroid_iou.numel() else 0.0
        )
        peak_iou_value = float(peak_iou[0]) if peak_iou.numel() else 0.0

        self.lanes += 1
        self.rows += int(weights.shape[0])
        self.mass_sum += float(attention_mass.sum())
        self.uniform_mass_sum += float(uniform_mass.sum())
        self.top1_inside += int(
            allowed.gather(1, peak_index[:, None]).sum()
        )
        self.centroid_error_sum += float((centroid_x - gt_x).abs().sum())
        self.peak_error_sum += float((peak_x - gt_x).abs().sum())
        self.centroid_iou_sum += centroid_iou_value
        self.peak_iou_sum += peak_iou_value
        self.centroid_hits_050 += int(centroid_iou_value >= 0.5)
        self.peak_hits_050 += int(peak_iou_value >= 0.5)

    def summary(self) -> dict[str, float | int]:
        rows = max(self.rows, 1)
        mean_mass = self.mass_sum / rows
        mean_uniform = self.uniform_mass_sum / rows
        return {
            "gt_lanes": self.lanes,
            "valid_lane_rows": self.rows,
            "corridor_attention_mass": mean_mass,
            "uniform_corridor_mass": mean_uniform,
            "corridor_enrichment": mean_mass / max(mean_uniform, 1e-12),
            "top1_inside_fraction": self.top1_inside / rows,
            "attention_centroid_error_px": self.centroid_error_sum / rows,
            "attention_peak_error_px": self.peak_error_sum / rows,
            "attention_centroid_mean_iou": self.centroid_iou_sum
            / max(self.lanes, 1),
            "attention_peak_mean_iou": self.peak_iou_sum / max(self.lanes, 1),
            "attention_centroid_recall_050": self.centroid_hits_050
            / max(self.lanes, 1),
            "attention_peak_recall_050": self.peak_hits_050
            / max(self.lanes, 1),
        }


class AssignedGeometryStats:
    """Measure whether a moved attention map actually moves the assigned row output."""

    def __init__(self) -> None:
        self.lanes = 0
        self.rows = 0
        self.before_iou_sum = 0.0
        self.after_iou_sum = 0.0
        self.abs_x_change_sum = 0.0
        self.error_reduction_sum = 0.0
        self.direction_eligible = 0
        self.direction_correct = 0

    def update(
        self,
        before_x: torch.Tensor,
        after_x: torch.Tensor,
        target: dict[str, torch.Tensor],
        record: LaneRecord,
        *,
        line_width: float,
    ) -> None:
        before = before_x[record.image_index, record.pred_index].float()
        after = after_x[record.image_index, record.pred_index].float()
        gt_x = target["x_rows"][record.gt_index].to(
            device=before.device,
            dtype=torch.float32,
        )
        valid = target["valid_mask"][record.gt_index].to(
            device=before.device
        ).bool()
        if not bool(valid.any()):
            return
        before_iou = line_iou_against_gt(
            before.unsqueeze(0),
            gt_x,
            valid,
            line_width=float(line_width),
        )
        after_iou = line_iou_against_gt(
            after.unsqueeze(0),
            gt_x,
            valid,
            line_width=float(line_width),
        )
        before_error = gt_x[valid] - before[valid]
        after_error = gt_x[valid] - after[valid]
        update = after[valid] - before[valid]
        direction_eligible = before_error.abs() > 4.0

        self.lanes += 1
        self.rows += int(valid.sum())
        self.before_iou_sum += float(before_iou[0]) if before_iou.numel() else 0.0
        self.after_iou_sum += float(after_iou[0]) if after_iou.numel() else 0.0
        self.abs_x_change_sum += float(update.abs().sum())
        self.error_reduction_sum += float(
            (before_error.abs() - after_error.abs()).sum()
        )
        self.direction_eligible += int(direction_eligible.sum())
        self.direction_correct += int(
            ((update * before_error > 0.0) & direction_eligible).sum()
        )

    def summary(self) -> dict[str, float | int]:
        lanes = max(self.lanes, 1)
        rows = max(self.rows, 1)
        return {
            "gt_lanes": self.lanes,
            "valid_lane_rows": self.rows,
            "assigned_iou_before": self.before_iou_sum / lanes,
            "assigned_iou_after": self.after_iou_sum / lanes,
            "assigned_iou_delta": (
                self.after_iou_sum - self.before_iou_sum
            )
            / lanes,
            "mean_abs_row_output_change_px": self.abs_x_change_sum / rows,
            "mean_row_error_reduction_px": self.error_reduction_sum / rows,
            "update_direction_accuracy_over_4px": self.direction_correct
            / max(self.direction_eligible, 1),
        }


class RecallStats:
    def __init__(self) -> None:
        self.lanes = 0
        self.before_hits_050 = 0
        self.after_hits_050 = 0
        self.before_hits_070 = 0
        self.after_hits_070 = 0
        self.recovered_050 = 0
        self.lost_050 = 0
        self.recovered_070 = 0
        self.lost_070 = 0
        self.iou_delta_sum = 0.0

    def update(self, before: float, after: float) -> None:
        self.lanes += 1
        for threshold, suffix in ((0.5, "050"), (0.7, "070")):
            before_hit = before >= threshold
            after_hit = after >= threshold
            self.__dict__[f"before_hits_{suffix}"] += int(before_hit)
            self.__dict__[f"after_hits_{suffix}"] += int(after_hit)
            self.__dict__[f"recovered_{suffix}"] += int(
                not before_hit and after_hit
            )
            self.__dict__[f"lost_{suffix}"] += int(before_hit and not after_hit)
        self.iou_delta_sum += float(after - before)

    def summary(self) -> dict[str, float | int]:
        lanes = max(self.lanes, 1)
        return {
            "gt_lanes": self.lanes,
            "before_recall_050": self.before_hits_050 / lanes,
            "after_recall_050": self.after_hits_050 / lanes,
            "net_gain_050_points": 100.0
            * (self.after_hits_050 - self.before_hits_050)
            / lanes,
            "recovered_050": self.recovered_050,
            "lost_050": self.lost_050,
            "before_recall_070": self.before_hits_070 / lanes,
            "after_recall_070": self.after_hits_070 / lanes,
            "net_gain_070_points": 100.0
            * (self.after_hits_070 - self.before_hits_070)
            / lanes,
            "recovered_070": self.recovered_070,
            "lost_070": self.lost_070,
            "mean_best_iou_delta": self.iou_delta_sum / lanes,
        }


class BaselineStageStats:
    def __init__(self, layers: int) -> None:
        self.lanes = 0
        self.hit_050 = [0] * layers
        self.hit_070 = [0] * layers
        self.iou_sum = [0.0] * layers

    def update(self, stage_ious: tuple[float, ...]) -> None:
        self.lanes += 1
        for index, value in enumerate(stage_ious):
            self.hit_050[index] += int(value >= 0.5)
            self.hit_070[index] += int(value >= 0.7)
            self.iou_sum[index] += float(value)

    def summary(self) -> dict[str, dict[str, float | int]]:
        lanes = max(self.lanes, 1)
        return {
            f"L{index + 1}": {
                "gt_lanes": self.lanes,
                "recall_050": self.hit_050[index] / lanes,
                "recall_070": self.hit_070[index] / lanes,
                "mean_best_iou": self.iou_sum[index] / lanes,
            }
            for index in range(len(self.hit_050))
        }


def _build_records(
    *,
    stages: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    group_size: int,
    line_width: float,
) -> list[LaneRecord]:
    stage_maps: list[list[dict[int, float]]] = []
    for stage in stages:
        stage_maps.append(
            [
                _best_iou_by_gt(
                    stage["pred_x_rows"][image_index, :group_size].float(),
                    target,
                    line_width=line_width,
                )
                for image_index, target in enumerate(targets)
            ]
        )

    records: list[LaneRecord] = []
    for image_index, (target, match) in enumerate(zip(targets, matches)):
        assignments = _group_zero_assignments(match, group_size=group_size)
        valid = target["valid_mask"]
        for gt_index in range(int(valid.shape[0])):
            if int(valid[gt_index].sum()) < 5 or gt_index not in assignments:
                continue
            records.append(
                LaneRecord(
                    image_index=image_index,
                    gt_index=gt_index,
                    pred_index=assignments[gt_index],
                    stage_ious=tuple(
                        stage_map[image_index].get(gt_index, 0.0)
                        for stage_map in stage_maps
                    ),
                )
            )
    return records


def _intervention_name(
    mode: str,
    target_layer: int,
    strength: float,
) -> str:
    strength_text = f"{strength:g}".replace(".", "p")
    if mode == "single":
        return f"L{target_layer}_bias_{strength_text}"
    return f"L{target_layer}_to_final_bias_{strength_text}"


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg.setdefault("dataloader", {})["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    head = model.structured_query_head
    if head is None:
        raise ValueError("attention acquisition audit requires structured_query")

    layer_count = len(head.layers)
    target_layer_index = int(args.target_layer) - 1
    if not 0 <= target_layer_index < layer_count:
        raise ValueError(
            f"--target-layer must be in [1, {layer_count}], got {args.target_layer}"
        )
    strengths = [float(value) for value in args.bias_strengths]
    if any(value <= 0.0 for value in strengths):
        raise ValueError("--bias-strengths must be positive")

    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    input_w = float(model_cfg.get("input_w", 800))
    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    bucket_names = (
        "all",
        "final_hit_050",
        "final_miss_050",
        "l2_hit_050",
        "l2_miss_l3_recovered_050",
        "l2_miss_l3_unresolved_050",
    )
    baseline_attention = {
        f"L{layer_index + 1}": {
            bucket: AttentionStats() for bucket in bucket_names
        }
        for layer_index in range(layer_count)
    }
    intervention_names = [
        _intervention_name(mode, int(args.target_layer), strength)
        for mode in args.bias_modes
        for strength in strengths
    ]
    intervention_recall = {
        name: {bucket: RecallStats() for bucket in bucket_names}
        for name in intervention_names
    }
    intervention_attention = {
        name: {bucket: AttentionStats() for bucket in bucket_names}
        for name in intervention_names
    }
    intervention_assigned_geometry = {
        name: {bucket: AssignedGeometryStats() for bucket in bucket_names}
        for name in intervention_names
    }
    baseline_stages = BaselineStageStats(layer_count)
    images_seen = 0
    lanes_without_group0_assignment = 0

    total_batches = len(loader)
    if int(args.max_batches) > 0:
        total_batches = min(total_batches, int(args.max_batches))
    progress = tqdm(
        enumerate(loader),
        total=total_batches,
        desc="attention acquisition audit",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            ),
        )
        amp_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if autocast_enabled
            else nullcontext()
        )
        with amp_context:
            features = model.encoder.forward_features(
                images,
                inference_only=True,
                structured_only=True,
            )["features"]
            baseline, stages, attention = _structured_forward_with_attention(
                head,
                features,
            )
        matches = matcher(baseline, targets)
        records = _build_records(
            stages=stages,
            targets=targets,
            matches=matches,
            group_size=group_size,
            line_width=float(args.line_width),
        )
        expected_lanes = sum(
            int((target["valid_mask"].sum(dim=1) >= 5).sum())
            for target in targets
        )
        lanes_without_group0_assignment += expected_lanes - len(records)
        for record in records:
            baseline_stages.update(record.stage_ious)
            for layer_index, layer_attention in enumerate(attention):
                for bucket in _record_buckets(record):
                    baseline_attention[f"L{layer_index + 1}"][bucket].update(
                        layer_attention,
                        targets[record.image_index],
                        record,
                        input_w=input_w,
                        radius_px=float(args.corridor_radius_px),
                        line_width=float(args.line_width),
                    )

        corridor = _build_corridor_indicator(
            targets=targets,
            matches=matches,
            batch=int(images.shape[0]),
            instances=num_instances,
            rows=head.num_rows,
            x_bins=head.evidence_x_bins,
            group_size=group_size,
            input_w=input_w,
            radius_px=float(args.corridor_radius_px),
            device=features.device,
            dtype=features.dtype,
        )
        for mode in args.bias_modes:
            if mode == "single":
                biased_layers = (target_layer_index,)
            else:
                biased_layers = tuple(range(target_layer_index, layer_count))
            for strength in strengths:
                name = _intervention_name(
                    mode,
                    int(args.target_layer),
                    strength,
                )
                bias_by_layer = {
                    layer_index: corridor * float(strength)
                    for layer_index in biased_layers
                }
                with amp_context:
                    after, _after_stages, after_attention = (
                        _structured_forward_with_attention(
                            head,
                            features,
                            cross_bias_by_layer=bias_by_layer,
                        )
                    )
                after_maps = [
                    _best_iou_by_gt(
                        after["pred_x_rows"][image_index, :group_size].float(),
                        target,
                        line_width=float(args.line_width),
                    )
                    for image_index, target in enumerate(targets)
                ]
                for record in records:
                    after_iou = after_maps[record.image_index].get(
                        record.gt_index, 0.0
                    )
                    for bucket in _record_buckets(record):
                        intervention_recall[name][bucket].update(
                            record.final_iou,
                            after_iou,
                        )
                        intervention_attention[name][bucket].update(
                            after_attention[target_layer_index],
                            targets[record.image_index],
                            record,
                            input_w=input_w,
                            radius_px=float(args.corridor_radius_px),
                            line_width=float(args.line_width),
                        )
                        intervention_assigned_geometry[name][bucket].update(
                            baseline["pred_x_rows"],
                            after["pred_x_rows"],
                            targets[record.image_index],
                            record,
                            line_width=float(args.line_width),
                        )
        images_seen += int(images.shape[0])

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "oracle_warning": (
            "GT lanes and final Hungarian group-0 assignments define additive "
            "attention-logit biases. Results diagnose a frozen decoder and are "
            "not deployable or benchmark scores."
        ),
        "interpretation": (
            "Low natural corridor attention on unresolved lanes indicates an "
            "acquisition failure. If soft bias raises corridor attention and recall, "
            "the decoder can use the evidence and supervision/routing is implicated. "
            "If attention moves without geometry recovery, the state update/interface "
            "cannot convert attended P2 evidence into a lane."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "split": args.split,
        "images": images_seen,
        "num_instances": num_instances,
        "num_groups": num_groups,
        "group_size": group_size,
        "decoder_layers": layer_count,
        "target_layer": int(args.target_layer),
        "corridor_radius_px": float(args.corridor_radius_px),
        "bias_strengths": strengths,
        "bias_modes": list(args.bias_modes),
        "lanes_without_group0_assignment": lanes_without_group0_assignment,
        "baseline_stage_geometry": baseline_stages.summary(),
        "baseline_attention": {
            layer_name: {
                bucket: values.summary()
                for bucket, values in buckets.items()
            }
            for layer_name, buckets in baseline_attention.items()
        },
        "soft_bias_interventions": {
            name: {
                "recall": {
                    bucket: values.summary()
                    for bucket, values in intervention_recall[name].items()
                },
                "target_layer_attention": {
                    bucket: values.summary()
                    for bucket, values in intervention_attention[name].items()
                },
                "assigned_row_output_response": {
                    bucket: values.summary()
                    for bucket, values in intervention_assigned_geometry[
                        name
                    ].items()
                },
            }
            for name in intervention_names
        },
    }
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
