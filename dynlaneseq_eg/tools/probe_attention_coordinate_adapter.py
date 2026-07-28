from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt
from dynlaneseq_eg.factory import build_dataloader, build_matcher, build_model
from dynlaneseq_eg.modeling.common import nested_to_device
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    _group_zero_assignments,
    _structured_forward_with_attention,
)


PROBE_MODES: dict[str, tuple[bool, bool]] = {
    "anchor_only": (False, False),
    "state_only": (True, False),
    "attention_only": (False, True),
    "state_attention": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a trained LaneRowNet checkpoint and train equal-capacity tiny "
            "adapters that receive either row state, natural attention coordinates, "
            "or both. The controlled comparison tests whether explicitly carrying "
            "attended x-position can recover geometry that the frozen decoder misses. "
            "This is a diagnostic, not a benchmark result."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-update-px", type=float, default=48.0)
    parser.add_argument("--smooth-l1-beta-px", type=float, default=3.0)
    parser.add_argument("--line-iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument(
        "--attention-layers",
        type=int,
        nargs="+",
        default=[3, 4],
        help="One-based decoder layers whose natural attention statistics are exposed.",
    )
    parser.add_argument(
        "--train-group-mode",
        choices=("all", "group0"),
        default="all",
        help="Use all one-to-many assignments or only inference group zero for probe training.",
    )
    parser.add_argument("--eval-max-batches", type=int, default=32)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-probes", default="")
    parser.add_argument("--load-probes", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CoordinateResidualAdapter(nn.Module):
    """Small, bounded lane-row correction head used only for diagnosis."""

    def __init__(
        self,
        *,
        state_dim: int,
        coordinate_dim: int,
        hidden_dim: int,
        num_rows: int,
        max_update_px: float,
        use_state: bool,
        use_attention: bool,
    ) -> None:
        super().__init__()
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be at least 8")
        self.state_dim = int(state_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_rows = int(num_rows)
        self.max_update_px = float(max_update_px)
        self.use_state = bool(use_state)
        self.use_attention = bool(use_attention)

        self.state_norm = nn.LayerNorm(self.state_dim)
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.coordinate_proj = nn.Linear(self.coordinate_dim, self.hidden_dim)
        self.anchor_proj = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.row_embedding = nn.Embedding(self.num_rows, self.hidden_dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=5,
                padding=2,
            ),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.residual_head = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        # Every control starts as an exact no-op. This makes early hit retention
        # comparable and prevents random probe initialization from moving lanes.
        nn.init.constant_(self.residual_head.weight, 0.0)
        nn.init.constant_(self.residual_head.bias, 0.0)

    def forward(
        self,
        row_state: torch.Tensor,
        coordinate_features: torch.Tensor,
        anchor_x: torch.Tensor,
        *,
        input_w: float,
    ) -> torch.Tensor:
        if row_state.ndim != 3:
            raise ValueError("row_state must have shape [lanes, rows, channels]")
        lanes, rows, state_dim = row_state.shape
        if state_dim != self.state_dim or rows != self.num_rows:
            raise ValueError(
                "row_state shape mismatch: "
                f"{tuple(row_state.shape)} vs [lanes,{self.num_rows},{self.state_dim}]"
            )
        if coordinate_features.shape != (
            lanes,
            rows,
            self.coordinate_dim,
        ):
            raise ValueError(
                "coordinate feature shape mismatch: "
                f"{tuple(coordinate_features.shape)} vs "
                f"{(lanes, rows, self.coordinate_dim)}"
            )
        if anchor_x.shape != (lanes, rows):
            raise ValueError(
                f"anchor_x must be {(lanes, rows)}, got {tuple(anchor_x.shape)}"
            )

        state_input = row_state.float()
        if not self.use_state:
            state_input = torch.zeros_like(state_input)
        coordinate_input = coordinate_features.float()
        if not self.use_attention:
            coordinate_input = torch.zeros_like(coordinate_input)

        width_scale = max(float(input_w) - 1.0, 1.0)
        anchor_norm = 2.0 * anchor_x.float() / width_scale - 1.0
        anchor_features = torch.stack(
            (
                anchor_norm,
                torch.sin(math.pi * anchor_norm),
                torch.cos(math.pi * anchor_norm),
            ),
            dim=-1,
        )
        hidden = (
            self.state_proj(self.state_norm(state_input))
            # These are already bounded physical statistics. LayerNorm across
            # heads would erase part of their common signed x displacement.
            + self.coordinate_proj(coordinate_input)
            + self.anchor_proj(anchor_features)
            + self.row_embedding.weight.view(1, rows, self.hidden_dim)
        )
        hidden = hidden + self.local_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        raw = self.residual_head(self.output_norm(hidden)).squeeze(-1)
        return self.max_update_px * torch.tanh(raw)


def _attention_coordinate_features(
    attention_by_layer: list[torch.Tensor],
    *,
    selected_layers: tuple[int, ...],
    anchor_x: torch.Tensor,
    input_w: float,
) -> torch.Tensor:
    """Return per-head position deltas and confidence for selected layers.

    Input attention tensors have shape [B, R, H, N, X]. Output is
    [B, N, R, len(layers) * H * 4], containing expected-x delta, peak-x
    delta, normalized entropy, and peak probability.
    """

    if anchor_x.ndim != 3:
        raise ValueError("anchor_x must have shape [B, N, R]")
    batch, instances, rows = anchor_x.shape
    width_scale = max(float(input_w) - 1.0, 1.0)
    anchor_norm = 2.0 * anchor_x.float() / width_scale - 1.0
    parts: list[torch.Tensor] = []
    for layer_index in selected_layers:
        attention = attention_by_layer[int(layer_index)]
        if attention.ndim != 5:
            raise ValueError("attention must have shape [B, R, H, N, X]")
        if (
            int(attention.shape[0]) != batch
            or int(attention.shape[1]) != rows
            or int(attention.shape[3]) != instances
        ):
            raise ValueError(
                "attention/anchor shape mismatch: "
                f"{tuple(attention.shape)} vs {tuple(anchor_x.shape)}"
            )
        weights = attention.float().permute(0, 3, 1, 2, 4).contiguous()
        x_bins = int(weights.shape[-1])
        centers_px = (
            torch.arange(
                x_bins,
                device=weights.device,
                dtype=weights.dtype,
            )
            + 0.5
        ) * float(input_w) / float(x_bins)
        centers = (
            2.0 * centers_px / width_scale - 1.0
        )
        expected = (weights * centers.view(1, 1, 1, 1, x_bins)).sum(dim=-1)
        peak_index = weights.argmax(dim=-1)
        peak = centers[peak_index]
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(
            dim=-1
        )
        entropy = entropy / max(math.log(max(x_bins, 2)), 1e-8)
        peak_probability = weights.amax(dim=-1)
        anchor = anchor_norm.unsqueeze(-1)
        parts.extend(
            (
                expected - anchor,
                peak - anchor,
                entropy,
                peak_probability,
            )
        )
    if not parts:
        raise ValueError("At least one attention layer must be selected")
    return torch.cat(parts, dim=-1)


def _extract_frozen_outputs(
    *,
    model: nn.Module,
    images: torch.Tensor,
    selected_layers: tuple[int, ...],
    input_w: float,
    amp_dtype: torch.dtype | None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[dict[str, torch.Tensor]],
]:
    amp_context = (
        torch.autocast(device_type=images.device.type, dtype=amp_dtype)
        if amp_dtype is not None and images.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_context:
        features = model.encoder.forward_features(
            images,
            inference_only=True,
            structured_only=True,
        )["features"]
        outputs, stages, attention = _structured_forward_with_attention(
            model.structured_query_head,
            features,
        )
    row_state = outputs["structured_row_tokens"].detach().float()
    anchor_x = outputs["pred_x_rows"].detach().float()
    coordinate_features = _attention_coordinate_features(
        attention,
        selected_layers=selected_layers,
        anchor_x=anchor_x,
        input_w=input_w,
    ).detach()
    detached_outputs = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in outputs.items()
    }
    return detached_outputs, row_state, coordinate_features, stages


@dataclass
class MatchedExamples:
    row_state: torch.Tensor
    coordinate_features: torch.Tensor
    anchor_x: torch.Tensor
    gt_x: torch.Tensor
    valid: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.anchor_x.shape[0])


def _matched_examples(
    *,
    row_state: torch.Tensor,
    coordinate_features: torch.Tensor,
    anchor_x: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    group_size: int,
    group_mode: str,
) -> MatchedExamples | None:
    state_parts: list[torch.Tensor] = []
    coordinate_parts: list[torch.Tensor] = []
    anchor_parts: list[torch.Tensor] = []
    gt_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    for batch_index, (target, match) in enumerate(zip(targets, matches)):
        gt_x = target["x_rows"].to(
            device=anchor_x.device,
            dtype=anchor_x.dtype,
        )
        valid = target["valid_mask"].to(device=anchor_x.device).bool()
        for pred_index, gt_index in zip(
            match["pred_indices"].tolist(),
            match["gt_indices"].tolist(),
        ):
            if group_mode == "group0" and int(pred_index) >= int(group_size):
                continue
            if int(valid[int(gt_index)].sum()) < 5:
                continue
            state_parts.append(row_state[batch_index, int(pred_index)])
            coordinate_parts.append(
                coordinate_features[batch_index, int(pred_index)]
            )
            anchor_parts.append(anchor_x[batch_index, int(pred_index)])
            gt_parts.append(gt_x[int(gt_index)])
            valid_parts.append(valid[int(gt_index)])
    if not state_parts:
        return None
    return MatchedExamples(
        row_state=torch.stack(state_parts),
        coordinate_features=torch.stack(coordinate_parts),
        anchor_x=torch.stack(anchor_parts),
        gt_x=torch.stack(gt_parts),
        valid=torch.stack(valid_parts),
    )


def _paired_line_iou_loss(
    pred_x: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    radius = 0.5 * float(line_width)
    overlap = (
        torch.minimum(pred_x + radius, gt_x + radius)
        - torch.maximum(pred_x - radius, gt_x - radius)
    ).clamp(min=0.0)
    union = (2.0 * float(line_width) - overlap).clamp_min(1e-6)
    mask = valid.float()
    iou = (overlap * mask).sum(dim=1) / (union * mask).sum(dim=1).clamp_min(
        1e-6
    )
    return (1.0 - iou).mean()


def _adapter_loss(
    corrected_x: torch.Tensor,
    examples: MatchedExamples,
    *,
    smooth_l1_beta_px: float,
    max_update_px: float,
    line_width: float,
    line_iou_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = examples.valid
    pred = corrected_x[valid]
    gt = examples.gt_x[valid]
    point = F.smooth_l1_loss(
        pred,
        gt,
        beta=float(smooth_l1_beta_px),
        reduction="mean",
    ) / max(float(max_update_px), 1.0)
    line = _paired_line_iou_loss(
        corrected_x,
        examples.gt_x,
        valid,
        line_width=float(line_width),
    )
    total = point + float(line_iou_loss_weight) * line
    with torch.no_grad():
        mae = float((corrected_x[valid] - examples.gt_x[valid]).abs().mean())
    return total, {
        "total": float(total.detach()),
        "point": float(point.detach()),
        "line_iou": float(line.detach()),
        "row_mae_px": mae,
    }


class GeometryStats:
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
        self.before_iou_sum = 0.0
        self.after_iou_sum = 0.0

    def update(self, before: float, after: float) -> None:
        self.lanes += 1
        self.before_iou_sum += float(before)
        self.after_iou_sum += float(after)
        for threshold, suffix in ((0.5, "050"), (0.7, "070")):
            before_hit = before >= threshold
            after_hit = after >= threshold
            self.__dict__[f"before_hits_{suffix}"] += int(before_hit)
            self.__dict__[f"after_hits_{suffix}"] += int(after_hit)
            self.__dict__[f"recovered_{suffix}"] += int(
                not before_hit and after_hit
            )
            self.__dict__[f"lost_{suffix}"] += int(before_hit and not after_hit)

    def summary(self) -> dict[str, float | int]:
        lanes = max(self.lanes, 1)
        return {
            "gt_lanes": self.lanes,
            "before_hits_050": self.before_hits_050,
            "after_hits_050": self.after_hits_050,
            "before_recall_050": self.before_hits_050 / lanes,
            "after_recall_050": self.after_hits_050 / lanes,
            "net_gain_050_points": 100.0
            * (self.after_hits_050 - self.before_hits_050)
            / lanes,
            "recovered_050": self.recovered_050,
            "lost_050": self.lost_050,
            "before_hits_070": self.before_hits_070,
            "after_hits_070": self.after_hits_070,
            "before_recall_070": self.before_hits_070 / lanes,
            "after_recall_070": self.after_hits_070 / lanes,
            "net_gain_070_points": 100.0
            * (self.after_hits_070 - self.before_hits_070)
            / lanes,
            "recovered_070": self.recovered_070,
            "lost_070": self.lost_070,
            "mean_iou_before": self.before_iou_sum / lanes,
            "mean_iou_after": self.after_iou_sum / lanes,
            "mean_iou_delta": (self.after_iou_sum - self.before_iou_sum) / lanes,
        }


class AssignedRowStats:
    def __init__(self) -> None:
        self.lanes = 0
        self.rows = 0
        self.before_mae_sum = 0.0
        self.after_mae_sum = 0.0
        self.update_abs_sum = 0.0

    def update(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        gt_x: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        valid = valid.bool()
        if int(valid.sum()) < 5:
            return
        self.lanes += 1
        self.rows += int(valid.sum())
        self.before_mae_sum += float((before[valid] - gt_x[valid]).abs().sum())
        self.after_mae_sum += float((after[valid] - gt_x[valid]).abs().sum())
        self.update_abs_sum += float((after[valid] - before[valid]).abs().sum())

    def summary(self) -> dict[str, float | int]:
        rows = max(self.rows, 1)
        return {
            "assigned_lanes": self.lanes,
            "valid_lane_rows": self.rows,
            "row_mae_before_px": self.before_mae_sum / rows,
            "row_mae_after_px": self.after_mae_sum / rows,
            "row_mae_reduction_px": (
                self.before_mae_sum - self.after_mae_sum
            )
            / rows,
            "mean_abs_update_px": self.update_abs_sum / rows,
        }


def _best_iou(
    candidates: torch.Tensor,
    gt_x: torch.Tensor,
    valid: torch.Tensor,
    *,
    line_width: float,
) -> float:
    values = line_iou_against_gt(
        candidates,
        gt_x,
        valid,
        line_width=float(line_width),
    )
    return float(values.max()) if values.numel() else 0.0


@torch.inference_mode()
def _evaluate(
    *,
    model: nn.Module,
    adapters: dict[str, CoordinateResidualAdapter],
    matcher,
    loader,
    device: torch.device,
    channels_last: bool,
    amp_dtype: torch.dtype | None,
    selected_layers: tuple[int, ...],
    input_w: float,
    group_size: int,
    max_batches: int,
    line_width: float,
) -> dict[str, Any]:
    for adapter in adapters.values():
        adapter.eval()
    bucket_names = ("all", "baseline_hit_050", "baseline_miss_050")
    geometry = {
        name: {bucket: GeometryStats() for bucket in bucket_names}
        for name in PROBE_MODES
    }
    assigned = {name: AssignedRowStats() for name in PROBE_MODES}
    images_seen = 0
    gt_lanes = 0
    displayed_batches = len(loader)
    if int(max_batches) > 0:
        displayed_batches = min(displayed_batches, int(max_batches))

    progress = tqdm(
        enumerate(loader),
        total=displayed_batches,
        desc="coordinate-adapter eval",
        ncols=100,
    )
    for batch_index, (images, targets, _metas) in progress:
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            ),
        )
        targets = nested_to_device(targets, device)
        outputs, row_state, coordinate_features, _stages = _extract_frozen_outputs(
            model=model,
            images=images,
            selected_layers=selected_layers,
            input_w=input_w,
            amp_dtype=amp_dtype,
        )
        matches = matcher(outputs, targets)
        baseline_x = outputs["pred_x_rows"].float()
        corrected_by_probe: dict[str, torch.Tensor] = {}
        flat_state = row_state.reshape(-1, row_state.shape[2], row_state.shape[3])
        flat_coordinates = coordinate_features.reshape(
            -1,
            coordinate_features.shape[2],
            coordinate_features.shape[3],
        )
        flat_anchor = baseline_x.reshape(-1, baseline_x.shape[2])
        for name, adapter in adapters.items():
            delta = adapter(
                flat_state,
                flat_coordinates,
                flat_anchor,
                input_w=input_w,
            ).reshape_as(baseline_x)
            corrected_by_probe[name] = (
                baseline_x + delta
            ).clamp(0.0, float(input_w - 1.0))

        for image_index, (target, match) in enumerate(zip(targets, matches)):
            gt_x = target["x_rows"].float()
            valid = target["valid_mask"].bool()
            group0_assignment = _group_zero_assignments(
                match,
                group_size=group_size,
            )
            base_candidates = baseline_x[image_index, :group_size]
            corrected_candidates = {
                name: values[image_index, :group_size]
                for name, values in corrected_by_probe.items()
            }
            for gt_index in range(int(gt_x.shape[0])):
                if int(valid[gt_index].sum()) < 5:
                    continue
                gt_lanes += 1
                before = _best_iou(
                    base_candidates,
                    gt_x[gt_index],
                    valid[gt_index],
                    line_width=line_width,
                )
                buckets = ["all"]
                buckets.append(
                    "baseline_hit_050"
                    if before >= 0.5
                    else "baseline_miss_050"
                )
                for name in PROBE_MODES:
                    after = _best_iou(
                        corrected_candidates[name],
                        gt_x[gt_index],
                        valid[gt_index],
                        line_width=line_width,
                    )
                    for bucket in buckets:
                        geometry[name][bucket].update(before, after)

                    pred_index = group0_assignment.get(gt_index)
                    if pred_index is not None:
                        assigned[name].update(
                            baseline_x[image_index, pred_index],
                            corrected_by_probe[name][image_index, pred_index],
                            gt_x[gt_index],
                            valid[gt_index],
                        )
        images_seen += int(images.shape[0])

    summaries = {
        name: {
            "raw_group0_geometry": {
                bucket: stats.summary()
                for bucket, stats in geometry[name].items()
            },
            "assigned_query_rows": assigned[name].summary(),
        }
        for name in PROBE_MODES
    }
    state_miss = summaries["state_only"]["raw_group0_geometry"][
        "baseline_miss_050"
    ]
    combined_miss = summaries["state_attention"]["raw_group0_geometry"][
        "baseline_miss_050"
    ]
    state_all = summaries["state_only"]["raw_group0_geometry"]["all"]
    combined_all = summaries["state_attention"]["raw_group0_geometry"]["all"]
    coordinate_increment = (
        float(combined_miss["after_recall_050"])
        - float(state_miss["after_recall_050"])
    )
    positive_signal = (
        coordinate_increment >= 0.10
        and int(combined_all["lost_050"])
        <= max(1, round(0.02 * int(combined_all["before_hits_050"])))
    )
    return {
        "images": images_seen,
        "gt_lanes": gt_lanes,
        "probes": summaries,
        "controlled_coordinate_signal": {
            "combined_minus_state_miss_recall_050_points": 100.0
            * coordinate_increment,
            "combined_minus_state_all_mean_iou": float(
                combined_all["mean_iou_after"]
            )
            - float(state_all["mean_iou_after"]),
            "positive_gate": bool(positive_signal),
            "gate_definition": (
                "state+attention must improve baseline-miss raw recall@0.50 by "
                "at least 10 points over the equal-capacity state-only control, "
                "while losing no more than 2% of all baseline hits."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg)
    checkpoint_iteration = load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.structured_query_head is None:
        raise ValueError("coordinate adapter probe requires structured_query")
    channels_last = (
        bool(cfg.get("training", {}).get("channels_last", False))
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    head = model.structured_query_head
    layer_count = len(head.layers)
    selected_layers = tuple(int(value) - 1 for value in args.attention_layers)
    if not selected_layers or any(
        value < 0 or value >= layer_count for value in selected_layers
    ):
        raise ValueError(
            f"--attention-layers must be within [1, {layer_count}], "
            f"got {args.attention_layers}"
        )
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("--attention-layers cannot contain duplicates")

    num_instances = int(head.num_instances)
    num_groups = int(head.num_groups)
    group_size = num_instances // max(num_groups, 1)
    input_w = float(model_cfg.get("input_w", head.input_w))
    num_heads = int(head.layers[0].cross_attn.num_heads)
    coordinate_dim = len(selected_layers) * num_heads * 4
    adapters = {
        name: CoordinateResidualAdapter(
            state_dim=int(head.dim),
            coordinate_dim=coordinate_dim,
            hidden_dim=int(args.hidden_dim),
            num_rows=int(head.num_rows),
            max_update_px=float(args.max_update_px),
            use_state=mode[0],
            use_attention=mode[1],
        ).to(device)
        for name, mode in PROBE_MODES.items()
    }
    if not args.load_probes:
        reference_state = adapters["anchor_only"].state_dict()
        for name in tuple(PROBE_MODES)[1:]:
            adapters[name].load_state_dict(reference_state, strict=True)
    parameter_counts = {
        name: sum(parameter.numel() for parameter in adapter.parameters())
        for name, adapter in adapters.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Adapter parameter counts differ: {parameter_counts}")

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    matcher = build_matcher(cfg)
    history: list[dict[str, Any]] = []
    trained_steps = int(args.train_steps)
    if args.load_probes:
        saved = torch.load(args.load_probes, map_location="cpu")
        for name, adapter in adapters.items():
            adapter.load_state_dict(saved["adapters"][name], strict=True)
        trained_steps = int(saved.get("args", {}).get("train_steps", trained_steps))
    elif int(args.train_steps) > 0:
        train_loader = build_dataloader(cfg, split="train", training=True)
        optimizer = torch.optim.AdamW(
            [
                parameter
                for adapter in adapters.values()
                for parameter in adapter.parameters()
            ],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        train_iterator = iter(train_loader)
        for adapter in adapters.values():
            adapter.train()
        progress = tqdm(
            range(1, int(args.train_steps) + 1),
            desc="coordinate-adapter train",
            ncols=100,
        )
        for step in progress:
            try:
                images, targets, _metas = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, targets, _metas = next(train_iterator)
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(
                    torch.channels_last
                    if channels_last
                    else torch.contiguous_format
                ),
            )
            targets = nested_to_device(targets, device)
            outputs, row_state, coordinate_features, _stages = (
                _extract_frozen_outputs(
                    model=model,
                    images=images,
                    selected_layers=selected_layers,
                    input_w=input_w,
                    amp_dtype=amp_dtype,
                )
            )
            matches = matcher(outputs, targets)
            examples = _matched_examples(
                row_state=row_state,
                coordinate_features=coordinate_features,
                anchor_x=outputs["pred_x_rows"].float(),
                targets=targets,
                matches=matches,
                group_size=group_size,
                group_mode=str(args.train_group_mode),
            )
            if examples is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            metrics: dict[str, dict[str, float]] = {}
            for name, adapter in adapters.items():
                delta = adapter(
                    examples.row_state,
                    examples.coordinate_features,
                    examples.anchor_x,
                    input_w=input_w,
                )
                corrected = (
                    examples.anchor_x + delta
                ).clamp(0.0, float(input_w - 1.0))
                loss, probe_metrics = _adapter_loss(
                    corrected,
                    examples,
                    smooth_l1_beta_px=float(args.smooth_l1_beta_px),
                    max_update_px=float(args.max_update_px),
                    line_width=float(args.line_width),
                    line_iou_loss_weight=float(args.line_iou_loss_weight),
                )
                losses[name] = loss
                metrics[name] = probe_metrics
            total_loss = torch.stack(list(losses.values())).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for adapter in adapters.values()
                    for parameter in adapter.parameters()
                ],
                max_norm=5.0,
            )
            optimizer.step()
            record = {
                "step": step,
                "matched_lane_examples": examples.count,
                "total_loss": float(total_loss.detach()),
                **{
                    f"{name}/{metric}": value
                    for name, values in metrics.items()
                    for metric, value in values.items()
                },
            }
            if (
                step == 1
                or step % int(args.log_interval) == 0
                or step == int(args.train_steps)
            ):
                history.append(record)
                progress.set_postfix(
                    {
                        name: f"{metrics[name]['row_mae_px']:.1f}px"
                        for name in PROBE_MODES
                    }
                )

    if args.save_probes:
        save_path = Path(args.save_probes)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapters": {
                    name: adapter.state_dict()
                    for name, adapter in adapters.items()
                },
                "args": vars(args),
                "parameter_counts": parameter_counts,
            },
            save_path,
        )

    eval_loader = build_dataloader(cfg, split="val", training=False)
    evaluation = _evaluate(
        model=model,
        adapters=adapters,
        matcher=matcher,
        loader=eval_loader,
        device=device,
        channels_last=channels_last,
        amp_dtype=amp_dtype,
        selected_layers=selected_layers,
        input_w=input_w,
        group_size=group_size,
        max_batches=int(args.eval_max_batches),
        line_width=float(args.line_width),
    )
    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "warning": (
            "All detector parameters are frozen, but tiny adapters are fitted on "
            "the CULane training split. Results are causal diagnostic evidence, "
            "not benchmark scores."
        ),
        "hypothesis": (
            "If the equal-capacity state+attention adapter outperforms state-only "
            "on held-out baseline misses without damaging hits, attended x-position "
            "is useful information missing from the production state/output interface."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "seed": int(args.seed),
        "train_steps": trained_steps,
        "train_group_mode": str(args.train_group_mode),
        "attention_layers": [value + 1 for value in selected_layers],
        "coordinate_features_per_row": coordinate_dim,
        "adapter_parameter_counts": parameter_counts,
        "history": history,
        "evaluation": evaluation,
    }
    print(json.dumps(payload, indent=2))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
