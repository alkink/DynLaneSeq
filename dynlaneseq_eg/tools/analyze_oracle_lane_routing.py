from __future__ import annotations

import argparse
from contextlib import nullcontext
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
            "Run GT-informed attention-corridor interventions on a frozen structured "
            "decoder. Results are oracle diagnostics, never benchmark numbers."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--corridor-radius-px", type=float, default=16.0)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _best_iou_per_gt(
    candidates: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    line_width: float,
) -> torch.Tensor:
    gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
    valid = target["valid_mask"].to(device=candidates.device).bool()
    values: list[torch.Tensor] = []
    for lane_index in range(int(gt_x.shape[0])):
        if int(valid[lane_index].sum()) < 5:
            continue
        ious = line_iou_against_gt(
            candidates,
            gt_x[lane_index],
            valid[lane_index],
            line_width=float(line_width),
        )
        values.append(ious.max() if ious.numel() else candidates.new_zeros(()))
    return torch.stack(values).float() if values else candidates.new_zeros((0,), dtype=torch.float32)


def _layer_forward_with_mask(
    layer: nn.Module,
    row_tokens: torch.Tensor,
    row_value_features: torch.Tensor,
    row_key_features: torch.Tensor,
    *,
    num_groups: int,
    cross_mask: torch.Tensor | None,
) -> torch.Tensor:
    b, n, r, c = row_tokens.shape
    _, _rv, x_bins, _ = row_value_features.shape
    q = row_tokens.permute(0, 2, 1, 3).reshape(b * r, n, c)
    q_norm = layer.norm_cross(q)
    key = row_key_features.reshape(b * r, x_bins, c)
    value = row_value_features.reshape(b * r, x_bins, c)
    mha_mask = None
    if cross_mask is not None:
        heads = int(layer.cross_attn.num_heads)
        mha_mask = (
            cross_mask.reshape(b * r, n, x_bins)
            .unsqueeze(1)
            .expand(-1, heads, -1, -1)
            .reshape(b * r * heads, n, x_bins)
        )
    q = q + layer.drop(
        layer.cross_attn(
            q_norm,
            key,
            value,
            attn_mask=mha_mask,
            need_weights=False,
        )[0]
    )
    q_norm = layer.norm_inter(q)
    q = q + layer.drop(
        layer._grouped_inter_attention(
            q_norm,
            batch_rows=b * r,
            num_instances=n,
            num_groups=num_groups,
        )
    )
    q = q.view(b, r, n, c).permute(0, 2, 1, 3).contiguous()
    lane_rows = q.reshape(b * n, r, c)
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
    return lane_rows.view(b, n, r, c).contiguous()


def _build_corridor_mask(
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
    endpoint_only: bool,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((batch, rows, instances, x_bins), device=device, dtype=torch.bool)
    bin_centers = (
        torch.arange(x_bins, device=device, dtype=torch.float32) + 0.5
    ) * float(input_w) / float(x_bins)
    for batch_index, match in enumerate(matches):
        pred_ids = match["pred_indices"].tolist()
        gt_ids = match["gt_indices"].tolist()
        target = targets[batch_index]
        x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
        valid_mask = target["valid_mask"].to(device=device).bool()
        for pred_id, gt_id in zip(pred_ids, gt_ids):
            if int(pred_id) >= int(group_size):
                continue
            valid_rows = valid_mask[int(gt_id)].nonzero(as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue
            selected_rows = valid_rows[-1:] if endpoint_only else valid_rows
            for row_id in selected_rows.tolist():
                center_x = x_rows[int(gt_id), int(row_id)]
                allowed = (bin_centers - center_x).abs() <= float(radius_px)
                if not bool(allowed.any()):
                    nearest = int((bin_centers - center_x).abs().argmin())
                    allowed[nearest] = True
                mask[batch_index, int(row_id), int(pred_id)] = ~allowed
    return mask


def _oracle_forward(
    head: nn.Module,
    features: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[dict[str, torch.Tensor]],
    *,
    group_size: int,
    radius_px: float,
    input_w: float,
    endpoint_only: bool,
    mask_every_layer: bool,
) -> dict[str, torch.Tensor]:
    batch = int(features.shape[0])
    instance = head.instance_tokens.weight.to(device=features.device, dtype=features.dtype)
    row = head.row_tokens.weight.to(device=features.device, dtype=features.dtype)
    row_tokens = instance[:, None, :] + row[None, :, :]
    row_tokens = row_tokens.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    row_values, row_keys = head._row_features(features)
    corridor = _build_corridor_mask(
        targets=targets,
        matches=matches,
        batch=batch,
        instances=head.num_instances,
        rows=head.num_rows,
        x_bins=head.evidence_x_bins,
        group_size=group_size,
        input_w=input_w,
        radius_px=radius_px,
        endpoint_only=endpoint_only,
        device=features.device,
    )
    for layer_index, layer in enumerate(head.layers):
        use_mask = corridor if mask_every_layer or layer_index == 0 else None
        row_tokens = _layer_forward_with_mask(
            layer,
            row_tokens,
            row_values,
            row_keys,
            num_groups=head.num_groups,
            cross_mask=use_mask,
        )
    return head._predict_from_row_tokens(row_tokens, instance, include_quality=True)


class TransitionStats:
    def __init__(self) -> None:
        self.gt = 0
        self.base_hit_050 = 0
        self.base_hit_070 = 0
        self.after_hit_050 = 0
        self.after_hit_070 = 0
        self.recovered_050 = 0
        self.recovered_070 = 0
        self.lost_050 = 0
        self.lost_070 = 0
        self.iou_delta_sum = 0.0

    def update(self, before: torch.Tensor, after: torch.Tensor) -> None:
        if before.shape != after.shape:
            raise ValueError("Oracle transition GT shape mismatch")
        self.gt += int(before.numel())
        for threshold in (0.5, 0.7):
            before_hit = before >= threshold
            after_hit = after >= threshold
            suffix = "050" if threshold == 0.5 else "070"
            setattr(self, f"base_hit_{suffix}", getattr(self, f"base_hit_{suffix}") + int(before_hit.sum()))
            setattr(self, f"after_hit_{suffix}", getattr(self, f"after_hit_{suffix}") + int(after_hit.sum()))
            setattr(
                self,
                f"recovered_{suffix}",
                getattr(self, f"recovered_{suffix}") + int((~before_hit & after_hit).sum()),
            )
            setattr(
                self,
                f"lost_{suffix}",
                getattr(self, f"lost_{suffix}") + int((before_hit & ~after_hit).sum()),
            )
        self.iou_delta_sum += float((after - before).sum())

    def summary(self) -> dict[str, float | int]:
        return {
            "gt_lanes": self.gt,
            "base_recall_050": self.base_hit_050 / max(self.gt, 1),
            "after_recall_050": self.after_hit_050 / max(self.gt, 1),
            "net_gain_050_points": 100.0 * (self.after_hit_050 - self.base_hit_050) / max(self.gt, 1),
            "recovered_base_misses_050": self.recovered_050,
            "lost_base_hits_050": self.lost_050,
            "base_recall_070": self.base_hit_070 / max(self.gt, 1),
            "after_recall_070": self.after_hit_070 / max(self.gt, 1),
            "net_gain_070_points": 100.0 * (self.after_hit_070 - self.base_hit_070) / max(self.gt, 1),
            "recovered_base_misses_070": self.recovered_070,
            "lost_base_hits_070": self.lost_070,
            "mean_best_iou_delta": self.iou_delta_sum / max(self.gt, 1),
        }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = args.dataset_root
    dataloader_cfg = cfg.setdefault("dataloader", {})
    dataloader_cfg["eval_batch_size"] = int(args.eval_batch_size)
    dataloader_cfg["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        dataloader_cfg["persistent_workers"] = False
    model_cfg = cfg.setdefault("model", {})
    model_cfg["pretrained_backbone"] = False
    model_cfg["require_pretrained_backbone"] = False
    structured_cfg = model_cfg.get("structured_query", {})
    num_instances = int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 0)))
    num_groups = int(structured_cfg.get("num_groups", 1))
    group_size = num_instances // max(num_groups, 1)
    input_w = float(model_cfg.get("input_w", 800))

    device = torch.device(args.device)
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, strict=False)
    model = model.to(device).eval()
    channels_last = bool(cfg.get("training", {}).get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    matcher = build_matcher(cfg)
    loader = build_dataloader(cfg, split=args.split, training=False)

    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(args.amp_dtype)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    stats = {
        "endpoint_first_layer": TransitionStats(),
        "endpoint_every_layer": TransitionStats(),
        "full_curve_first_layer": TransitionStats(),
    }
    images_seen = 0

    for batch_index, (images, targets, _metas) in enumerate(
        tqdm(loader, ncols=88, desc="oracle lane routing")
    ):
        if args.max_batches > 0 and batch_index >= args.max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last if channels_last else torch.contiguous_format,
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
            baseline = model.structured_query_head(features)
            matches = matcher(baseline, targets)
            interventions = {
                "endpoint_first_layer": _oracle_forward(
                    model.structured_query_head,
                    features,
                    targets,
                    matches,
                    group_size=group_size,
                    radius_px=float(args.corridor_radius_px),
                    input_w=input_w,
                    endpoint_only=True,
                    mask_every_layer=False,
                ),
                "endpoint_every_layer": _oracle_forward(
                    model.structured_query_head,
                    features,
                    targets,
                    matches,
                    group_size=group_size,
                    radius_px=float(args.corridor_radius_px),
                    input_w=input_w,
                    endpoint_only=True,
                    mask_every_layer=True,
                ),
                "full_curve_first_layer": _oracle_forward(
                    model.structured_query_head,
                    features,
                    targets,
                    matches,
                    group_size=group_size,
                    radius_px=float(args.corridor_radius_px),
                    input_w=input_w,
                    endpoint_only=False,
                    mask_every_layer=False,
                ),
            }
        images_seen += int(images.shape[0])
        for sample_index, target in enumerate(targets):
            before = _best_iou_per_gt(
                baseline["pred_x_rows"][sample_index, :group_size].float(),
                target,
                line_width=float(args.line_width),
            )
            for name, outputs in interventions.items():
                after = _best_iou_per_gt(
                    outputs["pred_x_rows"][sample_index, :group_size].float(),
                    target,
                    line_width=float(args.line_width),
                )
                stats[name].update(before, after)

    payload: dict[str, Any] = {
        "diagnostic_only": True,
        "oracle_warning": (
            "GT lanes define attention corridors for already matched group-0 queries. "
            "These are causal interventions on a frozen decoder, not deployable results."
        ),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "images": images_seen,
        "corridor_radius_px": float(args.corridor_radius_px),
        "group_size": group_size,
        "interventions": {name: value.summary() for name, value in stats.items()},
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
