from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ProposalRecallStats:
    thresholds: tuple[float, ...] = (0.5,)
    total_gt: int = 0
    hits: dict[float, int] = field(default_factory=dict)
    best_ious: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hits = {float(t): int(self.hits.get(float(t), 0)) for t in self.thresholds}

    def update(self, best_iou: float) -> None:
        self.total_gt += 1
        self.best_ious.append(float(best_iou))
        for threshold in self.thresholds:
            if best_iou >= threshold:
                self.hits[float(threshold)] += 1

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"gt": float(self.total_gt)}
        for threshold in self.thresholds:
            key = f"recall@{threshold:g}"
            out[key] = float(self.hits[float(threshold)]) / float(max(self.total_gt, 1))
        if self.best_ious:
            vals = sorted(self.best_ious)
            out["mean_best_iou"] = float(sum(vals) / len(vals))
            out["median_best_iou"] = float(vals[len(vals) // 2])
            out["p90_best_iou"] = float(vals[min(len(vals) - 1, int(0.9 * (len(vals) - 1)))])
        else:
            out["mean_best_iou"] = 0.0
            out["median_best_iou"] = 0.0
            out["p90_best_iou"] = 0.0
        return out


def collect_prediction_stages(outputs: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    stages: dict[str, dict[str, torch.Tensor]] = {}
    for key in ("s0_geometry_draft", "coarse", "stage1", "stage2", "final"):
        value = outputs.get(key)
        if isinstance(value, dict) and "pred_x_rows" in value:
            stages[key] = value
    if "pred_x_rows" in outputs and not any(name in stages for name in ("final", "stage2")):
        stages["main"] = outputs
    return stages


def select_candidates(
    stage_outputs: dict[str, torch.Tensor],
    batch_index: int,
    top_k: int = 0,
    rank_by: str = "none",
) -> torch.Tensor:
    pred_x = stage_outputs["pred_x_rows"][batch_index]
    if top_k <= 0 or top_k >= pred_x.shape[0]:
        return pred_x
    scores = None
    if rank_by in {"score", "score_quality"} and "exist_logits" in stage_outputs:
        scores = torch.softmax(stage_outputs["exist_logits"][batch_index].float(), dim=-1)[..., 0]
    if rank_by in {"quality", "score_quality"} and "quality_logits" in stage_outputs:
        quality = torch.sigmoid(stage_outputs["quality_logits"][batch_index].float())
        scores = quality if scores is None else scores * quality
    if scores is not None:
        order = torch.argsort(scores, descending=True)[:top_k]
        return pred_x[order]
    return pred_x[:top_k]


def line_iou_against_gt(
    pred_x_rows: torch.Tensor,
    gt_x_rows: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    line_width: float = 30.0,
) -> torch.Tensor:
    valid = gt_valid_mask.bool() & torch.isfinite(gt_x_rows)
    if pred_x_rows.numel() == 0 or not valid.any():
        return pred_x_rows.new_zeros((pred_x_rows.shape[0],))
    radius = float(line_width) * 0.5
    pred = pred_x_rows[:, valid].float()
    gt = gt_x_rows[valid].float().view(1, -1)
    pred_finite = torch.isfinite(pred)
    px1 = pred - radius
    px2 = pred + radius
    gx1 = gt - radius
    gx2 = gt + radius
    overlap = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0.0)
    overlap = overlap.masked_fill(~pred_finite, 0.0)
    union = (2.0 * float(line_width) - overlap).clamp(min=1e-6)
    union = union.masked_fill(~pred_finite, 0.0)
    denom = union.sum(dim=-1).clamp_min(1e-6)
    return overlap.sum(dim=-1) / denom


def update_stage_recall(
    stats: ProposalRecallStats,
    stage_outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    top_k: int = 0,
    rank_by: str = "none",
    line_width: float = 30.0,
    min_valid_rows: int = 5,
) -> None:
    for bi, target in enumerate(targets):
        candidates = select_candidates(stage_outputs, bi, top_k=top_k, rank_by=rank_by)
        gt_x = target["x_rows"].to(device=candidates.device, dtype=candidates.dtype)
        valid = target["valid_mask"].to(device=candidates.device).bool()
        for lane_idx in range(gt_x.shape[0]):
            if int(valid[lane_idx].sum().item()) < int(min_valid_rows):
                continue
            ious = line_iou_against_gt(candidates, gt_x[lane_idx], valid[lane_idx], line_width=line_width)
            best = float(ious.max().item()) if ious.numel() else 0.0
            stats.update(best)
