from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping

import torch


def build_discrete_offset_oracle_stage(
    stage: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    match: Mapping[str, torch.Tensor],
    offsets_px: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Move matched proposal rows by the closest available lateral offset."""

    oracle = {
        key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in stage.items()
        if key not in {"official_iou", "official_candidate_valid"}
    }
    pred_x = oracle["pred_x_rows"].float()
    target_x = target["x_rows"].to(device=pred_x.device, dtype=pred_x.dtype)
    valid_mask = target["valid_mask"].to(device=pred_x.device).bool()
    offsets = offsets_px.to(device=pred_x.device, dtype=pred_x.dtype).flatten()
    if offsets.numel() < 1:
        raise ValueError("offsets_px must not be empty")
    pred_idx = match["pred_indices"].to(device=pred_x.device, dtype=torch.long)
    gt_idx = match["gt_indices"].to(device=pred_x.device, dtype=torch.long)
    seen: set[int] = set()
    matched_slots = valid_rows = clamped_rows = 0
    for proposal_id, gt_id in zip(pred_idx.tolist(), gt_idx.tolist()):
        if proposal_id in seen:
            continue
        seen.add(proposal_id)
        mask = valid_mask[gt_id]
        if not bool(mask.any()):
            continue
        delta = target_x[gt_id] - pred_x[proposal_id]
        nearest_idx = (delta.unsqueeze(-1) - offsets.view(1, -1)).abs().argmin(dim=-1)
        nearest_delta = offsets[nearest_idx]
        pred_x[proposal_id, mask] = pred_x[proposal_id, mask] + nearest_delta[mask]
        matched_slots += 1
        valid_rows += int(mask.sum().item())
        clamped_rows += int(((delta < offsets.min()) | (delta > offsets.max()))[mask].sum().item())
    oracle["pred_x_rows"] = pred_x
    oracle["quality_pred_x_rows"] = pred_x
    return oracle, {
        "matched_slots": matched_slots,
        "valid_rows": valid_rows,
        "clamped_rows": clamped_rows,
    }


def matched_index_pairs(
    center_x: torch.Tensor,
    target: Mapping[str, torch.Tensor],
    match: Mapping[str, torch.Tensor],
    *,
    best_per_gt: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matched indices, optionally keeping the lowest-MAE proposal per GT."""

    pred_idx = match["pred_indices"].to(device=center_x.device, dtype=torch.long)
    gt_idx = match["gt_indices"].to(device=center_x.device, dtype=torch.long)
    if pred_idx.numel() == 0 or not best_per_gt:
        return pred_idx, gt_idx

    gt_x = target["x_rows"].to(device=center_x.device, dtype=center_x.dtype)
    valid_mask = target["valid_mask"].to(device=center_x.device).bool()
    keep: list[int] = []
    for gt_id in torch.unique(gt_idx, sorted=True).tolist():
        positions = torch.nonzero(gt_idx == int(gt_id), as_tuple=False).flatten()
        mask = valid_mask[int(gt_id)]
        if not bool(mask.any()):
            keep.append(int(positions[0]))
            continue
        candidates = center_x[pred_idx[positions]][:, mask]
        errors = (candidates - gt_x[int(gt_id), mask].unsqueeze(0)).abs().mean(dim=1)
        keep.append(int(positions[int(errors.argmin().item())]))
    keep_idx = torch.tensor(keep, device=center_x.device, dtype=torch.long)
    return pred_idx[keep_idx], gt_idx[keep_idx]


class OnlinePearson:
    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1)
        y = y.detach().float().reshape(-1)
        if x.numel() == 0:
            return
        self.count += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_x2 += float(x.square().sum().item())
        self.sum_y2 += float(y.square().sum().item())
        self.sum_xy += float((x * y).sum().item())

    def value(self) -> float:
        if self.count < 2:
            return 0.0
        n = float(self.count)
        covariance = self.sum_xy - (self.sum_x * self.sum_y / n)
        variance_x = self.sum_x2 - (self.sum_x * self.sum_x / n)
        variance_y = self.sum_y2 - (self.sum_y * self.sum_y / n)
        denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
        return covariance / denominator if denominator > 1e-12 else 0.0


class ActiveCorridorAccumulator:
    """Aggregate row- and lane-level Active Corridor behavior without duplicates."""

    def __init__(self, bin_edges: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 32.0, float("inf"))):
        if len(bin_edges) < 2:
            raise ValueError("bin_edges must contain at least two values")
        self.bin_edges = tuple(float(value) for value in bin_edges)
        self.rows = 0
        self.lanes = 0
        self.sums: dict[str, float] = {}
        self.correlation = OnlinePearson()
        self.bins = [self._empty_bin() for _ in range(len(self.bin_edges) - 1)]
        self.interventions: dict[str, dict[str, float]] = {}

    @staticmethod
    def _empty_bin() -> dict[str, float]:
        return {
            "count": 0.0,
            "coarse_error": 0.0,
            "active_error": 0.0,
            "final_error": 0.0,
            "oracle_error": 0.0,
            "active_improved": 0.0,
            "final_improved": 0.0,
        }

    def _add(self, name: str, value: torch.Tensor | float) -> None:
        number = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        self.sums[name] = self.sums.get(name, 0.0) + number

    def update(
        self,
        *,
        center_x: torch.Tensor,
        pred_delta: torch.Tensor,
        final_x: torch.Tensor,
        target_x: torch.Tensor,
        valid_mask: torch.Tensor,
        offsets: torch.Tensor,
        logits: torch.Tensor,
        interventions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        center_x = center_x.detach().float()
        pred_delta = pred_delta.detach().float()
        final_x = final_x.detach().float()
        target_x = target_x.detach().float()
        valid = valid_mask.detach().bool()
        offsets = offsets.detach().float().to(device=center_x.device)
        logits = logits.detach().float()
        if not bool(valid.any()):
            return

        target_delta = target_x - center_x
        active_x = center_x + pred_delta
        coarse_error = (center_x - target_x).abs()
        active_error = (active_x - target_x).abs()
        final_error = (final_x - target_x).abs()
        nearest_idx = (target_delta.unsqueeze(-1) - offsets.view(1, 1, -1)).abs().argmin(dim=-1)
        oracle_delta = offsets[nearest_idx]
        oracle_error = (center_x + oracle_delta - target_x).abs()
        probs = torch.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1)
        pred_offset = offsets[pred_idx]
        center_idx = int(offsets.abs().argmin().item())
        unique_offsets = torch.unique(offsets, sorted=True)
        if unique_offsets.numel() > 1:
            one_bin_px = (unique_offsets[1:] - unique_offsets[:-1]).abs().median()
        else:
            one_bin_px = offsets.new_tensor(0.0)

        row_count = int(valid.sum().item())
        self.rows += row_count
        for name, values in {
            "target_abs_delta": target_delta.abs(),
            "pred_abs_delta": pred_delta.abs(),
            "coarse_error": coarse_error,
            "active_error": active_error,
            "final_error": final_error,
            "oracle_discrete_error": oracle_error,
            "active_improved": (active_error < coarse_error).float(),
            "final_improved": (final_error < coarse_error).float(),
            "active_worsened": (active_error > coarse_error).float(),
            "final_worsened": (final_error > coarse_error).float(),
            "corridor_covered": ((target_delta >= offsets.min()) & (target_delta <= offsets.max())).float(),
            "offset_top1_correct": ((pred_offset - oracle_delta).abs() < 1e-4).float(),
            "offset_within_one_bin": ((pred_offset - oracle_delta).abs() <= one_bin_px + 1e-4).float(),
            "offset_entropy": -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1),
            "offset_max_prob": probs.max(dim=-1).values,
            "offset_center_prob": probs[..., center_idx],
        }.items():
            self._add(name, values[valid].sum())

        directional = valid & (target_delta.abs() >= 4.0)
        self._add("directional_count", directional.sum())
        if bool(directional.any()):
            self._add("directional_correct", (torch.sign(pred_delta) == torch.sign(target_delta))[directional].sum())
        self.correlation.update(target_delta[valid], pred_delta[valid])

        lane_valid = valid.sum(dim=1) > 0
        if bool(lane_valid.any()):
            denom = valid.sum(dim=1).clamp_min(1).float()
            coarse_lane = (coarse_error * valid).sum(dim=1) / denom
            active_lane = (active_error * valid).sum(dim=1) / denom
            final_lane = (final_error * valid).sum(dim=1) / denom
            self.lanes += int(lane_valid.sum().item())
            self._add("active_lane_improved", (active_lane < coarse_lane)[lane_valid].sum())
            self._add("final_lane_improved", (final_lane < coarse_lane)[lane_valid].sum())

        abs_target = target_delta.abs()
        for index, bucket in enumerate(self.bins):
            low, high = self.bin_edges[index], self.bin_edges[index + 1]
            bin_mask = valid & (abs_target >= low) & (abs_target < high)
            count = int(bin_mask.sum().item())
            if count == 0:
                continue
            bucket["count"] += count
            bucket["coarse_error"] += float(coarse_error[bin_mask].sum().item())
            bucket["active_error"] += float(active_error[bin_mask].sum().item())
            bucket["final_error"] += float(final_error[bin_mask].sum().item())
            bucket["oracle_error"] += float(oracle_error[bin_mask].sum().item())
            bucket["active_improved"] += float((active_error < coarse_error)[bin_mask].sum().item())
            bucket["final_improved"] += float((final_error < coarse_error)[bin_mask].sum().item())

        for mode, (shuffled_delta, shuffled_final_x) in (interventions or {}).items():
            shuffled_delta = shuffled_delta.detach().float()
            shuffled_final_x = shuffled_final_x.detach().float()
            shuffled_active_error = (center_x + shuffled_delta - target_x).abs()
            shuffled_final_error = (shuffled_final_x - target_x).abs()
            stats = self.interventions.setdefault(
                mode,
                {
                    "rows": 0.0,
                    "active_error": 0.0,
                    "final_error": 0.0,
                    "pred_delta_change": 0.0,
                    "final_x_change": 0.0,
                    "normal_active_better": 0.0,
                    "normal_final_better": 0.0,
                },
            )
            stats["rows"] += row_count
            stats["active_error"] += float(shuffled_active_error[valid].sum().item())
            stats["final_error"] += float(shuffled_final_error[valid].sum().item())
            stats["pred_delta_change"] += float((pred_delta - shuffled_delta).abs()[valid].sum().item())
            stats["final_x_change"] += float((final_x - shuffled_final_x).abs()[valid].sum().item())
            stats["normal_active_better"] += float((active_error < shuffled_active_error)[valid].sum().item())
            stats["normal_final_better"] += float((final_error < shuffled_final_error)[valid].sum().item())

    def as_dict(self) -> dict[str, object]:
        row_denom = max(self.rows, 1)
        lane_denom = max(self.lanes, 1)
        directional_count = max(self.sums.get("directional_count", 0.0), 1.0)
        out: dict[str, object] = {
            "matched_lanes": self.lanes,
            "valid_rows": self.rows,
            "corridor_coverage": self.sums.get("corridor_covered", 0.0) / row_denom,
            "coarse_mae_px": self.sums.get("coarse_error", 0.0) / row_denom,
            "active_mae_px": self.sums.get("active_error", 0.0) / row_denom,
            "final_mae_px": self.sums.get("final_error", 0.0) / row_denom,
            "oracle_discrete_mae_px": self.sums.get("oracle_discrete_error", 0.0) / row_denom,
            "active_row_improvement_rate": self.sums.get("active_improved", 0.0) / row_denom,
            "final_row_improvement_rate": self.sums.get("final_improved", 0.0) / row_denom,
            "active_row_worsening_rate": self.sums.get("active_worsened", 0.0) / row_denom,
            "final_row_worsening_rate": self.sums.get("final_worsened", 0.0) / row_denom,
            "active_lane_improvement_rate": self.sums.get("active_lane_improved", 0.0) / lane_denom,
            "final_lane_improvement_rate": self.sums.get("final_lane_improved", 0.0) / lane_denom,
            "offset_target_pearson": self.correlation.value(),
            "offset_direction_accuracy": self.sums.get("directional_correct", 0.0) / directional_count,
            "offset_top1_accuracy": self.sums.get("offset_top1_correct", 0.0) / row_denom,
            "offset_within_one_bin_accuracy": self.sums.get("offset_within_one_bin", 0.0) / row_denom,
            "offset_entropy": self.sums.get("offset_entropy", 0.0) / row_denom,
            "offset_max_probability": self.sums.get("offset_max_prob", 0.0) / row_denom,
            "offset_center_probability": self.sums.get("offset_center_prob", 0.0) / row_denom,
        }

        by_bin = []
        for index, bucket in enumerate(self.bins):
            count = max(bucket["count"], 1.0)
            high = self.bin_edges[index + 1]
            by_bin.append(
                {
                    "range_px": f"[{self.bin_edges[index]:g},{high:g})",
                    "rows": int(bucket["count"]),
                    "coarse_mae_px": bucket["coarse_error"] / count,
                    "active_mae_px": bucket["active_error"] / count,
                    "final_mae_px": bucket["final_error"] / count,
                    "oracle_discrete_mae_px": bucket["oracle_error"] / count,
                    "active_improvement_rate": bucket["active_improved"] / count,
                    "final_improvement_rate": bucket["final_improved"] / count,
                }
            )
        out["by_target_abs_delta"] = by_bin

        intervention_out = {}
        for mode, stats in self.interventions.items():
            count = max(stats["rows"], 1.0)
            intervention_out[mode] = {
                "shuffled_active_mae_px": stats["active_error"] / count,
                "shuffled_final_mae_px": stats["final_error"] / count,
                "pred_delta_mean_abs_change_px": stats["pred_delta_change"] / count,
                "final_x_mean_abs_change_px": stats["final_x_change"] / count,
                "normal_active_better_rate": stats["normal_active_better"] / count,
                "normal_final_better_rate": stats["normal_final_better"] / count,
            }
        out["evidence_interventions"] = intervention_out
        return out
