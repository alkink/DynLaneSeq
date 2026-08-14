from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v20_slot_owned_replacement import (
    slot_candidate_action_valid,
)
from dynlaneseq_eg.tools.v19_semantic_coverage_core import (
    _official_matching_maps,
    source_slot_gt_ownership,
)


def _official_matching_details(
    quality_sets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized exact maximum-total-IoU matching for tiny lane sets.

    Returns matched IoUs, total IoU and a ``[M,G]`` covered-GT value matrix
    whose unmatched entries are zero.  Thresholding that last tensor exactly
    reproduces the evaluator's post-Hungarian TP decision.
    """

    if quality_sets.ndim != 3:
        raise ValueError("quality_sets must have shape [M,G,P]")
    batch, gt_count, prediction_count = quality_sets.shape
    if gt_count == 0 or prediction_count == 0:
        return (
            quality_sets.new_zeros((batch, 0)),
            quality_sets.new_zeros((batch,)),
            quality_sets.new_zeros((batch, gt_count)),
        )
    mode, maps = _official_matching_maps(gt_count, prediction_count)
    maps = maps.to(quality_sets.device)
    if mode == "gt_to_prediction":
        gt_index = torch.arange(gt_count, device=quality_sets.device).view(
            1, gt_count
        )
        values = quality_sets[:, gt_index, maps]
    elif mode == "prediction_to_gt":
        prediction_index = torch.arange(
            prediction_count, device=quality_sets.device
        ).view(1, prediction_count)
        values = quality_sets[:, maps, prediction_index]
    else:
        raise RuntimeError("unexpected empty official matching mode")
    totals = values.sum(dim=-1)
    best_map_index = totals.argmax(dim=-1)
    row = torch.arange(batch, device=quality_sets.device)
    matched = values[row, best_map_index]
    best_map = maps[best_map_index]
    gt_values = quality_sets.new_zeros((batch, gt_count))
    if mode == "gt_to_prediction":
        gt_values = matched
    else:
        gt_values.scatter_(1, best_map, matched)
    return matched, totals[row, best_map_index], gt_values


def _semantic_collision_excess(
    quality_sets: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if quality_sets.shape[1] == 0 or quality_sets.shape[2] == 0:
        return torch.zeros(
            quality_sets.shape[0], dtype=torch.long, device=quality_sets.device
        )
    best_value, best_gt = quality_sets.max(dim=1)
    qualified = best_value > float(threshold)
    rows: list[int] = []
    for item in range(int(quality_sets.shape[0])):
        ids = best_gt[item][qualified[item]]
        rows.append(int(ids.numel() - ids.unique().numel()))
    return torch.tensor(rows, dtype=torch.long, device=quality_sets.device)


def build_one_edit_action_targets(
    quality: torch.Tensor,
    valid: torch.Tensor,
    source_route: torch.Tensor,
    active: torch.Tensor,
    *,
    policy_iou_support_delta: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Create exact official-raster supervision for KEEP + one edit.

    ``quality`` is the cached official-raster IoU matrix ``[G,S,N]`` for the
    4x32 frozen slot-conditioned V7-refined alternatives.  No differentiable
    geometry surrogate participates in these labels.
    """

    if quality.ndim != 3:
        raise ValueError("quality must have shape [G,S,N]")
    gt_count, slots, candidates = quality.shape
    if valid.shape != (slots, candidates):
        raise ValueError("valid must have shape [S,N]")
    if source_route.shape != (slots,) or active.shape != (slots,):
        raise ValueError("source route/activity shape mismatch")
    if float(policy_iou_support_delta) < 0.0:
        raise ValueError("policy IoU support delta must be non-negative")

    quality = quality.detach().float().cpu()
    valid = valid.detach().bool().cpu()
    source_route = source_route.detach().long().cpu()
    active = active.detach().bool().cpu()
    action_valid = slot_candidate_action_valid(
        valid.unsqueeze(0), source_route.unsqueeze(0), active.unsqueeze(0)
    ).squeeze(0)
    action_count = 1 + slots * candidates
    routes = source_route.view(1, slots).expand(action_count, slots).clone()
    flat_slot = torch.arange(slots).view(slots, 1).expand(slots, candidates)
    flat_candidate = torch.arange(candidates).view(1, candidates).expand(
        slots, candidates
    )
    action_rows = torch.arange(slots * candidates)
    routes[1 + action_rows, flat_slot.reshape(-1)] = flat_candidate.reshape(-1)

    active_slots = torch.nonzero(active, as_tuple=False).flatten().tolist()
    if active_slots:
        quality_sets = torch.stack(
            [
                quality[:, slot, routes[:, slot]].transpose(0, 1)
                for slot in active_slots
            ],
            dim=-1,
        )
    else:
        quality_sets = quality.new_zeros((action_count, gt_count, 0))
    matched, total_iou, gt_values = _official_matching_details(quality_sets)
    hit50 = (matched > 0.50).sum(dim=-1).long()
    hit75 = (matched > 0.75).sum(dim=-1).long()
    delta50 = hit50 - hit50[0]
    delta75 = hit75 - hit75[0]
    delta_iou = total_iou - total_iou[0]
    baseline_covered = gt_values[0] > 0.50
    covered = gt_values > 0.50
    abandon = (baseline_covered.unsqueeze(0) & ~covered).any(dim=-1)
    collision = _semantic_collision_excess(quality_sets, 0.50)
    duplicate = collision > collision[0]

    full_valid = torch.cat(
        (torch.ones(1, dtype=torch.bool), action_valid.reshape(-1)), dim=0
    )
    # Invalid actions receive neutral dense labels and zero policy mass.  The
    # action mask, not these placeholder values, controls every loss.
    delta50 = torch.where(full_valid, delta50, torch.zeros_like(delta50))
    delta75 = torch.where(full_valid, delta75, torch.zeros_like(delta75))
    delta_iou = torch.where(full_valid, delta_iou, torch.zeros_like(delta_iou))
    abandon = abandon & full_valid
    duplicate = duplicate & full_valid

    policy_target = torch.zeros(action_count, dtype=torch.float32)
    candidate_ids = torch.nonzero(full_valid, as_tuple=False).flatten()
    candidate_ids = candidate_ids[candidate_ids != 0]
    chosen = torch.empty(0, dtype=torch.long)
    if candidate_ids.numel():
        best_delta50 = int(delta50[candidate_ids].max())
        if best_delta50 > 0:
            chosen = candidate_ids[delta50[candidate_ids] == best_delta50]
            best_delta75 = int(delta75[chosen].max())
            chosen = chosen[delta75[chosen] == best_delta75]
        else:
            neutral = candidate_ids[delta50[candidate_ids] == 0]
            if neutral.numel() and int(delta75[neutral].max()) > 0:
                best_delta75 = int(delta75[neutral].max())
                chosen = neutral[delta75[neutral] == best_delta75]
    if chosen.numel():
        best_iou = float(delta_iou[chosen].max())
        chosen = chosen[
            delta_iou[chosen]
            >= best_iou - float(policy_iou_support_delta)
        ]
        policy_target[chosen] = 1.0 / float(chosen.numel())
    else:
        policy_target[0] = 1.0

    ownership = source_slot_gt_ownership(
        quality, valid, source_route, active
    )
    return {
        "action_valid": action_valid,
        "full_action_valid": full_valid,
        "routes": routes,
        "delta50": delta50,
        "delta75": delta75,
        "delta50_class": (delta50 + 1).clamp(0, 2),
        "delta75_class": (delta75 + 1).clamp(0, 2),
        "delta_iou": delta_iou,
        "duplicate": duplicate,
        "abandon": abandon,
        "policy_target": policy_target,
        "ownership": ownership,
        "source_tp50": hit50[0],
        "source_tp75": hit75[0],
        "best_one_edit_tp50": hit50[full_valid].max(),
        "best_one_edit_tp75": hit75[full_valid].max(),
        "gt_count": torch.tensor(gt_count, dtype=torch.long),
    }
