from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.nn import functional as F

from .v22_lane_field import sample_lane_field_rows, soft_range_weights


def _weighted_row_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.shape != weight.shape:
        raise ValueError("weighted row mean shape mismatch")
    return (value.float() * weight.float()).sum(dim=-1) / weight.float().sum(
        dim=-1
    ).clamp_min(1.0)


@torch.no_grad()
def candidate_field_component_scores(
    outputs: dict[str, torch.Tensor],
    *,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    input_w: int,
) -> dict[str, torch.Tensor]:
    """Score complete candidate curves by each Stage-A field head directly.

    Unlike the original Stage-A interface, this samples every candidate curve
    itself.  All three scores have a fixed, scale-defined interpretation:

    * ``centerline`` and ``support`` are mean log probabilities;
    * ``distance`` is negative mean absolute normalized displacement.

    No learned calibration, fitted scalar, threshold, or validation sweep is
    involved.
    """

    if candidate_x.ndim != 4 or candidate_range.ndim != 4:
        raise ValueError("candidate field scoring expects [B,S,K,R] curves")
    batch, slots, choices, rows = candidate_x.shape
    if candidate_range.shape != (batch, slots, choices, 2):
        raise ValueError("candidate range shape mismatch")
    if candidate_valid.shape != (batch, slots, choices):
        raise ValueError("candidate valid shape mismatch")
    flat_x = candidate_x.reshape(batch, slots * choices, rows)

    center = sample_lane_field_rows(
        outputs["centerline_logits"].float(), flat_x, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)
    distance = sample_lane_field_rows(
        torch.tanh(outputs["distance_raw"].float()), flat_x, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)
    support = sample_lane_field_rows(
        outputs["support_logits"].float(), flat_x, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)

    range_weight = soft_range_weights(candidate_range, rows=rows)
    coordinate_valid = (
        torch.isfinite(candidate_x)
        & (candidate_x >= 0.0)
        & (candidate_x < float(input_w))
    )
    weight = range_weight * coordinate_valid.float()
    raster_valid = candidate_valid.bool() & (weight.sum(dim=-1) >= 3.0)
    floor = -1.0e4
    scores = {
        "centerline": _weighted_row_mean(F.logsigmoid(center), weight),
        "distance": _weighted_row_mean(-distance.abs(), weight),
        "support": _weighted_row_mean(F.logsigmoid(support), weight),
    }
    for name in tuple(scores):
        scores[name] = scores[name].masked_fill(~raster_valid, floor)
    scores["valid"] = raster_valid
    return scores


def equal_rank_ensemble(
    scores: Iterable[torch.Tensor], valid: torch.Tensor
) -> torch.Tensor:
    """Average within-shortlist ranks without fitting component scales."""

    values = tuple(value.float() for value in scores)
    if not values:
        raise ValueError("rank ensemble requires at least one score")
    if any(value.shape != valid.shape for value in values):
        raise ValueError("rank ensemble shape mismatch")
    choices = int(valid.shape[-1])
    if choices <= 1:
        return torch.zeros_like(values[0]).masked_fill(~valid.bool(), -1.0e4)
    rank_values = []
    ordinal = torch.arange(
        choices, device=valid.device, dtype=torch.long
    ).view(*([1] * (valid.ndim - 1)), choices)
    for value in values:
        order = torch.argsort(
            value.masked_fill(~valid.bool(), -1.0e4),
            dim=-1,
            descending=True,
            stable=True,
        )
        ranks = torch.empty_like(order).scatter_(-1, order, ordinal.expand_as(order))
        rank_values.append(
            (float(choices - 1) - ranks.float()) / float(choices - 1)
        )
    result = torch.stack(rank_values, dim=0).mean(dim=0)
    return result.masked_fill(~valid.bool(), -1.0e4)


@torch.no_grad()
def correct_curves_from_distance(
    outputs: dict[str, torch.Tensor],
    *,
    x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    input_w: int,
    distance_limit_px: float,
    steps: int,
) -> list[torch.Tensor]:
    """Apply exactly ``steps`` deterministic Stage-A distance updates.

    The update is the field's natural closed-form use and intentionally
    matches the old interface's support gating:

    ``x <- x + sigmoid(support(x)) * tanh(distance(x)) * limit``.
    """

    if steps < 1:
        raise ValueError("distance correction requires at least one step")
    if x_rows.ndim < 3 or range_norm.shape != (*x_rows.shape[:-1], 2):
        raise ValueError("distance correction curve/range shape mismatch")
    batch, rows = int(x_rows.shape[0]), int(x_rows.shape[-1])
    prefix = tuple(int(value) for value in x_rows.shape[1:-1])
    curves = int(math.prod(prefix))
    current = x_rows.float().reshape(batch, curves, rows)
    original = current.clone()
    ranges = range_norm.float().reshape(batch, curves, 2)
    row_weight = soft_range_weights(ranges, rows=rows)
    visible = row_weight >= 0.5
    results: list[torch.Tensor] = []
    for _ in range(int(steps)):
        distance = sample_lane_field_rows(
            torch.tanh(outputs["distance_raw"].float())
            * float(distance_limit_px),
            current,
            input_w=input_w,
        )[..., 0]
        support = sample_lane_field_rows(
            outputs["support_logits"].float(), current, input_w=input_w
        )[..., 0].sigmoid()
        updated = (current + support * distance).clamp(
            0.0, float(input_w - 1)
        )
        coordinate_valid = torch.isfinite(current) & (current >= 0.0) & (
            current < float(input_w)
        )
        current = torch.where(visible & coordinate_valid, updated, current)
        results.append(current.reshape(batch, *prefix, rows).clone())
    # Preserve non-finite sentinel coordinates exactly outside visible rows.
    invalid_original = ~torch.isfinite(original)
    if bool(invalid_original.any()):
        for index in range(len(results)):
            flat = results[index].reshape(batch, curves, rows)
            flat[invalid_original] = original[invalid_original]
            results[index] = flat.reshape(batch, *prefix, rows)
    return results


@torch.no_grad()
def source_seeded_field_path(
    outputs: dict[str, torch.Tensor],
    *,
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    input_w: int,
    distance_limit_px: float,
    offset_step_px: float = 4.0,
    transition_scale_px: float = 8.0,
) -> torch.Tensor:
    """Extract a fixed source-relative continuous path with Viterbi DP.

    This is a training-free diagnostic, not a tuned deployment decoder.  The
    state grid is fixed to ``[-distance_limit, +distance_limit]`` around V7.
    Equal, dimensionless terms reward center/support probability, zero signed
    residual, proximity to the source, and row-to-row correction continuity.
    """

    if source_x.ndim != 3 or source_range.shape != (*source_x.shape[:2], 2):
        raise ValueError("source path expects [B,S,R] curves and [B,S,2] ranges")
    if offset_step_px <= 0.0 or transition_scale_px <= 0.0:
        raise ValueError("source path scales must be positive")
    batch, slots, rows = source_x.shape
    offsets = torch.arange(
        -float(distance_limit_px),
        float(distance_limit_px) + 0.5 * float(offset_step_px),
        float(offset_step_px),
        device=source_x.device,
        dtype=torch.float32,
    )
    choices = int(offsets.numel())
    positions = source_x.float().unsqueeze(2) + offsets.view(1, 1, choices, 1)
    flat = positions.reshape(batch, slots * choices, rows)
    center = sample_lane_field_rows(
        outputs["centerline_logits"].float(), flat, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)
    distance = sample_lane_field_rows(
        torch.tanh(outputs["distance_raw"].float()), flat, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)
    support = sample_lane_field_rows(
        outputs["support_logits"].float(), flat, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)

    emission = (
        F.logsigmoid(center)
        + F.logsigmoid(support)
        - distance.abs()
        - offsets.abs().view(1, 1, choices, 1) / float(distance_limit_px)
    ).permute(0, 1, 3, 2)
    position_valid = (
        torch.isfinite(positions)
        & (positions >= 0.0)
        & (positions < float(input_w))
    ).permute(0, 1, 3, 2)
    emission = emission.masked_fill(~position_valid, -1.0e4)

    row_axis = torch.linspace(
        0.0, 1.0, rows, device=source_x.device, dtype=torch.float32
    ).view(1, 1, rows)
    row_visible = (
        (row_axis >= source_range[..., 0].float().unsqueeze(-1))
        & (row_axis <= source_range[..., 1].float().unsqueeze(-1))
        & torch.isfinite(source_x)
        & (source_x >= 0.0)
        & (source_x < float(input_w))
    )
    zero_state = int(offsets.abs().argmin())
    forced = emission.new_full(emission.shape, -1.0e4)
    forced[..., zero_state] = 0.0
    emission = torch.where(row_visible.unsqueeze(-1), emission, forced)

    transition = -(
        offsets.view(choices, 1) - offsets.view(1, choices)
    ).abs() / float(transition_scale_px)
    state = emission[:, :, 0]
    backpointers: list[torch.Tensor] = []
    for row in range(1, rows):
        previous = state.unsqueeze(-1) + transition.view(1, 1, choices, choices)
        best, back = previous.max(dim=-2)
        state = best + emission[:, :, row]
        backpointers.append(back)
    chosen = state.argmax(dim=-1)
    path_states = [chosen]
    for back in reversed(backpointers):
        chosen = back.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        path_states.append(chosen)
    path_states.reverse()
    state_path = torch.stack(path_states, dim=-1)
    correction = offsets[state_path]
    result = (source_x.float() + correction).clamp(0.0, float(input_w - 1))
    return torch.where(row_visible, result, source_x.float())


@torch.no_grad()
def owned_component_centerline_score(
    outputs: dict[str, torch.Tensor],
    *,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    ownership: torch.Tensor,
    input_w: int,
) -> torch.Tensor:
    """GT-conditioned Voronoi-component upper bound for field utilization.

    This function is diagnostic-only.  It masks every candidate row whose
    nearest valid GT lane is not the source slot's cached owned GT.  It must
    never be described as an inference policy.
    """

    components = candidate_field_component_scores(
        outputs,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=candidate_valid,
        input_w=input_w,
    )
    batch, slots, choices, rows = candidate_x.shape
    flat = candidate_x.reshape(batch, slots * choices, rows)
    center = sample_lane_field_rows(
        outputs["centerline_logits"].float(), flat, input_w=input_w
    )[..., 0].reshape(batch, slots, choices, rows)
    center_log = F.logsigmoid(center)
    range_weight = soft_range_weights(candidate_range, rows=rows)
    coordinate_valid = (
        torch.isfinite(candidate_x)
        & (candidate_x >= 0.0)
        & (candidate_x < float(input_w))
    )
    result = center_log.new_full((batch, slots, choices), -1.0e4)
    off_component_floor = float(math.log(1.0e-6))
    for item, target in enumerate(targets):
        gt_x = target["x_rows"].to(candidate_x.device).float()[..., :rows]
        gt_valid = target["valid_mask"].to(candidate_x.device).bool()[..., :rows]
        gt_valid = (
            gt_valid
            & torch.isfinite(gt_x)
            & (gt_x >= 0.0)
            & (gt_x < float(input_w))
        )
        if int(gt_x.shape[0]) == 0:
            continue
        distance = (
            candidate_x[item].unsqueeze(0) - gt_x[:, None, None, :]
        ).abs()
        distance = distance.masked_fill(~gt_valid[:, None, None, :], float("inf"))
        nearest = distance.argmin(dim=0)
        any_gt = gt_valid.any(dim=0).view(1, 1, rows)
        for slot in range(slots):
            owned = int(ownership[item, slot])
            if owned < 0 or owned >= int(gt_x.shape[0]):
                continue
            owned_rows = (
                (nearest[slot] == owned)
                & gt_valid[owned].view(1, rows)
                & any_gt[0]
            )
            weight = (
                range_weight[item, slot]
                * coordinate_valid[item, slot].float()
            )
            row_score = torch.where(
                owned_rows,
                center_log[item, slot],
                center_log.new_full((choices, rows), off_component_floor),
            )
            value = _weighted_row_mean(row_score, weight)
            result[item, slot] = value
    return result.masked_fill(~components["valid"].bool(), -1.0e4)
