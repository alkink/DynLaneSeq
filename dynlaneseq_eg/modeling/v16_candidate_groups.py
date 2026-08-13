from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .common import fixed_row_fractions, sort_range_norm


REFERENCE_INPUT_WIDTH = 1600.0


@dataclass(frozen=True)
class V16CandidateGroupConfig:
    """GT-free variable-size groups around deployed V7 proposal anchors."""

    corridor_fraction: float = 0.60
    min_corridor_px: float = 72.0
    max_corridor_px: float = 256.0
    min_common_rows: int = 5
    min_overlap_fraction: float = 0.25

    def validate(self) -> None:
        if not 0.0 < float(self.corridor_fraction) <= 1.0:
            raise ValueError("V16 corridor_fraction must be in (0, 1]")
        if float(self.min_corridor_px) < 0.0:
            raise ValueError("V16 min_corridor_px must be non-negative")
        if float(self.max_corridor_px) < float(self.min_corridor_px):
            raise ValueError("V16 max corridor must be >= min corridor")
        if int(self.min_common_rows) < 1:
            raise ValueError("V16 min_common_rows must be positive")
        if not 0.0 <= float(self.min_overlap_fraction) <= 1.0:
            raise ValueError("V16 min_overlap_fraction must be in [0, 1]")


def _masked_linear_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match torch.quantile(linear) independently over masked final axes."""

    if values.shape != mask.shape:
        values = values.expand_as(mask)
    count = mask.sum(dim=-1)
    ordered = values.masked_fill(~mask, torch.inf).sort(dim=-1).values
    position = (count - 1).clamp_min(0).to(values.dtype) * float(quantile)
    lower = position.floor().long()
    upper = position.ceil().long()
    alpha = position - lower.to(position.dtype)
    lower_value = ordered.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_value = ordered.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    result = torch.lerp(lower_value, upper_value, alpha)
    result = torch.where(count > 0, result, torch.full_like(result, torch.inf))
    return result, count


def _masked_lower_median(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match torch.median for a variable number of valid final-axis values."""

    count = mask.sum(dim=-1)
    ordered = values.masked_fill(~mask, torch.inf).sort(dim=-1).values
    index = ((count - 1).clamp_min(0) // 2).long()
    result = ordered.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    result = torch.where(count > 0, result, torch.full_like(result, torch.inf))
    return result, count


def _masked_weighted_quantile(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    ordered_index = values.masked_fill(~mask, torch.inf).argsort(dim=-1)
    ordered_values = values.gather(-1, ordered_index)
    ordered_mask = mask.gather(-1, ordered_index)
    expanded_weights = weights.expand_as(values)
    ordered_weights = expanded_weights.gather(-1, ordered_index)
    ordered_weights = ordered_weights * ordered_mask.to(values.dtype)
    total = ordered_weights.sum(dim=-1, keepdim=True)
    reached = ordered_weights.cumsum(dim=-1) >= total * float(quantile)
    index = reached.to(torch.int64).argmax(dim=-1)
    selected = ordered_values.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return torch.where(
        total.squeeze(-1) > 0.0,
        selected,
        torch.full_like(selected, torch.inf),
    )


@torch.no_grad()
def build_v16_anchor_candidate_groups(
    *,
    proposal_x_rows: torch.Tensor,
    proposal_range_norm: torch.Tensor,
    candidate_valid: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_active: torch.Tensor,
    input_w: int,
    config: V16CandidateGroupConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Build the exact V16 adaptive Voronoi group mask on the input device.

    Geometry decides only which complete proposal curves may compete.  This
    function never averages coordinates and never pads a group to a fixed K.
    The output mask is disjoint over slots and always retains each active V7
    proposal anchor.
    """

    cfg = config or V16CandidateGroupConfig()
    cfg.validate()
    if proposal_x_rows.ndim != 3:
        raise ValueError("V16 proposal_x_rows must have shape [B,N,R]")
    batch, candidates, rows = proposal_x_rows.shape
    slots = int(anchor_indices.shape[1])
    if tuple(proposal_range_norm.shape) != (batch, candidates, 2):
        raise ValueError("V16 proposal_range_norm must have shape [B,N,2]")
    if tuple(candidate_valid.shape) != (batch, candidates):
        raise ValueError("V16 candidate_valid must have shape [B,N]")
    if tuple(anchor_indices.shape) != (batch, slots):
        raise ValueError("V16 anchor_indices must have shape [B,S]")
    if tuple(anchor_active.shape) != (batch, slots):
        raise ValueError("V16 anchor_active must have shape [B,S]")

    device = proposal_x_rows.device
    x = proposal_x_rows.detach().float()
    finite_x = torch.isfinite(x)
    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=float(max(int(input_w) - 1, 1)),
        neginf=0.0,
    )
    lane_range = sort_range_norm(proposal_range_norm.detach().float())
    valid_candidate = candidate_valid.detach().bool()
    row_y = fixed_row_fractions(
        rows,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, rows)
    proposal_visible = (
        (row_y >= lane_range[..., :1])
        & (row_y <= lane_range[..., 1:])
        & finite_x
        & valid_candidate.unsqueeze(-1)
    )

    active = anchor_active.detach().bool() & (anchor_indices >= 0)
    safe_anchor = anchor_indices.detach().long().clamp(
        min=0,
        max=max(candidates - 1, 0),
    )
    # An active V7 anchor must refer to a valid proposal.  Inactive slots use
    # a private numerical fallback but never own a candidate.
    anchor_is_valid = valid_candidate.gather(1, safe_anchor)
    if bool((active & ~anchor_is_valid).any()):
        raise ValueError("V16 active anchor refers to an invalid proposal")
    duplicate = (
        safe_anchor.unsqueeze(-1) == safe_anchor.unsqueeze(-2)
    ) & active.unsqueeze(-1) & active.unsqueeze(-2)
    duplicate &= ~torch.eye(slots, device=device, dtype=torch.bool).view(
        1, slots, slots
    )
    if bool(duplicate.any()):
        raise ValueError("V16 requires unique active V7 proposal anchors")

    anchor_x = x.gather(
        1,
        safe_anchor.unsqueeze(-1).expand(-1, -1, rows),
    )
    anchor_visible = proposal_visible.gather(
        1,
        safe_anchor.unsqueeze(-1).expand(-1, -1, rows),
    )
    common = anchor_visible[:, :, None, :] & proposal_visible[:, None, :, :]
    common_count = common.sum(dim=-1)
    anchor_count = anchor_visible.sum(dim=-1)
    proposal_count = proposal_visible.sum(dim=-1)
    shorter_count = torch.minimum(
        anchor_count.unsqueeze(-1),
        proposal_count.unsqueeze(1),
    )
    overlap = common_count.to(torch.float32) / shorter_count.clamp_min(1).to(
        torch.float32
    )
    gap = (anchor_x[:, :, None, :] - x[:, None, :, :]).abs()
    perspective = (0.10 + 0.90 * row_y.pow(3.0)).view(1, 1, 1, rows)
    weighted_median = _masked_weighted_quantile(
        gap, common, perspective, 0.50
    )
    weighted_q90 = _masked_weighted_quantile(
        gap, common, perspective, 0.90
    )

    pair_y = row_y.view(1, 1, 1, rows).expand_as(common)
    common_y_q65, _ = _masked_linear_quantile(pair_y, common, 0.65)
    lower_cut = torch.maximum(
        common_y_q65,
        common_y_q65.new_tensor(0.55),
    )
    lower_mask = common & (pair_y >= lower_cut.unsqueeze(-1))
    lower_median, lower_count = _masked_lower_median(gap, lower_mask)
    max_common_y = pair_y.masked_fill(~common, -torch.inf).amax(dim=-1)
    lower_available = (lower_count >= 3) & (max_common_y >= 0.55)
    lower = torch.where(lower_available, lower_median, weighted_median)

    bottom_index = pair_y.masked_fill(~common, -torch.inf).topk(
        k=min(3, rows), dim=-1
    ).indices
    bottom_values = gap.gather(-1, bottom_index)
    bottom = bottom_values.median(dim=-1).values
    bottom = torch.where(common_count >= min(3, rows), bottom, weighted_q90)

    distance = (
        0.40 * weighted_median
        + 0.20 * weighted_q90
        + 0.25 * lower
        + 0.15 * bottom
    )
    eligible = (
        active.unsqueeze(-1)
        & valid_candidate.unsqueeze(1)
        & (common_count >= int(cfg.min_common_rows))
        & (overlap >= float(cfg.min_overlap_fraction))
        & torch.isfinite(distance)
    )
    distance = distance.masked_fill(~eligible, torch.inf)

    # The exact deployed proposal always belongs to its own active group.
    distance.scatter_(
        2,
        safe_anchor.unsqueeze(-1),
        torch.where(
            active,
            torch.zeros_like(safe_anchor, dtype=distance.dtype),
            torch.full_like(safe_anchor, torch.inf, dtype=distance.dtype),
        ).unsqueeze(-1),
    )

    anchor_distance = distance.gather(
        2,
        safe_anchor.unsqueeze(1).expand(-1, slots, -1),
    )
    other = active.unsqueeze(1).expand(-1, slots, -1).clone()
    other &= ~torch.eye(slots, device=device, dtype=torch.bool).view(
        1, slots, slots
    )
    nearest_other = anchor_distance.masked_fill(~other, torch.inf).amin(dim=-1)
    scale = float(input_w) / REFERENCE_INPUT_WIDTH
    corridor = nearest_other * float(cfg.corridor_fraction)
    corridor = corridor.clamp(
        min=float(cfg.min_corridor_px) * scale,
        max=float(cfg.max_corridor_px) * scale,
    )
    corridor = torch.where(
        torch.isfinite(nearest_other),
        corridor,
        corridor.new_full(corridor.shape, float(cfg.max_corridor_px) * scale),
    )
    corridor = torch.where(active, corridor, torch.zeros_like(corridor))

    best_distance, owner = distance.amin(dim=1), distance.argmin(dim=1)
    owner = torch.where(
        torch.isfinite(best_distance), owner, torch.full_like(owner, -1)
    )
    # Resolve exact-anchor ties in favor of the slot that deployed that ID.
    for slot in range(slots):
        is_anchor = F.one_hot(
            safe_anchor[:, slot], num_classes=candidates
        ).bool()
        is_anchor &= active[:, slot : slot + 1]
        owner = torch.where(is_anchor, owner.new_full(owner.shape, slot), owner)
        best_distance = torch.where(
            is_anchor, torch.zeros_like(best_distance), best_distance
        )

    safe_owner = owner.clamp(min=0)
    owned_corridor = corridor.gather(1, safe_owner)
    assigned = (owner >= 0) & (best_distance <= owned_corridor)
    group_mask = F.one_hot(safe_owner, num_classes=slots).permute(0, 2, 1).bool()
    group_mask &= assigned.unsqueeze(1)
    group_mask &= active.unsqueeze(-1)
    # Reassert anchor retention after corridor filtering.
    for slot in range(slots):
        anchor_column = F.one_hot(
            safe_anchor[:, slot], num_classes=candidates
        ).bool()
        group_mask[:, :, :] &= ~(
            anchor_column[:, None, :] & active[:, slot : slot + 1, None]
        )
        group_mask[:, slot, :] |= anchor_column & active[:, slot : slot + 1]

    if bool((group_mask.sum(dim=1) > 1).any()):
        raise RuntimeError("V16 candidate groups are not disjoint")
    retained = group_mask.gather(2, safe_anchor.unsqueeze(-1)).squeeze(-1)
    if bool((active & ~retained).any()):
        raise RuntimeError("V16 failed to retain an active V7 anchor")

    return {
        "group_mask": group_mask,
        "anchor_distance_px": distance,
        "corridor_px": corridor,
        "group_size": group_mask.sum(dim=-1),
        "candidate_owner": torch.where(
            group_mask.any(dim=1),
            group_mask.to(torch.int64).argmax(dim=1),
            owner.new_full(owner.shape, -1),
        ),
        "proposal_visible": proposal_visible,
        "active_anchor": active,
    }
