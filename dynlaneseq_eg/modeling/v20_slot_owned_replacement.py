from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_row_fractions, sort_range_norm
from .v18_joint_exact_set_energy import _row_slope


def _gather_candidate(value: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """Gather one candidate per slot from ``[B,S,N,...]`` tensors."""

    if value.ndim < 3 or route.shape != value.shape[:2]:
        raise ValueError("candidate gather shape mismatch")
    safe = route.long().clamp(min=0, max=int(value.shape[2]) - 1)
    suffix = value.shape[3:]
    index = safe.view(*safe.shape, 1, *([1] * len(suffix))).expand(
        *safe.shape, 1, *suffix
    )
    return value.gather(2, index).squeeze(2)


def slot_candidate_action_valid(
    candidate_valid: torch.Tensor,
    source_route: torch.Tensor,
    source_active: torch.Tensor,
) -> torch.Tensor:
    """Return legal one-edit actions without changing cardinality or ID uniqueness."""

    if candidate_valid.ndim != 3:
        raise ValueError("candidate_valid must have shape [B,S,N]")
    batch, slots, candidates = candidate_valid.shape
    if source_route.shape != (batch, slots) or source_active.shape != (
        batch,
        slots,
    ):
        raise ValueError("source route/activity shape mismatch")
    route = source_route.long()
    active = source_active.bool()
    axis = torch.arange(candidates, device=route.device).view(1, 1, candidates)
    same = axis == route.unsqueeze(-1)
    used = (
        (axis.unsqueeze(1) == route.unsqueeze(1).unsqueeze(-1))
        & active.unsqueeze(1).unsqueeze(-1)
    )
    eye = torch.eye(slots, dtype=torch.bool, device=route.device).view(
        1, slots, slots, 1
    )
    used_by_other = (used & ~eye).any(dim=2)
    return (
        candidate_valid.bool()
        & active.unsqueeze(-1)
        & ~same
        & ~used_by_other
    )


def official_raster_candidate_valid(
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    *,
    min_valid_rows: int = 5,
) -> torch.Tensor:
    """Mirror the evaluator's range/finite candidate-validity contract.

    The exact raster target cache excludes curves with fewer than five visible
    rows.  Applying the same inexpensive test in deployment keeps the learned
    action space identical to the cached official-target action space.
    """

    if candidate_x.ndim != 4:
        raise ValueError("candidate_x must have shape [B,S,N,R]")
    if candidate_range.shape != (*candidate_x.shape[:3], 2):
        raise ValueError("candidate range shape mismatch")
    rows = int(candidate_x.shape[-1])
    row_fraction = fixed_row_fractions(
        rows, device=candidate_x.device, dtype=torch.float32
    ).view(1, 1, 1, rows)
    ranges = sort_range_norm(candidate_range.detach().float())
    visible = (
        (row_fraction >= ranges[..., :1])
        & (row_fraction <= ranges[..., 1:])
        & torch.isfinite(candidate_x.detach())
    )
    return visible.sum(dim=-1) >= int(min_valid_rows)


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def complete_curve_relations(
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    reference_x: torch.Tensor,
    reference_range: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    """Describe every candidate curve relative to every current V7 lane.

    Args:
        candidate_x: ``[B,S,N,R]`` slot-conditioned alternatives.
        candidate_range: ``[B,S,N,2]``.
        reference_x: ``[B,T,R]`` current frozen V7 lanes.
        reference_range: ``[B,T,2]``.

    Returns:
        ``[B,S,N,T,11]`` signed/range/continuity relations.
    """

    if candidate_x.ndim != 4 or reference_x.ndim != 3:
        raise ValueError("curve relation tensors have invalid rank")
    batch, slots, candidates, rows = candidate_x.shape
    target_slots = int(reference_x.shape[1])
    if reference_x.shape != (batch, target_slots, rows):
        raise ValueError("reference curve shape mismatch")
    if candidate_range.shape != (batch, slots, candidates, 2):
        raise ValueError("candidate range shape mismatch")
    if reference_range.shape != (batch, target_slots, 2):
        raise ValueError("reference range shape mismatch")

    candidate = candidate_x.detach().float().unsqueeze(3)
    reference = reference_x.detach().float().view(
        batch, 1, 1, target_slots, rows
    )
    candidate_range = sort_range_norm(candidate_range.detach().float()).unsqueeze(3)
    reference_range = sort_range_norm(reference_range.detach().float()).view(
        batch, 1, 1, target_slots, 2
    )
    row_fraction = fixed_row_fractions(
        rows, device=candidate.device, dtype=torch.float32
    ).view(1, 1, 1, 1, rows)
    candidate_visible = (
        (row_fraction >= candidate_range[..., :1])
        & (row_fraction <= candidate_range[..., 1:])
        & torch.isfinite(candidate)
    )
    reference_visible = (
        (row_fraction >= reference_range[..., :1])
        & (row_fraction <= reference_range[..., 1:])
        & torch.isfinite(reference)
    )
    common = candidate_visible & reference_visible
    either = candidate_visible | reference_visible
    safe_candidate = torch.where(
        torch.isfinite(candidate), candidate, torch.zeros_like(candidate)
    )
    safe_reference = torch.where(
        torch.isfinite(reference), reference, torch.zeros_like(reference)
    )
    width = float(max(int(input_w) - 1, 1))
    dx = (safe_candidate - safe_reference) / width
    abs_dx = dx.abs()

    bottom_weight = common.float() * row_fraction.pow(3)
    top_weight = common.float() * (1.0 - row_fraction).pow(3)
    bottom_signed = (dx * bottom_weight).sum(dim=-1) / bottom_weight.sum(
        dim=-1
    ).clamp_min(1.0)
    top_signed = (dx * top_weight).sum(dim=-1) / top_weight.sum(
        dim=-1
    ).clamp_min(1.0)

    candidate_slope = _row_slope(safe_candidate.squeeze(3)).unsqueeze(3) / width
    reference_slope = _row_slope(safe_reference.squeeze(1).squeeze(1)).view(
        batch, 1, 1, target_slots, rows
    ) / width
    slope_difference = (candidate_slope - reference_slope).abs()

    range_start = candidate_range[..., 0] - reference_range[..., 0]
    range_end = candidate_range[..., 1] - reference_range[..., 1]
    intersection = (
        torch.minimum(candidate_range[..., 1], reference_range[..., 1])
        - torch.maximum(candidate_range[..., 0], reference_range[..., 0])
    ).clamp_min(0.0)
    union = (
        torch.maximum(candidate_range[..., 1], reference_range[..., 1])
        - torch.minimum(candidate_range[..., 0], reference_range[..., 0])
    ).clamp_min(1.0e-6)
    range_iou = intersection / union

    sign_change = (
        (dx[..., 1:] * dx[..., :-1] < 0.0)
        & common[..., 1:]
        & common[..., :-1]
    )
    crossing = sign_change.float().sum(dim=-1) / (
        common[..., 1:] & common[..., :-1]
    ).float().sum(dim=-1).clamp_min(1.0)
    common_fraction = common.float().sum(dim=-1) / either.float().sum(
        dim=-1
    ).clamp_min(1.0)

    return torch.stack(
        (
            _masked_mean(dx, common),
            _masked_mean(abs_dx, common),
            _masked_mean(abs_dx.square(), common).sqrt(),
            bottom_signed,
            top_signed,
            _masked_mean(slope_difference, common),
            range_start,
            range_end,
            range_iou,
            common_fraction,
            crossing,
        ),
        dim=-1,
    )


class SlotOwnedSafeReplacementHead(nn.Module):
    """Choose KEEP or one safe slot-owned proposal replacement.

    V7 and V19 are immutable.  This head consumes V19's complete-curve
    candidate representation and explicitly conditions each action on the
    other three current V7 lanes.  The context-masked arm has identical
    parameters and inputs, but zeros only that set-context edge.
    """

    relation_dim = 11

    def __init__(
        self,
        candidate_dim: int,
        *,
        hidden_dim: int = 256,
        ff_dim: int = 512,
        input_w: int = 800,
        context_mode: str = "treatment",
    ) -> None:
        super().__init__()
        mode = str(context_mode).strip().lower()
        if mode not in {"treatment", "masked"}:
            raise ValueError("V20 context_mode must be treatment or masked")
        self.hidden_dim = int(hidden_dim)
        self.input_w = int(input_w)
        self.context_mode = mode

        self.candidate_norm = nn.LayerNorm(int(candidate_dim))
        self.candidate_projection = nn.Linear(
            int(candidate_dim), self.hidden_dim, bias=False
        )
        self.source_projection = nn.Linear(
            int(candidate_dim), self.hidden_dim, bias=False
        )
        self.own_relation_projection = nn.Linear(
            self.relation_dim, self.hidden_dim, bias=False
        )
        # candidate p50/p75/IoU, source values, their deltas, route log-p,
        # source route log-p, their delta and the original-route indicator.
        self.scalar_projection = nn.Sequential(
            nn.Linear(13, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.base_norm = nn.LayerNorm(self.hidden_dim)

        self.context_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.context_key = nn.Linear(
            int(candidate_dim), self.hidden_dim, bias=False
        )
        self.context_value = nn.Linear(
            int(candidate_dim), self.hidden_dim, bias=False
        )
        self.relation_key = nn.Linear(
            self.relation_dim, self.hidden_dim, bias=False
        )
        self.relation_value = nn.Linear(
            self.relation_dim, self.hidden_dim, bias=False
        )
        self.context_output = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        self.policy_output = nn.Linear(self.hidden_dim, 1)
        self.delta50_output = nn.Linear(self.hidden_dim, 3)
        self.delta75_output = nn.Linear(self.hidden_dim, 3)
        self.duplicate_output = nn.Linear(self.hidden_dim, 1)
        self.abandon_output = nn.Linear(self.hidden_dim, 1)
        self.delta_iou_output = nn.Linear(self.hidden_dim, 1)
        for output in (
            self.policy_output,
            self.delta50_output,
            self.delta75_output,
            self.duplicate_output,
            self.abandon_output,
            self.delta_iou_output,
        ):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(
        self,
        *,
        candidate_state: torch.Tensor,
        p50: torch.Tensor,
        p75: torch.Tensor,
        expected_iou: torch.Tensor,
        legacy_route_logits: torch.Tensor,
        counterfactual_valid: torch.Tensor,
        source_route: torch.Tensor,
        source_active: torch.Tensor,
        counterfactual_x: torch.Tensor | None = None,
        counterfactual_range: torch.Tensor | None = None,
        precomputed_relations: torch.Tensor | None = None,
        force_context_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_state.ndim != 4:
            raise ValueError("V20 candidate_state must have shape [B,S,N,C]")
        batch, slots, candidates, _channels = candidate_state.shape
        if p50.shape != (batch, slots, candidates):
            raise ValueError("V20 quality tensor shape mismatch")

        source_route = source_route.detach().long()
        source_active = source_active.detach().bool()
        state = candidate_state.detach().float()
        quality = torch.stack(
            (p50, p75, expected_iou), dim=-1
        ).detach().float()
        route_log_probability = F.log_softmax(
            legacy_route_logits.detach().float(), dim=-1
        )
        source_state = _gather_candidate(state, source_route)
        source_quality = _gather_candidate(quality, source_route)
        source_log_probability = _gather_candidate(
            route_log_probability.unsqueeze(-1), source_route
        ).squeeze(-1)
        if precomputed_relations is not None:
            relation = precomputed_relations.detach().float()
            if relation.shape != (
                batch,
                slots,
                candidates,
                slots,
                self.relation_dim,
            ):
                raise ValueError("V20 precomputed relation shape mismatch")
        else:
            if not isinstance(counterfactual_x, torch.Tensor) or not isinstance(
                counterfactual_range, torch.Tensor
            ):
                raise ValueError("V20 requires geometry or precomputed relations")
            if counterfactual_x.shape[:3] != (batch, slots, candidates):
                raise ValueError("V20 counterfactual geometry shape mismatch")
            if counterfactual_range.shape != (batch, slots, candidates, 2):
                raise ValueError("V20 counterfactual range shape mismatch")
            source_x = _gather_candidate(
                counterfactual_x.detach().float(), source_route
            )
            source_range = _gather_candidate(
                counterfactual_range.detach().float(), source_route
            )
            relation = complete_curve_relations(
                counterfactual_x,
                counterfactual_range,
                source_x,
                source_range,
                input_w=self.input_w,
            )
            # Match the exact target cache/evaluator action validity.  The
            # upstream V19 validity is intentionally retained as an
            # additional constraint.
            counterfactual_valid = (
                counterfactual_valid.detach().bool()
                & official_raster_candidate_valid(
                    counterfactual_x,
                    counterfactual_range,
                    min_valid_rows=5,
                )
            )
        own_index = torch.arange(slots, device=state.device).view(
            1, slots, 1, 1, 1
        ).expand(batch, slots, candidates, 1, self.relation_dim)
        own_relation = relation.gather(3, own_index).squeeze(3)

        source_state_expanded = source_state.unsqueeze(2).expand(
            batch, slots, candidates, source_state.shape[-1]
        )
        source_quality_expanded = source_quality.unsqueeze(2).expand_as(quality)
        source_log_expanded = source_log_probability.unsqueeze(-1).expand_as(
            route_log_probability
        )
        original = (
            torch.arange(candidates, device=state.device).view(1, 1, candidates)
            == source_route.unsqueeze(-1)
        ).float()
        scalar = torch.cat(
            (
                quality,
                source_quality_expanded,
                quality - source_quality_expanded,
                route_log_probability.unsqueeze(-1),
                source_log_expanded.unsqueeze(-1),
                (route_log_probability - source_log_expanded).unsqueeze(-1),
                original.unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.candidate_projection(self.candidate_norm(state))
        hidden = hidden + self.source_projection(source_state_expanded)
        hidden = hidden + self.own_relation_projection(own_relation)
        hidden = hidden + self.scalar_projection(scalar)
        hidden = self.base_norm(hidden)

        current_state = source_state.view(batch, 1, 1, slots, -1)
        key = self.context_key(current_state) + self.relation_key(relation)
        value = self.context_value(current_state) + self.relation_value(relation)
        query = self.context_query(hidden).unsqueeze(3)
        attention_logits = (query * key).sum(dim=-1) / math.sqrt(
            float(self.hidden_dim)
        )
        slot_axis = torch.arange(slots, device=state.device)
        other = slot_axis.view(1, 1, 1, slots) != slot_axis.view(
            1, slots, 1, 1
        )
        context_valid = other & source_active.view(batch, 1, 1, slots)
        attention_logits = attention_logits.masked_fill(~context_valid, -1.0e4)
        attention = torch.softmax(attention_logits.float(), dim=-1)
        attention = attention * context_valid.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        context = torch.einsum("bsnt,bsnth->bsnh", attention, value.float())
        mode = self.context_mode if force_context_mode is None else str(
            force_context_mode
        ).strip().lower()
        if mode not in {"treatment", "masked"}:
            raise ValueError("invalid forced V20 context mode")
        if mode == "masked":
            context = context * 0.0
        hidden = hidden + self.context_output(context)
        hidden = hidden + self.fusion(self.fusion_norm(hidden))

        action_valid = slot_candidate_action_valid(
            counterfactual_valid.detach().bool(), source_route, source_active
        )
        policy = self.policy_output(hidden).squeeze(-1)
        delta50 = self.delta50_output(hidden)
        delta75 = self.delta75_output(hidden)
        duplicate = self.duplicate_output(hidden).squeeze(-1)
        abandon = self.abandon_output(hidden).squeeze(-1)
        delta_iou = self.delta_iou_output(hidden).squeeze(-1)

        # Conservative deployment: outcome heads must predict a threshold
        # gain, abandonment/duplication risk must be below 0.5, and the policy
        # residual must beat exact KEEP=0.  Zero initialization therefore
        # deterministically preserves V7.
        class_value = policy.new_tensor((-1.0, 0.0, 1.0))
        expected_delta50 = (
            torch.softmax(delta50.float(), dim=-1) * class_value
        ).sum(dim=-1)
        expected_delta75 = (
            torch.softmax(delta75.float(), dim=-1) * class_value
        ).sum(dim=-1)
        gain_safe = (expected_delta50 > 0.0) | (
            (expected_delta50 >= 0.0) & (expected_delta75 > 0.0)
        )
        risk_safe = (torch.sigmoid(abandon) < 0.5) & (
            torch.sigmoid(duplicate) < 0.5
        )
        deployable = action_valid & gain_safe & risk_safe & (policy > 0.0)
        deploy_score = policy.masked_fill(~deployable, -1.0e4)
        flat_score = deploy_score.reshape(batch, slots * candidates)
        best_score, best_action = flat_score.max(dim=-1)
        replace = best_score > 0.0
        replace_slot = torch.div(best_action, candidates, rounding_mode="floor")
        replace_candidate = best_action.remainder(candidates)
        selected_route = source_route.clone()
        row = torch.arange(batch, device=state.device)
        selected_route[row[replace], replace_slot[replace]] = replace_candidate[
            replace
        ]
        edit_count = replace.long()
        return {
            "action_hidden": hidden,
            "action_valid": action_valid,
            "policy_logits": policy,
            "delta50_logits": delta50,
            "delta75_logits": delta75,
            "duplicate_logits": duplicate,
            "abandon_logits": abandon,
            "delta_iou": delta_iou,
            "expected_delta50": expected_delta50,
            "expected_delta75": expected_delta75,
            "set_attention": attention,
            "selected_route": selected_route,
            "replace": replace,
            "replace_slot": torch.where(
                replace, replace_slot, replace_slot.new_full(replace_slot.shape, -1)
            ),
            "replace_candidate": torch.where(
                replace,
                replace_candidate,
                replace_candidate.new_full(replace_candidate.shape, -1),
            ),
            "edit_count": edit_count,
        }
