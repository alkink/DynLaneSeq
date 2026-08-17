from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_row_fractions, sort_range_norm


SLOTS = 4
ALTERNATIVES = 3
ACTIONS = SLOTS * ALTERNATIVES
ROWS = 160
ROW_EVIDENCE_DIM = 6
ROW_GEOMETRY_DIM = 6
GLOBAL_EVIDENCE_DIM = 26
GLOBAL_GEOMETRY_DIM = 38
OUTCOME_NEUTRAL = 0
OUTCOME_BENEFICIAL = 1
OUTCOME_HARMFUL = 2


@dataclass(frozen=True)
class SelectorLossWeights:
    action: float = 1.0
    outcome: float = 0.5


def _path_indices(x_rows: torch.Tensor, *, input_w: int, bins: int) -> torch.Tensor:
    position = x_rows.float() / (float(input_w) / float(bins)) - 0.5
    return position.round().long().clamp(0, bins - 1)


def _masked_summary(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Six stable row summaries for tensors shaped [B,A,R]."""

    weight = mask.float()
    count = weight.sum(dim=-1).clamp_min(1.0)
    mean = (value * weight).sum(dim=-1) / count
    variance = ((value - mean.unsqueeze(-1)).square() * weight).sum(-1) / count
    minimum = torch.where(mask, value, torch.full_like(value, float("inf"))).amin(-1)
    maximum = torch.where(mask, value, torch.full_like(value, -float("inf"))).amax(-1)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    rows = int(value.shape[-1])
    first = max(rows // 3, 1)
    second = max(2 * rows // 3, first + 1)

    def band(start: int, end: int) -> torch.Tensor:
        local_mask = mask[..., start:end]
        local_weight = local_mask.float()
        local_count = local_weight.sum(-1).clamp_min(1.0)
        return (value[..., start:end] * local_weight).sum(-1) / local_count

    return torch.stack(
        (mean, variance.sqrt(), minimum, maximum, band(0, first), band(second, rows)),
        dim=-1,
    )


def _set_geometry_features(
    paths: torch.Tensor,
    ranges: torch.Tensor,
    active: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    """Compact order/duplicate geometry for [B,A,4,R] complete sets."""

    if paths.ndim != 4 or paths.shape[2] != SLOTS:
        raise ValueError("selector set paths must have shape [B,A,4,R]")
    rows = int(paths.shape[-1])
    fraction = fixed_row_fractions(
        rows, device=paths.device, dtype=torch.float32
    ).view(1, 1, 1, rows)
    visible = (
        active.unsqueeze(-1)
        & torch.isfinite(paths)
        & (fraction >= ranges[..., :1])
        & (fraction <= ranges[..., 1:])
    )
    signed_values: list[torch.Tensor] = []
    signed_masks: list[torch.Tensor] = []
    absolute_values: list[torch.Tensor] = []
    absolute_masks: list[torch.Tensor] = []
    for left in range(SLOTS - 1):
        right = left + 1
        common = visible[:, :, left] & visible[:, :, right]
        signed_values.append(
            (paths[:, :, right] - paths[:, :, left]) / float(input_w)
        )
        signed_masks.append(common)
    for left in range(SLOTS):
        for right in range(left + 1, SLOTS):
            common = visible[:, :, left] & visible[:, :, right]
            absolute_values.append(
                (paths[:, :, right] - paths[:, :, left]).abs() / float(input_w)
            )
            absolute_masks.append(common)
    signed = torch.cat(signed_values, dim=-1)
    signed_mask = torch.cat(signed_masks, dim=-1)
    absolute = torch.cat(absolute_values, dim=-1)
    absolute_mask = torch.cat(absolute_masks, dim=-1)
    signed_weight = signed_mask.float()
    absolute_weight = absolute_mask.float()
    signed_count = signed_weight.sum(-1).clamp_min(1.0)
    absolute_count = absolute_weight.sum(-1).clamp_min(1.0)
    crossing = ((signed < 0.0) & signed_mask).float().sum(-1) / signed_count
    signed_mean = (signed * signed_weight).sum(-1) / signed_count
    signed_min = torch.where(
        signed_mask, signed, torch.full_like(signed, float("inf"))
    ).amin(-1)
    signed_min = torch.where(torch.isfinite(signed_min), signed_min, torch.zeros_like(signed_min))
    absolute_mean = (absolute * absolute_weight).sum(-1) / absolute_count
    absolute_min = torch.where(
        absolute_mask, absolute, torch.full_like(absolute, float("inf"))
    ).amin(-1)
    absolute_min = torch.where(
        torch.isfinite(absolute_min), absolute_min, torch.zeros_like(absolute_min)
    )
    duplicate = (
        torch.exp(-absolute * float(input_w) / 12.0) * absolute_weight
    ).sum(-1) / absolute_count
    return torch.stack(
        (
            crossing,
            signed_mean,
            signed_min,
            absolute_mean,
            absolute_min,
            duplicate,
        ),
        dim=-1,
    )


def build_selector_features(
    *,
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    source_active: torch.Tensor,
    hypotheses: torch.Tensor,
    hypothesis_range: torch.Tensor,
    hypothesis_valid: torch.Tensor | None = None,
    evidence: dict[str, torch.Tensor],
    input_w: int,
) -> dict[str, torch.Tensor]:
    """Create path-native treatment/control inputs without mutating a curve.

    Candidate geometry always comes from ``hypotheses``. Passing logits from a
    cross-clip image as ``evidence`` changes only evidence features and is the
    paired causal wrong-image control used by V25-S1.
    """

    if source_x.shape[1:] != (SLOTS, ROWS):
        raise ValueError("source geometry must have shape [B,4,160]")
    if hypotheses.shape[1:] != (SLOTS, ALTERNATIVES, ROWS):
        raise ValueError("hypotheses must have shape [B,4,3,160]")
    unary = evidence["unary_logits"].float()
    if unary.shape[:3] != (source_x.shape[0], SLOTS, ROWS):
        raise ValueError("evidence unary tensor is not aligned to the bank")
    bins = int(unary.shape[-1])
    log_probability = unary.log_softmax(dim=-1)
    probability = log_probability.exp()
    entropy = -(probability * log_probability).sum(-1) / math.log(float(bins))
    top_two = log_probability.topk(k=2, dim=-1).values
    top_margin = top_two[..., 0] - top_two[..., 1]

    candidate_index = _path_indices(hypotheses, input_w=input_w, bins=bins)
    candidate_logp = log_probability.unsqueeze(2).expand(
        -1, -1, ALTERNATIVES, -1, -1
    ).gather(-1, candidate_index.unsqueeze(-1)).squeeze(-1)
    source_index = _path_indices(source_x, input_w=input_w, bins=bins)
    source_logp = log_probability.gather(-1, source_index.unsqueeze(-1)).squeeze(-1)
    source_logp = source_logp.unsqueeze(2).expand(-1, -1, ALTERNATIVES, -1)
    top_logp = top_two[..., 0].unsqueeze(2).expand_as(candidate_logp)
    margin = top_margin.unsqueeze(2).expand_as(candidate_logp)
    entropy = entropy.unsqueeze(2).expand_as(candidate_logp)
    delta = candidate_logp - source_logp

    batch = int(source_x.shape[0])
    signed_dx = (hypotheses.float() - source_x.float().unsqueeze(2)) / float(input_w)
    absolute_dx = signed_dx.abs()
    row_fraction = fixed_row_fractions(
        ROWS, device=source_x.device, dtype=torch.float32
    ).view(1, 1, 1, ROWS).expand_as(signed_dx)
    source_range = sort_range_norm(source_range.float())
    hypothesis_range = sort_range_norm(hypothesis_range.float())
    source_valid = (
        source_active.bool().unsqueeze(-1)
        & (row_fraction[:, :, 0] >= source_range[..., :1])
        & (row_fraction[:, :, 0] <= source_range[..., 1:])
    ).unsqueeze(2).expand_as(signed_dx)
    candidate_valid = (
        source_active.bool().unsqueeze(-1)
        & (row_fraction[:, :, 0] >= hypothesis_range[..., :1])
        & (row_fraction[:, :, 0] <= hypothesis_range[..., 1:])
    ).unsqueeze(2).expand_as(signed_dx)
    both = source_valid & candidate_valid

    row_evidence = torch.stack(
        (candidate_logp, source_logp, delta, candidate_logp - top_logp, margin, entropy),
        dim=-1,
    ).reshape(batch, ACTIONS, ROWS, ROW_EVIDENCE_DIM)
    row_geometry = torch.stack(
        (
            signed_dx,
            absolute_dx,
            row_fraction,
            source_valid.float(),
            candidate_valid.float(),
            both.float(),
        ),
        dim=-1,
    ).reshape(batch, ACTIONS, ROWS, ROW_GEOMETRY_DIM)

    flat_mask = candidate_valid.reshape(batch, ACTIONS, ROWS)
    candidate_flat = candidate_logp.reshape(batch, ACTIONS, ROWS)
    source_flat = source_logp.reshape(batch, ACTIONS, ROWS)
    delta_flat = delta.reshape(batch, ACTIONS, ROWS)
    entropy_flat = entropy.reshape(batch, ACTIONS, ROWS)
    margin_flat = margin.reshape(batch, ACTIONS, ROWS)
    evidence_summary = torch.cat(
        (
            _masked_summary(candidate_flat, flat_mask),
            _masked_summary(source_flat, flat_mask),
            _masked_summary(delta_flat, flat_mask),
            _masked_summary(entropy_flat, flat_mask)[..., :2],
            _masked_summary(margin_flat, flat_mask)[..., :2],
        ),
        dim=-1,
    )
    slot_q50 = torch.sigmoid(evidence["quality50_logits"].float()).unsqueeze(2)
    slot_q75 = torch.sigmoid(evidence["quality75_logits"].float()).unsqueeze(2)
    slot_exist = torch.softmax(evidence["exist_logits"].float(), dim=-1)[..., 0].unsqueeze(2)
    confidence = torch.cat((slot_q50, slot_q75, slot_exist), dim=-1)
    confidence = confidence.unsqueeze(2).expand(-1, -1, ALTERNATIVES, -1)
    confidence = confidence.reshape(batch, ACTIONS, 3)
    candidate_score = candidate_logp.mean(-1)
    alternative_margin = candidate_score - candidate_score.amax(2, keepdim=True)
    global_evidence = torch.cat(
        (evidence_summary, confidence, alternative_margin.reshape(batch, ACTIONS, 1)),
        dim=-1,
    )
    if int(global_evidence.shape[-1]) != GLOBAL_EVIDENCE_DIM:
        raise RuntimeError("V25-S1 global evidence dimension drifted")

    slot_ids = torch.arange(SLOTS, device=source_x.device).view(1, SLOTS, 1)
    path_ids = torch.arange(ALTERNATIVES, device=source_x.device).view(1, 1, ALTERNATIVES)
    slot_one_hot = F.one_hot(slot_ids.expand(batch, -1, ALTERNATIVES), SLOTS).float()
    path_one_hot = F.one_hot(path_ids.expand(batch, SLOTS, -1), ALTERNATIVES).float()
    dx_summary = _masked_summary(
        signed_dx.reshape(batch, ACTIONS, ROWS), flat_mask
    )
    abs_summary = _masked_summary(
        absolute_dx.reshape(batch, ACTIONS, ROWS), flat_mask
    )
    source_range_action = source_range.unsqueeze(2).expand(-1, -1, ALTERNATIVES, -1)
    candidate_range_action = hypothesis_range.unsqueeze(2).expand_as(source_range_action)
    overlap = (
        torch.minimum(source_range_action[..., 1], candidate_range_action[..., 1])
        - torch.maximum(source_range_action[..., 0], candidate_range_action[..., 0])
    ).clamp_min(0.0)
    range_features = torch.cat(
        (
            source_range_action,
            candidate_range_action,
            overlap.unsqueeze(-1),
            candidate_range_action - source_range_action,
        ),
        dim=-1,
    ).reshape(batch, ACTIONS, 7)

    source_paths = source_x.float().unsqueeze(1).expand(-1, ACTIONS, -1, -1).clone()
    selected_paths = source_paths.clone()
    selected_ranges = source_range.unsqueeze(1).expand(-1, ACTIONS, -1, -1).clone()
    selected_active = source_active.bool().unsqueeze(1).expand(-1, ACTIONS, -1).clone()
    for slot in range(SLOTS):
        for path in range(ALTERNATIVES):
            action = slot * ALTERNATIVES + path
            selected_paths[:, action, slot] = hypotheses[:, slot, path]
            selected_ranges[:, action, slot] = hypothesis_range[:, slot]
    source_set = _set_geometry_features(
        source_x.float().unsqueeze(1),
        source_range.unsqueeze(1),
        source_active.bool().unsqueeze(1),
        input_w=input_w,
    ).expand(-1, ACTIONS, -1)
    action_set = _set_geometry_features(
        selected_paths,
        selected_ranges,
        selected_active,
        input_w=input_w,
    )
    global_geometry = torch.cat(
        (
            slot_one_hot.reshape(batch, ACTIONS, SLOTS),
            path_one_hot.reshape(batch, ACTIONS, ALTERNATIVES),
            dx_summary,
            abs_summary,
            range_features,
            source_set,
            action_set - source_set,
        ),
        dim=-1,
    )
    if int(global_geometry.shape[-1]) != GLOBAL_GEOMETRY_DIM:
        raise RuntimeError(
            f"V25-S1 global geometry dimension drifted: {global_geometry.shape[-1]}"
        )
    action_valid = source_active.bool().unsqueeze(-1).expand(
        -1, -1, ALTERNATIVES
    )
    if hypothesis_valid is not None:
        if hypothesis_valid.shape != action_valid.shape:
            raise ValueError("hypothesis_valid must have shape [B,4,3]")
        action_valid = action_valid & hypothesis_valid.bool()
    action_valid = action_valid.reshape(batch, ACTIONS)
    return {
        "row_evidence": row_evidence,
        "row_geometry": row_geometry,
        "global_evidence": global_evidence,
        "global_geometry": global_geometry,
        "action_valid": action_valid,
    }


class ImmutableBankSingleEditSelector(nn.Module):
    """Risk-aware KEEP-or-one-edit selector over immutable writer paths."""

    def __init__(self, *, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        row_dim = ROW_EVIDENCE_DIM + ROW_GEOMETRY_DIM
        self.row_encoder = nn.Sequential(
            nn.Conv1d(row_dim, hidden_dim, 5, padding=2, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(GLOBAL_EVIDENCE_DIM + GLOBAL_GEOMETRY_DIM),
            nn.Linear(GLOBAL_EVIDENCE_DIM + GLOBAL_GEOMETRY_DIM, hidden_dim),
            nn.GELU(),
        )
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 3),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        *,
        evidence_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        row_evidence = features["row_evidence"].float()
        global_evidence = features["global_evidence"].float()
        if not evidence_enabled:
            row_evidence = torch.zeros_like(row_evidence)
            global_evidence = torch.zeros_like(global_evidence)
        row = torch.cat((row_evidence, features["row_geometry"].float()), dim=-1)
        batch = int(row.shape[0])
        encoded = self.row_encoder(
            row.reshape(batch * ACTIONS, ROWS, -1).transpose(1, 2)
        )
        mean = encoded.mean(-1)
        maximum = encoded.amax(-1)
        global_state = self.global_encoder(
            torch.cat((global_evidence, features["global_geometry"].float()), dim=-1)
        ).reshape(batch * ACTIONS, -1)
        outcome_logits = self.outcome_head(
            torch.cat((mean, maximum, global_state), dim=-1)
        ).reshape(batch, ACTIONS, 3)
        log_probability = outcome_logits.log_softmax(dim=-1)
        # A fixed 2:1 cost for harmful edits creates explicit abstention. KEEP
        # has utility zero; an edit is emitted only when predicted beneficial
        # mass exceeds neutral plus twice harmful mass.
        edit_score = log_probability[..., OUTCOME_BENEFICIAL] - torch.logaddexp(
            log_probability[..., OUTCOME_NEUTRAL],
            math.log(2.0) + log_probability[..., OUTCOME_HARMFUL],
        )
        valid = features["action_valid"].bool()
        edit_score = torch.where(valid, edit_score, torch.full_like(edit_score, -1.0e9))
        action_scores = torch.cat((torch.zeros_like(edit_score[:, :1]), edit_score), dim=1)
        winner = action_scores.argmax(dim=-1)
        return {
            "outcome_logits": outcome_logits,
            "edit_scores": edit_score,
            "action_scores": action_scores,
            "selected_action": winner,
        }


def immutable_selector_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_action: torch.Tensor,
    action_outcome: torch.Tensor,
    action_valid: torch.Tensor,
    weights: SelectorLossWeights = SelectorLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_image_action = F.cross_entropy(
        outputs["action_scores"], target_action.long(), reduction="none"
    )
    # Useful edits are deliberately rare. A fixed three-to-one weight keeps
    # the task learnable without changing the official population or sampling
    # a cherry-picked subset.
    image_weight = torch.where(
        target_action.long() > 0,
        per_image_action.new_tensor(3.0),
        per_image_action.new_tensor(1.0),
    )
    action = (per_image_action * image_weight).sum() / image_weight.sum().clamp_min(1.0)
    valid = action_valid.bool().reshape(-1)
    logits = outputs["outcome_logits"].reshape(-1, 3)[valid]
    target = action_outcome.long().reshape(-1)[valid]
    if int(logits.shape[0]) > 0:
        class_weight = logits.new_tensor((0.25, 1.0, 2.0))
        outcome = F.cross_entropy(logits, target, weight=class_weight)
    else:
        # Empty-road frames still have the valid KEEP action, but no edit
        # outcome exists to supervise.
        outcome = outputs["outcome_logits"].sum() * 0.0
    total = float(weights.action) * action + float(weights.outcome) * outcome
    return total, {
        "loss_total": total,
        "loss_action": action,
        "loss_outcome": outcome,
        "selected_edit_fraction": (outputs["selected_action"] > 0).float().mean(),
        "target_edit_fraction": (target_action > 0).float().mean(),
        "action_accuracy": (outputs["selected_action"] == target_action).float().mean(),
    }
