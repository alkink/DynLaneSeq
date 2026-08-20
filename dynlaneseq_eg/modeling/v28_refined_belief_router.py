from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
)

from .backbone_dla import DLA34Backbone
from .common import fixed_indices, fixed_row_fractions, sort_range_norm
from .fpn import SimpleFPN
from .four_slot_selection import decode_unique_real_slot_routes
from .v23_ordered_slot_cost_volume import build_v23_owned_targets


@dataclass(frozen=True)
class V28BeliefLossWeights:
    """Fixed Stage-S0 objective weights.

    Arm B uses only ``route``.  Arm C uses the identical route objective and
    adds ``field``.  Keeping the two terms explicit makes the causal
    comparison auditable and prevents the direct-geometry objectives used by
    V23 from re-entering this selection-only experiment.
    """

    route: float = 1.0
    field: float = 1.0
    target_temperature: float = 0.05
    target_delta: float = 0.05
    target_floor: float = 0.0
    line_width: float = 30.0
    minimum_valid_rows: int = 5


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def gather_canonical_slots(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Gather a public V7 slot tensor into canonical left-to-right order."""

    if value.ndim < 2 or order.ndim != 2:
        raise ValueError("canonical slot gather expects [B,S,...] and [B,S]")
    if tuple(value.shape[:2]) != tuple(order.shape):
        raise ValueError("canonical slot order does not match tensor slots")
    index = order
    while index.ndim < value.ndim:
        index = index.unsqueeze(-1)
    return value.gather(1, index.expand(*order.shape, *value.shape[2:]))


def restore_public_slots(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Undo :func:`gather_canonical_slots` at the writer boundary."""

    inverse = order.argsort(dim=1, stable=True)
    return gather_canonical_slots(value, inverse)


@torch.no_grad()
def decode_v28_unique_routes(
    scores: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Exact four-slot assignment with slot-specific candidate validity.

    The mature V7 decoder accepts one shared validity mask.  Counterfactual
    refinement adds a slot axis to validity, so invalid slot/candidate pairs
    are first removed with ``-inf`` and the existing exact decoder then
    enforces proposal-ID uniqueness.  No NMS or greedy repair is introduced.
    """

    if scores.ndim != 3 or candidate_valid.shape != scores.shape:
        raise ValueError("V28 route decode expects matching [B,S,N] tensors")
    slot_masked = scores.float().masked_fill(
        ~candidate_valid.bool(), float("-inf")
    )
    shared_valid = candidate_valid.bool().any(dim=1)
    decoded = decode_unique_real_slot_routes(slot_masked, shared_valid)
    indices = decoded["indices"]
    selected_valid = candidate_valid.gather(
        -1, indices.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    if not bool(selected_valid.all()):
        raise RuntimeError("V28 could not find a valid injective route assignment")
    return indices


def gather_slot_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather one candidate per slot from ``[B,S,N,...]``."""

    if value.ndim < 3 or indices.ndim != 2:
        raise ValueError("candidate gather expects [B,S,N,...] and [B,S]")
    if tuple(value.shape[:2]) != tuple(indices.shape):
        raise ValueError("candidate gather batch/slot mismatch")
    index = indices.unsqueeze(-1)
    while index.ndim < value.ndim:
        index = index.unsqueeze(-1)
    expanded = index.expand(*indices.shape, 1, *value.shape[3:])
    return value.gather(2, expanded).squeeze(2)


def score_slot_candidate_paths(
    log_probability: torch.Tensor,
    *,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    input_w: int,
    minimum_valid_rows: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score final-ready slot/candidate curves on a row belief field.

    ``candidate_x`` is slot-specific: V7's frozen counterfactual refiner can
    map the same proposal ID to a different final curve for each output slot.
    V23's old proposal scorer accepted one shared raw bank and therefore could
    not express this contract.
    """

    if log_probability.ndim != 4:
        raise ValueError("belief field must have shape [B,S,R,X]")
    if candidate_x.ndim != 4 or candidate_range.ndim != 4:
        raise ValueError("refined bank must have shape [B,S,N,R]/[B,S,N,2]")
    batch, slots, rows, bins = log_probability.shape
    if tuple(candidate_x.shape[:2]) != (batch, slots):
        raise ValueError("refined bank batch/slot mismatch")
    candidates = int(candidate_x.shape[2])
    if int(candidate_x.shape[-1]) != rows:
        raise ValueError("refined bank row count mismatch")
    if tuple(candidate_range.shape) != (batch, slots, candidates, 2):
        raise ValueError("refined candidate range shape mismatch")
    if tuple(candidate_valid.shape) != (batch, slots, candidates):
        raise ValueError("refined candidate validity shape mismatch")

    position = candidate_x.float() / (float(input_w) / float(bins)) - 0.5
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = (position - lower.float()).clamp(0.0, 1.0)
    # [B,S,R,N] indices gather all candidates from each row without creating
    # a [B,S,N,R,X] expansion of the belief field.
    gather_lower = lower.permute(0, 1, 3, 2)
    gather_upper = upper.permute(0, 1, 3, 2)
    lower_value = log_probability.gather(-1, gather_lower).permute(0, 1, 3, 2)
    upper_value = log_probability.gather(-1, gather_upper).permute(0, 1, 3, 2)
    sampled = lower_value * (1.0 - fraction) + upper_value * fraction

    row_fraction = fixed_row_fractions(
        rows, device=log_probability.device, dtype=torch.float32
    ).view(1, 1, 1, rows)
    ranges = sort_range_norm(candidate_range.float())
    row_valid = (
        torch.isfinite(candidate_x)
        & (candidate_x >= 0.0)
        & (candidate_x < float(input_w))
        & (row_fraction >= ranges[..., :1])
        & (row_fraction <= ranges[..., 1:])
    )
    valid = candidate_valid.bool() & (
        row_valid.sum(dim=-1) >= int(minimum_valid_rows)
    )
    weight = row_valid.float()
    score = (sampled * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    return score.masked_fill(~valid, -1.0e4), valid


class V28RefinedBeliefRouter(nn.Module):
    """Selection-only image belief network over an immutable refined V7 bank.

    V7 owns proposal generation, activity/count, and counterfactual
    refinement.  This module owns only a slot-conditioned row belief field and
    the resulting ordering of the 32 final-ready candidates.  It cannot move
    a proposal or create a new curve, so a failure cannot damage the proposal
    oracle and a success is attributable to route representation learning.
    """

    def __init__(
        self,
        *,
        input_h: int = 640,
        input_w: int = 1600,
        num_rows: int = 160,
        x_bins: int = 800,
        fpn_channels: int = 256,
        hidden_dim: int = 96,
        query_dim: int = 96,
        vertical_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        source_prior_sigma_px: float = 64.0,
        source_prior_weight: float = 0.25,
        freeze_batch_norm_stats: bool = True,
    ) -> None:
        super().__init__()
        if int(query_dim) % int(num_heads):
            raise ValueError("V28 query_dim must be divisible by num_heads")
        if int(num_rows) < 2 or int(x_bins) < 2:
            raise ValueError("V28 row/bin counts must exceed one")
        if float(source_prior_sigma_px) <= 0.0:
            raise ValueError("V28 source prior sigma must be positive")
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.source_prior_sigma_px = float(source_prior_sigma_px)
        self.source_prior_weight = float(source_prior_weight)
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)

        self.backbone = DLA34Backbone(pretrained=False)
        self.fpn = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=int(fpn_channels),
        )
        self.image_projection = nn.Sequential(
            nn.Conv2d(int(fpn_channels), int(hidden_dim), 1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(query_dim), 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(query_dim)), int(query_dim)),
            nn.GELU(),
        )
        self.key_projection = nn.Conv2d(
            int(query_dim), int(query_dim), 1, bias=False
        )
        self.slot_embedding = nn.Parameter(torch.empty(4, int(query_dim)))
        self.row_embedding = nn.Parameter(
            torch.empty(self.num_rows, int(query_dim))
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(5, int(query_dim)),
            nn.GELU(),
            nn.Linear(int(query_dim), int(query_dim)),
        )
        self.context_projection = nn.Linear(int(query_dim), int(query_dim))
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=int(query_dim),
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            norm=nn.LayerNorm(int(query_dim)),
        )
        self.inter_attention = nn.MultiheadAttention(
            int(query_dim), int(num_heads), dropout=0.1, batch_first=True
        )
        self.inter_norm = nn.LayerNorm(int(query_dim))
        self.query_norm = nn.LayerNorm(int(query_dim))
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            for module in (*self.backbone.modules(), *self.fpn.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def copy_v7_encoder_(self, v7_model: nn.Module) -> dict[str, float | int]:
        encoder = getattr(v7_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "backbone") or not hasattr(
            encoder, "fpn"
        ):
            raise TypeError("V28 warm start requires V7 encoder.backbone/fpn")
        self.backbone.load_state_dict(encoder.backbone.state_dict(), strict=True)
        self.fpn.load_state_dict(encoder.fpn.state_dict(), strict=True)
        maximum = 0.0
        count = 0
        for target, source in (
            (self.backbone, encoder.backbone),
            (self.fpn, encoder.fpn),
        ):
            source_parameters = dict(source.named_parameters())
            for name, value in target.named_parameters():
                maximum = max(
                    maximum,
                    float(
                        (value.detach() - source_parameters[name].detach())
                        .abs()
                        .max()
                    ),
                )
                count += 1
        return {"parameter_tensors": count, "maximum_copy_difference": maximum}

    def _source_prior(self, source_x: torch.Tensor) -> torch.Tensor:
        bin_width = float(self.input_w) / float(self.x_bins)
        centers = (
            fixed_indices(
                self.x_bins, device=source_x.device, dtype=torch.float32
            )
            + 0.5
        ) * bin_width
        delta = centers.view(1, 1, 1, -1) - source_x.float().unsqueeze(-1)
        return -0.5 * (delta / self.source_prior_sigma_px).pow(2)

    def forward(
        self,
        images: torch.Tensor,
        *,
        source_x: torch.Tensor,
        source_range: torch.Tensor,
        source_active: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        minimum_valid_rows: int = 5,
    ) -> dict[str, torch.Tensor]:
        batch, slots, rows = source_x.shape
        if slots != 4 or rows != self.num_rows:
            raise ValueError("V28 requires four source slots and configured rows")
        if tuple(source_range.shape) != (batch, slots, 2):
            raise ValueError("V28 source range shape mismatch")
        if tuple(source_active.shape) != (batch, slots):
            raise ValueError("V28 source activity shape mismatch")

        p2 = self.fpn(self.backbone(images))
        image_features = self.image_projection(p2)
        image_features = F.interpolate(
            image_features,
            size=(self.num_rows, self.x_bins),
            mode="bilinear",
            align_corners=False,
        )
        keys = self.key_projection(image_features)
        row_fraction = fixed_row_fractions(
            rows, device=images.device, dtype=torch.float32
        ).view(1, 1, rows)
        source_range = sort_range_norm(source_range.float())
        geometry = torch.stack(
            (
                source_x.float() / float(self.input_w),
                row_fraction.expand(batch, slots, rows),
                source_range[..., 0:1].expand(batch, slots, rows),
                source_range[..., 1:2].expand(batch, slots, rows),
                source_active.float().unsqueeze(-1).expand(batch, slots, rows),
            ),
            dim=-1,
        )
        query = (
            self.slot_embedding.view(1, slots, 1, -1)
            + self.row_embedding.view(1, 1, rows, -1)
            + self.geometry_projection(geometry)
        )
        scale = 1.0 / math.sqrt(float(query.shape[-1]))
        source_prior = self._source_prior(source_x)
        first_image_logits = torch.einsum(
            "bsrc,bcrx->bsrx", query, keys
        ) * scale
        first_logits = (
            first_image_logits + self.source_prior_weight * source_prior
        )
        first_probability = first_logits.softmax(dim=-1)
        context = torch.einsum(
            "bsrx,bcrx->bsrc", first_probability, image_features
        )
        query = query + self.context_projection(context)
        query = self.vertical_encoder(
            query.reshape(batch * slots, rows, -1)
        ).reshape(batch, slots, rows, -1)
        inter_input = query.permute(0, 2, 1, 3).reshape(
            batch * rows, slots, -1
        )
        inter_output, _ = self.inter_attention(
            inter_input, inter_input, inter_input, need_weights=False
        )
        query = self.inter_norm(inter_input + inter_output).reshape(
            batch, rows, slots, -1
        ).permute(0, 2, 1, 3)
        query = self.query_norm(query)
        second_image_logits = torch.einsum(
            "bsrc,bcrx->bsrx", query, keys
        ) * scale
        field_logits = (
            first_image_logits
            + second_image_logits
            + self.source_prior_weight * source_prior
        )
        log_probability = F.log_softmax(field_logits.float(), dim=-1)
        candidate_scores, valid = score_slot_candidate_paths(
            log_probability,
            candidate_x=candidate_x,
            candidate_range=candidate_range,
            candidate_valid=candidate_valid,
            input_w=self.input_w,
            minimum_valid_rows=int(minimum_valid_rows),
        )
        return {
            "field_logits": field_logits,
            "field_log_probability": log_probability,
            "candidate_scores": candidate_scores,
            "candidate_valid": valid,
            "source_x_rows": source_x,
            "source_range_norm": source_range,
            "source_active": source_active,
        }


@torch.no_grad()
def build_v28_refined_route_targets(
    *,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    owned_x: torch.Tensor,
    owned_valid: torch.Tensor,
    owned_matched: torch.Tensor,
    input_h: int,
    line_width: float,
    minimum_valid_rows: int,
    temperature: float,
    support_delta: float,
    support_floor: float,
) -> dict[str, torch.Tensor]:
    """Build slot-owned soft route targets from final-ready candidate curves."""

    batch, slots, candidates, rows = candidate_x.shape
    flat_x = candidate_x.reshape(batch * slots, candidates, rows)
    flat_range = candidate_range.reshape(batch * slots, candidates, 2)
    flat_gt = owned_x.reshape(batch * slots, 1, rows)
    flat_valid = owned_valid.reshape(batch * slots, 1, rows)
    quality, geometric_valid, _ = batched_pairwise_range_aware_row_strip_iou(
        flat_x,
        flat_range,
        flat_gt,
        flat_valid,
        input_h=int(input_h),
        line_width=float(line_width),
        min_valid_rows=int(minimum_valid_rows),
    )
    quality = quality[..., 0].reshape(batch, slots, candidates)
    valid = (
        geometric_valid.reshape(batch, slots, candidates)
        & candidate_valid.bool()
        & owned_matched.bool().unsqueeze(-1)
    )
    masked_quality = quality.masked_fill(~valid, float("-inf"))
    best = masked_quality.amax(dim=-1)
    has_target = owned_matched.bool() & valid.any(dim=-1) & torch.isfinite(best)
    effective_floor = torch.minimum(
        best, best.new_full(best.shape, float(support_floor))
    )
    cutoff = torch.maximum(effective_floor, best - float(support_delta))
    support = valid & (quality >= cutoff.unsqueeze(-1))
    logits = (quality / float(temperature)).masked_fill(~support, -1.0e4)
    probability = logits.softmax(dim=-1)
    probability = torch.where(
        has_target.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    return {
        "probability": probability,
        "quality": quality,
        "valid": valid,
        "support": support,
        "matched": has_target,
        "best_quality": torch.where(has_target, best, torch.zeros_like(best)),
    }


def _field_row_loss(
    log_probability: torch.Tensor,
    *,
    target_x: torch.Tensor,
    target_valid: torch.Tensor,
    matched: torch.Tensor,
    input_w: int,
) -> torch.Tensor:
    bins = int(log_probability.shape[-1])
    position = target_x.float() / (float(input_w) / float(bins)) - 0.5
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = (position - lower.float()).clamp(0.0, 1.0)
    lower_log = log_probability.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_log = log_probability.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    # Linear interpolation is the same two-bin row target used by DFL.  It is
    # a row-wise position distribution, not a foreground/background bitmap.
    nll = -((1.0 - fraction) * lower_log + fraction * upper_log)
    valid = target_valid.bool() & matched.bool().unsqueeze(-1)
    return (nll * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def v28_refined_belief_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    candidate_x: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_valid: torch.Tensor,
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    source_active: torch.Tensor,
    source_route: torch.Tensor,
    input_h: int,
    input_w: int,
    arm: str,
    weights: V28BeliefLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Selection-only B/C causal objective."""

    cfg = weights or V28BeliefLossWeights()
    arm_name = str(arm).strip().upper()
    if arm_name not in {"B", "C"}:
        raise ValueError("V28 arm must be B (route only) or C (route + field)")
    owned = build_v23_owned_targets(
        targets,
        source_x=source_x,
        source_range=source_range,
        source_active=source_active,
        input_h=int(input_h),
        input_w=int(input_w),
        minimum_valid_rows=int(cfg.minimum_valid_rows),
    )
    route_target = build_v28_refined_route_targets(
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=candidate_valid,
        owned_x=owned["x_rows"],
        owned_valid=owned["valid_mask"],
        owned_matched=owned["matched"],
        input_h=int(input_h),
        line_width=float(cfg.line_width),
        minimum_valid_rows=int(cfg.minimum_valid_rows),
        temperature=float(cfg.target_temperature),
        support_delta=float(cfg.target_delta),
        support_floor=float(cfg.target_floor),
    )
    model_log_probability = outputs["candidate_scores"].float().masked_fill(
        ~route_target["valid"], -1.0e4
    ).log_softmax(dim=-1)
    per_slot = -(
        route_target["probability"] * model_log_probability
    ).sum(dim=-1)
    matched = route_target["matched"].float()
    route_loss = (per_slot * matched).sum() / matched.sum().clamp_min(1.0)
    field_loss = _field_row_loss(
        outputs["field_log_probability"],
        target_x=owned["x_rows"],
        target_valid=owned["valid_mask"],
        matched=route_target["matched"],
        input_w=int(input_w),
    )
    total = float(cfg.route) * route_loss
    if arm_name == "C":
        total = total + float(cfg.field) * field_loss

    with torch.no_grad():
        predicted = outputs["candidate_scores"].argmax(dim=-1)
        target_best = route_target["quality"].masked_fill(
            ~route_target["valid"], -1.0e4
        ).argmax(dim=-1)
        top1 = ((predicted == target_best) & route_target["matched"]).float()
        source_safe = source_route.long().clamp(min=0)
        source_mass = route_target["probability"].gather(
            -1, source_safe.unsqueeze(-1)
        ).squeeze(-1)
        diagnostics = {
            "loss_total": total.detach(),
            "loss_route": route_loss.detach(),
            "loss_field": field_loss.detach(),
            "matched_slots": matched.sum().detach(),
            "route_target_top1": (
                top1.sum() / matched.sum().clamp_min(1.0)
            ).detach(),
            "route_target_source_mass": (
                (source_mass * matched).sum() / matched.sum().clamp_min(1.0)
            ).detach(),
            "route_target_support": (
                (route_target["support"].sum(dim=-1).float() * matched).sum()
                / matched.sum().clamp_min(1.0)
            ).detach(),
            "route_target_best_quality": (
                (route_target["best_quality"] * matched).sum()
                / matched.sum().clamp_min(1.0)
            ).detach(),
        }
    return total, diagnostics


def v28_model_contract(model: V28RefinedBeliefRouter) -> dict[str, object]:
    return {
        "architecture": "V28RefinedBeliefRouter",
        "input_h": model.input_h,
        "input_w": model.input_w,
        "num_rows": model.num_rows,
        "x_bins": model.x_bins,
        "slots": 4,
        "continuous_geometry_head": False,
        "candidate_geometry_trainable": False,
        "writer_geometry": "frozen_v7_counterfactual_refined_candidate",
        "activity_owner": "exact_frozen_v7",
    }
