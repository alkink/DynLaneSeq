from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
)

from .backbone_dla import DLA34Backbone
from .common import fixed_indices, fixed_row_fractions, fixed_y_rows, sort_range_norm
from .fpn import SimpleFPN


@dataclass(frozen=True)
class V23LossWeights:
    row_distribution: float = 5.0
    point: float = 1.0
    strip_iou: float = 2.0
    quality50: float = 0.5
    quality75: float = 0.25
    smoothness: float = 0.25
    order: float = 0.25
    proposal_path: float = 1.0
    nondegradation: float = 2.0
    gate_regularization: float = 0.01
    tail_scale: float = 1.0
    proposal_temperature: float = 0.08
    line_width: float = 30.0
    minimum_valid_rows: int = 5


def _gather_slots(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    index = order
    while index.ndim < value.ndim:
        index = index.unsqueeze(-1)
    return value.gather(1, index.expand(*order.shape, *value.shape[2:]))


def _last_valid_x(x_rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    rows = int(x_rows.shape[-1])
    indices = fixed_indices(
        rows, device=x_rows.device, dtype=torch.long
    ).view(*([1] * (x_rows.ndim - 1)), rows)
    last = torch.where(valid, indices, torch.full_like(indices, -1)).amax(dim=-1)
    safe = last.clamp(min=0)
    value = x_rows.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return torch.where(last >= 0, value, torch.full_like(value, float("inf")))


def canonicalize_v7_slots(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Sort V7's deployed lane objects from left to right.

    The order key is each active lane's bottom-most row inside its deployed
    range.  Inactive slots are placed after all active slots.  This turns V7's
    arbitrary internal slot permutation into a stable inference-time semantic.
    """

    required = (
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_active",
    )
    missing = [name for name in required if name not in outputs]
    if missing:
        raise KeyError(f"V23 requires V7 four-slot outputs; missing {missing}")
    x_rows = outputs[required[0]].float()
    ranges = sort_range_norm(outputs[required[1]].float())
    active = outputs[required[2]].bool()
    if x_rows.ndim != 3 or x_rows.shape[1] != 4:
        raise ValueError("V23 requires exactly four V7 deployed slots")
    if ranges.shape != (*x_rows.shape[:2], 2) or active.shape != x_rows.shape[:2]:
        raise ValueError("V7 deployed slot tensors have incompatible shapes")
    rows = int(x_rows.shape[-1])
    row_fraction = fixed_row_fractions(
        rows, device=x_rows.device, dtype=x_rows.dtype
    ).view(1, 1, rows)
    visible = (
        active.unsqueeze(-1)
        & torch.isfinite(x_rows)
        & (row_fraction >= ranges[..., :1])
        & (row_fraction <= ranges[..., 1:])
    )
    bottom_x = _last_valid_x(x_rows, visible)
    bottom_x = torch.where(active, bottom_x, torch.full_like(bottom_x, float("inf")))
    order = bottom_x.argsort(dim=1, stable=True)
    result = {
        "x_rows": _gather_slots(x_rows, order),
        "range_norm": _gather_slots(ranges, order),
        "active": _gather_slots(active, order),
        "source_slot_indices": order,
    }
    active_logits = outputs.get("selection_slot_active_logits")
    if isinstance(active_logits, torch.Tensor) and active_logits.shape[:2] == active.shape:
        result["active_logits"] = _gather_slots(active_logits, order)
    return result


def _canonical_target_lanes(
    target: dict[str, torch.Tensor],
    *,
    device: torch.device,
    rows: int,
    input_w: int,
    minimum_valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_rows = target["x_rows"].to(device=device, dtype=torch.float32)
    valid = target["valid_mask"].to(device=device).bool()
    if x_rows.ndim != 2 or valid.shape != x_rows.shape:
        raise ValueError("CULane targets must contain x_rows/valid_mask [G,R]")
    if int(x_rows.shape[-1]) != rows:
        raise ValueError(f"target rows={x_rows.shape[-1]} but V23 rows={rows}")
    valid = valid & torch.isfinite(x_rows) & (x_rows >= 0.0) & (x_rows < float(input_w))
    keep = valid.sum(dim=-1) >= int(minimum_valid_rows)
    x_rows = x_rows[keep]
    valid = valid[keep]
    if int(x_rows.shape[0]) > 4:
        raise ValueError(
            "V23's first arm has exactly four ordered lane slots, but an "
            f"augmented training target contains {x_rows.shape[0]} valid lanes"
        )
    if int(x_rows.shape[0]) == 0:
        return x_rows, valid, x_rows.new_zeros((0, 2))
    bottom_x = _last_valid_x(x_rows, valid)
    order = bottom_x.argsort(stable=True)
    x_rows = x_rows[order]
    valid = valid[order]
    index = fixed_indices(rows, device=device, dtype=torch.long).view(1, rows)
    first = torch.where(valid, index, torch.full_like(index, rows)).amin(dim=-1)
    last = torch.where(valid, index, torch.full_like(index, -1)).amax(dim=-1)
    ranges = torch.stack(
        (first.float() / float(rows), last.float() / float(rows)), dim=-1
    )
    return x_rows, valid, ranges


def _batched_pairwise_common_row_distance(
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    target_x: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    input_h: int,
) -> torch.Tensor:
    """Vectorized [B,S,G] slot/GT distance without per-image syncs."""

    rows = int(source_x.shape[-1])
    y = fixed_y_rows(rows, input_h, device=source_x.device, dtype=torch.float32)
    source_valid = (
        torch.isfinite(source_x)
        & (y.view(1, 1, rows) >= source_range[..., 0:1] * float(input_h))
        & (y.view(1, 1, rows) <= source_range[..., 1:2] * float(input_h))
    )
    common = source_valid[:, :, None, :] & target_valid[:, None, :, :]
    count = common.sum(dim=-1)
    distance = (source_x[:, :, None, :] - target_x[:, None, :, :]).abs()
    mean = (distance * common.float()).sum(dim=-1) / count.clamp_min(1).float()
    return torch.where(count > 0, mean, torch.full_like(mean, 1.0e6))


def _monotonic_assignment_from_cost(
    pair_cost: list[list[float]],
    active: list[bool],
    gt_count: int,
) -> list[int]:
    """Assign ordered V7 slots to ordered GT without crossing identities."""

    active_ids = [index for index, is_active in enumerate(active) if is_active]
    assignment = [-1] * len(active)
    if not active_ids or gt_count == 0:
        return assignment
    best_cost = float("inf")
    best_pairs: tuple[tuple[int, int], ...] = tuple()
    if len(active_ids) <= gt_count:
        for gt_ids in combinations(range(gt_count), len(active_ids)):
            pairs = tuple(zip(range(len(active_ids)), gt_ids))
            cost = sum(pair_cost[active_ids[a]][g] for a, g in pairs)
            if cost < best_cost:
                best_cost, best_pairs = cost, pairs
    else:
        for active_positions in combinations(range(len(active_ids)), gt_count):
            pairs = tuple(zip(active_positions, range(gt_count)))
            cost = sum(pair_cost[active_ids[a]][g] for a, g in pairs)
            if cost < best_cost:
                best_cost, best_pairs = cost, pairs
    for active_position, gt in best_pairs:
        assignment[active_ids[active_position]] = int(gt)
    return assignment


def build_v23_owned_targets(
    targets: list[dict[str, torch.Tensor]],
    *,
    source_x: torch.Tensor,
    source_range: torch.Tensor,
    source_active: torch.Tensor,
    input_h: int,
    input_w: int,
    minimum_valid_rows: int = 5,
) -> dict[str, torch.Tensor]:
    """Create GT-owned targets for the canonical V7 lane objects.

    When V7 and GT counts differ, an ordered minimum-distance assignment is
    used.  It preserves left/right topology and avoids silently shifting every
    lane identity after a missed outer lane.
    """

    batch, slots, rows = source_x.shape
    if slots != 4 or len(targets) != batch:
        raise ValueError("V23 owned-target batch/slot mismatch")
    # Targets arrive as a CPU list with at most four lanes. Canonicalize and
    # pad there, then perform one batched host-to-device copy instead of two
    # tiny transfers plus two synchronizations for every image.
    canonical_x_cpu = torch.zeros((batch, slots, rows), dtype=torch.float32)
    canonical_valid_cpu = torch.zeros((batch, slots, rows), dtype=torch.bool)
    canonical_range_cpu = torch.zeros((batch, slots, 2), dtype=torch.float32)
    target_counts: list[int] = []
    for batch_index, target in enumerate(targets):
        target_x, target_valid, target_range = _canonical_target_lanes(
            target,
            device=torch.device("cpu"),
            rows=rows,
            input_w=input_w,
            minimum_valid_rows=minimum_valid_rows,
        )
        target_count = int(target_x.shape[0])
        target_counts.append(target_count)
        if target_count:
            canonical_x_cpu[batch_index, :target_count] = target_x
            canonical_valid_cpu[batch_index, :target_count] = target_valid
            canonical_range_cpu[batch_index, :target_count] = target_range

    canonical_x = canonical_x_cpu.to(device=source_x.device, non_blocking=True)
    canonical_valid = canonical_valid_cpu.to(
        device=source_x.device, non_blocking=True
    )
    canonical_range = canonical_range_cpu.to(
        device=source_x.device, non_blocking=True
    )
    pair_cost = _batched_pairwise_common_row_distance(
        source_x.detach().float(),
        source_range.detach().float(),
        canonical_x,
        canonical_valid,
        input_h=input_h,
    )
    # Copy the complete 4x4 cost matrices and four activity flags together.
    # This is the only GPU-to-CPU synchronization in ownership construction.
    assignment_payload = torch.cat(
        (pair_cost.reshape(batch, -1), source_active.detach().float()), dim=-1
    ).cpu().tolist()
    assignment_rows: list[list[int]] = []
    for batch_index, payload in enumerate(assignment_payload):
        flat_cost = payload[: slots * slots]
        cost = [
            flat_cost[row * slots : (row + 1) * slots]
            for row in range(slots)
        ]
        active = [bool(value) for value in payload[slots * slots :]]
        assignment_rows.append(
            _monotonic_assignment_from_cost(
                cost,
                active,
                target_counts[batch_index],
            )
        )
    owned_index = torch.tensor(
        assignment_rows, dtype=torch.long, device=source_x.device
    )
    has_target = owned_index >= 0
    safe_index = owned_index.clamp_min(0)
    owned_x = canonical_x.gather(
        1, safe_index.unsqueeze(-1).expand(batch, slots, rows)
    )
    owned_valid = canonical_valid.gather(
        1, safe_index.unsqueeze(-1).expand(batch, slots, rows)
    )
    owned_range = canonical_range.gather(
        1, safe_index.unsqueeze(-1).expand(batch, slots, 2)
    )
    owned_x = torch.where(has_target.unsqueeze(-1), owned_x, torch.zeros_like(owned_x))
    owned_valid = owned_valid & has_target.unsqueeze(-1)
    owned_range = torch.where(
        has_target.unsqueeze(-1), owned_range, torch.zeros_like(owned_range)
    )
    matched = (owned_index >= 0) & source_active.bool()
    return {
        "x_rows": owned_x,
        "valid_mask": owned_valid,
        "range_norm": owned_range,
        "target_indices": owned_index,
        "matched": matched,
    }


def _shifted_transition(
    value: torch.Tensor,
    *,
    radius: int,
    transition_penalty: float,
    penalty_vector: torch.Tensor | None = None,
) -> torch.Tensor:
    radius = int(radius)
    if radius < 0:
        raise ValueError("transition radius must be non-negative")
    if radius == 0:
        return value
    # The previous implementation launched one slice, pad, subtraction and
    # stack input for every offset at every one of 319 recurrent row steps.
    # A padded sliding window contains exactly the same predecessor states and
    # preserves the original log-sum-exp transition while issuing one batched
    # GPU operation per row.
    window = 2 * radius + 1
    neighbours = F.pad(value, (radius, radius), value=-1.0e4).unfold(
        -1, window, 1
    )
    penalty = penalty_vector
    if penalty is None:
        penalty = (
            fixed_indices(window, device=value.device, dtype=value.dtype) - radius
        ).abs() * float(transition_penalty)
    return torch.logsumexp(neighbours - penalty, dim=-1)


def soft_viterbi_marginals(
    unary_logits: torch.Tensor,
    *,
    transition_radius_bins: int = 8,
    transition_penalty: float = 0.15,
) -> torch.Tensor:
    """Bidirectional differentiable path messages for [B,S,R,X] unaries."""

    if unary_logits.ndim != 4:
        raise ValueError("V23 path decoder expects [B,S,R,X] unary logits")
    if int(transition_radius_bins) < 0:
        raise ValueError("transition radius must be non-negative")
    unary = unary_logits.float()
    rows = int(unary.shape[-2])
    window = 2 * int(transition_radius_bins) + 1
    penalty_vector = (
        fixed_indices(window, device=unary.device, dtype=unary.dtype)
        - int(transition_radius_bins)
    ).abs() * float(transition_penalty)
    forward: list[torch.Tensor] = [unary[..., 0, :]]
    for row in range(1, rows):
        message = _shifted_transition(
            forward[-1],
            radius=transition_radius_bins,
            transition_penalty=transition_penalty,
            penalty_vector=penalty_vector,
        )
        state = unary[..., row, :] + message
        forward.append(state - torch.logsumexp(state, dim=-1, keepdim=True))
    backward: list[torch.Tensor] = [torch.zeros_like(unary[..., -1, :])]
    for row in range(rows - 2, -1, -1):
        next_state = unary[..., row + 1, :] + backward[-1]
        message = _shifted_transition(
            next_state,
            radius=transition_radius_bins,
            transition_penalty=transition_penalty,
            penalty_vector=penalty_vector,
        )
        backward.append(
            message - torch.logsumexp(message, dim=-1, keepdim=True)
        )
    backward.reverse()
    forward_tensor = torch.stack(forward, dim=-2)
    backward_tensor = torch.stack(backward, dim=-2)
    marginal = forward_tensor + backward_tensor
    return marginal - torch.logsumexp(marginal, dim=-1, keepdim=True)


class V23OrderedSlotCostVolume(nn.Module):
    """End-to-end ordered four-lane cost-volume student.

    V7 supplies count, ranges, an initialization path and proposal memory.  It
    never supplies the final coordinates.  The student creates one global row
    distribution for each ordered lane object and decodes a continuous path.
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
        teacher_prior_sigma_px: float = 18.0,
        teacher_prior_weight: float = 1.0,
        transition_radius_bins: int = 8,
        transition_penalty: float = 0.15,
        freeze_batch_norm_stats: bool = True,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("V23 query_dim must be divisible by num_heads")
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.teacher_prior_sigma_px = float(teacher_prior_sigma_px)
        self.teacher_prior_weight = float(teacher_prior_weight)
        self.transition_radius_bins = int(transition_radius_bins)
        self.transition_penalty = float(transition_penalty)
        self.freeze_batch_norm_stats = bool(freeze_batch_norm_stats)

        self.backbone = DLA34Backbone(pretrained=False)
        self.fpn = SimpleFPN(
            in_channels=self.backbone.out_channels,
            out_channels=int(fpn_channels),
        )
        # P2 is stride four. This independent stride-two branch keeps genuine
        # two-pixel image evidence instead of merely interpolating P2 to 800.
        self.fine_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.p2_projection = nn.Sequential(
            nn.Conv2d(int(fpn_channels), int(hidden_dim), 1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
        )
        self.fine_projection = nn.Sequential(
            nn.Conv2d(32, int(hidden_dim), 1, bias=False),
            nn.GroupNorm(_group_count(int(hidden_dim)), int(hidden_dim)),
            nn.GELU(),
        )
        self.image_fusion = nn.Sequential(
            nn.Conv2d(2 * int(hidden_dim), int(query_dim), 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(int(query_dim)), int(query_dim)),
            nn.GELU(),
            nn.Conv2d(int(query_dim), int(query_dim), 1, bias=False),
        )
        self.key_projection = nn.Conv2d(int(query_dim), int(query_dim), 1, bias=False)
        self.slot_embedding = nn.Parameter(torch.empty(4, int(query_dim)))
        self.row_embedding = nn.Parameter(torch.empty(self.num_rows, int(query_dim)))
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
            vertical_layer, num_layers=int(vertical_layers), norm=nn.LayerNorm(int(query_dim))
        )
        self.inter_attention = nn.MultiheadAttention(
            int(query_dim), int(num_heads), dropout=0.1, batch_first=True
        )
        self.inter_norm = nn.LayerNorm(int(query_dim))
        self.query_norm = nn.LayerNorm(int(query_dim))
        self.quality_head = nn.Sequential(
            nn.LayerNorm(int(query_dim)),
            nn.Linear(int(query_dim), int(query_dim)),
            nn.GELU(),
            nn.Linear(int(query_dim), 2),
        )
        # A zero forward gate makes the untrained V23 endpoint exactly V7,
        # while the straight-through training path still teaches the complete
        # cost volume from the first step.
        self.geometry_gate = nn.Parameter(torch.zeros(4))

        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm_stats:
            for module in (*self.backbone.modules(), *self.fpn.modules()):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def copy_v22_encoder_(self, v22_model: nn.Module) -> dict[str, float | int]:
        if not hasattr(v22_model, "backbone") or not hasattr(v22_model, "fpn"):
            raise TypeError("V23 warm start requires V22 backbone/fpn")
        self.backbone.load_state_dict(v22_model.backbone.state_dict(), strict=True)
        self.fpn.load_state_dict(v22_model.fpn.state_dict(), strict=True)
        difference = 0.0
        tensors = 0
        for target_module, source_module in (
            (self.backbone, v22_model.backbone),
            (self.fpn, v22_model.fpn),
        ):
            source = dict(source_module.named_parameters())
            for name, value in target_module.named_parameters():
                difference = max(
                    difference,
                    float((value.detach() - source[name].detach()).abs().max()),
                )
                tensors += 1
        return {"parameter_tensors": tensors, "maximum_copy_difference": difference}

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        p2 = self.fpn(self.backbone(images))
        coarse = self.p2_projection(p2)
        fine = self.fine_projection(self.fine_stem(images))
        target_size = (self.num_rows, self.x_bins)
        coarse = F.interpolate(coarse, size=target_size, mode="bilinear", align_corners=False)
        fine = F.interpolate(fine, size=target_size, mode="bilinear", align_corners=False)
        return self.image_fusion(torch.cat((coarse, fine), dim=1))

    def _teacher_prior(self, source_x: torch.Tensor) -> torch.Tensor:
        bin_width = float(self.input_w) / float(self.x_bins)
        centres = (
            fixed_indices(
                self.x_bins, device=source_x.device, dtype=torch.float32
            )
            + 0.5
        ) * bin_width
        delta = centres.view(1, 1, 1, -1) - source_x.float().unsqueeze(-1)
        return -0.5 * (delta / self.teacher_prior_sigma_px).pow(2)

    def _proposal_scores(
        self,
        path_logits: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, rows, bins = path_logits.shape
        proposals = int(proposal_x.shape[1])
        if proposal_x.shape != (batch, proposals, rows):
            raise ValueError("V23 proposal rows mismatch")
        # soft_viterbi_marginals already returns normalized log-probability.
        # Re-normalizing this full [B,S,R,X] tensor here duplicated a costly
        # reduction and only changed round-off at the 1e-6 scale.
        log_probability = path_logits.float()
        position = proposal_x.float() / (float(self.input_w) / float(bins)) - 0.5
        lower = position.floor().long().clamp(0, bins - 1)
        upper = (lower + 1).clamp(max=bins - 1)
        fraction = (position - lower.float()).clamp(0.0, 1.0)
        expanded_lower = lower[:, None].expand(batch, slots, proposals, rows)
        expanded_upper = upper[:, None].expand_as(expanded_lower)
        # Gather K proposal coordinates from each row without materialising a
        # prohibitively large [B,S,K,R,X] copy of the cost volume.
        lower_value = log_probability.gather(
            -1, expanded_lower.permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)
        upper_value = log_probability.gather(
            -1, expanded_upper.permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)
        sampled = lower_value * (1.0 - fraction[:, None]) + upper_value * fraction[:, None]
        row_fraction = fixed_row_fractions(
            rows, device=path_logits.device, dtype=torch.float32
        ).view(1, 1, rows)
        proposal_range = sort_range_norm(proposal_range.float())
        valid = (
            torch.isfinite(proposal_x)
            & (proposal_x >= 0.0)
            & (proposal_x < float(self.input_w))
            & (row_fraction >= proposal_range[..., :1])
            & (row_fraction <= proposal_range[..., 1:])
        )
        weight = valid[:, None].float()
        score = (sampled * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
        candidate_valid = valid.sum(dim=-1) >= 5
        return score.masked_fill(~candidate_valid[:, None], -1.0e4), candidate_valid

    def forward(
        self,
        images: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor],
        *,
        include_proposal_scores: bool = True,
    ) -> dict[str, torch.Tensor]:
        source = canonicalize_v7_slots(teacher_outputs)
        # Clone the detached teacher state so a caller may safely obtain it
        # under inference_mode; inference tensors cannot be saved by the
        # trainable geometry projection during autograd.
        source_x = source["x_rows"].detach().to(
            device=images.device, dtype=torch.float32
        ).clone()
        source_range = source["range_norm"].detach().to(
            device=images.device, dtype=torch.float32
        ).clone()
        source_active = source["active"].detach().to(device=images.device).bool().clone()
        batch, slots, rows = source_x.shape
        if rows != self.num_rows or slots != 4:
            raise ValueError("V23/V7 row or slot contract mismatch")

        image_features = self._image_features(images)
        keys = self.key_projection(image_features)
        row_fraction = fixed_row_fractions(
            rows, device=images.device, dtype=torch.float32
        ).view(1, 1, rows)
        geometry = torch.stack(
            (
                source_x / float(self.input_w),
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
        teacher_prior = self._teacher_prior(source_x)
        initial_image_logits = torch.einsum("bsrc,bcrx->bsrx", query, keys) * scale
        initial_logits = initial_image_logits + self.teacher_prior_weight * teacher_prior
        initial_probability = initial_logits.softmax(dim=-1)
        context = torch.einsum("bsrx,bcrx->bsrc", initial_probability, image_features)
        query = query + self.context_projection(context)
        query = self.vertical_encoder(query.reshape(batch * slots, rows, -1)).reshape(
            batch, slots, rows, -1
        )
        inter_input = query.permute(0, 2, 1, 3).reshape(batch * rows, slots, -1)
        inter_output, _ = self.inter_attention(
            inter_input, inter_input, inter_input, need_weights=False
        )
        inter_output = self.inter_norm(inter_input + inter_output)
        query = inter_output.reshape(batch, rows, slots, -1).permute(0, 2, 1, 3)
        query = self.query_norm(query)
        refined_image_logits = torch.einsum("bsrc,bcrx->bsrx", query, keys) * scale
        unary_logits = (
            initial_image_logits
            + refined_image_logits
            + self.teacher_prior_weight * teacher_prior
        )
        path_logits = soft_viterbi_marginals(
            unary_logits,
            transition_radius_bins=self.transition_radius_bins,
            transition_penalty=self.transition_penalty,
        )
        posterior = path_logits.softmax(dim=-1)
        bin_width = float(self.input_w) / float(self.x_bins)
        centres = (
            fixed_indices(self.x_bins, device=images.device, dtype=torch.float32) + 0.5
        ) * bin_width
        student_x = (posterior * centres.view(1, 1, 1, -1)).sum(dim=-1)
        correction = student_x - source_x
        gate = self.geometry_gate.clamp(-1.0, 1.0).view(1, slots, 1)
        deployed_correction = gate * correction
        if self.training:
            deployed_correction = deployed_correction + (
                1.0 - gate.detach()
            ) * (correction - correction.detach())
        final_x = (source_x + deployed_correction).clamp(0.0, float(self.input_w - 1))

        pooled_query = query.mean(dim=2)
        quality = self.quality_head(pooled_query)
        ordered_exist_logits = torch.stack(
            (
                torch.where(source_active, torch.full_like(source_x[..., 0], 20.0), torch.full_like(source_x[..., 0], -20.0)),
                torch.where(source_active, torch.full_like(source_x[..., 0], -20.0), torch.full_like(source_x[..., 0], 20.0)),
            ),
            dim=-1,
        )
        # The internal semantic is canonical left-to-right. Restore V7's
        # public slot order at the writer boundary so zero-step prediction
        # files remain byte-identical to the source detector.
        inverse_order = source["source_slot_indices"].argsort(dim=1, stable=True)
        public_x = _gather_slots(final_x, inverse_order)
        public_range = _gather_slots(source_range, inverse_order)
        public_exist_logits = _gather_slots(ordered_exist_logits, inverse_order)
        public_quality = _gather_slots(quality, inverse_order)
        output: dict[str, torch.Tensor] = {
            "exist_logits": public_exist_logits,
            "pred_x_rows": public_x,
            "range_norm": public_range,
            "quality_logits": public_quality[..., 0],
            "quality50_logits": quality[..., 0],
            "quality75_logits": quality[..., 1],
            "ordered_pred_x_rows": final_x,
            "ordered_range_norm": source_range,
            "unary_logits": unary_logits,
            "path_logits": path_logits,
            "path_posterior": posterior,
            "student_x_rows": student_x,
            "source_x_rows": source_x,
            "source_range_norm": source_range,
            "source_active": source_active,
            "source_slot_indices": source["source_slot_indices"],
            "geometry_gate": gate.expand(batch, slots, 1),
        }
        proposal_x = teacher_outputs.get("pred_x_rows")
        proposal_range = teacher_outputs.get("range_norm")
        if (
            include_proposal_scores
            and isinstance(proposal_x, torch.Tensor)
            and isinstance(proposal_range, torch.Tensor)
        ):
            proposal_x = proposal_x.detach().to(device=images.device, dtype=torch.float32)
            proposal_range = proposal_range.detach().to(device=images.device, dtype=torch.float32)
            proposal_scores, proposal_valid = self._proposal_scores(
                path_logits, proposal_x, proposal_range
            )
            output.update(
                {
                    "proposal_path_scores": proposal_scores,
                    "teacher_proposal_x_rows": proposal_x,
                    "teacher_proposal_range_norm": proposal_range,
                    "teacher_proposal_valid": proposal_valid,
                }
            )
        return output


def _aligned_strip_iou(
    pred_x: torch.Tensor,
    pred_range: torch.Tensor,
    target_x: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    input_h: int,
    line_width: float,
) -> torch.Tensor:
    batch, slots, rows = pred_x.shape
    y = fixed_y_rows(rows, input_h, device=pred_x.device, dtype=torch.float32)
    pred_range = sort_range_norm(pred_range.float())
    pred_valid = (
        torch.isfinite(pred_x)
        & (y.view(1, 1, rows) >= pred_range[..., :1] * float(input_h))
        & (y.view(1, 1, rows) <= pred_range[..., 1:] * float(input_h))
    )
    target_valid = target_valid.bool() & torch.isfinite(target_x)
    both = pred_valid & target_valid
    either = pred_valid | target_valid
    overlap = (float(line_width) - (pred_x.float() - target_x.float()).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(either, torch.full_like(overlap, float(line_width)), torch.zeros_like(overlap)),
    )
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1.0e-6)


def _row_distribution_loss(
    logits: torch.Tensor,
    target_x: torch.Tensor,
    valid: torch.Tensor,
    slot_weight: torch.Tensor,
    *,
    input_w: int,
) -> torch.Tensor:
    bins = int(logits.shape[-1])
    position = target_x.float() / (float(input_w) / float(bins)) - 0.5
    lower = position.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = (position - lower.float()).clamp(0.0, 1.0)
    # V23's path decoder contract is normalized log-probability. Avoid a
    # second full-width log_softmax on every row during loss construction.
    log_probability = logits.float()
    lower_value = log_probability.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_value = log_probability.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    nll = -((1.0 - fraction) * lower_value + fraction * upper_value)
    weight = valid.float() * slot_weight.unsqueeze(-1)
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


def _proposal_path_ranking_loss(
    outputs: dict[str, torch.Tensor],
    owned: dict[str, torch.Tensor],
    *,
    input_h: int,
    temperature: float,
    line_width: float,
    minimum_valid_rows: int,
) -> torch.Tensor:
    scores = outputs.get("proposal_path_scores")
    proposal_x = outputs.get("teacher_proposal_x_rows")
    proposal_range = outputs.get("teacher_proposal_range_norm")
    candidate_valid = outputs.get("teacher_proposal_valid")
    if not all(isinstance(value, torch.Tensor) for value in (scores, proposal_x, proposal_range, candidate_valid)):
        return outputs["pred_x_rows"].sum() * 0.0
    batch, slots, candidates = scores.shape
    rows = int(proposal_x.shape[-1])
    expanded_x = proposal_x[:, None].expand(batch, slots, candidates, rows).reshape(
        batch * slots, candidates, rows
    )
    expanded_range = proposal_range[:, None].expand(
        batch, slots, candidates, 2
    ).reshape(batch * slots, candidates, 2)
    target_x = owned["x_rows"].reshape(batch * slots, 1, rows)
    target_valid = owned["valid_mask"].reshape(batch * slots, 1, rows)
    with torch.no_grad():
        quality, _, _ = batched_pairwise_range_aware_row_strip_iou(
            expanded_x,
            expanded_range,
            target_x,
            target_valid,
            input_h=input_h,
            line_width=line_width,
            min_valid_rows=minimum_valid_rows,
        )
        quality = quality[..., 0].reshape(batch, slots, candidates)
        valid = candidate_valid[:, None].expand_as(quality) & owned["matched"].unsqueeze(-1)
        target_logits = (quality / float(temperature)).masked_fill(~valid, -1.0e4)
        target_probability = target_logits.softmax(dim=-1)
    model_log_probability = scores.float().masked_fill(~valid, -1.0e4).log_softmax(dim=-1)
    per_slot = -(target_probability * model_log_probability).sum(dim=-1)
    matched = owned["matched"].float()
    return (per_slot * matched).sum() / matched.sum().clamp_min(1.0)


def v23_ordered_cost_volume_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    input_h: int,
    input_w: int,
    weights: V23LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = weights or V23LossWeights()
    owned = build_v23_owned_targets(
        targets,
        source_x=outputs["source_x_rows"],
        source_range=outputs["source_range_norm"],
        source_active=outputs["source_active"],
        input_h=input_h,
        input_w=input_w,
        minimum_valid_rows=cfg.minimum_valid_rows,
    )
    matched = owned["matched"]
    source_iou = _aligned_strip_iou(
        outputs["source_x_rows"],
        outputs["source_range_norm"],
        owned["x_rows"],
        owned["valid_mask"],
        input_h=input_h,
        line_width=cfg.line_width,
    ).detach()
    predicted_iou = _aligned_strip_iou(
        outputs["ordered_pred_x_rows"],
        outputs["ordered_range_norm"],
        owned["x_rows"],
        owned["valid_mask"],
        input_h=input_h,
        line_width=cfg.line_width,
    )
    tail_weight = (1.0 + float(cfg.tail_scale) * (1.0 - source_iou)).clamp(1.0, 3.0)
    tail_weight = tail_weight * matched.float()

    row_distribution = _row_distribution_loss(
        outputs["path_logits"],
        owned["x_rows"],
        owned["valid_mask"],
        tail_weight,
        input_w=input_w,
    )
    point_weight = owned["valid_mask"].float() * tail_weight.unsqueeze(-1)
    point = (
        F.smooth_l1_loss(
            outputs["ordered_pred_x_rows"].float() / float(input_w),
            owned["x_rows"].float() / float(input_w),
            reduction="none",
            beta=0.01,
        )
        * point_weight
    ).sum() / point_weight.sum().clamp_min(1.0)
    strip_iou = ((1.0 - predicted_iou) * tail_weight).sum() / tail_weight.sum().clamp_min(1.0)

    quality_target50 = (predicted_iou.detach() >= 0.50).float()
    quality_target75 = (predicted_iou.detach() >= 0.75).float()
    quality50 = F.binary_cross_entropy_with_logits(
        outputs["quality50_logits"].float(), quality_target50, reduction="none"
    )
    quality75 = F.binary_cross_entropy_with_logits(
        outputs["quality75_logits"].float(), quality_target75, reduction="none"
    )
    quality50 = (quality50 * matched.float()).sum() / matched.float().sum().clamp_min(1.0)
    quality75 = (quality75 * matched.float()).sum() / matched.float().sum().clamp_min(1.0)

    pred_delta = outputs["ordered_pred_x_rows"][..., 1:] - outputs["ordered_pred_x_rows"][..., :-1]
    target_delta = owned["x_rows"][..., 1:] - owned["x_rows"][..., :-1]
    pair_valid = owned["valid_mask"][..., 1:] & owned["valid_mask"][..., :-1]
    smooth_weight = pair_valid.float() * tail_weight.unsqueeze(-1)
    smoothness = (
        F.smooth_l1_loss(
            pred_delta / float(input_w),
            target_delta / float(input_w),
            reduction="none",
            beta=0.005,
        )
        * smooth_weight
    ).sum() / smooth_weight.sum().clamp_min(1.0)

    row_valid = owned["valid_mask"] & matched.unsqueeze(-1)
    adjacent_valid = row_valid[:, :-1] & row_valid[:, 1:]
    spacing = outputs["ordered_pred_x_rows"][:, 1:] - outputs["ordered_pred_x_rows"][:, :-1]
    order = (
        F.relu(4.0 - spacing) * adjacent_valid.float()
    ).sum() / adjacent_valid.float().sum().clamp_min(1.0)

    good50 = matched & (source_iou >= 0.50)
    good75 = matched & (source_iou >= 0.75)
    nondegradation = (
        F.relu(source_iou - predicted_iou) * good50.float()
        + 2.0 * F.relu(source_iou - predicted_iou) * good75.float()
    ).sum() / (good50.float().sum() + 2.0 * good75.float().sum()).clamp_min(1.0)

    proposal_path = _proposal_path_ranking_loss(
        outputs,
        owned,
        input_h=input_h,
        temperature=cfg.proposal_temperature,
        line_width=cfg.line_width,
        minimum_valid_rows=cfg.minimum_valid_rows,
    )
    gate_regularization = outputs["geometry_gate"].float().pow(2).mean()
    total = (
        cfg.row_distribution * row_distribution
        + cfg.point * point
        + cfg.strip_iou * strip_iou
        + cfg.quality50 * quality50
        + cfg.quality75 * quality75
        + cfg.smoothness * smoothness
        + cfg.order * order
        + cfg.proposal_path * proposal_path
        + cfg.nondegradation * nondegradation
        + cfg.gate_regularization * gate_regularization
    )
    with torch.no_grad():
        diagnostics = {
            "loss_total": total.detach(),
            "loss_row_distribution": row_distribution.detach(),
            "loss_point": point.detach(),
            "loss_strip_iou": strip_iou.detach(),
            "loss_quality50": quality50.detach(),
            "loss_quality75": quality75.detach(),
            "loss_smoothness": smoothness.detach(),
            "loss_order": order.detach(),
            "loss_proposal_path": proposal_path.detach(),
            "loss_nondegradation": nondegradation.detach(),
            "loss_gate_regularization": gate_regularization.detach(),
            "mean_source_iou": (source_iou * matched.float()).sum()
            / matched.float().sum().clamp_min(1.0),
            "mean_predicted_iou": (predicted_iou * matched.float()).sum()
            / matched.float().sum().clamp_min(1.0),
            "matched_slots": matched.sum().float(),
            "geometry_gate_mean": outputs["geometry_gate"].mean(),
        }
    return total, diagnostics


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def v23_model_contract(model: V23OrderedSlotCostVolume) -> dict[str, Any]:
    return {
        "input_h": model.input_h,
        "input_w": model.input_w,
        "num_rows": model.num_rows,
        "x_bins": model.x_bins,
        "slots": 4,
        "output_semantics": "ordered_slot_conditioned_row_cost_volume",
        "final_geometry_owner": "student_cost_volume",
        "teacher_activity_count_frozen": True,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
