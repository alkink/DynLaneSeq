from __future__ import annotations

import math
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    fixed_indices,
    fixed_row_fractions,
    fixed_sample_indices,
    sort_range_norm,
)
from .v16_candidate_reranker import FourSlotCandidateAlignedReranker
from .v17_iterative_slot_geometry import FourSlotIterativeMultiScaleGeometry
from .v18_joint_exact_set_energy import FourSlotJointExactSetEnergy
from .v19_counterfactual_fidelity import (
    FourSlotCounterfactualProposalFidelity,
    frozen_v7_counterfactual_anchors,
)
from .v20_slot_owned_replacement import SlotOwnedSafeReplacementHead
from .v30_joint_slot_field import FourSlotJointBeliefField


def _count_repeated_real_indices(indices: torch.Tensor) -> torch.Tensor:
    """Count repeated non-negative indices without per-image Python loops."""

    if indices.ndim != 2:
        raise ValueError("slot indices must have shape [B,S]")
    slots = int(indices.shape[1])
    if slots <= 1:
        return indices.new_zeros((indices.shape[0],))
    same = indices.unsqueeze(-1) == indices.unsqueeze(-2)
    same = same & (indices.unsqueeze(-1) >= 0)
    # Row ``s`` is a duplicate iff the same real proposal appeared in an
    # earlier slot.  This is exactly ``active_count - unique_count`` while
    # avoiding one CUDA ``unique`` launch per image.
    repeated = torch.tril(same, diagonal=-1).any(dim=-1)
    return repeated.sum(dim=-1)


@torch.no_grad()
def decode_unique_four_slot_routes(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    combinations: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Exactly decode a tiny slot/proposal assignment on the input device.

    With ``S`` slots, an optimal unique assignment can only use a slot's top
    ``S`` real proposals (or its dustbin): at most ``S - 1`` better proposals
    can be occupied by the other slots.  Enumerating ``(S + 1) ** S`` choices
    is therefore exact while avoiding a GPU-to-CPU SciPy round trip in every
    training forward.  Dustbin choices are private and may repeat.
    """

    squeeze = False
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        candidate_valid = candidate_valid.unsqueeze(0)
        squeeze = True
    if logits.ndim != 3:
        raise ValueError("four-slot logits must have shape [B,S,N+1]")
    batch, slots, classes = logits.shape
    candidates = int(classes) - 1
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("four-slot candidate_valid must have shape [B,N]")
    if int(slots) < 1 or candidates < 1:
        raise ValueError("four-slot decoding requires slots and candidates")

    real_logits = logits[..., :candidates].float().masked_fill(
        ~candidate_valid[:, None, :],
        float("-inf"),
    )
    top_count = min(int(slots), candidates)
    top_values, top_indices = real_logits.topk(top_count, dim=-1)
    top_valid = torch.isfinite(top_values)
    if top_count < int(slots):
        padding = int(slots) - top_count
        top_values = F.pad(top_values, (0, padding), value=float("-inf"))
        top_indices = F.pad(top_indices, (0, padding), value=0)
        top_valid = F.pad(top_valid, (0, padding), value=False)

    dustbin_index = torch.full(
        (batch, int(slots), 1),
        candidates,
        dtype=torch.long,
        device=logits.device,
    )
    option_indices = torch.cat((top_indices, dustbin_index), dim=-1)
    option_values = torch.cat((top_values, logits[..., -1:].float()), dim=-1)
    option_valid = torch.cat(
        (
            top_valid,
            torch.ones(
                (batch, int(slots), 1),
                dtype=torch.bool,
                device=logits.device,
            ),
        ),
        dim=-1,
    )

    if combinations is None:
        combinations = torch.tensor(
            tuple(product(range(int(slots) + 1), repeat=int(slots))),
            dtype=torch.long,
            device=logits.device,
        )
    else:
        combinations = combinations.to(device=logits.device, dtype=torch.long)
        expected_shape = ((int(slots) + 1) ** int(slots), int(slots))
        if tuple(combinations.shape) != expected_shape:
            raise ValueError(
                "invalid precomputed four-slot route combinations: "
                f"{tuple(combinations.shape)} != {expected_shape}"
            )
    combination_count = int(combinations.shape[0])
    gather_index = combinations.view(
        1,
        combination_count,
        int(slots),
        1,
    ).expand(batch, -1, -1, -1)
    chosen_indices = option_indices.unsqueeze(1).expand(
        -1,
        combination_count,
        -1,
        -1,
    ).gather(3, gather_index).squeeze(-1)
    chosen_values = option_values.unsqueeze(1).expand(
        -1,
        combination_count,
        -1,
        -1,
    ).gather(3, gather_index).squeeze(-1)
    chosen_valid = option_valid.unsqueeze(1).expand(
        -1,
        combination_count,
        -1,
        -1,
    ).gather(3, gather_index).squeeze(-1)
    valid_combination = chosen_valid.all(dim=-1)
    for left in range(int(slots)):
        for right in range(left + 1, int(slots)):
            duplicate_real = (
                (chosen_indices[..., left] < candidates)
                & (chosen_indices[..., left] == chosen_indices[..., right])
            )
            valid_combination &= ~duplicate_real
    score = chosen_values.sum(dim=-1).masked_fill(
        ~valid_combination,
        float("-inf"),
    )
    best_combination = score.argmax(dim=-1)
    batch_ids = fixed_indices(
        batch,
        device=logits.device,
        dtype=torch.long,
    )
    assigned_class = chosen_indices[batch_ids, best_combination]
    selected = torch.where(
        assigned_class < candidates,
        assigned_class,
        assigned_class.new_full(assigned_class.shape, -1),
    )
    probability = torch.softmax(logits.float(), dim=-1)
    probability_class = assigned_class.clamp(max=candidates)
    selected_probability = probability.gather(
        -1,
        probability_class.unsqueeze(-1),
    ).squeeze(-1)
    raw_class = logits.argmax(dim=-1)
    raw_indices = torch.where(
        raw_class < candidates,
        raw_class,
        raw_class.new_full(raw_class.shape, -1),
    )
    repair_count = (selected != raw_indices).sum(dim=-1)
    raw_collision_count = _count_repeated_real_indices(raw_indices)
    result = {
        "indices": selected,
        "scores": selected_probability,
        "raw_collision_count": raw_collision_count,
        "repair_count": repair_count,
    }
    if squeeze:
        return {name: value[0] for name, value in result.items()}
    return result


@torch.no_grad()
def decode_unique_real_slot_routes(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    combinations: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Decode one globally-unique *real* proposal for every lane slot.

    Cardinality is deliberately absent from this operator.  A separate active
    head decides whether a routed slot is emitted, while this assignment gives
    every slot a concrete geometry hypothesis.  Consequently an inactive slot
    cannot steal geometry supervision by selecting a dustbin class.

    For ``S`` slots an optimum can only use each slot's top ``S`` candidates:
    at most ``S - 1`` better candidates can be occupied by other slots.  The
    resulting ``S ** S`` search is exact for the normal ``N >= S`` case.
    """

    squeeze = False
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        candidate_valid = candidate_valid.unsqueeze(0)
        squeeze = True
    if logits.ndim != 3:
        raise ValueError("real route logits must have shape [B,S,N]")
    batch, slots, candidates = logits.shape
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("real route candidate_valid must have shape [B,N]")
    if int(slots) < 1 or int(candidates) < 1:
        raise ValueError("real route decoding requires slots and candidates")

    masked = logits.float().masked_fill(
        ~candidate_valid[:, None, :].bool(),
        float("-inf"),
    )
    top_count = min(int(slots), int(candidates))
    top_values, top_indices = masked.topk(top_count, dim=-1)
    if top_count < int(slots):
        padding = int(slots) - top_count
        top_values = F.pad(top_values, (0, padding), value=float("-inf"))
        top_indices = F.pad(top_indices, (0, padding), value=0)

    if combinations is None:
        combinations = torch.tensor(
            tuple(product(range(int(slots)), repeat=int(slots))),
            dtype=torch.long,
            device=logits.device,
        )
    else:
        combinations = combinations.to(device=logits.device, dtype=torch.long)
        expected_shape = (int(slots) ** int(slots), int(slots))
        if tuple(combinations.shape) != expected_shape:
            raise ValueError(
                "invalid precomputed real-route combinations: "
                f"{tuple(combinations.shape)} != {expected_shape}"
            )
    combination_count = int(combinations.shape[0])
    gather_index = combinations.view(
        1,
        combination_count,
        int(slots),
        1,
    ).expand(batch, -1, -1, -1)
    chosen_indices = top_indices.unsqueeze(1).expand(
        -1,
        combination_count,
        -1,
        -1,
    ).gather(3, gather_index).squeeze(-1)
    chosen_values = top_values.unsqueeze(1).expand(
        -1,
        combination_count,
        -1,
        -1,
    ).gather(3, gather_index).squeeze(-1)
    valid_combination = torch.isfinite(chosen_values).all(dim=-1)
    for left in range(int(slots)):
        for right in range(left + 1, int(slots)):
            valid_combination &= (
                chosen_indices[..., left] != chosen_indices[..., right]
            )
    score = chosen_values.sum(dim=-1).masked_fill(
        ~valid_combination,
        float("-inf"),
    )
    has_assignment = torch.isfinite(score).any(dim=-1)
    best = score.argmax(dim=-1)
    batch_ids = fixed_indices(
        batch,
        device=logits.device,
        dtype=torch.long,
    )
    assigned = chosen_indices[batch_ids, best]
    assigned = torch.where(
        has_assignment.unsqueeze(-1),
        assigned,
        assigned.new_full(assigned.shape, -1),
    )
    probability = torch.softmax(masked, dim=-1)
    safe = assigned.clamp(min=0)
    selected_probability = probability.gather(
        -1,
        safe.unsqueeze(-1),
    ).squeeze(-1)
    selected_probability = torch.where(
        assigned >= 0,
        selected_probability,
        torch.zeros_like(selected_probability),
    )
    raw = masked.argmax(dim=-1)
    raw = torch.where(
        candidate_valid.any(dim=-1, keepdim=True),
        raw,
        raw.new_full(raw.shape, -1),
    )
    repair_count = (assigned != raw).sum(dim=-1)
    result = {
        "indices": assigned,
        "scores": selected_probability,
        "raw_indices": raw,
        "repair_count": repair_count,
    }
    if squeeze:
        return {name: value[0] for name, value in result.items()}
    return result


def structured_unique_route_marginals(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    temperature: float = 1.0,
    iterations: int = 20,
) -> torch.Tensor:
    """Differentiable rectangular-Sinkhorn real-route marginals.

    Unlike an independent per-slot softmax, these marginals solve the same
    injective assignment problem as the hard forward.  Each slot has unit row
    mass and each proposal has column mass at most one.  Dustbin/cardinality is
    intentionally not represented, so geometry gradients cannot alter it.

    A square transport matrix is formed by adding ``N-S`` dummy rows.  Sinkhorn
    normalization makes every real slot row sum to one while all rows together
    give each proposal column unit capacity; therefore the four real rows have
    column mass at most one.  This is a vectorized relaxation of the exact hard
    assignment, not the independent per-slot softmax that collapsed V6-C.

    If an unusual sample has fewer valid proposals than slots, it returns the
    detached hard assignment for that sample.  Normal CULane batches have 32
    valid proposal positions and retain the differentiable path.
    """

    if logits.ndim != 3:
        raise ValueError("structured route logits must have shape [B,S,N]")
    batch, slots, candidates = logits.shape
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("structured route validity must have shape [B,N]")
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError("structured route temperature must be positive")
    sinkhorn_iterations = int(iterations)
    if sinkhorn_iterations < 1:
        raise ValueError("structured route iterations must be positive")

    if int(candidates) < int(slots):
        return logits.new_zeros(
            (batch, slots, candidates),
            dtype=torch.float32,
        )

    # Build one masked NxN transport per image.  The upper block contains the
    # S real slot rows and V valid proposal columns.  Exactly V-S dummy rows
    # are connected to that block; the remaining dummy rows are connected
    # only to the invalid columns.  The two disconnected blocks make the real
    # marginal identical to running the historical VxV Sinkhorn separately,
    # without a host boolean, CUDA nonzero, or Python image loop.
    candidate_valid = candidate_valid.bool()
    valid_count = candidate_valid.sum(dim=-1)
    enough = valid_count >= int(slots)
    safe_valid = torch.where(
        enough.unsqueeze(-1),
        candidate_valid,
        torch.ones_like(candidate_valid),
    )
    safe_count = safe_valid.sum(dim=-1)
    scores = (logits.float() / tau).masked_fill(
        ~safe_valid[:, None, :],
        float("-inf"),
    )
    dummy_count = int(candidates) - int(slots)
    dummy_index = fixed_indices(
        dummy_count,
        device=logits.device,
        dtype=torch.long,
    ).view(1, dummy_count, 1)
    valid_dummy_count = (safe_count - int(slots)).view(batch, 1, 1)
    dummy_for_valid = dummy_index < valid_dummy_count
    dummy_connection = torch.where(
        dummy_for_valid,
        safe_valid[:, None, :],
        ~safe_valid[:, None, :],
    )
    dummy = scores.new_zeros((batch, dummy_count, candidates)).masked_fill(
        ~dummy_connection,
        float("-inf"),
    )
    log_transport = torch.cat((scores, dummy), dim=1)
    for _ in range(sinkhorn_iterations):
        log_transport = log_transport - torch.logsumexp(
            log_transport,
            dim=2,
            keepdim=True,
        )
        log_transport = log_transport - torch.logsumexp(
            log_transport,
            dim=1,
            keepdim=True,
        )
    marginal = log_transport[:, :slots].exp().masked_fill(
        ~candidate_valid[:, None, :],
        0.0,
    )
    # Fewer than S valid candidates has no injective assignment.  Preserve
    # the historical detached hard fallback (all zeros in that case) while
    # keeping the common path entirely on device.
    return torch.where(
        enough.view(batch, 1, 1),
        marginal,
        torch.zeros_like(marginal),
    )


def structured_unique_route_marginals_with_private_dustbins(
    real_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    private_dustbin_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    iterations: int = 20,
) -> torch.Tensor:
    """Injective proposal transport with one private dustbin per slot.

    The returned tensor has shape ``[B,S,N+S]``.  Every real slot row has
    unit mass, every real proposal column has capacity at most one, and slot
    ``s`` is the only real row that can use private dustbin ``N+s``.  ``N``
    dummy rows complete a square ``(N+S) x (N+S)`` Sinkhorn problem; their
    mass is discarded after normalization.

    This operator is used for both the learned V14 association and its target
    transport.  Using the same feasible polytope prevents independent GT
    target rows from asking two slots to own the same proposal.
    """

    if real_logits.ndim != 3:
        raise ValueError("private-dustbin logits must have shape [B,S,N]")
    batch, slots, candidates = real_logits.shape
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("private-dustbin validity must have shape [B,N]")
    if private_dustbin_logits.shape != (batch, slots):
        raise ValueError("private dustbin logits must have shape [B,S]")
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError("private-dustbin temperature must be positive")
    sinkhorn_iterations = int(iterations)
    if sinkhorn_iterations < 1:
        raise ValueError("private-dustbin Sinkhorn iterations must be positive")

    real_scores = (real_logits.float() / tau).masked_fill(
        ~candidate_valid.bool()[:, None, :],
        float("-inf"),
    )
    private_scores = real_scores.new_full(
        (batch, slots, slots),
        float("-inf"),
    )
    diagonal = fixed_indices(
        slots,
        device=real_logits.device,
        dtype=torch.long,
    ).view(1, slots, 1).expand(batch, -1, -1)
    private_scores.scatter_(
        2,
        diagonal,
        (private_dustbin_logits.float() / tau).unsqueeze(-1),
    )
    real_rows = torch.cat((real_scores, private_scores), dim=-1)

    # There are N+S columns and S real rows, hence exactly N dummy rows make
    # the transport square.  Dummy rows may fill any unused/invalid proposal
    # or private-dustbin column, while the masked real rows retain the desired
    # capacity constraints.
    dummy_rows = real_rows.new_zeros(
        (batch, candidates, candidates + slots)
    )
    log_transport = torch.cat((real_rows, dummy_rows), dim=1)
    for _ in range(sinkhorn_iterations):
        log_transport = log_transport - torch.logsumexp(
            log_transport,
            dim=2,
            keepdim=True,
        )
        log_transport = log_transport - torch.logsumexp(
            log_transport,
            dim=1,
            keepdim=True,
        )
    marginal = log_transport[:, :slots].exp()
    real_marginal = marginal[..., :candidates].masked_fill(
        ~candidate_valid.bool()[:, None, :],
        0.0,
    )
    private_mask = torch.eye(
        slots,
        device=real_logits.device,
        dtype=torch.bool,
    ).view(1, slots, slots)
    private_marginal = marginal[..., candidates:].masked_fill(
        ~private_mask,
        0.0,
    )
    result = torch.cat((real_marginal, private_marginal), dim=-1)
    # The finite iteration budget ends on a column normalization.  Restore
    # exact real-slot row mass; the following audits still enforce the real
    # column-capacity tolerance independently.
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


class FourSlotBoundedRefinement(nn.Module):
    """Locally refine routed proposal curves without touching the detector.

    Proposal/P2 inputs are always detached.  V6-B also detaches the hard
    route and slot state, while V6-C can preserve the exact hard forward and
    attach a straight-through soft route so slot geometry supervises the
    router without reaching the proposal generator.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        hidden_dim: int,
        delta_offsets_px: tuple[float, ...],
        evidence_offsets_px: tuple[float, ...] | None = None,
        straight_through_routing: bool = False,
        detach_slot_states: bool = True,
        slot_state_gradient_scale: float = 1.0,
        route_temperature: float = 1.0,
        structured_unique_routing: bool = False,
        route_gradient_scale: float = 1.0,
        reference_mode: str = "hard_st",
        neighborhood_max_candidates: int = 4,
        neighborhood_max_mean_distance_px: float = 48.0,
        neighborhood_min_common_fraction: float = 0.50,
        neighborhood_distance_temperature_px: float = 24.0,
        neighborhood_gradient_scale: float = 0.10,
        range_refinement: bool = False,
        range_delta_offsets_norm: tuple[float, ...] = (
            -0.10,
            -0.05,
            -0.025,
            0.0,
            0.025,
            0.05,
            0.10,
        ),
        vertical_layers: int = 0,
        vertical_num_heads: int = 8,
        vertical_ff_dim: int | None = None,
        vertical_dropout: float = 0.0,
        zero_init_delta_heads: bool = True,
        delta_head_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        offsets = tuple(float(value) for value in delta_offsets_px)
        if len(offsets) < 3 or tuple(sorted(offsets)) != offsets:
            raise ValueError("slot refinement offsets must be sorted")
        if not any(abs(value) < 1.0e-12 for value in offsets):
            raise ValueError("slot refinement offsets must include zero")
        if any(
            abs(left + right) > 1.0e-6
            for left, right in zip(offsets, reversed(offsets))
        ):
            raise ValueError("slot refinement offsets must be symmetric")
        evidence_offsets = (
            offsets
            if evidence_offsets_px is None
            else tuple(float(value) for value in evidence_offsets_px)
        )
        if len(evidence_offsets) < 1:
            raise ValueError("slot evidence offsets must not be empty")
        if tuple(sorted(evidence_offsets)) != evidence_offsets:
            raise ValueError("slot evidence offsets must be sorted")
        if not any(abs(value) < 1.0e-12 for value in evidence_offsets):
            raise ValueError("slot evidence offsets must include zero")
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.hidden_dim = int(hidden_dim)
        self.straight_through_routing = bool(straight_through_routing)
        self.detach_slot_states = bool(detach_slot_states)
        self.slot_state_gradient_scale = float(slot_state_gradient_scale)
        self.route_temperature = float(route_temperature)
        self.structured_unique_routing = bool(structured_unique_routing)
        self.route_gradient_scale = float(route_gradient_scale)
        self.reference_mode = str(reference_mode).strip().lower()
        self.neighborhood_max_candidates = int(neighborhood_max_candidates)
        self.neighborhood_max_mean_distance_px = float(
            neighborhood_max_mean_distance_px
        )
        self.neighborhood_min_common_fraction = float(
            neighborhood_min_common_fraction
        )
        self.neighborhood_distance_temperature_px = float(
            neighborhood_distance_temperature_px
        )
        self.neighborhood_gradient_scale = float(neighborhood_gradient_scale)
        self.range_refinement = bool(range_refinement)
        self.vertical_layers = int(vertical_layers)
        self.zero_init_delta_heads = bool(zero_init_delta_heads)
        self.delta_head_init_std = float(delta_head_init_std)
        if self.route_temperature <= 0.0:
            raise ValueError("slot refinement route_temperature must be positive")
        if not 0.0 <= self.slot_state_gradient_scale <= 1.0:
            raise ValueError(
                "slot state gradient scale must be in [0, 1]"
            )
        if not 0.0 <= self.route_gradient_scale <= 1.0:
            raise ValueError(
                "slot refinement route_gradient_scale must be in [0, 1]"
            )
        if self.reference_mode not in {"hard_st", "soft", "neighborhood_soft"}:
            raise ValueError(
                "slot refinement reference_mode must be 'hard_st', 'soft', "
                "or 'neighborhood_soft'"
            )
        if self.neighborhood_max_candidates < 1:
            raise ValueError("slot neighborhood size must be positive")
        if self.neighborhood_max_mean_distance_px <= 0.0:
            raise ValueError("slot neighborhood distance must be positive")
        if not 0.0 <= self.neighborhood_min_common_fraction <= 1.0:
            raise ValueError(
                "slot neighborhood common fraction must be in [0, 1]"
            )
        if self.neighborhood_distance_temperature_px <= 0.0:
            raise ValueError(
                "slot neighborhood distance temperature must be positive"
            )
        if not 0.0 <= self.neighborhood_gradient_scale <= 1.0:
            raise ValueError(
                "slot neighborhood gradient scale must be in [0, 1]"
            )
        if self.vertical_layers < 0:
            raise ValueError("slot vertical layer count must be non-negative")
        if int(vertical_num_heads) < 1 or self.hidden_dim % int(
            vertical_num_heads
        ):
            raise ValueError(
                "slot vertical attention heads must divide hidden_dim"
            )
        if float(vertical_dropout) < 0.0:
            raise ValueError("slot vertical dropout must be non-negative")
        if self.delta_head_init_std < 0.0:
            raise ValueError("slot delta-head init std must be non-negative")
        range_offsets = tuple(float(value) for value in range_delta_offsets_norm)
        if self.range_refinement:
            if len(range_offsets) < 3 or tuple(sorted(range_offsets)) != range_offsets:
                raise ValueError("slot range refinement offsets must be sorted")
            if not any(abs(value) < 1.0e-12 for value in range_offsets):
                raise ValueError("slot range refinement offsets must include zero")
            if any(
                abs(left + right) > 1.0e-6
                for left, right in zip(range_offsets, reversed(range_offsets))
            ):
                raise ValueError("slot range refinement offsets must be symmetric")
        self.row_norm = nn.LayerNorm(self.dim)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.row_projection = nn.Linear(self.dim, self.hidden_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.query_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.evidence_norm = nn.LayerNorm(self.dim)
        self.evidence_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.evidence_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.offset_embedding = nn.Parameter(
            torch.empty(len(evidence_offsets), self.hidden_dim)
        )
        self.context_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        if self.vertical_layers:
            self.row_position_projection = nn.Linear(
                2,
                self.hidden_dim,
                bias=False,
            )
            vertical_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(vertical_num_heads),
                dim_feedforward=int(vertical_ff_dim or 2 * self.hidden_dim),
                dropout=float(vertical_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.vertical_encoder = nn.TransformerEncoder(
                vertical_layer,
                num_layers=self.vertical_layers,
                enable_nested_tensor=False,
            )
        else:
            self.row_position_projection = None
            self.vertical_encoder = None
        self.delta_norm = nn.LayerNorm(
            self.hidden_dim,
            elementwise_affine=False,
        )
        self.delta_head = nn.Linear(
            self.hidden_dim,
            len(offsets),
            bias=False,
        )
        # V8 treats the production route only as a coarse lane-cluster anchor.
        # A geometry-only row-wise arbitrator may combine at most K nearby
        # proposals before the existing bounded P2 refiner.  Every input is
        # detached; therefore final geometry can train this module without
        # changing activity, global routing, proposal geometry or the visual
        # backbone.  The scalar blend starts at zero, preserving the exact V7
        # forward at initialization.  A gradient-only straight-through term
        # lets the local scorer learn before that blend opens.
        if self.reference_mode == "neighborhood_soft":
            self.neighborhood_row_norm = nn.LayerNorm(self.dim)
            self.neighborhood_anchor_projection = nn.Linear(
                self.dim,
                self.hidden_dim,
                bias=False,
            )
            self.neighborhood_candidate_projection = nn.Linear(
                self.dim,
                self.hidden_dim,
                bias=False,
            )
            self.neighborhood_slot_projection = nn.Linear(
                self.slot_dim,
                self.hidden_dim,
                bias=False,
            )
            self.neighborhood_mix = nn.Parameter(torch.zeros(()))
        else:
            self.neighborhood_row_norm = None
            self.neighborhood_anchor_projection = None
            self.neighborhood_candidate_projection = None
            self.neighborhood_slot_projection = None
            self.register_parameter("neighborhood_mix", None)
        self.range_delta_head = (
            nn.Linear(
                self.hidden_dim,
                2 * len(range_offsets),
                bias=False,
            )
            if self.range_refinement
            else None
        )
        nn.init.normal_(self.offset_embedding, std=0.02)
        # Uniform probability over symmetric offsets has exactly zero expected
        # displacement, so the new graph starts as the verified V6-A model.
        if self.zero_init_delta_heads:
            nn.init.zeros_(self.delta_head.weight)
        else:
            nn.init.normal_(
                self.delta_head.weight,
                std=self.delta_head_init_std,
            )
        if self.range_delta_head is not None:
            if self.zero_init_delta_heads:
                nn.init.zeros_(self.range_delta_head.weight)
            else:
                nn.init.normal_(
                    self.range_delta_head.weight,
                    std=self.delta_head_init_std,
                )
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
        )
        self.register_buffer(
            "evidence_offsets_px",
            torch.tensor(evidence_offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "range_delta_offsets_norm",
            torch.tensor(range_offsets, dtype=torch.float32),
        )

    @torch.no_grad()
    def _proposal_neighborhood(
        self,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        route_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a GT-free sparse neighborhood around every route anchor."""

        batch, candidates, rows = proposal_x.shape
        slots = int(route_indices.shape[1])
        ranges = sort_range_norm(proposal_range.float())
        row_y = fixed_row_fractions(
            rows,
            device=proposal_x.device,
            dtype=torch.float32,
        ).view(1, 1, rows)
        visible = (
            (row_y >= ranges[..., :1])
            & (row_y <= ranges[..., 1:])
            & torch.isfinite(proposal_x)
        )
        common = visible[:, :, None, :] & visible[:, None, :, :]
        common_count = common.sum(dim=-1)
        distance = (
            proposal_x[:, :, None, :].float()
            - proposal_x[:, None, :, :].float()
        ).abs()
        mean_distance = distance.masked_fill(~common, 0.0).sum(dim=-1)
        mean_distance = mean_distance / common_count.clamp_min(1).float()
        mean_distance = mean_distance.masked_fill(common_count == 0, torch.inf)
        shorter_visible = torch.minimum(
            visible.sum(dim=-1)[:, :, None],
            visible.sum(dim=-1)[:, None, :],
        ).clamp_min(1)
        common_fraction = common_count.float() / shorter_visible.float()

        safe_anchor = route_indices.clamp(min=0, max=max(candidates - 1, 0))
        gather_index = safe_anchor.unsqueeze(-1).expand(-1, -1, candidates)
        anchor_distance = mean_distance.gather(1, gather_index)
        anchor_common = common_fraction.gather(1, gather_index)
        route_valid = route_indices >= 0
        eligible = candidate_valid[:, None, :].bool().expand(
            batch,
            slots,
            candidates,
        ).clone()
        eligible &= route_valid.unsqueeze(-1)
        eligible &= torch.isfinite(anchor_distance)
        eligible &= anchor_common >= self.neighborhood_min_common_fraction
        eligible &= anchor_distance <= self.neighborhood_max_mean_distance_px
        eligible.scatter_(
            2,
            safe_anchor.unsqueeze(-1),
            route_valid.unsqueeze(-1),
        )

        neighbor_count = min(self.neighborhood_max_candidates, candidates)
        ranked_distance = anchor_distance.masked_fill(~eligible, torch.inf)
        top_distance, top_indices = ranked_distance.topk(
            neighbor_count,
            dim=-1,
            largest=False,
            sorted=True,
        )
        top_valid = torch.isfinite(top_distance)
        neighborhood_mask = torch.zeros_like(eligible)
        neighborhood_mask.scatter_(2, top_indices, top_valid)
        # An invalid route still needs one finite softmax entry; its public
        # geometry is zeroed later, so this private fallback is unobservable.
        fallback = ~neighborhood_mask.any(dim=-1)
        neighborhood_mask.scatter_(
            2,
            safe_anchor.unsqueeze(-1),
            fallback.unsqueeze(-1),
        )
        return {
            "mask": neighborhood_mask,
            "distance": anchor_distance,
            "support": top_valid.sum(dim=-1),
        }

    def _neighborhood_reference(
        self,
        *,
        slot_states: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        route_indices: torch.Tensor,
        hard_reference_x: torch.Tensor,
        hard_slot_range: torch.Tensor,
        hard_routed_rows: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            self.neighborhood_row_norm is None
            or self.neighborhood_anchor_projection is None
            or self.neighborhood_candidate_projection is None
            or self.neighborhood_slot_projection is None
            or self.neighborhood_mix is None
        ):
            raise RuntimeError("slot neighborhood modules are not initialized")
        neighborhood = self._proposal_neighborhood(
            proposal_x,
            proposal_range,
            candidate_valid,
            route_indices,
        )
        mask = neighborhood["mask"]
        distance = neighborhood["distance"]
        anchor_query = self.neighborhood_anchor_projection(
            self.neighborhood_row_norm(hard_routed_rows.float())
        )
        anchor_query = anchor_query + self.neighborhood_slot_projection(
            slot_states.detach().float()
        ).unsqueeze(2)
        candidate_key = self.neighborhood_candidate_projection(
            self.neighborhood_row_norm(proposal_rows.float())
        )
        logits = torch.einsum(
            "bsrh,bnrh->bsrn",
            anchor_query,
            candidate_key,
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits - distance.unsqueeze(2) / float(
            self.neighborhood_distance_temperature_px
        )
        logits = logits.masked_fill(~mask.unsqueeze(2), -1.0e4)
        weight = torch.softmax(logits.float(), dim=-1)
        soft_reference_x = torch.einsum(
            "bsrn,bnr->bsr",
            weight,
            proposal_x.float(),
        )
        soft_routed_rows = torch.einsum(
            "bsrn,bnrd->bsrd",
            weight,
            proposal_rows.float(),
        )
        candidate_weight = weight.mean(dim=2)
        soft_slot_range = torch.einsum(
            "bsn,bnd->bsd",
            candidate_weight,
            proposal_range.float(),
        )

        # Signed bounded residual, not a one-sided convex gate.  ``tanh`` is
        # exactly zero at initialization and retains a non-zero derivative on
        # both sides, so the source forward is preserved without the dead-zone
        # created by clamp(0, 1).  The neighborhood stays local (<=48 px) and
        # final x/range remain clipped/bounded by the production operator.
        mix = torch.tanh(self.neighborhood_mix)
        reference_x = hard_reference_x.float() + mix * (
            soft_reference_x - hard_reference_x.float()
        )
        slot_range = hard_slot_range.float() + mix * (
            soft_slot_range - hard_slot_range.float()
        )
        routed_rows = hard_routed_rows.float() + mix * (
            soft_routed_rows - hard_routed_rows.float()
        )
        if self.training and torch.is_grad_enabled():
            scale = self.neighborhood_gradient_scale
            reference_x = reference_x + scale * (
                soft_reference_x - soft_reference_x.detach()
            )
            slot_range = slot_range + scale * (
                soft_slot_range - soft_slot_range.detach()
            )
            routed_rows = routed_rows + scale * (
                soft_routed_rows - soft_routed_rows.detach()
            )
        entropy = -(weight * weight.clamp_min(1.0e-12).log()).sum(dim=-1)
        active = (route_indices >= 0).unsqueeze(-1).expand_as(entropy)
        denominator = active.float().sum(dim=(1, 2)).clamp_min(1.0)
        mean_entropy = (entropy * active.float()).sum(dim=(1, 2)) / denominator
        mean_top1 = (weight.amax(dim=-1) * active.float()).sum(dim=(1, 2))
        mean_top1 = mean_top1 / denominator
        return {
            "reference_x": reference_x,
            "slot_range": slot_range,
            "routed_rows": routed_rows,
            "support": neighborhood["support"],
            "weight": weight,
            "mean_entropy": mean_entropy,
            "mean_top1": mean_top1,
            "mix": mix,
        }

    def _sample_local_evidence(
        self,
        row_value_features: torch.Tensor,
        reference_x_rows: torch.Tensor,
    ) -> torch.Tensor:
        batch, rows, x_bins, channels = row_value_features.shape
        if reference_x_rows.ndim != 3 or int(reference_x_rows.shape[0]) != batch:
            raise ValueError("slot reference rows must have shape [B,S,R]")
        if int(reference_x_rows.shape[-1]) != rows:
            raise ValueError("slot references must share the P2 row grid")
        slots = int(reference_x_rows.shape[1])
        offsets = self.evidence_offsets_px.to(
            device=reference_x_rows.device,
            dtype=torch.float32,
        )
        with torch.autocast(
            device_type=row_value_features.device.type,
            enabled=False,
        ):
            sample_x = reference_x_rows.detach().float().unsqueeze(-1)
            sample_x = (sample_x + offsets.view(1, 1, 1, -1)).clamp(
                0.0,
                float(max(self.input_w - 1, 1)),
            )
            feature_x = sample_x * float(max(x_bins - 1, 0)) / float(
                max(self.input_w - 1, 1)
            )
            left = feature_x.floor().long()
            right = (left + 1).clamp(max=max(x_bins - 1, 0))
            alpha = feature_x - left.to(dtype=feature_x.dtype)
        flat = row_value_features.detach().reshape(
            batch * rows,
            x_bins,
            channels,
        )
        sample_count = int(offsets.numel())
        left = left.permute(0, 2, 1, 3).reshape(
            batch * rows,
            slots * sample_count,
        )
        right = right.permute(0, 2, 1, 3).reshape(
            batch * rows,
            slots * sample_count,
        )
        alpha = alpha.permute(0, 2, 1, 3).reshape(
            batch * rows,
            slots * sample_count,
            1,
        )
        row_index = fixed_indices(
            batch * rows,
            device=row_value_features.device,
            dtype=torch.long,
        ).view(-1, 1)
        paired = torch.stack((left, right), dim=-1).reshape(
            batch * rows,
            slots * sample_count * 2,
        )
        sampled = flat[row_index, paired].view(
            batch * rows,
            slots * sample_count,
            2,
            channels,
        )
        sampled = torch.lerp(
            sampled[:, :, 0],
            sampled[:, :, 1],
            alpha.to(dtype=row_value_features.dtype),
        )
        return sampled.view(
            batch,
            rows,
            slots,
            sample_count,
            channels,
        ).permute(0, 2, 1, 3, 4).contiguous()

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        route_indices: torch.Tensor,
        route_logits: torch.Tensor | None = None,
        candidate_valid: torch.Tensor | None = None,
        slot_active: torch.Tensor | None = None,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, slots = route_indices.shape
        candidates = int(proposal_x_rows.shape[1])
        rows = int(proposal_x_rows.shape[-1])
        route_valid = route_indices >= 0
        if slot_active is None:
            deployment_active = route_valid
        else:
            if tuple(slot_active.shape) != (batch, slots):
                raise ValueError("slot_active must have shape [B,S]")
            deployment_active = slot_active.bool() & route_valid
        proposal_x = proposal_x_rows.detach()
        proposal_range = proposal_range_norm.detach().float()
        proposal_rows = proposal_row_tokens.detach()
        safe = route_indices.clamp(
            min=0,
            max=max(candidates - 1, 0),
        )
        x_index = safe.unsqueeze(-1).expand(-1, -1, rows)
        hard_reference_x = proposal_x.gather(1, x_index)
        range_index = safe.unsqueeze(-1).expand(-1, -1, 2)
        hard_slot_range = proposal_range.gather(1, range_index)
        token_index = safe.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            rows,
            int(proposal_rows.shape[-1]),
        )
        hard_routed_rows = proposal_rows.gather(1, token_index)
        hard_reference_x = torch.where(
            route_valid.unsqueeze(-1),
            hard_reference_x,
            torch.zeros_like(hard_reference_x),
        )
        hard_slot_range = torch.where(
            route_valid.unsqueeze(-1),
            hard_slot_range,
            torch.zeros_like(hard_slot_range),
        )
        hard_routed_rows = torch.where(
            route_valid.unsqueeze(-1).unsqueeze(-1),
            hard_routed_rows,
            torch.zeros_like(hard_routed_rows),
        )
        neighborhood_result: dict[str, torch.Tensor] | None = None
        soft_route_weight: torch.Tensor | None = None
        if self.reference_mode == "neighborhood_soft":
            if candidate_valid is None:
                raise ValueError(
                    "neighborhood slot refinement requires candidate validity"
                )
            if tuple(candidate_valid.shape) != (batch, candidates):
                raise ValueError("slot refinement candidate-valid shape mismatch")
            neighborhood_result = self._neighborhood_reference(
                slot_states=slot_states,
                proposal_rows=proposal_rows,
                proposal_x=proposal_x,
                proposal_range=proposal_range,
                candidate_valid=candidate_valid,
                route_indices=route_indices,
                hard_reference_x=hard_reference_x,
                hard_slot_range=hard_slot_range,
                hard_routed_rows=hard_routed_rows,
            )
            reference_x = neighborhood_result["reference_x"]
            slot_range = neighborhood_result["slot_range"]
            routed_rows = neighborhood_result["routed_rows"]
        soft_forward = self.reference_mode == "soft"
        needs_soft_route = self.reference_mode != "neighborhood_soft" and (
            soft_forward or (
            self.straight_through_routing
            and self.training
            and torch.is_grad_enabled()
            )
        )
        if needs_soft_route:
            if route_logits is None or candidate_valid is None:
                raise ValueError(
                    "straight-through slot refinement requires route logits "
                    "and candidate validity"
                )
            if tuple(route_logits.shape[:2]) != (batch, slots) or int(
                route_logits.shape[-1]
            ) not in {candidates, candidates + 1}:
                raise ValueError("slot refinement route logit shape mismatch")
            if tuple(candidate_valid.shape) != (batch, candidates):
                raise ValueError("slot refinement candidate-valid shape mismatch")
            if int(route_logits.shape[-1]) == candidates + 1:
                real_logits = route_logits[..., :candidates].float()
            elif int(route_logits.shape[-1]) == candidates:
                real_logits = route_logits.float()
            else:
                raise ValueError("slot refinement real-route shape mismatch")
            real_logits = real_logits.masked_fill(
                ~candidate_valid[:, None, :].bool(),
                -1.0e4,
            )
            if self.structured_unique_routing:
                candidate_weight = structured_unique_route_marginals(
                    real_logits,
                    candidate_valid,
                    temperature=self.route_temperature,
                )
            else:
                candidate_weight = torch.softmax(
                    real_logits / self.route_temperature,
                    dim=-1,
                )
            if soft_forward and self.route_gradient_scale != 1.0:
                candidate_weight = candidate_weight.detach() + (
                    self.route_gradient_scale
                    * (candidate_weight - candidate_weight.detach())
                )
            soft_route_weight = candidate_weight
            soft_reference_x = torch.einsum(
                "bsn,bnr->bsr",
                candidate_weight,
                proposal_x.float(),
            )
            soft_slot_range = torch.einsum(
                "bsn,bnd->bsd",
                candidate_weight,
                proposal_range,
            )
            soft_routed_rows = torch.einsum(
                "bsn,bnrd->bsrd",
                candidate_weight,
                proposal_rows.float(),
            )
            if soft_forward:
                # Direct-slot arm: the final object state consumes the
                # constrained real-proposal distribution itself.  There is no
                # hard proposal identity in the geometry-producing forward,
                # and activity/dustbin remains a separate head.
                reference_x = soft_reference_x
                slot_range = soft_slot_range
                routed_rows = soft_routed_rows
            else:
                # Structured hard-reference arm: preserve the exact unique
                # hard gather in forward, with only a controlled gradient from
                # the capacity-constrained soft route.
                reference_x = hard_reference_x.float() + (
                    soft_reference_x - soft_reference_x.detach()
                ) * self.route_gradient_scale
                slot_range = hard_slot_range + (
                    soft_slot_range - soft_slot_range.detach()
                ) * self.route_gradient_scale
                routed_rows = hard_routed_rows.float() + (
                    soft_routed_rows - soft_routed_rows.detach()
                ) * self.route_gradient_scale
        elif self.reference_mode != "neighborhood_soft":
            reference_x = hard_reference_x
            slot_range = hard_slot_range
            routed_rows = hard_routed_rows
        slot_range = sort_range_norm(slot_range.float())
        reference_x = torch.where(
            route_valid.unsqueeze(-1),
            reference_x,
            torch.zeros_like(reference_x),
        )
        slot_range = torch.where(
            route_valid.unsqueeze(-1),
            slot_range,
            torch.zeros_like(slot_range),
        )
        input_slot_range = slot_range

        if self.detach_slot_states:
            slot_input = slot_states.detach()
        else:
            slot_input = slot_states.detach() + self.slot_state_gradient_scale * (
                slot_states - slot_states.detach()
            )
        query_state = self.row_projection(self.row_norm(routed_rows.float()))
        query_state = query_state + self.slot_projection(
            self.slot_norm(slot_input.float())
        ).unsqueeze(2)
        query_state = self.query_norm(query_state)
        evidence = self._sample_local_evidence(
            row_value_features.detach(),
            reference_x,
        ).float()
        evidence = self.evidence_norm(evidence)
        keys = self.evidence_key(evidence) + self.offset_embedding.view(
            1,
            1,
            1,
            -1,
            self.hidden_dim,
        )
        values = self.evidence_value(evidence)
        query = self.query_projection(query_state).unsqueeze(-2)
        attention = torch.softmax(
            (query * keys).sum(dim=-1) / math.sqrt(float(self.hidden_dim)),
            dim=-1,
        )
        context = (attention.unsqueeze(-1) * values).sum(dim=-2)
        hidden = query_state + self.context_projection(context)
        hidden = hidden + self.ffn(self.output_norm(hidden))
        if self.vertical_encoder is not None:
            if self.row_position_projection is None:
                raise RuntimeError("slot row position projection is unavailable")
            row_fraction = fixed_row_fractions(
                rows,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            row_position = torch.stack(
                (row_fraction, row_fraction.square()),
                dim=-1,
            )
            hidden = hidden + self.row_position_projection(row_position).view(
                1,
                1,
                rows,
                self.hidden_dim,
            )
            hidden = self.vertical_encoder(
                hidden.reshape(batch * slots, rows, self.hidden_dim)
            ).reshape(batch, slots, rows, self.hidden_dim)
        delta_logits = self.delta_head(self.delta_norm(hidden))
        probability = torch.softmax(delta_logits.float(), dim=-1)
        offsets = self.delta_offsets_px.to(
            device=probability.device,
            dtype=probability.dtype,
        )
        delta = (probability * offsets).sum(dim=-1)
        delta = torch.where(
            route_valid.unsqueeze(-1),
            delta,
            torch.zeros_like(delta),
        )
        refined_x = (reference_x.float() + delta).clamp(
            0.0,
            float(max(self.input_w - 1, 1)),
        )
        range_delta_logits: torch.Tensor | None = None
        range_delta = slot_range.new_zeros(slot_range.shape)
        range_boundary_mass = slot_range.new_zeros((batch, slots))
        if self.range_delta_head is not None:
            pooled_hidden = hidden.mean(dim=2)
            range_delta_logits = self.range_delta_head(pooled_hidden).view(
                batch,
                slots,
                2,
                int(self.range_delta_offsets_norm.numel()),
            )
            range_probability = torch.softmax(
                range_delta_logits.float(),
                dim=-1,
            )
            range_offsets = self.range_delta_offsets_norm.to(
                device=range_probability.device,
                dtype=range_probability.dtype,
            )
            range_delta = (range_probability * range_offsets).sum(dim=-1)
            range_delta = torch.where(
                route_valid.unsqueeze(-1),
                range_delta,
                torch.zeros_like(range_delta),
            )
            slot_range = sort_range_norm(
                (input_slot_range.float() + range_delta).clamp(0.0, 1.0)
            )
            range_boundary_mass = range_probability[..., (0, -1)].sum(dim=-1)
        active_rows = route_valid.unsqueeze(-1).expand(-1, -1, rows)
        active_count = active_rows.float().sum(dim=(1, 2)).clamp_min(1.0)
        mean_abs = (delta.abs() * active_rows.float()).sum(dim=(1, 2))
        mean_abs = mean_abs / active_count
        max_abs = delta.abs().masked_fill(~active_rows, 0.0).amax(dim=(1, 2))
        boundary_mass = probability[..., (0, -1)].sum(dim=-1)
        boundary_mass = (
            boundary_mass * active_rows.float()
        ).sum(dim=(1, 2)) / active_count
        result = {
            "selection_slot_pred_x_rows": refined_x,
            "selection_slot_range_norm": slot_range,
            "selection_slot_active": deployment_active,
            "selection_slot_geometry_valid": route_valid,
            "selection_slot_input_reference_x_rows": reference_x,
            "selection_slot_input_range_norm": input_slot_range,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta": range_delta,
            "selection_slot_range_delta_boundary_mass": range_boundary_mass,
            "selection_slot_delta_mean_abs": mean_abs,
            "selection_slot_delta_max_abs": max_abs,
            "selection_slot_delta_boundary_mass": boundary_mass,
        }
        if range_delta_logits is not None:
            result["selection_slot_range_delta_logits"] = range_delta_logits
            result["selection_slot_range_delta_offsets_norm"] = (
                self.range_delta_offsets_norm
            )
        if neighborhood_result is not None:
            support = neighborhood_result["support"].float()
            valid_slots = route_valid.float()
            valid_count = valid_slots.sum(dim=-1).clamp_min(1.0)
            mean_support = (support * valid_slots).sum(dim=-1) / valid_count
            alternative_fraction = (
                ((support > 1).float() * valid_slots).sum(dim=-1) / valid_count
            )
            reference_shift = (
                reference_x.float() - hard_reference_x.float()
            ).abs()
            reference_shift = (
                reference_shift * active_rows.float()
            ).sum(dim=(1, 2)) / active_count
            result.update(
                {
                    "selection_slot_neighborhood_support": support,
                    "selection_slot_neighborhood_mean_support": mean_support,
                    "selection_slot_neighborhood_alternative_fraction": (
                        alternative_fraction
                    ),
                    "selection_slot_neighborhood_weight": neighborhood_result[
                        "weight"
                    ],
                    "selection_slot_neighborhood_entropy": neighborhood_result[
                        "mean_entropy"
                    ],
                    "selection_slot_neighborhood_top1_mass": neighborhood_result[
                        "mean_top1"
                    ],
                    "selection_slot_neighborhood_mix": neighborhood_result[
                        "mix"
                    ].expand(batch),
                    "selection_slot_neighborhood_reference_shift_px": (
                        reference_shift
                    ),
                }
            )
        if self.reference_mode == "soft" and soft_route_weight is not None:
            soft_entropy = -(
                soft_route_weight
                * soft_route_weight.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            valid_slots = route_valid.float()
            valid_count = valid_slots.sum(dim=-1).clamp_min(1.0)
            result.update(
                {
                    "selection_slot_owned_weight": soft_route_weight,
                    "selection_slot_owned_entropy": (
                        soft_entropy * valid_slots
                    ).sum(dim=-1)
                    / valid_count,
                    "selection_slot_owned_top1_mass": (
                        soft_route_weight.amax(dim=-1) * valid_slots
                    ).sum(dim=-1)
                    / valid_count,
                    "selection_slot_owned_reference_shift_px": (
                        (reference_x.float() - hard_reference_x.float()).abs()
                        * active_rows.float()
                    ).sum(dim=(1, 2))
                    / active_count,
                }
            )
        return result


class FourSlotGlobalVisualGeometry(nn.Module):
    """Predict four slot-owned lanes from the complete P2 row grid.

    Unlike the V7--V9 geometry paths, this module never gathers or averages
    proposal coordinates.  Every slot attends over every horizontal P2 bin
    on every output row, exchanges information vertically, attends a second
    time, and then predicts a bounded residual and an absolute visible range.
    Proposal routing remains available for activity/scoring diagnostics, but
    a wrong proposal ID cannot choose the geometry-producing visual region.

    The P2 tensor is deliberately detached in the first causal gate.  Final
    geometry can shape the new visual projections and (optionally) the live
    persistent slot state, while it cannot update the proposal detector,
    backbone/FPN, or the separate active/no-lane head.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        delta_offsets_px: tuple[float, ...],
        detach_slot_states: bool = False,
        slot_state_gradient_scale: float = 1.0,
        spatial_prior_strength: float = 1.0,
        spatial_prior_sigma: float = 0.22,
        range_start_prior: float = 0.05,
        range_end_prior: float = 0.95,
        zero_init_delta_head: bool = True,
        delta_head_init_std: float = 1.0e-3,
    ) -> None:
        super().__init__()
        offsets = tuple(float(value) for value in delta_offsets_px)
        if len(offsets) < 3 or tuple(sorted(offsets)) != offsets:
            raise ValueError("global visual geometry offsets must be sorted")
        if not any(abs(value) < 1.0e-12 for value in offsets):
            raise ValueError("global visual geometry offsets must include zero")
        if any(
            abs(left + right) > 1.0e-6
            for left, right in zip(offsets, reversed(offsets))
        ):
            raise ValueError("global visual geometry offsets must be symmetric")
        if int(num_slots) < 1:
            raise ValueError("global visual geometry needs at least one slot")
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError(
                "global visual geometry heads must divide hidden_dim"
            )
        if int(vertical_layers) < 0:
            raise ValueError("global visual vertical layer count is invalid")
        if not 0.0 <= float(slot_state_gradient_scale) <= 1.0:
            raise ValueError("visual slot gradient scale must be in [0, 1]")
        if float(spatial_prior_strength) < 0.0:
            raise ValueError("visual spatial-prior strength must be non-negative")
        if float(spatial_prior_sigma) <= 0.0:
            raise ValueError("visual spatial-prior sigma must be positive")
        if not 0.0 < float(range_start_prior) < float(range_end_prior) < 1.0:
            raise ValueError("visual range priors must satisfy 0 < start < end < 1")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.detach_slot_states = bool(detach_slot_states)
        self.slot_state_gradient_scale = float(slot_state_gradient_scale)
        self.spatial_prior_strength = float(spatial_prior_strength)
        self.spatial_prior_sigma = float(spatial_prior_sigma)

        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.x_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.visual_slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.first_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.second_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.first_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.second_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.first_output_norm = nn.LayerNorm(self.hidden_dim)
        self.first_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        if int(vertical_layers):
            vertical_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ff_dim),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.vertical_encoder = nn.TransformerEncoder(
                vertical_layer,
                num_layers=int(vertical_layers),
                enable_nested_tensor=False,
            )
        else:
            self.vertical_encoder = None
        self.second_output_norm = nn.LayerNorm(self.hidden_dim)
        self.second_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        self.delta_norm = nn.LayerNorm(
            self.hidden_dim,
            elementwise_affine=False,
        )
        self.delta_head = nn.Linear(
            self.hidden_dim,
            len(offsets),
            bias=False,
        )
        self.range_norm = nn.LayerNorm(self.hidden_dim)
        self.range_head = nn.Linear(self.hidden_dim, 2)

        # A weak, learnable left-to-right discovery prior breaks the four-way
        # permutation symmetry at cold start.  It biases attention only; the
        # full image width remains reachable and the anchors/slope are live.
        initial_anchors = torch.linspace(
            0.15,
            0.85,
            self.num_slots,
            dtype=torch.float32,
        )
        self.anchor_logits = nn.Parameter(
            torch.logit(initial_anchors.clamp(1.0e-4, 1.0 - 1.0e-4))
        )
        self.anchor_slopes = nn.Parameter(torch.zeros(self.num_slots))

        nn.init.normal_(self.visual_slot_tokens.weight, std=0.02)
        if bool(zero_init_delta_head):
            nn.init.zeros_(self.delta_head.weight)
        else:
            nn.init.normal_(self.delta_head.weight, std=float(delta_head_init_std))
        nn.init.zeros_(self.range_head.weight)
        nn.init.constant_(
            self.range_head.bias[0],
            math.log(float(range_start_prior) / (1.0 - float(range_start_prior))),
        )
        nn.init.constant_(
            self.range_head.bias[1],
            math.log(float(range_end_prior) / (1.0 - float(range_end_prior))),
        )
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
        )

    @staticmethod
    def _position_basis(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )

    def _spatial_prior(
        self,
        row_fraction: torch.Tensor,
        x_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = torch.sigmoid(
            self.anchor_logits[:, None]
            + self.anchor_slopes[:, None] * (row_fraction[None, :] - 0.5)
        )
        distance = x_fraction.view(1, 1, -1) - anchor.unsqueeze(-1)
        prior = -0.5 * distance.square() / (self.spatial_prior_sigma**2)
        return self.spatial_prior_strength * prior, anchor

    def _attend(
        self,
        query_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        spatial_prior: torch.Tensor,
        projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.einsum(
            "bsrh,brxh->bsrx",
            projection(query_state),
            keys,
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits + spatial_prior.unsqueeze(0).to(dtype=logits.dtype)
        probability = torch.softmax(logits.float(), dim=-1)
        context = torch.einsum("bsrx,brxh->bsrh", probability, values.float())
        return logits, probability, context

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        slot_active: torch.Tensor,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if row_value_features.ndim != 4:
            raise ValueError("P2 row features must have shape [B,R,X,C]")
        batch, rows, x_bins, channels = row_value_features.shape
        if int(channels) != self.dim:
            raise ValueError("P2 row feature dimension mismatch")
        if slot_states.ndim != 3 or int(slot_states.shape[0]) != batch:
            raise ValueError("visual slot states must have shape [B,S,D]")
        slots = int(slot_states.shape[1])
        if slots != self.num_slots:
            raise ValueError("visual slot count does not match configuration")
        if tuple(slot_active.shape) != (batch, slots):
            raise ValueError("visual slot activity must have shape [B,S]")

        # The causal gate freezes the visual producer.  Only these fresh P2
        # projections and the live slot-side graph receive geometry gradient.
        features = row_value_features.detach().float()
        feature_state = self.feature_norm(features)
        x_fraction = torch.linspace(
            0.0,
            1.0,
            x_bins,
            device=features.device,
            dtype=torch.float32,
        )
        row_fraction = fixed_row_fractions(
            rows,
            device=features.device,
            dtype=torch.float32,
        )
        x_position = self.x_position_projection(
            self._position_basis(x_fraction)
        ).view(1, 1, x_bins, self.hidden_dim)
        keys = self.feature_key(feature_state) + x_position
        values = self.feature_value(feature_state) + x_position

        if self.detach_slot_states:
            slot_input = slot_states.detach()
        else:
            scale = self.slot_state_gradient_scale
            slot_input = slot_states.detach() + scale * (
                slot_states - slot_states.detach()
            )
        row_position = self.row_position_projection(
            self._position_basis(row_fraction)
        ).view(1, 1, rows, self.hidden_dim)
        visual_tokens = self.visual_slot_tokens.weight.view(
            1,
            slots,
            1,
            self.hidden_dim,
        )
        query_state = self.slot_projection(
            self.slot_norm(slot_input.float())
        ).unsqueeze(2)
        query_state = self.query_norm(query_state + visual_tokens + row_position)
        spatial_prior, anchor = self._spatial_prior(row_fraction, x_fraction)

        _first_logits, first_probability, first_context = self._attend(
            query_state,
            keys,
            values,
            spatial_prior,
            self.first_query,
        )
        hidden = query_state + self.first_context(first_context)
        hidden = hidden + self.first_ffn(self.first_output_norm(hidden))
        if self.vertical_encoder is not None:
            hidden = self.vertical_encoder(
                hidden.reshape(batch * slots, rows, self.hidden_dim)
            ).reshape(batch, slots, rows, self.hidden_dim)

        _second_logits, probability, second_context = self._attend(
            hidden,
            keys,
            values,
            spatial_prior,
            self.second_query,
        )
        hidden = hidden + self.second_context(second_context)
        hidden = hidden + self.second_ffn(self.second_output_norm(hidden))
        x_pixels = x_fraction * float(max(self.input_w - 1, 1))
        reference_x = torch.einsum("bsrx,x->bsr", probability, x_pixels)

        delta_logits = self.delta_head(self.delta_norm(hidden))
        delta_probability = torch.softmax(delta_logits.float(), dim=-1)
        delta_offsets = self.delta_offsets_px.to(
            device=delta_probability.device,
            dtype=delta_probability.dtype,
        )
        delta = (delta_probability * delta_offsets).sum(dim=-1)
        refined_x = (reference_x + delta).clamp(
            0.0,
            float(max(self.input_w - 1, 1)),
        )

        pooled_hidden = hidden.mean(dim=2)
        predicted_range = sort_range_norm(
            torch.sigmoid(self.range_head(self.range_norm(pooled_hidden))).float()
        )
        geometry_valid = torch.ones(
            (batch, slots),
            dtype=torch.bool,
            device=slot_states.device,
        )
        active_rows = geometry_valid.unsqueeze(-1).expand(-1, -1, rows)
        active_count = active_rows.float().sum(dim=(1, 2)).clamp_min(1.0)
        entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(
            dim=-1
        )
        top1 = probability.amax(dim=-1)
        mean_abs = (delta.abs() * active_rows.float()).sum(dim=(1, 2))
        mean_abs = mean_abs / active_count
        max_abs = delta.abs().amax(dim=(1, 2))
        boundary_mass = delta_probability[..., (0, -1)].sum(dim=-1).mean(
            dim=(1, 2)
        )
        return {
            "selection_slot_pred_x_rows": refined_x,
            "selection_slot_range_norm": predicted_range,
            "selection_slot_active": slot_active.bool(),
            "selection_slot_geometry_valid": geometry_valid,
            "selection_slot_input_reference_x_rows": reference_x,
            "selection_slot_input_range_norm": predicted_range,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta": torch.zeros_like(predicted_range),
            "selection_slot_range_delta_boundary_mass": predicted_range.new_zeros(
                (batch, slots)
            ),
            "selection_slot_delta_mean_abs": mean_abs,
            "selection_slot_delta_max_abs": max_abs,
            "selection_slot_delta_boundary_mass": boundary_mass,
            "selection_slot_visual_attention": probability,
            "selection_slot_visual_first_attention": first_probability,
            "selection_slot_visual_attention_entropy": entropy.mean(dim=(1, 2)),
            "selection_slot_visual_attention_top1_mass": top1.mean(dim=(1, 2)),
            "selection_slot_visual_anchor_fraction": anchor,
        }


class FourSlotUnifiedProposalVisualDecoder(nn.Module):
    """Turn four slots into row-level lane objects over proposal and P2 memory.

    V7 exposes a strong proposal population but commits final geometry to one
    hard proposal ID.  V10 went too far in the opposite direction and asked a
    cold P2-only branch to rediscover the lane set.  This decoder instead uses
    a soft global retrieval over all 32 proposal row memories as its coarse
    geometry, then consumes the complete P2 row grid.  Its final residual
    support spans the full image width; the legacy hard ID is diagnostic only.

    Proposal/backbone tensors are detached in the first causal gate.  Geometry
    nevertheless reaches every fresh row-state, proposal-attention and visual
    projection parameter.  A post-geometry activity residual is emitted from
    the same lane-object state; route probability is deliberately absent from
    the public deployment score.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        delta_offsets_px: tuple[float, ...],
        range_delta_offsets_norm: tuple[float, ...],
        proposal_logit_residual_scale: float = 1.0,
        proposal_attention_temperature: float = 1.0,
        proposal_attention_sinkhorn_iterations: int = 64,
        visual_prior_strength: float = 0.25,
        visual_prior_sigma: float = 0.35,
        output_head_init_std: float = 1.0e-5,
        activity_head_init_std: float = 1.0e-7,
    ) -> None:
        super().__init__()
        offsets = tuple(float(value) for value in delta_offsets_px)
        range_offsets = tuple(float(value) for value in range_delta_offsets_norm)
        for values, label in ((offsets, "x"), (range_offsets, "range")):
            if len(values) < 3 or tuple(sorted(values)) != values:
                raise ValueError(f"unified slot {label} offsets must be sorted")
            if not any(abs(value) < 1.0e-12 for value in values):
                raise ValueError(f"unified slot {label} offsets need zero")
            if any(
                abs(left + right) > 1.0e-6
                for left, right in zip(values, reversed(values))
            ):
                raise ValueError(
                    f"unified slot {label} offsets must be symmetric"
                )
        if int(num_slots) < 1:
            raise ValueError("unified slot decoder needs at least one slot")
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("unified slot heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("unified slot decoder needs vertical interaction")
        if float(proposal_logit_residual_scale) <= 0.0:
            raise ValueError("proposal residual scale must be positive")
        if float(proposal_attention_temperature) <= 0.0:
            raise ValueError("proposal attention temperature must be positive")
        if int(proposal_attention_sinkhorn_iterations) < 1:
            raise ValueError(
                "proposal attention Sinkhorn iterations must be positive"
            )
        if float(visual_prior_strength) < 0.0:
            raise ValueError("visual prior strength must be non-negative")
        if float(visual_prior_sigma) <= 0.0:
            raise ValueError("visual prior sigma must be positive")
        if float(output_head_init_std) <= 0.0:
            raise ValueError("output-head init std must be positive")
        if float(activity_head_init_std) <= 0.0:
            raise ValueError("activity-head init std must be positive")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.proposal_logit_residual_scale = float(
            proposal_logit_residual_scale
        )
        self.proposal_attention_temperature = float(
            proposal_attention_temperature
        )
        self.proposal_attention_sinkhorn_iterations = int(
            proposal_attention_sinkhorn_iterations
        )
        self.visual_prior_strength = float(visual_prior_strength)
        self.visual_prior_sigma = float(visual_prior_sigma)

        # Persistent [B,S,R,H] lane-object initialization.  No hard proposal
        # row is gathered here: proposal identity is memory, never ownership.
        self.proposal_row_norm = nn.LayerNorm(self.dim)
        self.proposal_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)

        # Full-width row-aligned P2 evidence.  No anchor-local crop is used.
        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.x_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.first_visual_query = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.second_visual_query = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.first_visual_context = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.second_visual_context = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )

        # Evidence-aware global attention over *all* proposals.  The old V7
        # route logit is a warm-start prior, not a geometry gather operator.
        self.global_proposal_query = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.global_proposal_context = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.coarse_geometry_projection = nn.Linear(
            3,
            self.hidden_dim,
            bias=False,
        )
        self.proposal_fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.proposal_fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.final_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        # The wide distribution can move a soft proposal-memory lane anywhere
        # in an 800-pixel frame.  Symmetric zero initialization initially
        # exposes the predicted-soft reference without a scalar blend gate.
        self.delta_head = nn.Linear(
            self.hidden_dim,
            len(offsets),
            bias=False,
        )
        self.range_delta_head = nn.Linear(
            self.hidden_dim,
            2 * len(range_offsets),
            bias=False,
        )
        self.post_geometry_activity = nn.Linear(self.hidden_dim, 1)
        # Tiny but non-zero heads make every edge a genuine forward derivative
        # from the first backward pass while keeping the predicted-soft start
        # and legacy activity essentially unchanged.
        nn.init.normal_(self.delta_head.weight, std=float(output_head_init_std))
        nn.init.normal_(
            self.range_delta_head.weight,
            std=float(output_head_init_std),
        )
        nn.init.normal_(
            self.post_geometry_activity.weight,
            std=float(activity_head_init_std),
        )
        nn.init.zeros_(self.post_geometry_activity.bias)
        nn.init.normal_(self.slot_tokens.weight, std=0.02)
        # Start from the measured V7 predicted-soft policy, not from an
        # arbitrary random re-ranking of 32 proposals.  This is a small
        # trainable residual (not a mix gate): it has nonzero gradient on the
        # first step and can grow without a bounded scalar bottleneck.
        nn.init.normal_(self.global_proposal_query.weight, std=1.0e-3)
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
        )
        self.register_buffer(
            "range_delta_offsets_norm",
            torch.tensor(range_offsets, dtype=torch.float32),
        )

    @staticmethod
    def _position_basis(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )

    def _visual_attention(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        center_x: torch.Tensor,
        x_fraction: torch.Tensor,
        *,
        query_projection: nn.Linear,
        context_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum(
            "bsrh,brxh->bsrx",
            query_projection(hidden),
            keys,
        ) / math.sqrt(float(self.hidden_dim))
        distance = x_fraction.view(1, 1, 1, -1) - center_x.unsqueeze(-1)
        prior = -0.5 * distance.square() / (self.visual_prior_sigma**2)
        logits = logits + self.visual_prior_strength * prior.to(logits.dtype)
        probability = torch.softmax(logits.float(), dim=-1)
        context = torch.einsum(
            "bsrx,brxh->bsrh",
            probability,
            values.float(),
        )
        return hidden + context_projection(context), probability

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        legacy_active_logits: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        route_indices: torch.Tensor,
        legacy_route_logits: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if row_value_features.ndim != 4:
            raise ValueError("unified P2 rows must have shape [B,R,X,C]")
        batch, rows, x_bins, channels = row_value_features.shape
        if int(channels) != self.dim:
            raise ValueError("unified P2 feature dimension mismatch")
        if tuple(slot_states.shape[:2]) != (batch, self.num_slots):
            raise ValueError("unified slot state shape mismatch")
        slots = self.num_slots
        candidates = int(proposal_x_rows.shape[1])
        if tuple(route_indices.shape) != (batch, slots):
            raise ValueError("unified route index shape mismatch")
        if tuple(legacy_active_logits.shape) != (batch, slots):
            raise ValueError("unified activity shape mismatch")
        if tuple(legacy_route_logits.shape) != (batch, slots, candidates):
            raise ValueError("unified legacy route-logit shape mismatch")
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("unified candidate-valid shape mismatch")
        # The first gate protects every successful upstream tensor.  Fresh
        # projections still receive both final-geometry and cluster-attention
        # gradients through this decoder.
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_range = sort_range_norm(
            proposal_range_norm.detach().float()
        )
        # The old unique ID is retained only as public provenance.  It does
        # not gate or initialize geometry; every slot is valid whenever the
        # proposal population contains at least one valid memory row.
        geometry_valid = candidate_valid.bool().any(dim=-1, keepdim=True).expand(
            -1,
            slots,
        )

        row_fraction = fixed_row_fractions(
            rows,
            device=slot_states.device,
            dtype=torch.float32,
        )
        row_position = self.row_position_projection(
            self._position_basis(row_fraction)
        ).view(1, 1, rows, self.hidden_dim)
        slot_seed = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2)
        slot_seed = slot_seed + self.slot_tokens.weight.view(
            1,
            slots,
            1,
            self.hidden_dim,
        )
        normalized_proposal_rows = self.proposal_row_norm(proposal_rows)
        hidden = slot_seed + row_position
        hidden = self.initial_norm(hidden)

        # First retrieve one coherent global proposal distribution per slot
        # from all 32 full row memories.  The frozen V7 logits are merely a
        # useful prior; the differentiable correction sees every proposal row.
        proposal_keys = self.proposal_key(normalized_proposal_rows)
        proposal_values = self.proposal_value(normalized_proposal_rows)
        row_logits = torch.einsum(
            "bsrh,bnrh->bsnr",
            self.global_proposal_query(hidden),
            proposal_keys,
        ) / math.sqrt(float(self.hidden_dim))
        row_y = row_fraction.view(1, 1, rows)
        proposal_visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & torch.isfinite(proposal_x)
        )
        visible_weight = proposal_visible[:, None].to(row_logits.dtype)
        learned_logits = (row_logits * visible_weight).sum(dim=-1)
        learned_logits = learned_logits / visible_weight.sum(dim=-1).clamp_min(
            1.0
        )
        proposal_logits = legacy_route_logits.detach().float() + (
            self.proposal_logit_residual_scale * learned_logits
        )
        proposal_logits = proposal_logits.masked_fill(
            ~candidate_valid[:, None, :].bool(),
            -1.0e4,
        )
        # Four slots compete through a differentiable injective relaxation:
        # every slot has unit mass and every proposal has total capacity <= 1.
        # This is cross-slot set reasoning without reinstating a hard ID.
        proposal_attention = structured_unique_route_marginals(
            proposal_logits,
            candidate_valid,
            temperature=self.proposal_attention_temperature,
            iterations=self.proposal_attention_sinkhorn_iterations,
        )
        proposal_context = torch.einsum(
            "bsn,bnrh->bsrh",
            proposal_attention,
            proposal_values,
        )
        coarse_x = torch.einsum(
            "bsn,bnr->bsr",
            proposal_attention,
            proposal_x,
        )
        coarse_range = sort_range_norm(
            torch.einsum(
                "bsn,bnd->bsd",
                proposal_attention,
                proposal_range,
            )
        )
        coarse_geometry = torch.cat(
            (
                coarse_x.unsqueeze(-1)
                / float(max(self.input_w - 1, 1)),
                coarse_range.unsqueeze(2).expand(-1, -1, rows, -1),
            ),
            dim=-1,
        )
        hidden = hidden + self.global_proposal_context(proposal_context)
        hidden = hidden + self.coarse_geometry_projection(coarse_geometry)
        hidden = hidden + self.proposal_fusion_ffn(
            self.proposal_fusion_norm(hidden)
        )

        # Only after global memory retrieval does the slot read live,
        # full-width row-aligned P2 evidence.  Both passes attend over every x
        # bin; the Gaussian term is a broad positional prior, not a crop.
        features = self.feature_norm(row_value_features.detach().float())
        x_fraction = torch.linspace(
            0.0,
            1.0,
            x_bins,
            device=slot_states.device,
            dtype=torch.float32,
        )
        x_position = self.x_position_projection(
            self._position_basis(x_fraction)
        ).view(1, 1, x_bins, self.hidden_dim)
        feature_keys = self.feature_key(features) + x_position
        feature_values = self.feature_value(features) + x_position
        coarse_center = (
            coarse_x / float(max(self.input_w - 1, 1))
        ).clamp(0.0, 1.0)
        hidden, first_visual_probability = self._visual_attention(
            hidden,
            feature_keys,
            feature_values,
            coarse_center,
            x_fraction,
            query_projection=self.first_visual_query,
            context_projection=self.first_visual_context,
        )
        hidden = self.vertical_encoder(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        hidden, visual_probability = self._visual_attention(
            hidden,
            feature_keys,
            feature_values,
            coarse_center,
            x_fraction,
            query_projection=self.second_visual_query,
            context_projection=self.second_visual_context,
        )
        hidden = hidden + self.final_ffn(self.final_norm(hidden))

        delta_logits = self.delta_head(self.final_norm(hidden))
        delta_probability = torch.softmax(delta_logits.float(), dim=-1)
        delta_offsets = self.delta_offsets_px.to(
            device=hidden.device,
            dtype=delta_probability.dtype,
        )
        # Subtract the explicit uniform distribution so zero logits produce
        # an exactly-zero residual in floating point, not merely a symmetric
        # sum that can leave a sub-ULP remainder.
        delta = (
            (
                delta_probability
                - 1.0 / float(int(delta_probability.shape[-1]))
            )
            * delta_offsets
        ).sum(dim=-1)
        delta = torch.where(
            geometry_valid.unsqueeze(-1),
            delta,
            torch.zeros_like(delta),
        )
        # Final geometry is owned by the soft proposal-memory slot.  The old
        # hard-routed/refined V7 curve is not part of this equation.
        raw_final_x = coarse_x + delta
        final_x = raw_final_x.clamp(
            0.0,
            float(max(self.input_w - 1, 1)),
        )

        pooled_hidden = hidden.mean(dim=2)
        range_logits = self.range_delta_head(
            self.final_norm(pooled_hidden)
        ).view(batch, slots, 2, int(self.range_delta_offsets_norm.numel()))
        range_probability = torch.softmax(range_logits.float(), dim=-1)
        range_offsets = self.range_delta_offsets_norm.to(
            device=hidden.device,
            dtype=range_probability.dtype,
        )
        range_delta = (
            (
                range_probability
                - 1.0 / float(int(range_probability.shape[-1]))
            )
            * range_offsets
        ).sum(dim=-1)
        range_delta = torch.where(
            geometry_valid.unsqueeze(-1),
            range_delta,
            torch.zeros_like(range_delta),
        )
        raw_final_range = coarse_range + range_delta
        final_range = sort_range_norm(raw_final_range.clamp(0.0, 1.0))

        activity_residual = self.post_geometry_activity(
            self.final_norm(pooled_hidden)
        ).squeeze(-1)
        final_active_logits = legacy_active_logits.detach().float() + (
            activity_residual
        )
        final_active = (final_active_logits >= 0.0) & geometry_valid
        public_indices = torch.where(
            final_active,
            route_indices,
            route_indices.new_full(route_indices.shape, -1),
        )
        final_scores = torch.sigmoid(final_active_logits)

        valid_rows = geometry_valid.unsqueeze(-1).expand(-1, -1, rows)
        valid_count = valid_rows.float().sum(dim=(1, 2)).clamp_min(1.0)
        mean_abs_delta = (
            delta.abs() * valid_rows.float()
        ).sum(dim=(1, 2)) / valid_count
        proposal_entropy = -(
            proposal_attention
            * proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        visual_entropy = -(
            visual_probability
            * visual_probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "selection_slot_pred_x_rows": final_x,
            "selection_slot_range_norm": final_range,
            "selection_slot_active_logits": final_active_logits,
            "selection_slot_active": final_active,
            "selection_slot_geometry_valid": geometry_valid,
            "selection_slot_indices": public_indices,
            "selection_slot_scores": final_scores,
            "selection_slot_input_reference_x_rows": coarse_x,
            "selection_slot_input_range_norm": coarse_range,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta": range_delta,
            "selection_slot_range_delta_logits": range_logits,
            "selection_slot_range_delta_offsets_norm": (
                self.range_delta_offsets_norm
            ),
            "selection_slot_range_delta_boundary_mass": (
                range_probability[..., (0, -1)].sum(dim=-1)
            ),
            "selection_slot_delta_mean_abs": mean_abs_delta,
            "selection_slot_delta_max_abs": delta.abs().amax(dim=(1, 2)),
            "selection_slot_delta_boundary_mass": (
                delta_probability[..., (0, -1)].sum(dim=-1).mean(dim=(1, 2))
            ),
            "selection_slot_unified_aux_x_rows": coarse_x,
            "selection_slot_unified_aux_range_norm": coarse_range,
            "selection_slot_unified_proposal_logits": proposal_logits,
            "selection_slot_unified_proposal_attention": proposal_attention,
            "selection_slot_unified_proposal_entropy": proposal_entropy.mean(
                dim=-1
            ),
            "selection_slot_unified_visual_attention": visual_probability,
            "selection_slot_unified_first_visual_attention": (
                first_visual_probability
            ),
            "selection_slot_unified_visual_entropy": visual_entropy.mean(
                dim=(1, 2)
            ),
            "selection_slot_unified_activity_residual": activity_residual,
            "selection_slot_unified_base_x_rows": coarse_x,
            "selection_slot_unified_base_range_norm": coarse_range,
        }


class FourSlotVisualFirstAssociation(nn.Module):
    """Associate four persistent lane slots only after reading image evidence.

    V11 retrieved proposal memory before P2 could influence a slot.  Its late
    visual branch could refine coordinates, but it could not move a slot to a
    different global proposal cluster.  V12 reverses that causal order:

    ``V7 slot/curve anchor -> P2 -> cross-slot/vertical state -> proposals``.

    This module is intentionally an association-only Stage-A sidecar.  The
    deployed V7 x/range/activity/score tensors remain untouched while direct
    row-visual and proposal-cluster losses train the new state.  Geometry is
    allowed to consume this state only after association generalizes on
    clip-disjoint and validation images.

    The frozen V7 route logits are deliberately absent from the proposal
    equation.  V7 contributes slot identity and a broad visual center through
    its final curve, but it cannot dominate candidate ranking by addition.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        proposal_attention_temperature: float = 1.0,
        proposal_attention_sinkhorn_iterations: int = 64,
        visual_prior_strength: float = 0.25,
        visual_prior_sigma: float = 0.35,
        proposal_curve_distance_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if int(num_slots) < 1:
            raise ValueError("visual-first association needs at least one slot")
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("visual-first heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("visual-first association needs vertical layers")
        if float(proposal_attention_temperature) <= 0.0:
            raise ValueError("visual-first proposal temperature must be positive")
        if int(proposal_attention_sinkhorn_iterations) < 1:
            raise ValueError("visual-first Sinkhorn iterations must be positive")
        if float(visual_prior_strength) < 0.0:
            raise ValueError("visual-first prior strength must be non-negative")
        if float(visual_prior_sigma) <= 0.0:
            raise ValueError("visual-first prior sigma must be positive")
        if float(proposal_curve_distance_scale) < 0.0:
            raise ValueError("proposal curve-distance scale must be non-negative")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.proposal_attention_temperature = float(
            proposal_attention_temperature
        )
        self.proposal_attention_sinkhorn_iterations = int(
            proposal_attention_sinkhorn_iterations
        )
        self.visual_prior_strength = float(visual_prior_strength)
        self.visual_prior_sigma = float(visual_prior_sigma)
        self.proposal_curve_distance_scale = float(
            proposal_curve_distance_scale
        )

        # Persistent row state.  Exact V7 geometry is an input anchor, never
        # the candidate-ranking logit used by the new association policy.
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.anchor_geometry_projection = nn.Linear(3, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)

        # P2 content is read before proposal memory.  Position is represented
        # only by an explicit broad logit prior; it is not injected into P2
        # values, so zero-image and wrong-image interventions remain causal.
        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.first_visual_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.second_visual_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.first_visual_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.second_visual_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )

        # Explicit four-slot interaction occurs while the state is visual,
        # before candidates are scored.  This is the set-level mechanism that
        # prevents four independent visual queries from claiming one lane.
        self.cross_slot_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_slot_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.visual_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        # All-32 proposal association consumes the already visual state.  No
        # frozen V7 route logit is added here.  Geometry compatibility is an
        # explicit normalized curve distance, not a learned identity shortcut.
        self.proposal_row_norm = nn.LayerNorm(self.dim)
        self.proposal_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.proposal_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.association_norm = nn.LayerNorm(self.hidden_dim)
        self.association_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        nn.init.normal_(self.slot_tokens.weight, std=0.02)

    @staticmethod
    def _position_basis(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )

    def _visual_attention(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        center_x: torch.Tensor,
        x_fraction: torch.Tensor,
        *,
        query_projection: nn.Linear,
        context_projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.einsum(
            "bsrh,brxh->bsrx",
            query_projection(hidden),
            keys,
        ) / math.sqrt(float(self.hidden_dim))
        distance = x_fraction.view(1, 1, 1, -1) - center_x.unsqueeze(-1)
        prior = -0.5 * distance.square() / (self.visual_prior_sigma**2)
        logits = logits + self.visual_prior_strength * prior.to(logits.dtype)
        probability = torch.softmax(logits.float(), dim=-1)
        context = torch.einsum(
            "bsrx,brxh->bsrh",
            probability,
            values.float(),
        )
        expected_x = torch.einsum("bsrx,x->bsr", probability, x_fraction)
        return hidden + context_projection(context), logits, expected_x

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        anchor_x_rows: torch.Tensor,
        anchor_range_norm: torch.Tensor,
        anchor_geometry_valid: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if row_value_features.ndim != 4:
            raise ValueError("visual-first P2 rows must have shape [B,R,X,C]")
        batch, rows, x_bins, channels = row_value_features.shape
        if int(channels) != self.dim:
            raise ValueError("visual-first P2 feature dimension mismatch")
        if tuple(slot_states.shape[:2]) != (batch, self.num_slots):
            raise ValueError("visual-first slot state shape mismatch")
        slots = self.num_slots
        candidates = int(proposal_x_rows.shape[1])
        if tuple(anchor_x_rows.shape) != (batch, slots, rows):
            raise ValueError("visual-first anchor x shape mismatch")
        if tuple(anchor_range_norm.shape) != (batch, slots, 2):
            raise ValueError("visual-first anchor range shape mismatch")
        if tuple(anchor_geometry_valid.shape) != (batch, slots):
            raise ValueError("visual-first anchor validity shape mismatch")
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("visual-first candidate validity shape mismatch")

        # Protect all proven V7/proposal/P2 producers in Stage A.  The fresh
        # consumer receives gradients, while upstream coverage cannot regress.
        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        geometry_valid = anchor_geometry_valid.detach().bool()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        features = self.feature_norm(row_value_features.detach().float())

        row_fraction = fixed_row_fractions(
            rows,
            device=slot_states.device,
            dtype=torch.float32,
        )
        row_position = self.row_position_projection(
            self._position_basis(row_fraction)
        ).view(1, 1, rows, self.hidden_dim)
        anchor_geometry = torch.cat(
            (
                anchor_x.unsqueeze(-1) / float(max(self.input_w - 1, 1)),
                anchor_range.unsqueeze(2).expand(-1, -1, rows, -1),
            ),
            dim=-1,
        )
        hidden = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2)
        hidden = hidden + self.slot_tokens.weight.view(
            1, slots, 1, self.hidden_dim
        )
        hidden = hidden + row_position
        hidden = hidden + self.anchor_geometry_projection(anchor_geometry)
        hidden = self.initial_norm(hidden)

        # There is intentionally no absolute-position vector in the values.
        # The only positional shortcut is the measured V7 curve prior in the
        # logits; correct/wrong/zero P2 replay can therefore test appearance.
        feature_keys = self.feature_key(features)
        feature_values = self.feature_value(features)
        x_fraction = torch.linspace(
            0.0,
            1.0,
            x_bins,
            device=slot_states.device,
            dtype=torch.float32,
        )
        anchor_center = (
            anchor_x / float(max(self.input_w - 1, 1))
        ).clamp(0.0, 1.0)
        hidden, first_logits, first_expected_x = self._visual_attention(
            hidden,
            feature_keys,
            feature_values,
            anchor_center,
            x_fraction,
            query_projection=self.first_visual_query,
            context_projection=self.first_visual_context,
        )

        # Cross the four slots at every row, then enforce within-lane vertical
        # coherence.  The second P2 pass follows the first visual expectation,
        # rather than being locked to the original proposal/anchor center.
        by_row = hidden.permute(0, 2, 1, 3).reshape(
            batch * rows, slots, self.hidden_dim
        )
        normalized_by_row = self.cross_slot_norm(by_row)
        cross_slot, _weights = self.cross_slot_attention(
            normalized_by_row,
            normalized_by_row,
            normalized_by_row,
            need_weights=False,
        )
        hidden = (by_row + cross_slot).reshape(
            batch, rows, slots, self.hidden_dim
        ).permute(0, 2, 1, 3)
        hidden = self.vertical_encoder(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        hidden, visual_logits, visual_expected_x = self._visual_attention(
            hidden,
            feature_keys,
            feature_values,
            first_expected_x,
            x_fraction,
            query_projection=self.second_visual_query,
            context_projection=self.second_visual_context,
        )
        hidden = hidden + self.visual_ffn(self.visual_norm(hidden))

        normalized_proposal_rows = self.proposal_row_norm(proposal_rows)
        proposal_keys = self.proposal_key(normalized_proposal_rows)
        proposal_values = self.proposal_value(normalized_proposal_rows)
        row_logits = torch.einsum(
            "bsrh,bnrh->bsnr",
            self.proposal_query(self.visual_norm(hidden)),
            proposal_keys,
        ) / math.sqrt(float(self.hidden_dim))
        row_y = row_fraction.view(1, 1, rows)
        proposal_visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & torch.isfinite(proposal_x)
        )
        visible_weight = proposal_visible[:, None].to(row_logits.dtype)
        content_logits = (row_logits * visible_weight).sum(dim=-1)
        content_logits = content_logits / visible_weight.sum(dim=-1).clamp_min(
            1.0
        )
        proposal_x_fraction = proposal_x / float(max(self.input_w - 1, 1))
        curve_distance = (
            (proposal_x_fraction[:, None] - visual_expected_x.unsqueeze(2)).abs()
            * visible_weight
        ).sum(dim=-1) / visible_weight.sum(dim=-1).clamp_min(1.0)
        proposal_logits = content_logits - (
            self.proposal_curve_distance_scale * curve_distance
        )
        proposal_logits = proposal_logits.masked_fill(
            ~candidate_valid[:, None, :].bool(),
            -1.0e4,
        )
        proposal_attention = structured_unique_route_marginals(
            proposal_logits,
            candidate_valid,
            temperature=self.proposal_attention_temperature,
            iterations=self.proposal_attention_sinkhorn_iterations,
        )
        proposal_context = torch.einsum(
            "bsn,bnrh->bsrh",
            proposal_attention,
            proposal_values,
        )
        associated_hidden = hidden + self.proposal_context(proposal_context)
        associated_hidden = associated_hidden + self.association_ffn(
            self.association_norm(associated_hidden)
        )

        first_probability = torch.softmax(first_logits.float(), dim=-1)
        visual_probability = torch.softmax(visual_logits.float(), dim=-1)
        proposal_entropy = -(
            proposal_attention
            * proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        visual_entropy = -(
            visual_probability
            * visual_probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "selection_slot_v12_anchor_x_rows": anchor_x,
            "selection_slot_v12_anchor_range_norm": anchor_range,
            "selection_slot_v12_geometry_valid": geometry_valid,
            "selection_slot_v12_first_visual_logits": first_logits,
            "selection_slot_v12_first_visual_attention": first_probability,
            "selection_slot_v12_first_visual_x_rows": (
                first_expected_x * float(max(self.input_w - 1, 1))
            ),
            "selection_slot_v12_visual_logits": visual_logits,
            "selection_slot_v12_visual_attention": visual_probability,
            "selection_slot_v12_visual_x_rows": (
                visual_expected_x * float(max(self.input_w - 1, 1))
            ),
            "selection_slot_v12_visual_entropy": visual_entropy.mean(
                dim=(1, 2)
            ),
            "selection_slot_v12_proposal_logits": proposal_logits,
            "selection_slot_v12_proposal_attention": proposal_attention,
            "selection_slot_v12_proposal_entropy": proposal_entropy.mean(
                dim=-1
            ),
            # Retained for the future Stage-B geometry consumer and topology
            # audits.  It is not a deployed output in association-only mode.
            "selection_slot_v12_associated_state": associated_hidden,
            # Pre-proposal visual lane-object state.  V13 consumes this state
            # directly so the failed global proposal-ID interface cannot
            # become the owner of final geometry again.
            "selection_slot_v12_visual_state": hidden,
        }


class FourSlotCorrectedVisualFirstAssociation(nn.Module):
    """V14 Stage-A image-causal association with feasible joint targets.

    One P2 pass precedes all proposal ranking.  The learned transport has 32
    real proposal columns plus one private dustbin per slot; no V7 route logit
    is added.  Exact V7 geometry/activity/score stay on the public deployment
    path, so Stage A changes only diagnostic sidecar tensors.
    """

    FEATURE_POLICIES = {
        "correct",
        "zero_content",
        "position_only",
        "zero_content_zero_position",
        "x_reversed",
        "row_reversed",
    }

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        min_valid_rows: int = 5,
        proposal_attention_temperature: float = 1.0,
        proposal_attention_sinkhorn_iterations: int = 64,
        visual_prior_strength: float = 0.25,
        visual_prior_sigma: float = 0.35,
    ) -> None:
        super().__init__()
        if int(num_slots) < 1:
            raise ValueError("V14 association needs at least one slot")
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("V14 attention heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("V14 association needs vertical interaction")
        if int(min_valid_rows) < 1:
            raise ValueError("V14 min_valid_rows must be positive")
        if float(proposal_attention_temperature) <= 0.0:
            raise ValueError("V14 proposal temperature must be positive")
        if int(proposal_attention_sinkhorn_iterations) < 1:
            raise ValueError("V14 Sinkhorn iterations must be positive")
        if float(visual_prior_strength) < 0.0:
            raise ValueError("V14 visual prior strength must be non-negative")
        if float(visual_prior_sigma) <= 0.0:
            raise ValueError("V14 visual prior sigma must be positive")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.min_valid_rows = int(min_valid_rows)
        self.proposal_attention_temperature = float(
            proposal_attention_temperature
        )
        self.proposal_attention_sinkhorn_iterations = int(
            proposal_attention_sinkhorn_iterations
        )
        self.visual_prior_strength = float(visual_prior_strength)
        self.visual_prior_sigma = float(visual_prior_sigma)

        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.anchor_geometry_projection = nn.Linear(3, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)

        # Image content and x position have deliberately separate paths.  P2
        # values never receive positional embeddings; the intervention audit
        # can therefore retain position keys while deleting image content.
        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.x_position_key = nn.Linear(4, self.hidden_dim, bias=False)
        self.visual_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.visual_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.visual_x_projection = nn.Linear(4, self.hidden_dim, bias=False)

        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.visual_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        # Proposal row content and geometry share one key only after the slot
        # has read P2.  No legacy route logit or hard proposal ID is accepted.
        self.proposal_row_norm = nn.LayerNorm(self.dim)
        self.proposal_content_key = nn.Linear(
            self.dim, self.hidden_dim, bias=False
        )
        self.proposal_geometry_key = nn.Linear(
            4, self.hidden_dim, bias=False
        )
        self.proposal_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.private_dustbin = nn.Linear(self.hidden_dim, 1)
        self.proposal_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.association_norm = nn.LayerNorm(self.hidden_dim)
        self.association_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        nn.init.normal_(self.slot_tokens.weight, std=0.02)

    @staticmethod
    def _position_basis(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        anchor_x_rows: torch.Tensor,
        anchor_range_norm: torch.Tensor,
        anchor_geometry_valid: torch.Tensor,
        anchor_active: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_value_features: torch.Tensor,
        feature_policy: str = "correct",
    ) -> dict[str, torch.Tensor]:
        policy = str(feature_policy).strip().lower()
        if policy not in self.FEATURE_POLICIES:
            raise ValueError(f"unsupported V14 feature policy: {feature_policy}")
        if row_value_features.ndim != 4:
            raise ValueError("V14 P2 rows must have shape [B,R,X,C]")
        batch, rows, x_bins, channels = row_value_features.shape
        slots = self.num_slots
        candidates = int(proposal_x_rows.shape[1])
        if int(channels) != self.dim:
            raise ValueError("V14 P2 feature dimension mismatch")
        if tuple(slot_states.shape[:2]) != (batch, slots):
            raise ValueError("V14 slot state shape mismatch")
        if tuple(anchor_x_rows.shape) != (batch, slots, rows):
            raise ValueError("V14 anchor x shape mismatch")
        if tuple(anchor_range_norm.shape) != (batch, slots, 2):
            raise ValueError("V14 anchor range shape mismatch")
        if tuple(anchor_geometry_valid.shape) != (batch, slots):
            raise ValueError("V14 geometry-valid shape mismatch")
        if tuple(anchor_active.shape) != (batch, slots):
            raise ValueError("V14 source-active shape mismatch")
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("V14 candidate validity shape mismatch")

        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        geometry_valid = anchor_geometry_valid.detach().bool()
        source_active = anchor_active.detach().bool()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        candidate_valid = candidate_valid.detach().bool()
        raw_features = row_value_features.detach().float()
        if policy == "x_reversed":
            raw_features = raw_features.flip(dims=(2,))
        elif policy == "row_reversed":
            raw_features = raw_features.flip(dims=(1,))
        elif policy in {
            "zero_content",
            "position_only",
            "zero_content_zero_position",
        }:
            raw_features = torch.zeros_like(raw_features)

        row_fraction = fixed_row_fractions(
            rows,
            device=slot_states.device,
            dtype=torch.float32,
        )
        row_position = self.row_position_projection(
            self._position_basis(row_fraction)
        ).view(1, 1, rows, self.hidden_dim)
        anchor_geometry = torch.cat(
            (
                anchor_x.unsqueeze(-1) / float(max(self.input_w - 1, 1)),
                anchor_range.unsqueeze(2).expand(-1, -1, rows, -1),
            ),
            dim=-1,
        )
        initial = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2)
        initial = initial + self.slot_tokens.weight.view(
            1, slots, 1, self.hidden_dim
        )
        initial = initial + row_position
        initial = initial + self.anchor_geometry_projection(anchor_geometry)
        initial = self.initial_norm(initial)

        features = self.feature_norm(raw_features)
        content_keys = self.feature_key(features)
        content_values = self.feature_value(features)
        if policy in {"position_only", "zero_content_zero_position"}:
            content_keys = torch.zeros_like(content_keys)
            content_values = torch.zeros_like(content_values)
        x_fraction = torch.linspace(
            0.0,
            1.0,
            x_bins,
            device=slot_states.device,
            dtype=torch.float32,
        )
        position_keys = self.x_position_key(
            self._position_basis(x_fraction)
        ).view(1, 1, x_bins, self.hidden_dim)
        if policy == "zero_content_zero_position":
            position_keys = torch.zeros_like(position_keys)
        feature_keys = content_keys + position_keys

        # The supervised visual-localization distribution may train U0.  The
        # association branch replays the exact same query with U0 detached:
        # its forward values are identical, while L_assoc cannot turn the V7
        # anchor/slot projection into a direct proposal-ranking shortcut.
        visual_logits = torch.einsum(
            "bsrh,brxh->bsrx",
            self.visual_query(initial),
            feature_keys,
        ) / math.sqrt(float(self.hidden_dim))
        association_visual_logits = torch.einsum(
            "bsrh,brxh->bsrx",
            self.visual_query(initial.detach()),
            feature_keys,
        ) / math.sqrt(float(self.hidden_dim))
        anchor_center = (
            anchor_x / float(max(self.input_w - 1, 1))
        ).clamp(0.0, 1.0)
        distance = x_fraction.view(1, 1, 1, -1) - anchor_center.unsqueeze(-1)
        prior = -0.5 * distance.square() / (self.visual_prior_sigma**2)
        visual_logits = visual_logits + self.visual_prior_strength * prior
        association_visual_logits = (
            association_visual_logits + self.visual_prior_strength * prior
        )
        visual_probability = torch.softmax(
            association_visual_logits.float(), dim=-1
        )
        visual_context = torch.einsum(
            "bsrx,brxh->bsrh",
            visual_probability,
            content_values.float(),
        )
        visual_x_fraction = torch.einsum(
            "bsrx,x->bsr", visual_probability, x_fraction
        )
        # Preserve U0 in the forward value but stop association supervision
        # from learning a proposal/anchor-only shortcut through it.  The
        # association path must improve through the P2 context, expected-x
        # state and the vertical consumer; the direct visual DFL objective
        # still trains U0/query/key localization above.
        visual_hidden = initial.detach() + self.visual_context(visual_context)
        visual_hidden = visual_hidden + self.visual_x_projection(
            self._position_basis(visual_x_fraction)
        )
        visual_hidden = self.vertical_encoder(
            visual_hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        visual_hidden = visual_hidden + self.visual_ffn(
            self.visual_norm(visual_hidden)
        )

        normalized_proposal_rows = self.proposal_row_norm(proposal_rows)
        row_y = row_fraction.view(1, 1, rows)
        proposal_visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & torch.isfinite(proposal_x)
        )
        proposal_geometry = torch.stack(
            (
                proposal_x / float(max(self.input_w - 1, 1)),
                proposal_range[..., 0].unsqueeze(-1).expand(-1, -1, rows),
                proposal_range[..., 1].unsqueeze(-1).expand(-1, -1, rows),
                proposal_visible.float(),
            ),
            dim=-1,
        )
        proposal_keys = self.proposal_content_key(normalized_proposal_rows)
        proposal_keys = proposal_keys + self.proposal_geometry_key(
            proposal_geometry
        )
        proposal_values = self.proposal_value(normalized_proposal_rows)
        row_logits = torch.einsum(
            "bsrh,bnrh->bsnr",
            self.proposal_query(visual_hidden),
            proposal_keys,
        ) / math.sqrt(float(self.hidden_dim))
        visible_weight = proposal_visible[:, None].to(row_logits.dtype)
        proposal_logits = (row_logits * visible_weight).sum(dim=-1)
        proposal_logits = proposal_logits / visible_weight.sum(
            dim=-1
        ).clamp_min(1.0)
        proposal_logits = proposal_logits.masked_fill(
            ~candidate_valid[:, None, :],
            -1.0e4,
        )
        private_logits = self.private_dustbin(
            visual_hidden.mean(dim=2)
        ).squeeze(-1)
        proposal_attention = (
            structured_unique_route_marginals_with_private_dustbins(
                proposal_logits,
                candidate_valid,
                private_logits,
                temperature=self.proposal_attention_temperature,
                iterations=self.proposal_attention_sinkhorn_iterations,
            )
        )
        real_attention = proposal_attention[..., :candidates]
        proposal_context = torch.einsum(
            "bsn,bnrh->bsrh",
            real_attention,
            proposal_values,
        )
        associated_hidden = visual_hidden + self.proposal_context(
            proposal_context
        )
        associated_hidden = associated_hidden + self.association_ffn(
            self.association_norm(associated_hidden)
        )

        writer_rows = (
            (row_fraction.view(1, 1, rows) >= anchor_range[..., :1])
            & (row_fraction.view(1, 1, rows) <= anchor_range[..., 1:])
            & torch.isfinite(anchor_x)
        )
        writer_valid = (
            source_active
            & geometry_valid
            & (writer_rows.sum(dim=-1) >= self.min_valid_rows)
        )
        visual_entropy = -(
            visual_probability
            * visual_probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        proposal_entropy = -(
            proposal_attention
            * proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "selection_slot_v14_anchor_x_rows": anchor_x,
            "selection_slot_v14_anchor_range_norm": anchor_range,
            "selection_slot_v14_geometry_valid": geometry_valid,
            "selection_slot_v14_source_active": source_active,
            "selection_slot_v14_writer_valid": writer_valid,
            "selection_slot_v14_visual_logits": visual_logits,
            "selection_slot_v14_visual_attention": visual_probability,
            "selection_slot_v14_visual_x_rows": (
                visual_x_fraction * float(max(self.input_w - 1, 1))
            ),
            "selection_slot_v14_visual_state": visual_hidden,
            "selection_slot_v14_proposal_logits": proposal_logits,
            "selection_slot_v14_private_dustbin_logits": private_logits,
            "selection_slot_v14_proposal_attention": proposal_attention,
            "selection_slot_v14_real_proposal_attention": real_attention,
            "selection_slot_v14_associated_state": associated_hidden,
            "selection_slot_v14_visual_entropy": visual_entropy.mean(
                dim=(1, 2)
            ),
            "selection_slot_v14_proposal_entropy": proposal_entropy.mean(
                dim=-1
            ),
            "selection_slot_v14_feature_policy_id": visual_logits.new_full(
                (batch,),
                float(sorted(self.FEATURE_POLICIES).index(policy)),
            ),
        }


class FourSlotV14ParityAnchoredGeometry(nn.Module):
    """Stage-B geometry owned by frozen, image-causal V14 lane states.

    The exact deployed V7 curve/range are immutable anchors.  Stage-A visual
    and proposal transport tensors are memory only; a fresh row consumer emits
    symmetric residual distributions whose uniform initialization is exactly
    zero.  This module cannot alter activity, score, public route identity, or
    the proposal detector.
    """

    def __init__(
        self,
        proposal_dim: int,
        *,
        input_w: int,
        visual_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        delta_offsets_px: tuple[float, ...],
        range_offsets_norm: tuple[float, ...],
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("V14 Stage-B heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("V14 Stage-B needs vertical interaction")
        for label, values in (
            ("x", delta_offsets_px),
            ("range", range_offsets_norm),
        ):
            if not values or len(values) % 2 != 1:
                raise ValueError(f"V14 Stage-B {label} offsets must be odd")
            if float(values[len(values) // 2]) != 0.0:
                raise ValueError(f"V14 Stage-B {label} offsets need zero center")
            if any(
                abs(float(values[index]) + float(values[-1 - index])) > 1.0e-8
                for index in range(len(values) // 2)
            ):
                raise ValueError(f"V14 Stage-B {label} offsets must be symmetric")
        self.input_w = int(input_w)
        self.visual_dim = int(visual_dim)
        self.proposal_dim = int(proposal_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(delta_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "range_offsets_norm",
            torch.tensor(range_offsets_norm, dtype=torch.float32),
        )

        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.visual_projection = nn.Linear(self.visual_dim, self.hidden_dim)
        self.proposal_norm = nn.LayerNorm(self.proposal_dim)
        self.proposal_value = nn.Linear(
            self.proposal_dim, self.hidden_dim, bias=False
        )
        self.proposal_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        # anchor x, associated coarse x, displacement, real proposal mass,
        # and proposal-visible mass at this row.
        self.geometry_projection = nn.Linear(5, self.hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = nn.Linear(
            self.hidden_dim, len(delta_offsets_px), bias=False
        )
        self.range_head = nn.Linear(
            self.hidden_dim, 2 * len(range_offsets_norm), bias=False
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.range_head.weight)

    @staticmethod
    def _symmetric_expectation(
        logits: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        probability = torch.softmax(logits.float(), dim=-1)
        centered = probability - 1.0 / float(probability.shape[-1])
        return torch.einsum(
            "...k,k->...", centered, offsets.to(probability)
        )

    def forward(
        self,
        *,
        visual_state: torch.Tensor,
        proposal_attention: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        anchor_x_rows: torch.Tensor,
        anchor_range_norm: torch.Tensor,
        anchor_geometry_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual = visual_state.detach().float()
        attention = proposal_attention.detach().float()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        geometry_valid = anchor_geometry_valid.detach().bool()
        batch, slots, rows, _channels = visual.shape
        candidates = int(proposal_x.shape[1])
        if tuple(attention.shape) != (batch, slots, candidates + slots):
            raise ValueError("V14 Stage-B attention shape mismatch")
        if tuple(anchor_x.shape) != (batch, slots, rows):
            raise ValueError("V14 Stage-B anchor shape mismatch")

        real_attention = attention[..., :candidates]
        real_mass = real_attention.sum(dim=-1).clamp_min(1.0e-6)
        normalized_attention = real_attention / real_mass.unsqueeze(-1)
        proposal_values = self.proposal_value(
            self.proposal_norm(proposal_rows)
        )
        proposal_context = torch.einsum(
            "bsn,bnrh->bsrh", normalized_attention, proposal_values
        )
        coarse_x = torch.einsum(
            "bsn,bnr->bsr", normalized_attention, proposal_x
        )
        row_fraction = fixed_row_fractions(
            rows, device=visual.device, dtype=torch.float32
        )
        proposal_visible = (
            (row_fraction.view(1, 1, rows) >= proposal_range[..., :1])
            & (row_fraction.view(1, 1, rows) <= proposal_range[..., 1:])
            & torch.isfinite(proposal_x)
        )
        visible_mass = torch.einsum(
            "bsn,bnr->bsr", normalized_attention, proposal_visible.float()
        )
        scale = float(max(self.input_w - 1, 1))
        geometry = torch.stack(
            (
                anchor_x / scale,
                coarse_x / scale,
                (coarse_x - anchor_x) / scale,
                real_mass.unsqueeze(-1).expand(-1, -1, rows),
                visible_mass,
            ),
            dim=-1,
        )
        hidden = self.visual_projection(self.visual_norm(visual))
        hidden = hidden + self.proposal_context(proposal_context)
        hidden = hidden + self.geometry_projection(geometry)
        hidden = hidden + self.fusion_ffn(self.fusion_norm(hidden))
        hidden = self.vertical_encoder(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        normalized = self.output_norm(hidden)
        delta_logits = self.delta_head(normalized)
        delta = self._symmetric_expectation(
            delta_logits, self.delta_offsets_px
        )
        final_x = (anchor_x + delta).clamp(0.0, scale)
        pooled = normalized.mean(dim=2)
        range_logits = self.range_head(pooled).view(
            batch, slots, 2, -1
        )
        range_delta = self._symmetric_expectation(
            range_logits, self.range_offsets_norm
        )
        final_range = sort_range_norm(
            (anchor_range + range_delta).clamp(0.0, 1.0)
        )
        final_x = torch.where(
            geometry_valid.unsqueeze(-1), final_x, anchor_x
        )
        final_range = torch.where(
            geometry_valid.unsqueeze(-1), final_range, anchor_range
        )
        return {
            "selection_slot_v14_stage_b_anchor_x_rows": anchor_x,
            "selection_slot_v14_stage_b_anchor_range_norm": anchor_range,
            "selection_slot_v14_stage_b_geometry_valid": geometry_valid,
            "selection_slot_v14_stage_b_hidden": hidden,
            "selection_slot_v14_stage_b_real_attention_mass": real_mass,
            "selection_slot_v14_stage_b_coarse_x_rows": coarse_x,
            "selection_slot_v14_stage_b_delta_x_rows": delta,
            "selection_slot_pred_x_rows": final_x,
            "selection_slot_range_norm": final_range,
            "selection_slot_input_reference_x_rows": anchor_x,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta_logits": range_logits,
            "selection_slot_range_delta_offsets_norm": self.range_offsets_norm,
        }


class FourSlotBottomAwareRelationalGeometry(nn.Module):
    """V15 visual-first lane state with soft bottom-aware proposal context.

    Proposal geometry defines continuous graph edges, never hard clusters.
    Every proposal remains an independent memory node and final geometry is a
    residual owned by the persistent slot-row state, not a proposal prototype
    or coordinate barycenter.  Exact V7 geometry/activity/score remain the
    initialization and deployment-control anchors.
    """

    FEATURE_POLICIES = {
        "correct",
        "zero_content",
        "position_only",
        "zero_content_zero_position",
        "x_reversed",
        "row_reversed",
    }
    GRAPH_POLICIES = {"correct", "identity", "geometry_shuffled"}
    CONTEXT_POLICIES = {"correct", "none"}

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        visual_vertical_layers: int,
        fusion_vertical_layers: int,
        slot_interaction_layers: int,
        dropout: float,
        min_valid_rows: int,
        visual_prior_strength: float,
        visual_prior_sigma: float,
        graph_prior_strength: float,
        graph_self_bias: float,
        slot_curve_prior_strength: float,
        delta_offsets_px: tuple[float, ...],
        range_offsets_norm: tuple[float, ...],
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("V15 attention heads must divide hidden_dim")
        if int(num_slots) < 1:
            raise ValueError("V15 needs at least one slot")
        if int(visual_vertical_layers) < 1 or int(fusion_vertical_layers) < 1:
            raise ValueError("V15 needs visual and fusion vertical interaction")
        if int(slot_interaction_layers) < 1:
            raise ValueError("V15 needs cross-slot interaction")
        if int(min_valid_rows) < 1:
            raise ValueError("V15 min_valid_rows must be positive")
        if float(visual_prior_strength) < 0.0:
            raise ValueError("V15 visual prior strength must be non-negative")
        if float(visual_prior_sigma) <= 0.0:
            raise ValueError("V15 visual prior sigma must be positive")
        if float(graph_prior_strength) < 0.0:
            raise ValueError("V15 graph prior strength must be non-negative")
        if float(slot_curve_prior_strength) < 0.0:
            raise ValueError("V15 slot curve prior strength must be non-negative")
        for label, values in (
            ("x", delta_offsets_px),
            ("range", range_offsets_norm),
        ):
            if not values or len(values) % 2 != 1:
                raise ValueError(f"V15 {label} offsets must be nonempty and odd")
            if float(values[len(values) // 2]) != 0.0:
                raise ValueError(f"V15 {label} offsets require a zero center")
            if any(
                abs(float(values[index]) + float(values[-1 - index])) > 1.0e-8
                for index in range(len(values) // 2)
            ):
                raise ValueError(f"V15 {label} offsets must be symmetric")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.min_valid_rows = int(min_valid_rows)
        self.visual_prior_strength = float(visual_prior_strength)
        self.visual_prior_sigma = float(visual_prior_sigma)
        self.graph_prior_strength = float(graph_prior_strength)
        self.graph_self_bias = float(graph_self_bias)
        self.slot_curve_prior_strength = float(slot_curve_prior_strength)
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(delta_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "range_offsets_norm",
            torch.tensor(range_offsets_norm, dtype=torch.float32),
        )

        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.anchor_geometry_projection = nn.Linear(3, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)

        # P2 content and position remain causally separable.  Position is a key
        # and an explicit expected-x state, never part of the visual values.
        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.x_position_key = nn.Linear(4, self.hidden_dim, bias=False)
        self.visual_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.visual_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.visual_x_projection = nn.Linear(4, self.hidden_dim, bias=False)
        visual_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_vertical = nn.TransformerEncoder(
            visual_layer,
            num_layers=int(visual_vertical_layers),
            enable_nested_tensor=False,
        )
        self.visual_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        # Proposal nodes retain row content and their own coordinates.  The
        # edge MLP receives continuous perspective-aware pair features.  Its
        # final zero initialization starts from the fixed monotonic prior while
        # allowing geometry loss to learn deviations from that prior.
        self.proposal_row_norm = nn.LayerNorm(self.dim)
        self.proposal_content = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_geometry = nn.Linear(5, self.hidden_dim, bias=False)
        self.graph_edge_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, 1, bias=False),
        )
        nn.init.zeros_(self.graph_edge_mlp[-1].weight)
        self.graph_message = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.graph_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )

        self.slot_memory_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.proposal_memory_key = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.proposal_memory_value = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.proposal_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        # anchor x, visual x, contextual proposal x, proposal-anchor delta,
        # visual-proposal delta, and proposal visible mass.
        self.context_geometry_projection = nn.Linear(
            6, self.hidden_dim, bias=False
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        slot_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_interaction = nn.TransformerEncoder(
            slot_layer,
            num_layers=int(slot_interaction_layers),
            enable_nested_tensor=False,
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_vertical = nn.TransformerEncoder(
            fusion_layer,
            num_layers=int(fusion_vertical_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = nn.Linear(
            self.hidden_dim, len(delta_offsets_px), bias=False
        )
        self.range_head = nn.Linear(
            self.hidden_dim, 2 * len(range_offsets_norm), bias=False
        )
        nn.init.normal_(self.slot_tokens.weight, std=0.02)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.range_head.weight)

    @staticmethod
    def _position_basis(values: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                values,
                values.square(),
                torch.sin(math.pi * values),
                torch.cos(math.pi * values),
            ),
            dim=-1,
        )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = mask.sum(dim=-1)
        mean = (values * mask.to(values.dtype)).sum(dim=-1)
        mean = mean / count.clamp_min(1).to(values.dtype)
        return mean, count

    @staticmethod
    def _masked_quantile(
        values: torch.Tensor,
        mask: torch.Tensor,
        quantile: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = mask.sum(dim=-1)
        filled = values.masked_fill(~mask, float("inf"))
        sorted_values = filled.sort(dim=-1).values
        index = (
            torch.ceil(count.to(values.dtype) * float(quantile)).long() - 1
        ).clamp(min=0, max=max(int(values.shape[-1]) - 1, 0))
        selected = sorted_values.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        selected = torch.where(count > 0, selected, torch.ones_like(selected))
        return selected, count

    @staticmethod
    def _masked_weighted_quantile(
        values: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        filled = values.masked_fill(~mask, float("inf"))
        order = filled.argsort(dim=-1)
        sorted_values = filled.gather(-1, order)
        expanded_weights = weights.expand_as(values)
        sorted_weights = expanded_weights.gather(-1, order)
        sorted_weights = sorted_weights * mask.gather(-1, order).to(values.dtype)
        total = sorted_weights.sum(dim=-1, keepdim=True)
        cumulative = sorted_weights.cumsum(dim=-1)
        cutoff = total * float(quantile)
        reached = cumulative >= cutoff
        index = reached.to(torch.int64).argmax(dim=-1)
        selected = sorted_values.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        return torch.where(
            total.squeeze(-1) > 0,
            selected,
            torch.ones_like(selected),
        )

    @staticmethod
    def _symmetric_expectation(
        logits: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        probability = torch.softmax(logits.float(), dim=-1)
        centered = probability - 1.0 / float(probability.shape[-1])
        return torch.einsum(
            "...k,k->...", centered, offsets.to(probability)
        )

    def _proposal_graph(
        self,
        *,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        proposal_visible: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_fraction: torch.Tensor,
        graph_policy: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return edge features, pair validity and row-stochastic weights."""

        batch, candidates, rows = proposal_x.shape
        scale = float(max(self.input_w - 1, 1))
        common = proposal_visible[:, :, None, :] & proposal_visible[:, None, :, :]
        x_gap = (
            proposal_x[:, :, None, :] - proposal_x[:, None, :, :]
        ).abs() / scale
        global_mean, common_count = self._masked_mean(x_gap, common)
        global_q90, _ = self._masked_quantile(x_gap, common, 0.90)
        row_y = row_fraction.view(1, 1, 1, rows)
        perspective_weight = 0.10 + 0.90 * row_y.pow(3.0)
        weighted_total = (perspective_weight * common).sum(dim=-1)
        weighted_mean = (
            x_gap * perspective_weight * common.to(x_gap.dtype)
        ).sum(dim=-1) / weighted_total.clamp_min(1.0)
        weighted_q90 = self._masked_weighted_quantile(
            x_gap, common, perspective_weight, 0.90
        )

        lower_mask = common & (row_y >= 0.58)
        upper_mask = common & (row_y <= 0.35)
        lower_mean, lower_count = self._masked_mean(x_gap, lower_mask)
        lower_q90, _ = self._masked_quantile(x_gap, lower_mask, 0.90)
        upper_mean, upper_count = self._masked_mean(x_gap, upper_mask)
        lower_available = lower_count >= 3
        # Missing lower evidence is not a rejection.  The hard perspective
        # audit over-fragmented partial lanes; V15 smoothly falls back to the
        # full shared curve whenever the lower road is not observed.
        lower_mean = torch.where(lower_available, lower_mean, weighted_mean)
        lower_q90 = torch.where(lower_available, lower_q90, weighted_q90)
        upper_mean = torch.where(upper_count > 0, upper_mean, global_mean)

        row_rank = row_y.expand_as(x_gap).masked_fill(~common, -1.0)
        bottom_index = row_rank.topk(k=min(3, rows), dim=-1).indices
        bottom_values = x_gap.gather(-1, bottom_index)
        bottom_endpoint = bottom_values.median(dim=-1).values
        bottom_endpoint = torch.where(
            common_count >= min(3, rows), bottom_endpoint, global_q90
        )
        lower_minus_upper = (lower_mean - upper_mean).clamp_min(0.0)

        range_start = (
            proposal_range[:, :, None, 0] - proposal_range[:, None, :, 0]
        ).abs()
        range_end = (
            proposal_range[:, :, None, 1] - proposal_range[:, None, :, 1]
        ).abs()
        range_length = (
            (proposal_range[..., 1] - proposal_range[..., 0])[:, :, None]
            - (proposal_range[..., 1] - proposal_range[..., 0])[:, None, :]
        ).abs()
        valid_rows = proposal_visible.sum(dim=-1)
        overlap = common_count.to(x_gap.dtype) / torch.minimum(
            valid_rows[:, :, None], valid_rows[:, None, :]
        ).clamp_min(1).to(x_gap.dtype)
        edge_features = torch.stack(
            (
                overlap,
                global_mean,
                global_q90,
                weighted_mean,
                weighted_q90,
                lower_mean,
                lower_q90,
                bottom_endpoint,
                lower_minus_upper,
                range_start,
                range_end,
                range_length,
            ),
            dim=-1,
        )
        edge_features = torch.nan_to_num(
            edge_features, nan=1.0, posinf=1.0, neginf=0.0
        )
        pair_valid = (
            candidate_valid[:, :, None]
            & candidate_valid[:, None, :]
            & (common_count >= 3)
        )
        eye = torch.eye(
            candidates, device=proposal_x.device, dtype=torch.bool
        ).view(1, candidates, candidates)
        pair_valid = pair_valid | (
            eye & candidate_valid[:, :, None] & candidate_valid[:, None, :]
        )

        reference_scale = float(self.input_w) / 1600.0
        flat_term = 0.5 * (
            global_mean / max(48.0 * reference_scale / scale, 1.0e-6)
            + global_q90 / max(96.0 * reference_scale / scale, 1.0e-6)
        )
        lower_term = 0.25 * (
            weighted_mean / max(48.0 * reference_scale / scale, 1.0e-6)
            + weighted_q90 / max(96.0 * reference_scale / scale, 1.0e-6)
            + bottom_endpoint / max(80.0 * reference_scale / scale, 1.0e-6)
            + lower_minus_upper / max(80.0 * reference_scale / scale, 1.0e-6)
        )
        fixed_prior = -(flat_term + lower_term)
        fixed_prior = fixed_prior + overlap.clamp_min(1.0e-4).log()
        fixed_prior = fixed_prior + eye.to(fixed_prior.dtype) * self.graph_self_bias
        learned_prior = self.graph_edge_mlp(edge_features).squeeze(-1)
        logits = self.graph_prior_strength * fixed_prior + learned_prior
        logits = logits.masked_fill(~pair_valid, -1.0e4)
        correct = torch.softmax(logits.float(), dim=-1)
        correct = correct * pair_valid.to(correct.dtype)
        correct = correct / correct.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        if graph_policy == "correct":
            graph = correct
        elif graph_policy == "identity":
            graph = eye.to(correct.dtype) * candidate_valid[:, :, None].to(
                correct.dtype
            )
        elif graph_policy == "geometry_shuffled":
            graph = correct.roll(shifts=1, dims=-1)
            graph = graph * candidate_valid[:, None, :].to(graph.dtype)
            graph = graph * candidate_valid[:, :, None].to(graph.dtype)
            graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        else:
            raise ValueError(f"unsupported V15 graph policy: {graph_policy}")
        return edge_features, pair_valid, graph

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        anchor_x_rows: torch.Tensor,
        anchor_range_norm: torch.Tensor,
        anchor_geometry_valid: torch.Tensor,
        anchor_active: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_value_features: torch.Tensor,
        feature_policy: str = "correct",
        graph_policy: str = "correct",
        context_policy: str = "correct",
    ) -> dict[str, torch.Tensor]:
        feature_policy = str(feature_policy).strip().lower()
        graph_policy = str(graph_policy).strip().lower()
        context_policy = str(context_policy).strip().lower()
        if feature_policy not in self.FEATURE_POLICIES:
            raise ValueError(f"unsupported V15 feature policy: {feature_policy}")
        if graph_policy not in self.GRAPH_POLICIES:
            raise ValueError(f"unsupported V15 graph policy: {graph_policy}")
        if context_policy not in self.CONTEXT_POLICIES:
            raise ValueError(f"unsupported V15 context policy: {context_policy}")
        if row_value_features.ndim != 4:
            raise ValueError("V15 P2 rows must have shape [B,R,X,C]")
        batch, rows, x_bins, channels = row_value_features.shape
        slots = self.num_slots
        candidates = int(proposal_x_rows.shape[1])
        if int(channels) != self.dim:
            raise ValueError("V15 P2 feature dimension mismatch")
        if tuple(slot_states.shape[:2]) != (batch, slots):
            raise ValueError("V15 slot state shape mismatch")
        if tuple(anchor_x_rows.shape) != (batch, slots, rows):
            raise ValueError("V15 anchor x shape mismatch")
        if tuple(anchor_range_norm.shape) != (batch, slots, 2):
            raise ValueError("V15 anchor range shape mismatch")
        if tuple(anchor_geometry_valid.shape) != (batch, slots):
            raise ValueError("V15 geometry-valid shape mismatch")
        if tuple(anchor_active.shape) != (batch, slots):
            raise ValueError("V15 source-active shape mismatch")
        if tuple(proposal_row_tokens.shape[:3]) != (batch, candidates, rows):
            raise ValueError("V15 proposal row-token shape mismatch")
        if tuple(proposal_x_rows.shape) != (batch, candidates, rows):
            raise ValueError("V15 proposal x shape mismatch")
        if tuple(proposal_range_norm.shape) != (batch, candidates, 2):
            raise ValueError("V15 proposal range shape mismatch")
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("V15 candidate validity shape mismatch")

        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        geometry_valid = anchor_geometry_valid.detach().bool()
        source_active = anchor_active.detach().bool()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_x_finite = torch.isfinite(proposal_x)
        proposal_x_safe = torch.nan_to_num(
            proposal_x,
            nan=0.0,
            posinf=float(max(self.input_w - 1, 1)),
            neginf=0.0,
        )
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        candidate_valid = candidate_valid.detach().bool()
        raw_features = row_value_features.detach().float()
        if feature_policy == "x_reversed":
            raw_features = raw_features.flip(dims=(2,))
        elif feature_policy == "row_reversed":
            raw_features = raw_features.flip(dims=(1,))
        elif feature_policy in {
            "zero_content",
            "position_only",
            "zero_content_zero_position",
        }:
            raw_features = torch.zeros_like(raw_features)

        row_fraction = fixed_row_fractions(
            rows, device=slot_states.device, dtype=torch.float32
        )
        row_position = self.row_position_projection(
            self._position_basis(row_fraction)
        ).view(1, 1, rows, self.hidden_dim)
        scale = float(max(self.input_w - 1, 1))
        anchor_geometry = torch.cat(
            (
                anchor_x.unsqueeze(-1) / scale,
                anchor_range.unsqueeze(2).expand(-1, -1, rows, -1),
            ),
            dim=-1,
        )
        initial = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2)
        initial = initial + self.slot_tokens.weight.view(
            1, slots, 1, self.hidden_dim
        )
        initial = initial + row_position
        initial = initial + self.anchor_geometry_projection(anchor_geometry)
        initial = self.initial_norm(initial)

        normalized_features = self.feature_norm(raw_features)
        content_keys = self.feature_key(normalized_features)
        content_values = self.feature_value(normalized_features)
        if feature_policy in {"position_only", "zero_content_zero_position"}:
            content_keys = torch.zeros_like(content_keys)
            content_values = torch.zeros_like(content_values)
        x_fraction = torch.linspace(
            0.0,
            1.0,
            x_bins,
            device=slot_states.device,
            dtype=torch.float32,
        )
        position_keys = self.x_position_key(
            self._position_basis(x_fraction)
        ).view(1, 1, x_bins, self.hidden_dim)
        if feature_policy == "zero_content_zero_position":
            position_keys = torch.zeros_like(position_keys)
        feature_keys = content_keys + position_keys
        visual_logits = torch.einsum(
            "bsrh,brxh->bsrx", self.visual_query(initial), feature_keys
        ) / math.sqrt(float(self.hidden_dim))
        anchor_center = (anchor_x / scale).clamp(0.0, 1.0)
        distance = x_fraction.view(1, 1, 1, -1) - anchor_center.unsqueeze(-1)
        visual_logits = visual_logits - (
            0.5
            * self.visual_prior_strength
            * distance.square()
            / (self.visual_prior_sigma**2)
        )
        visual_probability = torch.softmax(visual_logits.float(), dim=-1)
        visual_context = torch.einsum(
            "bsrx,brxh->bsrh", visual_probability, content_values.float()
        )
        visual_x_fraction = torch.einsum(
            "bsrx,x->bsr", visual_probability, x_fraction
        )
        visual_hidden = initial + self.visual_context(visual_context)
        visual_hidden = visual_hidden + self.visual_x_projection(
            self._position_basis(visual_x_fraction)
        )
        visual_hidden = self.visual_vertical(
            visual_hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        visual_hidden = visual_hidden + self.visual_ffn(
            self.visual_norm(visual_hidden)
        )

        row_y = row_fraction.view(1, 1, rows)
        proposal_visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & proposal_x_finite
        )
        proposal_geometry = torch.stack(
            (
                proposal_x_safe / scale,
                proposal_range[..., 0].unsqueeze(-1).expand(-1, -1, rows),
                proposal_range[..., 1].unsqueeze(-1).expand(-1, -1, rows),
                proposal_visible.float(),
                row_y.expand(batch, candidates, rows),
            ),
            dim=-1,
        )
        proposal_hidden = self.proposal_content(
            self.proposal_row_norm(proposal_rows)
        ) + self.proposal_geometry(proposal_geometry)
        edge_features, pair_valid, graph_attention = self._proposal_graph(
            proposal_x=proposal_x_safe,
            proposal_range=proposal_range,
            proposal_visible=proposal_visible,
            candidate_valid=candidate_valid,
            row_fraction=row_fraction,
            graph_policy=graph_policy,
        )
        graph_message = torch.einsum(
            "bnm,bmrh->bnrh", graph_attention, proposal_hidden
        )
        graph_hidden = proposal_hidden + self.graph_message(graph_message)
        graph_hidden = graph_hidden + self.graph_ffn(self.graph_norm(graph_hidden))

        proposal_keys = self.proposal_memory_key(graph_hidden)
        proposal_values = self.proposal_memory_value(graph_hidden)
        row_logits = torch.einsum(
            "bsrh,bnrh->bsnr",
            self.slot_memory_query(visual_hidden),
            proposal_keys,
        ) / math.sqrt(float(self.hidden_dim))
        perspective_weight = (
            0.10 + 0.90 * row_fraction.pow(3.0)
        ).view(1, 1, 1, rows)
        visible_weight = (
            proposal_visible[:, None].to(row_logits.dtype) * perspective_weight
        )
        content_logits = (row_logits * visible_weight).sum(dim=-1)
        content_logits = content_logits / visible_weight.sum(dim=-1).clamp_min(1.0)
        visual_x = visual_x_fraction * scale
        visual_gap = (
            visual_x[:, :, None, :] - proposal_x_safe[:, None, :, :]
        ).abs() / scale
        visual_gap_mean = (visual_gap * visible_weight).sum(dim=-1)
        visual_gap_mean = visual_gap_mean / visible_weight.sum(dim=-1).clamp_min(1.0)
        proposal_logits = (
            content_logits - self.slot_curve_prior_strength * visual_gap_mean
        )
        proposal_logits = proposal_logits.masked_fill(
            ~candidate_valid[:, None, :], -1.0e4
        )
        proposal_attention = torch.softmax(proposal_logits.float(), dim=-1)
        proposal_attention = proposal_attention * candidate_valid[:, None].to(
            proposal_attention.dtype
        )
        proposal_attention = proposal_attention / proposal_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        row_attention = (
            proposal_attention.unsqueeze(-1)
            * proposal_visible[:, None].to(proposal_attention.dtype)
        )
        visible_mass = row_attention.sum(dim=2)
        normalized_row_attention = row_attention / visible_mass.unsqueeze(
            2
        ).clamp_min(1.0e-12)
        normalized_row_attention = torch.where(
            (visible_mass > 0).unsqueeze(2),
            normalized_row_attention,
            proposal_attention.unsqueeze(-1).expand(-1, -1, -1, rows),
        )
        proposal_context = torch.einsum(
            "bsnr,bnrh->bsrh", normalized_row_attention, proposal_values
        )
        contextual_x = torch.einsum(
            "bsnr,bnr->bsr",
            normalized_row_attention,
            proposal_x_safe,
        )
        context_geometry = torch.stack(
            (
                anchor_x / scale,
                visual_x / scale,
                contextual_x / scale,
                (contextual_x - anchor_x) / scale,
                (contextual_x - visual_x) / scale,
                visible_mass,
            ),
            dim=-1,
        )
        if context_policy == "none":
            # This is an endpoint-only causal replay.  Anchor and visual
            # coordinates remain available to the slot geometry trunk, while
            # every proposal-derived value is removed from the public
            # geometry computation.  Proposal logits are still returned as
            # diagnostics, so this intervention does not silently redefine
            # the proposal population or its geometry.
            proposal_context = torch.zeros_like(proposal_context)
            contextual_x = torch.zeros_like(contextual_x)
            context_geometry = torch.cat(
                (
                    (anchor_x / scale).unsqueeze(-1),
                    (visual_x / scale).unsqueeze(-1),
                    torch.zeros_like(context_geometry[..., 2:]),
                ),
                dim=-1,
            )
        hidden = visual_hidden + self.proposal_context(proposal_context)
        hidden = hidden + self.context_geometry_projection(context_geometry)
        hidden = hidden + self.fusion_ffn(self.fusion_norm(hidden))
        hidden = self.slot_interaction(
            hidden.permute(0, 2, 1, 3).reshape(
                batch * rows, slots, self.hidden_dim
            )
        ).reshape(batch, rows, slots, self.hidden_dim).permute(0, 2, 1, 3)
        hidden = self.fusion_vertical(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        normalized = self.output_norm(hidden)
        delta_logits = self.delta_head(normalized)
        delta = self._symmetric_expectation(delta_logits, self.delta_offsets_px)
        final_x = (anchor_x + delta).clamp(0.0, scale)
        pooled = normalized.mean(dim=2)
        range_logits = self.range_head(pooled).view(batch, slots, 2, -1)
        range_delta = self._symmetric_expectation(
            range_logits, self.range_offsets_norm
        )
        final_range = sort_range_norm(
            (anchor_range + range_delta).clamp(0.0, 1.0)
        )
        final_x = torch.where(geometry_valid.unsqueeze(-1), final_x, anchor_x)
        final_range = torch.where(
            geometry_valid.unsqueeze(-1), final_range, anchor_range
        )

        writer_rows = (
            (row_fraction.view(1, 1, rows) >= anchor_range[..., :1])
            & (row_fraction.view(1, 1, rows) <= anchor_range[..., 1:])
            & torch.isfinite(anchor_x)
        )
        writer_valid = (
            source_active
            & geometry_valid
            & (writer_rows.sum(dim=-1) >= self.min_valid_rows)
        )
        visual_entropy = -(
            visual_probability * visual_probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        proposal_entropy = -(
            proposal_attention * proposal_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        graph_entropy = -(
            graph_attention * graph_attention.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "selection_slot_v15_anchor_x_rows": anchor_x,
            "selection_slot_v15_anchor_range_norm": anchor_range,
            "selection_slot_v15_geometry_valid": geometry_valid,
            "selection_slot_v15_source_active": source_active,
            "selection_slot_v15_writer_valid": writer_valid,
            "selection_slot_v15_visual_logits": visual_logits,
            "selection_slot_v15_visual_attention": visual_probability,
            "selection_slot_v15_visual_x_rows": visual_x,
            "selection_slot_v15_visual_state": visual_hidden,
            "selection_slot_v15_graph_edge_features": edge_features.detach(),
            "selection_slot_v15_graph_pair_valid": pair_valid,
            "selection_slot_v15_graph_attention": graph_attention,
            "selection_slot_v15_graph_state": graph_hidden,
            "selection_slot_v15_proposal_logits": proposal_logits,
            "selection_slot_v15_proposal_attention": proposal_attention,
            "selection_slot_v15_context_x_rows": contextual_x,
            "selection_slot_v15_hidden": hidden,
            "selection_slot_v15_delta_x_rows": delta,
            "selection_slot_v15_visual_entropy": visual_entropy.mean(dim=(1, 2)),
            "selection_slot_v15_proposal_entropy": proposal_entropy.mean(dim=-1),
            "selection_slot_v15_graph_entropy": graph_entropy.mean(dim=-1),
            "selection_slot_v15_feature_policy_id": visual_logits.new_full(
                (batch,), float(sorted(self.FEATURE_POLICIES).index(feature_policy))
            ),
            "selection_slot_v15_graph_policy_id": visual_logits.new_full(
                (batch,), float(sorted(self.GRAPH_POLICIES).index(graph_policy))
            ),
            "selection_slot_v15_context_policy_id": visual_logits.new_full(
                (batch,),
                float(sorted(self.CONTEXT_POLICIES).index(context_policy)),
            ),
            "selection_slot_pred_x_rows": final_x,
            "selection_slot_range_norm": final_range,
            "selection_slot_input_reference_x_rows": anchor_x,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta_logits": range_logits,
            "selection_slot_range_delta_offsets_norm": self.range_offsets_norm,
        }


class FourSlotVisualPrecisionGeometry(nn.Module):
    """Own final lane geometry without selecting a proposal identity.

    V12 proved that full-width P2 evidence localizes the correct lane on
    unseen images, but mapping that visual state back to one of 32 proposal
    IDs erased almost all of the visual advantage.  V13 therefore keeps the
    proposal population as row-wise *feature context* only.  It retrieves
    proposal row tokens independently at every row, reads precise local P2
    evidence around both the visual and V7 centers, reasons across the four
    slots and vertically, and emits full-width x/range corrections itself.

    The V7 curve is a parity anchor, not a movement bound.  Zero-initialized
    per-row gates and symmetric full-width offsets make initialization exactly
    V7 while retaining enough support to move a slot to another global lane.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        visual_dim: int,
        num_slots: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        local_offsets_px: tuple[float, ...],
        delta_offsets_px: tuple[float, ...],
        range_delta_offsets_norm: tuple[float, ...],
        proposal_distance_scale: float = 8.0,
        invisible_row_logit_bias: float = -2.0,
        gradient_only_candidate_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("V13 heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("V13 needs at least one vertical layer")
        if not local_offsets_px:
            raise ValueError("V13 local P2 offsets cannot be empty")
        if not delta_offsets_px or not range_delta_offsets_norm:
            raise ValueError("V13 geometry offset supports cannot be empty")
        for label, values in (
            ("x", delta_offsets_px),
            ("range", range_delta_offsets_norm),
        ):
            if len(values) % 2 != 1 or float(values[len(values) // 2]) != 0.0:
                raise ValueError(f"V13 {label} offsets must have a zero center")
            if any(
                abs(float(values[index]) + float(values[-1 - index]))
                > 1.0e-8
                for index in range(len(values) // 2)
            ):
                raise ValueError(f"V13 {label} offsets must be symmetric")
        if float(gradient_only_candidate_scale) < 0.0:
            raise ValueError("V13 gradient-only scale must be non-negative")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.visual_dim = int(visual_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.proposal_distance_scale = float(proposal_distance_scale)
        self.invisible_row_logit_bias = float(invisible_row_logit_bias)
        self.gradient_only_candidate_scale = float(
            gradient_only_candidate_scale
        )
        self.register_buffer(
            "local_offsets_px",
            torch.tensor(local_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(delta_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "range_delta_offsets_norm",
            torch.tensor(range_delta_offsets_norm, dtype=torch.float32),
        )

        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.visual_projection = nn.Linear(self.visual_dim, self.hidden_dim)
        self.geometry_projection = nn.Linear(5, self.hidden_dim, bias=False)

        # Row-wise proposal memory: no global [S,N] identity and no coordinate
        # averaging in the final output.  A proposal may provide a useful row
        # token even when no single proposal is the correct whole-lane member.
        self.proposal_norm = nn.LayerNorm(self.dim)
        self.proposal_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.proposal_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.proposal_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )

        # Fine P2 evidence is sampled around two independent centers: the V12
        # visual curve and the exact V7 anchor.  Relative positions influence
        # keys, never values, preserving correct/wrong/zero-image causality.
        sample_count = 2 * len(local_offsets_px)
        self.local_feature_norm = nn.LayerNorm(self.dim)
        self.local_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.local_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.local_query = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.local_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.local_position = nn.Parameter(
            torch.empty(sample_count, self.hidden_dim)
        )
        nn.init.normal_(self.local_position, std=0.02)

        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Linear(int(ff_dim), self.hidden_dim),
        )
        self.cross_slot_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_slot_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical_encoder = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

        # Every output is row/slot conditioned.  The global scalar gate that
        # invalidated V8 is gone.  Symmetric uniform offset probabilities have
        # exactly zero expectation at initialization.
        self.candidate_mix = nn.Linear(self.hidden_dim, 1)
        self.candidate_gate = nn.Linear(self.hidden_dim, 1)
        self.delta_head = nn.Linear(
            self.hidden_dim, len(delta_offsets_px), bias=False
        )
        self.range_delta_head = nn.Linear(
            self.hidden_dim,
            2 * len(range_delta_offsets_norm),
            bias=False,
        )
        # Candidate mixing is hidden behind the zero candidate gate, so its
        # ordinary random weight cannot change initialization.  Keeping a
        # live derivative here lets the gradient-only candidate surrogate
        # reach the local-P2 and vertical trunks on the very first step.
        nn.init.zeros_(self.candidate_mix.bias)
        nn.init.zeros_(self.candidate_gate.weight)
        nn.init.zeros_(self.candidate_gate.bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.range_delta_head.weight)

    def _sample_local_evidence(
        self,
        features: torch.Tensor,
        visual_x: torch.Tensor,
        anchor_x: torch.Tensor,
    ) -> torch.Tensor:
        batch, rows, x_bins, channels = features.shape
        slots = int(visual_x.shape[1])
        offsets = self.local_offsets_px.to(
            device=features.device, dtype=torch.float32
        )
        centers = torch.stack((visual_x, anchor_x), dim=-1)
        sample_x = centers.unsqueeze(-1) + offsets.view(1, 1, 1, 1, -1)
        sample_x = sample_x.clamp(0.0, float(max(self.input_w - 1, 1)))
        sample_x = sample_x.flatten(-2)
        feature_x = sample_x * float(max(x_bins - 1, 0)) / float(
            max(self.input_w - 1, 1)
        )
        left = feature_x.floor().long()
        right = (left + 1).clamp(max=max(x_bins - 1, 0))
        alpha = feature_x - left.float()
        samples = int(sample_x.shape[-1])

        flat = features.reshape(batch * rows, x_bins, channels)
        left = left.permute(0, 2, 1, 3).reshape(
            batch * rows, slots * samples
        )
        right = right.permute(0, 2, 1, 3).reshape(
            batch * rows, slots * samples
        )
        alpha = alpha.permute(0, 2, 1, 3).reshape(
            batch * rows, slots * samples, 1
        )
        row_index = fixed_indices(
            batch * rows, device=features.device, dtype=torch.long
        ).view(-1, 1)
        left_value = flat[row_index, left]
        right_value = flat[row_index, right]
        sampled = torch.lerp(left_value, right_value, alpha.to(features.dtype))
        return sampled.view(
            batch, rows, slots, samples, channels
        ).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _symmetric_expectation(
        probability: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """Compute an odd symmetric expectation with exact zero at uniform.

        Pairwise probability differences avoid the tiny FP32 cancellation
        residue produced by a long dot product of symmetric offsets.  This is
        mathematically identical to the ordinary expectation for the
        constructor-validated supports.
        """

        midpoint = int(probability.shape[-1]) // 2
        positive_probability = probability[..., midpoint + 1 :]
        negative_probability = probability[..., :midpoint].flip(-1)
        positive_offsets = offsets[midpoint + 1 :].to(probability)
        return torch.einsum(
            "...k,k->...",
            positive_probability - negative_probability,
            positive_offsets,
        )

    def forward(
        self,
        *,
        visual_state: torch.Tensor,
        visual_x_rows: torch.Tensor,
        anchor_x_rows: torch.Tensor,
        anchor_range_norm: torch.Tensor,
        anchor_geometry_valid: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, slots, rows, _visual_channels = visual_state.shape
        if slots != self.num_slots:
            raise ValueError("V13 visual slot count mismatch")
        if tuple(visual_x_rows.shape) != (batch, slots, rows):
            raise ValueError("V13 visual x shape mismatch")
        if tuple(anchor_x_rows.shape) != (batch, slots, rows):
            raise ValueError("V13 anchor x shape mismatch")
        candidates = int(proposal_x_rows.shape[1])
        if tuple(candidate_valid.shape) != (batch, candidates):
            raise ValueError("V13 candidate-valid shape mismatch")
        if tuple(row_value_features.shape[:2]) != (batch, rows):
            raise ValueError("V13 P2 row geometry mismatch")

        visual_state = visual_state.detach().float()
        visual_x = visual_x_rows.detach().float()
        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        anchor_valid = anchor_geometry_valid.detach().bool()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_x_finite = torch.isfinite(proposal_x)
        safe_proposal_x = torch.where(
            proposal_x_finite, proposal_x, torch.zeros_like(proposal_x)
        )
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        features = row_value_features.detach().float()

        hidden = self.visual_projection(self.visual_norm(visual_state))
        proposal_key = self.proposal_key(self.proposal_norm(proposal_rows))
        proposal_value = self.proposal_value(
            self.proposal_norm(proposal_rows)
        )
        proposal_logits = torch.einsum(
            "bsrh,bnrh->bsrn",
            self.proposal_query(hidden),
            proposal_key,
        ) / math.sqrt(float(self.hidden_dim))
        row_fraction = fixed_row_fractions(
            rows, device=hidden.device, dtype=torch.float32
        ).view(1, 1, rows)
        proposal_visible = (
            (row_fraction >= proposal_range[..., :1])
            & (row_fraction <= proposal_range[..., 1:])
            & proposal_x_finite
        )
        valid = candidate_valid[:, :, None].bool() & proposal_x_finite
        distance = (
            safe_proposal_x[:, None] - visual_x.unsqueeze(2)
        ).abs().permute(0, 1, 3, 2) / float(max(self.input_w - 1, 1))
        proposal_logits = proposal_logits - self.proposal_distance_scale * distance
        proposal_logits = proposal_logits + torch.where(
            proposal_visible.permute(0, 2, 1)[:, None],
            torch.zeros_like(proposal_logits),
            proposal_logits.new_full((), self.invisible_row_logit_bias),
        )
        proposal_logits = proposal_logits.masked_fill(
            ~valid.permute(0, 2, 1)[:, None], -1.0e4
        )
        proposal_attention = torch.softmax(proposal_logits.float(), dim=-1)
        proposal_context = torch.einsum(
            "bsrn,bnrh->bsrh", proposal_attention, proposal_value
        )
        proposal_expected_x = torch.einsum(
            "bsrn,bnr->bsr", proposal_attention, safe_proposal_x
        )
        hidden = hidden + self.proposal_context(proposal_context)
        width = float(max(self.input_w - 1, 1))
        geometry = torch.stack(
            (
                anchor_x / width,
                visual_x / width,
                proposal_expected_x / width,
                (visual_x - anchor_x) / width,
                (proposal_expected_x - visual_x) / width,
            ),
            dim=-1,
        )
        hidden = hidden + self.geometry_projection(geometry)

        local = self._sample_local_evidence(
            features, visual_x, anchor_x
        )
        local = self.local_feature_norm(local)
        local_key = self.local_key(local) + self.local_position.view(
            1, 1, 1, -1, self.hidden_dim
        )
        local_value = self.local_value(local)
        local_logits = torch.einsum(
            "bsrh,bsrkh->bsrk",
            self.local_query(hidden),
            local_key,
        ) / math.sqrt(float(self.hidden_dim))
        local_attention = torch.softmax(local_logits.float(), dim=-1)
        local_context = torch.einsum(
            "bsrk,bsrkh->bsrh", local_attention, local_value
        )
        hidden = hidden + self.local_context(local_context)
        hidden = hidden + self.fusion_ffn(self.fusion_norm(hidden))

        by_row = hidden.permute(0, 2, 1, 3).reshape(
            batch * rows, slots, self.hidden_dim
        )
        normalized = self.cross_slot_norm(by_row)
        cross_slot, _weight = self.cross_slot_attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = (by_row + cross_slot).reshape(
            batch, rows, slots, self.hidden_dim
        ).permute(0, 2, 1, 3)
        hidden = self.vertical_encoder(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        normalized_hidden = self.output_norm(hidden)

        candidate_mix = torch.sigmoid(
            self.candidate_mix(normalized_hidden).squeeze(-1)
        )
        candidate_x = candidate_mix * proposal_expected_x + (
            1.0 - candidate_mix
        ) * visual_x
        candidate_gate = torch.tanh(
            self.candidate_gate(normalized_hidden).squeeze(-1)
        )
        delta_logits = self.delta_head(normalized_hidden)
        delta_probability = torch.softmax(delta_logits.float(), dim=-1)
        delta = self._symmetric_expectation(
            delta_probability,
            self.delta_offsets_px.to(delta_probability),
        )
        correction = candidate_gate * (candidate_x - anchor_x) + delta
        if self.training and torch.is_grad_enabled():
            surrogate = candidate_x - anchor_x
            correction = correction + self.gradient_only_candidate_scale * (
                surrogate - surrogate.detach()
            )
        final_x = (anchor_x + correction).clamp(0.0, width)

        pooled = normalized_hidden.mean(dim=2)
        range_logits = self.range_delta_head(pooled).view(
            batch, slots, 2, -1
        )
        range_probability = torch.softmax(range_logits.float(), dim=-1)
        range_delta = self._symmetric_expectation(
            range_probability,
            self.range_delta_offsets_norm.to(range_probability),
        )
        final_range = sort_range_norm(
            (anchor_range + range_delta).clamp(0.0, 1.0)
        )
        final_x = torch.where(
            anchor_valid.unsqueeze(-1), final_x, torch.zeros_like(final_x)
        )
        final_range = torch.where(
            anchor_valid.unsqueeze(-1),
            final_range,
            torch.zeros_like(final_range),
        )
        return {
            "selection_slot_v13_anchor_x_rows": anchor_x,
            "selection_slot_v13_anchor_range_norm": anchor_range,
            "selection_slot_v13_visual_x_rows": visual_x,
            "selection_slot_v13_proposal_expected_x_rows": proposal_expected_x,
            "selection_slot_v13_candidate_x_rows": candidate_x,
            "selection_slot_v13_candidate_gate": candidate_gate,
            "selection_slot_v13_proposal_row_attention": proposal_attention,
            "selection_slot_v13_local_attention": local_attention,
            "selection_slot_v13_hidden": hidden,
            "selection_slot_input_reference_x_rows": anchor_x,
            "selection_slot_input_range_norm": anchor_range,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_range_delta_logits": range_logits,
            "selection_slot_range_delta_offsets_norm": (
                self.range_delta_offsets_norm
            ),
            "selection_slot_range_delta": range_delta,
            "selection_slot_pred_x_rows": final_x,
            "selection_slot_range_norm": final_range,
            "selection_slot_geometry_valid": anchor_valid,
        }


class FourSlotLaneSelectionHead(nn.Module):
    """Route four persistent lane-object slots over frozen V5 proposals.

    The 32-query detector remains proposal memory, while four explicit object
    slots retain the lane/GT axis until the final routing decision.  Optional
    V6-B/V6-C refinement then performs bounded slot-owned geometry updates;
    V6-C additionally allows those geometry losses to train routing through
    a straight-through path while preserving hard unique inference.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        hidden_dim: int = 256,
        num_slots: int = 4,
        proposal_layers: int = 2,
        slot_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        curve_samples: int = 20,
        range_temperature: float = 0.02,
        min_valid_rows: int = 5,
        refinement_enabled: bool = False,
        refinement_hidden_dim: int | None = None,
        refinement_delta_offsets_px: tuple[float, ...] = (
            -24.0,
            -12.0,
            -6.0,
            0.0,
            6.0,
            12.0,
            24.0,
        ),
        refinement_straight_through_routing: bool = False,
        refinement_detach_slot_states: bool = True,
        refinement_route_temperature: float = 1.0,
        factorized_routing: bool = False,
        active_prior_prob: float = 0.80,
        geometry_detach_router_states: bool = True,
        geometry_router_state_gradient_scale: float = 1.0,
        refinement_structured_unique_routing: bool = False,
        refinement_route_gradient_scale: float = 1.0,
        refinement_reference_mode: str = "hard_st",
        refinement_neighborhood_max_candidates: int = 4,
        refinement_neighborhood_max_mean_distance_px: float = 48.0,
        refinement_neighborhood_min_common_fraction: float = 0.50,
        refinement_neighborhood_distance_temperature_px: float = 24.0,
        refinement_neighborhood_gradient_scale: float = 0.10,
        range_refinement_enabled: bool = False,
        range_delta_offsets_norm: tuple[float, ...] = (
            -0.10,
            -0.05,
            -0.025,
            0.0,
            0.025,
            0.05,
            0.10,
        ),
        slot_owned_geometry_enabled: bool = False,
        slot_owned_geometry_hidden_dim: int | None = None,
        slot_owned_geometry_delta_offsets_px: tuple[float, ...] = (
            -160.0,
            -96.0,
            -48.0,
            -24.0,
            0.0,
            24.0,
            48.0,
            96.0,
            160.0,
        ),
        slot_owned_geometry_evidence_offsets_px: tuple[float, ...] = (
            -48.0,
            -24.0,
            -12.0,
            0.0,
            12.0,
            24.0,
            48.0,
        ),
        slot_owned_geometry_route_temperature: float = 1.0,
        slot_owned_geometry_route_gradient_scale: float = 1.0,
        slot_owned_geometry_structured_unique_routing: bool = True,
        slot_owned_geometry_vertical_layers: int = 2,
        slot_owned_geometry_vertical_num_heads: int = 8,
        slot_owned_geometry_vertical_ff_dim: int | None = None,
        slot_owned_geometry_vertical_dropout: float = 0.0,
        slot_owned_geometry_zero_init_delta_heads: bool = False,
        slot_owned_geometry_delta_head_init_std: float = 1.0e-3,
        global_visual_geometry_enabled: bool = False,
        global_visual_geometry_hidden_dim: int | None = None,
        global_visual_geometry_num_heads: int = 8,
        global_visual_geometry_ff_dim: int | None = None,
        global_visual_geometry_vertical_layers: int = 2,
        global_visual_geometry_dropout: float = 0.0,
        global_visual_geometry_delta_offsets_px: tuple[float, ...] = (
            -96.0,
            -48.0,
            -24.0,
            0.0,
            24.0,
            48.0,
            96.0,
        ),
        global_visual_geometry_detach_slot_states: bool = False,
        global_visual_geometry_slot_gradient_scale: float = 1.0,
        global_visual_geometry_spatial_prior_strength: float = 1.0,
        global_visual_geometry_spatial_prior_sigma: float = 0.22,
        global_visual_geometry_range_start_prior: float = 0.05,
        global_visual_geometry_range_end_prior: float = 0.95,
        global_visual_geometry_zero_init_delta_head: bool = True,
        global_visual_geometry_delta_head_init_std: float = 1.0e-3,
        unified_slot_decoder_enabled: bool = False,
        unified_slot_decoder_hidden_dim: int | None = None,
        unified_slot_decoder_num_heads: int = 8,
        unified_slot_decoder_ff_dim: int | None = None,
        unified_slot_decoder_vertical_layers: int = 2,
        unified_slot_decoder_dropout: float = 0.0,
        unified_slot_decoder_delta_offsets_px: tuple[float, ...] = (
            -800.0,
            -600.0,
            -400.0,
            -300.0,
            -200.0,
            -128.0,
            -64.0,
            -32.0,
            0.0,
            32.0,
            64.0,
            128.0,
            200.0,
            300.0,
            400.0,
            600.0,
            800.0,
        ),
        unified_slot_decoder_range_delta_offsets_norm: tuple[float, ...] = (
            -1.0,
            -0.50,
            -0.25,
            -0.10,
            0.0,
            0.10,
            0.25,
            0.50,
            1.0,
        ),
        unified_slot_decoder_proposal_logit_residual_scale: float = 1.0,
        unified_slot_decoder_proposal_attention_temperature: float = 1.0,
        unified_slot_decoder_proposal_attention_sinkhorn_iterations: int = 64,
        unified_slot_decoder_visual_prior_strength: float = 0.25,
        unified_slot_decoder_visual_prior_sigma: float = 0.35,
        unified_slot_decoder_output_head_init_std: float = 1.0e-5,
        unified_slot_decoder_activity_head_init_std: float = 1.0e-7,
        visual_first_association_enabled: bool = False,
        visual_first_association_hidden_dim: int | None = None,
        visual_first_association_num_heads: int = 8,
        visual_first_association_ff_dim: int | None = None,
        visual_first_association_vertical_layers: int = 2,
        visual_first_association_dropout: float = 0.0,
        visual_first_association_proposal_temperature: float = 1.0,
        visual_first_association_sinkhorn_iterations: int = 64,
        visual_first_association_visual_prior_strength: float = 0.25,
        visual_first_association_visual_prior_sigma: float = 0.35,
        visual_first_association_curve_distance_scale: float = 4.0,
        corrected_visual_first_association_enabled: bool = False,
        corrected_visual_first_association_hidden_dim: int | None = None,
        corrected_visual_first_association_num_heads: int = 8,
        corrected_visual_first_association_ff_dim: int | None = None,
        corrected_visual_first_association_vertical_layers: int = 2,
        corrected_visual_first_association_dropout: float = 0.0,
        corrected_visual_first_association_proposal_temperature: float = 1.0,
        corrected_visual_first_association_sinkhorn_iterations: int = 64,
        corrected_visual_first_association_visual_prior_strength: float = 0.25,
        corrected_visual_first_association_visual_prior_sigma: float = 0.35,
        corrected_visual_first_geometry_enabled: bool = False,
        corrected_visual_first_geometry_hidden_dim: int | None = None,
        corrected_visual_first_geometry_num_heads: int = 8,
        corrected_visual_first_geometry_ff_dim: int | None = None,
        corrected_visual_first_geometry_vertical_layers: int = 2,
        corrected_visual_first_geometry_dropout: float = 0.0,
        corrected_visual_first_geometry_delta_offsets_px: tuple[float, ...] = (
            -1600.0,
            -1200.0,
            -800.0,
            -600.0,
            -400.0,
            -300.0,
            -200.0,
            -128.0,
            -64.0,
            -32.0,
            0.0,
            32.0,
            64.0,
            128.0,
            200.0,
            300.0,
            400.0,
            600.0,
            800.0,
            1200.0,
            1600.0,
        ),
        corrected_visual_first_geometry_range_offsets_norm: tuple[float, ...] = (
            -1.0,
            -0.50,
            -0.25,
            -0.10,
            0.0,
            0.10,
            0.25,
            0.50,
            1.0,
        ),
        bottom_aware_relational_geometry_enabled: bool = False,
        bottom_aware_relational_geometry_hidden_dim: int | None = None,
        bottom_aware_relational_geometry_num_heads: int = 8,
        bottom_aware_relational_geometry_ff_dim: int | None = None,
        bottom_aware_relational_geometry_visual_vertical_layers: int = 2,
        bottom_aware_relational_geometry_fusion_vertical_layers: int = 2,
        bottom_aware_relational_geometry_slot_interaction_layers: int = 1,
        bottom_aware_relational_geometry_dropout: float = 0.0,
        bottom_aware_relational_geometry_visual_prior_strength: float = 0.25,
        bottom_aware_relational_geometry_visual_prior_sigma: float = 0.35,
        bottom_aware_relational_geometry_graph_prior_strength: float = 1.0,
        bottom_aware_relational_geometry_graph_self_bias: float = 1.0,
        bottom_aware_relational_geometry_slot_curve_prior_strength: float = 8.0,
        bottom_aware_relational_geometry_delta_offsets_px: tuple[float, ...] = (
            -1600.0,
            -1200.0,
            -800.0,
            -600.0,
            -400.0,
            -300.0,
            -200.0,
            -128.0,
            -64.0,
            -32.0,
            0.0,
            32.0,
            64.0,
            128.0,
            200.0,
            300.0,
            400.0,
            600.0,
            800.0,
            1200.0,
            1600.0,
        ),
        bottom_aware_relational_geometry_range_offsets_norm: tuple[float, ...] = (
            -1.0,
            -0.50,
            -0.25,
            -0.10,
            0.0,
            0.10,
            0.25,
            0.50,
            1.0,
        ),
        candidate_aligned_reranker_enabled: bool = False,
        candidate_aligned_reranker_hidden_dim: int = 128,
        candidate_aligned_reranker_row_dilations: tuple[int, ...] = (1, 2, 4),
        candidate_aligned_reranker_evidence_offsets_px: tuple[float, ...] = (
            -24.0,
            0.0,
            24.0,
        ),
        candidate_aligned_reranker_dropout: float = 0.0,
        candidate_aligned_reranker_corridor_fraction: float = 0.60,
        candidate_aligned_reranker_min_corridor_px: float = 72.0,
        candidate_aligned_reranker_max_corridor_px: float = 256.0,
        candidate_aligned_reranker_min_overlap_fraction: float = 0.25,
        iterative_slot_geometry_enabled: bool = False,
        iterative_slot_geometry_hidden_dim: int | None = None,
        iterative_slot_geometry_num_heads: int = 8,
        iterative_slot_geometry_ff_dim: int | None = None,
        iterative_slot_geometry_num_stages: int = 3,
        iterative_slot_geometry_vertical_layers_per_stage: int = 1,
        iterative_slot_geometry_dropout: float = 0.0,
        iterative_slot_geometry_scale_names: tuple[str, ...] = (
            "p2",
            "p3",
            "p4",
        ),
        iterative_slot_geometry_visual_offsets_px: tuple[float, ...] = (
            -96.0,
            -64.0,
            -32.0,
            -16.0,
            0.0,
            16.0,
            32.0,
            64.0,
            96.0,
        ),
        iterative_slot_geometry_delta_offsets_px: tuple[float, ...] = (
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
            64.0,
        ),
        iterative_slot_geometry_range_offsets_norm: tuple[float, ...] = (
            -0.025,
            -0.0125,
            0.0,
            0.0125,
            0.025,
        ),
        joint_exact_set_energy_enabled: bool = False,
        joint_exact_set_energy_hidden_dim: int | None = None,
        joint_exact_set_energy_num_heads: int = 8,
        joint_exact_set_energy_ff_dim: int | None = None,
        joint_exact_set_energy_dropout: float = 0.0,
        joint_exact_set_energy_scale_names: tuple[str, ...] = (
            "p2",
            "p3",
            "p4",
        ),
        joint_exact_set_energy_association_offsets_px: tuple[float, ...] = (
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
        ),
        joint_exact_set_energy_visual_offsets_px: tuple[float, ...] = (
            -64.0,
            -32.0,
            -16.0,
            0.0,
            16.0,
            32.0,
            64.0,
        ),
        joint_exact_set_energy_delta_offsets_px: tuple[float, ...] = (
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
            64.0,
        ),
        joint_exact_set_energy_range_offsets_norm: tuple[float, ...] = (
            -0.025,
            -0.0125,
            0.0,
            0.0125,
            0.025,
        ),
        joint_exact_set_energy_permutation_temperature: float = 1.0,
        joint_exact_set_energy_keep_prior_probability: float = 0.997,
        joint_exact_set_energy_detach_association_for_set_loss: bool = False,
        joint_exact_set_energy_sampling_backend: str = "grid_sample",
        counterfactual_fidelity_enabled: bool = False,
        counterfactual_fidelity_hidden_dim: int | None = None,
        counterfactual_fidelity_num_heads: int = 8,
        counterfactual_fidelity_ff_dim: int | None = None,
        counterfactual_fidelity_vertical_layers: int = 1,
        counterfactual_fidelity_dropout: float = 0.0,
        counterfactual_fidelity_scale_names: tuple[str, ...] = (
            "p2",
            "p3",
            "p4",
        ),
        counterfactual_fidelity_evidence_offsets_px: tuple[float, ...] = (
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
        ),
        counterfactual_fidelity_sampling_backend: str = "linear_gather",
        slot_owned_safe_replacement_enabled: bool = False,
        slot_owned_safe_replacement_hidden_dim: int | None = None,
        slot_owned_safe_replacement_ff_dim: int | None = None,
        slot_owned_safe_replacement_context_mode: str = "treatment",
        visual_precision_geometry_enabled: bool = False,
        visual_precision_geometry_hidden_dim: int | None = None,
        visual_precision_geometry_num_heads: int = 8,
        visual_precision_geometry_ff_dim: int | None = None,
        visual_precision_geometry_vertical_layers: int = 2,
        visual_precision_geometry_dropout: float = 0.0,
        visual_precision_geometry_local_offsets_px: tuple[float, ...] = (
            -128.0,
            -64.0,
            -32.0,
            -16.0,
            -8.0,
            0.0,
            8.0,
            16.0,
            32.0,
            64.0,
            128.0,
        ),
        visual_precision_geometry_delta_offsets_px: tuple[float, ...] = (
            -1600.0,
            -1200.0,
            -800.0,
            -600.0,
            -400.0,
            -300.0,
            -200.0,
            -128.0,
            -64.0,
            -32.0,
            0.0,
            32.0,
            64.0,
            128.0,
            200.0,
            300.0,
            400.0,
            600.0,
            800.0,
            1200.0,
            1600.0,
        ),
        visual_precision_geometry_range_offsets_norm: tuple[float, ...] = (
            -1.0,
            -0.50,
            -0.25,
            -0.10,
            0.0,
            0.10,
            0.25,
            0.50,
            1.0,
        ),
        visual_precision_geometry_proposal_distance_scale: float = 8.0,
        visual_precision_geometry_invisible_row_logit_bias: float = -2.0,
        visual_precision_geometry_gradient_only_candidate_scale: float = 0.10,
        joint_slot_field_enabled: bool = False,
        joint_slot_field_num_rows: int = 160,
        joint_slot_field_hidden_dim: int = 64,
        joint_slot_field_route_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads):
            raise ValueError("four-slot hidden_dim must be divisible by num_heads")
        if int(num_slots) < 1:
            raise ValueError("four-slot num_slots must be positive")
        if int(curve_samples) < 1:
            raise ValueError("four-slot curve_samples must be positive")
        if int(min_valid_rows) < 1:
            raise ValueError("four-slot min_valid_rows must be positive")
        if float(range_temperature) <= 0.0:
            raise ValueError("four-slot range_temperature must be positive")
        if not 0.0 < float(active_prior_prob) < 1.0:
            raise ValueError("four-slot active_prior_prob must be in (0, 1)")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.curve_samples = int(curve_samples)
        self.range_temperature = float(range_temperature)
        self.min_valid_rows = int(min_valid_rows)
        self.refinement_enabled = bool(refinement_enabled)
        self.slot_owned_geometry_enabled = bool(slot_owned_geometry_enabled)
        self.global_visual_geometry_enabled = bool(
            global_visual_geometry_enabled
        )
        self.unified_slot_decoder_enabled = bool(
            unified_slot_decoder_enabled
        )
        self.visual_first_association_enabled = bool(
            visual_first_association_enabled
        )
        self.corrected_visual_first_association_enabled = bool(
            corrected_visual_first_association_enabled
        )
        self.corrected_visual_first_geometry_enabled = bool(
            corrected_visual_first_geometry_enabled
        )
        self.bottom_aware_relational_geometry_enabled = bool(
            bottom_aware_relational_geometry_enabled
        )
        self.candidate_aligned_reranker_enabled = bool(
            candidate_aligned_reranker_enabled
        )
        self.iterative_slot_geometry_enabled = bool(
            iterative_slot_geometry_enabled
        )
        self.joint_exact_set_energy_enabled = bool(
            joint_exact_set_energy_enabled
        )
        self.counterfactual_fidelity_enabled = bool(
            counterfactual_fidelity_enabled
        )
        self.slot_owned_safe_replacement_enabled = bool(
            slot_owned_safe_replacement_enabled
        )
        self.visual_precision_geometry_enabled = bool(
            visual_precision_geometry_enabled
        )
        self.joint_slot_field_enabled = bool(joint_slot_field_enabled)
        if sum(
            (
                self.refinement_enabled,
                self.slot_owned_geometry_enabled,
                self.global_visual_geometry_enabled,
            )
        ) > 1:
            raise ValueError(
                "legacy, proposal-memory, and global-visual geometry are "
                "mutually exclusive"
            )
        self.factorized_routing = bool(factorized_routing)
        if self.unified_slot_decoder_enabled and not self.factorized_routing:
            raise ValueError(
                "unified slot decoder requires factorized activity/routing"
            )
        if self.visual_first_association_enabled:
            if not self.factorized_routing:
                raise ValueError(
                    "visual-first association requires factorized routing"
                )
            if not self.refinement_enabled:
                raise ValueError(
                    "visual-first Stage A requires exact V7 refined anchors"
                )
            if self.unified_slot_decoder_enabled:
                raise ValueError(
                    "V11 unified and V12 visual-first modules are exclusive"
                )
        if self.corrected_visual_first_association_enabled:
            if not self.factorized_routing:
                raise ValueError(
                    "V14 corrected visual-first association requires "
                    "factorized routing"
                )
            if not self.refinement_enabled:
                raise ValueError(
                    "V14 Stage A requires exact V7 refined anchors"
                )
            if self.unified_slot_decoder_enabled:
                raise ValueError(
                    "V11 unified and V14 corrected modules are exclusive"
                )
            if self.visual_first_association_enabled:
                raise ValueError(
                    "V12 and V14 visual-first modules are mutually exclusive"
                )
            if self.visual_precision_geometry_enabled:
                raise ValueError(
                    "V13 precision geometry and V14 are mutually exclusive"
                )
        if self.corrected_visual_first_geometry_enabled:
            if not self.corrected_visual_first_association_enabled:
                raise ValueError(
                    "V14 Stage B requires the corrected Stage-A association"
                )
            if not self.refinement_enabled:
                raise ValueError("V14 Stage B requires exact V7 anchors")
        if self.bottom_aware_relational_geometry_enabled:
            if not self.factorized_routing:
                raise ValueError("V15 relational geometry requires factorized routing")
            if not self.refinement_enabled:
                raise ValueError("V15 relational geometry requires exact V7 anchors")
            if any(
                (
                    self.unified_slot_decoder_enabled,
                    self.visual_first_association_enabled,
                    self.corrected_visual_first_association_enabled,
                    self.corrected_visual_first_geometry_enabled,
                    self.visual_precision_geometry_enabled,
                )
            ):
                raise ValueError(
                    "V15 relational geometry is exclusive with V11-V14 modules"
                )
        if self.candidate_aligned_reranker_enabled:
            if not self.factorized_routing:
                raise ValueError("V16 candidate reranker requires factorized routing")
            if not self.refinement_enabled:
                raise ValueError("V16 candidate reranker requires exact V7 anchors")
            if any(
                (
                    self.unified_slot_decoder_enabled,
                    self.visual_first_association_enabled,
                    self.corrected_visual_first_association_enabled,
                    self.corrected_visual_first_geometry_enabled,
                    self.visual_precision_geometry_enabled,
                    self.bottom_aware_relational_geometry_enabled,
                    self.slot_owned_geometry_enabled,
                    self.global_visual_geometry_enabled,
                )
            ):
                raise ValueError("V16 candidate reranker is exclusive with V9-V15 modules")
        if self.iterative_slot_geometry_enabled:
            if not self.factorized_routing:
                raise ValueError("V17 iterative geometry requires factorized routing")
            if not self.refinement_enabled:
                raise ValueError("V17 iterative geometry requires exact V7 anchors")
            if any(
                (
                    self.unified_slot_decoder_enabled,
                    self.visual_first_association_enabled,
                    self.corrected_visual_first_association_enabled,
                    self.corrected_visual_first_geometry_enabled,
                    self.visual_precision_geometry_enabled,
                    self.bottom_aware_relational_geometry_enabled,
                    self.candidate_aligned_reranker_enabled,
                    self.slot_owned_geometry_enabled,
                    self.global_visual_geometry_enabled,
                )
            ):
                raise ValueError("V17 iterative geometry is exclusive with V9-V16 modules")
        if self.joint_exact_set_energy_enabled:
            if not self.factorized_routing:
                raise ValueError("V18 exact set energy requires factorized routing")
            if not self.refinement_enabled:
                raise ValueError("V18 exact set energy requires the V7 bounded anchor")
            if any(
                (
                    self.unified_slot_decoder_enabled,
                    self.visual_first_association_enabled,
                    self.corrected_visual_first_association_enabled,
                    self.corrected_visual_first_geometry_enabled,
                    self.visual_precision_geometry_enabled,
                    self.bottom_aware_relational_geometry_enabled,
                    self.candidate_aligned_reranker_enabled,
                    self.iterative_slot_geometry_enabled,
                    self.slot_owned_geometry_enabled,
                    self.global_visual_geometry_enabled,
                )
            ):
                raise ValueError("V18 exact set energy is exclusive with V9-V17 modules")
        if self.counterfactual_fidelity_enabled:
            if not self.factorized_routing:
                raise ValueError("V19 fidelity requires factorized V7 routing")
            if not self.refinement_enabled:
                raise ValueError("V19 fidelity requires the frozen V7 refiner")
            if any(
                (
                    self.unified_slot_decoder_enabled,
                    self.visual_first_association_enabled,
                    self.corrected_visual_first_association_enabled,
                    self.corrected_visual_first_geometry_enabled,
                    self.visual_precision_geometry_enabled,
                    self.bottom_aware_relational_geometry_enabled,
                    self.candidate_aligned_reranker_enabled,
                    self.iterative_slot_geometry_enabled,
                    self.joint_exact_set_energy_enabled,
                    self.slot_owned_geometry_enabled,
                    self.global_visual_geometry_enabled,
                )
            ):
                raise ValueError("V19 fidelity is exclusive with V9-V18 modules")
        if self.slot_owned_safe_replacement_enabled:
            if not self.counterfactual_fidelity_enabled:
                raise ValueError("V20 safe replacement requires frozen V19 fidelity")
            if not self.factorized_routing or not self.refinement_enabled:
                raise ValueError("V20 safe replacement requires exact factorized V7")
        if self.visual_precision_geometry_enabled:
            if not self.visual_first_association_enabled:
                raise ValueError(
                    "V13 precision geometry requires V12 visual-first state"
                )
            if not self.refinement_enabled:
                raise ValueError(
                    "V13 precision geometry requires exact V7 anchors"
                )
        self.geometry_detach_router_states = bool(
            geometry_detach_router_states
        )
        self.geometry_router_state_gradient_scale = float(
            geometry_router_state_gradient_scale
        )
        if not 0.0 <= self.geometry_router_state_gradient_scale <= 1.0:
            raise ValueError(
                "four-slot geometry router-state gradient scale must be in "
                "[0, 1]"
            )
        self.register_buffer(
            "_route_combinations",
            torch.tensor(
                tuple(
                    product(
                        range(self.num_slots + 1),
                        repeat=self.num_slots,
                    )
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_real_route_combinations",
            torch.tensor(
                tuple(
                    product(
                        range(self.num_slots),
                        repeat=self.num_slots,
                    )
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )

        # Compatibility attributes consumed by StructuredLaneQueryHead's
        # shared set-selection integration path.
        self.use_curve_evidence = False
        self.use_semantic_decision = False
        self.detach_geometry_features = True
        self.retain_pointer_diagnostic_tensors = False
        self.requires_row_value_features = (
            self.refinement_enabled
            or self.slot_owned_geometry_enabled
            or self.global_visual_geometry_enabled
            or self.unified_slot_decoder_enabled
            or self.visual_first_association_enabled
            or self.corrected_visual_first_association_enabled
            or self.corrected_visual_first_geometry_enabled
            or self.visual_precision_geometry_enabled
            or self.bottom_aware_relational_geometry_enabled
            or self.candidate_aligned_reranker_enabled
            or self.iterative_slot_geometry_enabled
            or self.joint_exact_set_energy_enabled
            or self.counterfactual_fidelity_enabled
            or self.joint_slot_field_enabled
        )
        self.requires_multi_scale_features = (
            self.iterative_slot_geometry_enabled
            or self.joint_exact_set_energy_enabled
            or self.counterfactual_fidelity_enabled
        )
        # V18 must receive a live P2 tensor. Candidate coordinates remain
        # detached inside the module, while image/proposal representations are
        # deliberately reachable from the exact final-set objective.
        self.requires_live_row_value_features = (
            self.joint_exact_set_energy_enabled
            or self.joint_slot_field_enabled
        )

        # Exact successful probe descriptor:
        # query + range-masked row state + ten geometry scalars + sampled
        # x/confidence + persistent V5 ownership state + foreground margin.
        input_dim = 3 * self.dim + 11 + 2 * self.curve_samples
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        proposal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.proposal_encoder = nn.TransformerEncoder(
            proposal_layer,
            num_layers=int(proposal_layers),
            enable_nested_tensor=False,
        )
        slot_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            slot_layer,
            num_layers=int(slot_layers),
        )
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.slot_norm = nn.LayerNorm(self.hidden_dim)
        self.candidate_norm = nn.LayerNorm(self.hidden_dim)
        self.slot_query = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.candidate_key = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.dustbin = (
            None if self.factorized_routing else nn.Linear(self.hidden_dim, 1)
        )
        self.active = (
            nn.Linear(self.hidden_dim, 1)
            if self.factorized_routing
            else None
        )
        self.slot_refinement = (
            FourSlotBoundedRefinement(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                hidden_dim=int(refinement_hidden_dim or self.hidden_dim),
                delta_offsets_px=tuple(refinement_delta_offsets_px),
                straight_through_routing=bool(
                    refinement_straight_through_routing
                ),
                detach_slot_states=bool(refinement_detach_slot_states),
                route_temperature=float(refinement_route_temperature),
                structured_unique_routing=bool(
                    refinement_structured_unique_routing
                ),
                route_gradient_scale=float(refinement_route_gradient_scale),
                reference_mode=str(refinement_reference_mode),
                neighborhood_max_candidates=int(
                    refinement_neighborhood_max_candidates
                ),
                neighborhood_max_mean_distance_px=float(
                    refinement_neighborhood_max_mean_distance_px
                ),
                neighborhood_min_common_fraction=float(
                    refinement_neighborhood_min_common_fraction
                ),
                neighborhood_distance_temperature_px=float(
                    refinement_neighborhood_distance_temperature_px
                ),
                neighborhood_gradient_scale=float(
                    refinement_neighborhood_gradient_scale
                ),
                range_refinement=bool(range_refinement_enabled),
                range_delta_offsets_norm=tuple(range_delta_offsets_norm),
            )
            if self.refinement_enabled
            else None
        )
        # V9 is deliberately a distinct module/prefix.  Loading a V7/V8
        # checkpoint therefore initializes this row decoder from scratch
        # instead of silently importing weights trained around a hard routed
        # proposal.  Its forward consumes a global structured soft proposal
        # memory; hard proposal IDs remain only for activity/diagnostics.
        self.slot_owned_geometry = (
            FourSlotBoundedRefinement(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                hidden_dim=int(
                    slot_owned_geometry_hidden_dim or self.hidden_dim
                ),
                delta_offsets_px=tuple(slot_owned_geometry_delta_offsets_px),
                evidence_offsets_px=tuple(
                    slot_owned_geometry_evidence_offsets_px
                ),
                straight_through_routing=False,
                detach_slot_states=self.geometry_detach_router_states,
                slot_state_gradient_scale=(
                    self.geometry_router_state_gradient_scale
                ),
                route_temperature=float(
                    slot_owned_geometry_route_temperature
                ),
                structured_unique_routing=bool(
                    slot_owned_geometry_structured_unique_routing
                ),
                route_gradient_scale=float(
                    slot_owned_geometry_route_gradient_scale
                ),
                reference_mode="soft",
                range_refinement=bool(range_refinement_enabled),
                range_delta_offsets_norm=tuple(range_delta_offsets_norm),
                vertical_layers=int(slot_owned_geometry_vertical_layers),
                vertical_num_heads=int(
                    slot_owned_geometry_vertical_num_heads
                ),
                vertical_ff_dim=(
                    None
                    if slot_owned_geometry_vertical_ff_dim is None
                    else int(slot_owned_geometry_vertical_ff_dim)
                ),
                vertical_dropout=float(slot_owned_geometry_vertical_dropout),
                zero_init_delta_heads=bool(
                    slot_owned_geometry_zero_init_delta_heads
                ),
                delta_head_init_std=float(
                    slot_owned_geometry_delta_head_init_std
                ),
            )
            if self.slot_owned_geometry_enabled
            else None
        )
        # V10 removes proposal identity from the geometry-producing forward.
        # Four persistent slots directly scan the complete P2 row grid and
        # own their final x/range predictions.  This distinct prefix prevents
        # accidental reuse of V7--V9 weights trained around proposal anchors.
        self.global_visual_geometry = (
            FourSlotGlobalVisualGeometry(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    global_visual_geometry_hidden_dim or self.hidden_dim
                ),
                num_heads=int(global_visual_geometry_num_heads),
                ff_dim=int(
                    global_visual_geometry_ff_dim
                    or 2 * int(
                        global_visual_geometry_hidden_dim or self.hidden_dim
                    )
                ),
                vertical_layers=int(global_visual_geometry_vertical_layers),
                dropout=float(global_visual_geometry_dropout),
                delta_offsets_px=tuple(
                    global_visual_geometry_delta_offsets_px
                ),
                detach_slot_states=bool(
                    global_visual_geometry_detach_slot_states
                ),
                slot_state_gradient_scale=float(
                    global_visual_geometry_slot_gradient_scale
                ),
                spatial_prior_strength=float(
                    global_visual_geometry_spatial_prior_strength
                ),
                spatial_prior_sigma=float(
                    global_visual_geometry_spatial_prior_sigma
                ),
                range_start_prior=float(
                    global_visual_geometry_range_start_prior
                ),
                range_end_prior=float(
                    global_visual_geometry_range_end_prior
                ),
                zero_init_delta_head=bool(
                    global_visual_geometry_zero_init_delta_head
                ),
                delta_head_init_std=float(
                    global_visual_geometry_delta_head_init_std
                ),
            )
            if self.global_visual_geometry_enabled
            else None
        )
        # V11 keeps the proven V7 selector only as a frozen soft-distribution
        # prior and public provenance.  All 32 proposal row memories and the
        # full P2 grid feed a new lane-object state whose soft coarse geometry
        # and full-width residual own the final curve.
        self.unified_slot_decoder = (
            FourSlotUnifiedProposalVisualDecoder(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    unified_slot_decoder_hidden_dim or self.hidden_dim
                ),
                num_heads=int(unified_slot_decoder_num_heads),
                ff_dim=int(
                    unified_slot_decoder_ff_dim
                    or 2
                    * int(
                        unified_slot_decoder_hidden_dim or self.hidden_dim
                    )
                ),
                vertical_layers=int(unified_slot_decoder_vertical_layers),
                dropout=float(unified_slot_decoder_dropout),
                delta_offsets_px=tuple(
                    unified_slot_decoder_delta_offsets_px
                ),
                range_delta_offsets_norm=tuple(
                    unified_slot_decoder_range_delta_offsets_norm
                ),
                proposal_logit_residual_scale=float(
                    unified_slot_decoder_proposal_logit_residual_scale
                ),
                proposal_attention_temperature=float(
                    unified_slot_decoder_proposal_attention_temperature
                ),
                proposal_attention_sinkhorn_iterations=int(
                    unified_slot_decoder_proposal_attention_sinkhorn_iterations
                ),
                visual_prior_strength=float(
                    unified_slot_decoder_visual_prior_strength
                ),
                visual_prior_sigma=float(
                    unified_slot_decoder_visual_prior_sigma
                ),
                output_head_init_std=float(
                    unified_slot_decoder_output_head_init_std
                ),
                activity_head_init_std=float(
                    unified_slot_decoder_activity_head_init_std
                ),
            )
            if self.unified_slot_decoder_enabled
            else None
        )
        # V12 is a distinct association-first subtree.  Unlike V11, it reads
        # full-width P2 evidence and performs cross-slot reasoning before any
        # all-proposal ranking.  Stage A leaves the complete V7 deployment
        # path intact and trains this module only through diagnostic targets.
        self.visual_first_association = (
            FourSlotVisualFirstAssociation(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    visual_first_association_hidden_dim or self.hidden_dim
                ),
                num_heads=int(visual_first_association_num_heads),
                ff_dim=int(
                    visual_first_association_ff_dim
                    or 2
                    * int(
                        visual_first_association_hidden_dim or self.hidden_dim
                    )
                ),
                vertical_layers=int(
                    visual_first_association_vertical_layers
                ),
                dropout=float(visual_first_association_dropout),
                proposal_attention_temperature=float(
                    visual_first_association_proposal_temperature
                ),
                proposal_attention_sinkhorn_iterations=int(
                    visual_first_association_sinkhorn_iterations
                ),
                visual_prior_strength=float(
                    visual_first_association_visual_prior_strength
                ),
                visual_prior_sigma=float(
                    visual_first_association_visual_prior_sigma
                ),
                proposal_curve_distance_scale=float(
                    visual_first_association_curve_distance_scale
                ),
            )
            if self.visual_first_association_enabled
            else None
        )
        # V14 is the corrected, capacity-compatible Stage-A contract.  It is
        # intentionally separate from V12 so the earlier result remains
        # reproducible and the new target/dustbin semantics are auditable.
        self.corrected_visual_first_association = (
            FourSlotCorrectedVisualFirstAssociation(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    corrected_visual_first_association_hidden_dim
                    or self.hidden_dim
                ),
                num_heads=int(corrected_visual_first_association_num_heads),
                ff_dim=int(
                    corrected_visual_first_association_ff_dim
                    or (2 * self.hidden_dim)
                ),
                vertical_layers=int(
                    corrected_visual_first_association_vertical_layers
                ),
                dropout=float(corrected_visual_first_association_dropout),
                min_valid_rows=self.min_valid_rows,
                proposal_attention_temperature=float(
                    corrected_visual_first_association_proposal_temperature
                ),
                proposal_attention_sinkhorn_iterations=int(
                    corrected_visual_first_association_sinkhorn_iterations
                ),
                visual_prior_strength=float(
                    corrected_visual_first_association_visual_prior_strength
                ),
                visual_prior_sigma=float(
                    corrected_visual_first_association_visual_prior_sigma
                ),
            )
            if self.corrected_visual_first_association_enabled
            else None
        )
        self.corrected_visual_first_geometry = (
            FourSlotV14ParityAnchoredGeometry(
                self.dim,
                input_w=self.input_w,
                visual_dim=int(
                    corrected_visual_first_association_hidden_dim
                    or self.hidden_dim
                ),
                num_slots=self.num_slots,
                hidden_dim=int(
                    corrected_visual_first_geometry_hidden_dim
                    or self.hidden_dim
                ),
                num_heads=int(corrected_visual_first_geometry_num_heads),
                ff_dim=int(
                    corrected_visual_first_geometry_ff_dim
                    or 2
                    * int(
                        corrected_visual_first_geometry_hidden_dim
                        or self.hidden_dim
                    )
                ),
                vertical_layers=int(
                    corrected_visual_first_geometry_vertical_layers
                ),
                dropout=float(corrected_visual_first_geometry_dropout),
                delta_offsets_px=tuple(
                    corrected_visual_first_geometry_delta_offsets_px
                ),
                range_offsets_norm=tuple(
                    corrected_visual_first_geometry_range_offsets_norm
                ),
            )
            if self.corrected_visual_first_geometry_enabled
            else None
        )
        # V15 keeps every proposal member intact.  Bottom-aware curve geometry
        # creates a continuous message-passing graph, while a visual-first
        # persistent slot-row state owns final parity-anchored geometry.  No
        # hard cluster, prototype, proposal-ID target, or public route is added.
        self.bottom_aware_relational_geometry = (
            FourSlotBottomAwareRelationalGeometry(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    bottom_aware_relational_geometry_hidden_dim
                    or self.hidden_dim
                ),
                num_heads=int(bottom_aware_relational_geometry_num_heads),
                ff_dim=int(
                    bottom_aware_relational_geometry_ff_dim
                    or 2
                    * int(
                        bottom_aware_relational_geometry_hidden_dim
                        or self.hidden_dim
                    )
                ),
                visual_vertical_layers=int(
                    bottom_aware_relational_geometry_visual_vertical_layers
                ),
                fusion_vertical_layers=int(
                    bottom_aware_relational_geometry_fusion_vertical_layers
                ),
                slot_interaction_layers=int(
                    bottom_aware_relational_geometry_slot_interaction_layers
                ),
                dropout=float(bottom_aware_relational_geometry_dropout),
                min_valid_rows=self.min_valid_rows,
                visual_prior_strength=float(
                    bottom_aware_relational_geometry_visual_prior_strength
                ),
                visual_prior_sigma=float(
                    bottom_aware_relational_geometry_visual_prior_sigma
                ),
                graph_prior_strength=float(
                    bottom_aware_relational_geometry_graph_prior_strength
                ),
                graph_self_bias=float(
                    bottom_aware_relational_geometry_graph_self_bias
                ),
                slot_curve_prior_strength=float(
                    bottom_aware_relational_geometry_slot_curve_prior_strength
                ),
                delta_offsets_px=tuple(
                    bottom_aware_relational_geometry_delta_offsets_px
                ),
                range_offsets_norm=tuple(
                    bottom_aware_relational_geometry_range_offsets_norm
                ),
            )
            if self.bottom_aware_relational_geometry_enabled
            else None
        )
        # V16 keeps V7 deployment bit-exact and trains only a private hard
        # representative selector.  The candidate groups are GT-free,
        # variable-size and disjoint; every emitted sidecar curve is one
        # complete proposal member, never a coordinate mixture.
        self.candidate_aligned_reranker = (
            FourSlotCandidateAlignedReranker(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(candidate_aligned_reranker_hidden_dim),
                row_dilations=tuple(
                    int(value)
                    for value in candidate_aligned_reranker_row_dilations
                ),
                evidence_offsets_px=tuple(
                    float(value)
                    for value in candidate_aligned_reranker_evidence_offsets_px
                ),
                dropout=float(candidate_aligned_reranker_dropout),
                min_valid_rows=self.min_valid_rows,
                corridor_fraction=float(
                    candidate_aligned_reranker_corridor_fraction
                ),
                min_corridor_px=float(
                    candidate_aligned_reranker_min_corridor_px
                ),
                max_corridor_px=float(
                    candidate_aligned_reranker_max_corridor_px
                ),
                min_overlap_fraction=float(
                    candidate_aligned_reranker_min_overlap_fraction
                ),
            )
            if self.candidate_aligned_reranker_enabled
            else None
        )
        # V17 keeps exact V7 deployment geometry as its initialization and
        # performs three bounded, re-centered updates. P2/P3/P4 and all 32
        # proposal rows are feature context; no proposal coordinate mixture
        # or proposal ID can own the output curve.
        self.iterative_slot_geometry = (
            FourSlotIterativeMultiScaleGeometry(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                num_slots=self.num_slots,
                hidden_dim=int(
                    iterative_slot_geometry_hidden_dim or self.hidden_dim
                ),
                num_heads=int(iterative_slot_geometry_num_heads),
                ff_dim=int(
                    iterative_slot_geometry_ff_dim
                    or 2
                    * int(
                        iterative_slot_geometry_hidden_dim or self.hidden_dim
                    )
                ),
                num_stages=int(iterative_slot_geometry_num_stages),
                vertical_layers_per_stage=int(
                    iterative_slot_geometry_vertical_layers_per_stage
                ),
                dropout=float(iterative_slot_geometry_dropout),
                min_valid_rows=self.min_valid_rows,
                scale_names=tuple(iterative_slot_geometry_scale_names),
                visual_offsets_px=tuple(
                    iterative_slot_geometry_visual_offsets_px
                ),
                delta_offsets_px=tuple(
                    iterative_slot_geometry_delta_offsets_px
                ),
                range_offsets_norm=tuple(
                    iterative_slot_geometry_range_offsets_norm
                ),
            )
            if self.iterative_slot_geometry_enabled
            else None
        )
        # V18 replaces independent unary routing with an exact higher-order
        # score over every 32P4 assignment.  The existing V7 refiner remains
        # the parity anchor; one new branch proposes a single alternative and
        # a categorical KEEP/REFINE policy decides whether to deploy it.
        self.joint_exact_set_energy = (
            FourSlotJointExactSetEnergy(
                self.dim,
                feature_dim=self.dim,
                slot_dim=self.hidden_dim,
                hidden_dim=int(
                    joint_exact_set_energy_hidden_dim or self.hidden_dim
                ),
                input_w=self.input_w,
                candidate_count=32,
                num_slots=self.num_slots,
                num_heads=int(joint_exact_set_energy_num_heads),
                ff_dim=int(
                    joint_exact_set_energy_ff_dim
                    or 2
                    * int(
                        joint_exact_set_energy_hidden_dim or self.hidden_dim
                    )
                ),
                dropout=float(joint_exact_set_energy_dropout),
                scale_names=tuple(joint_exact_set_energy_scale_names),
                association_offsets_px=tuple(
                    joint_exact_set_energy_association_offsets_px
                ),
                visual_offsets_px=tuple(
                    joint_exact_set_energy_visual_offsets_px
                ),
                delta_offsets_px=tuple(
                    joint_exact_set_energy_delta_offsets_px
                ),
                range_offsets_norm=tuple(
                    joint_exact_set_energy_range_offsets_norm
                ),
                permutation_temperature=float(
                    joint_exact_set_energy_permutation_temperature
                ),
                keep_prior_probability=float(
                    joint_exact_set_energy_keep_prior_probability
                ),
                detach_association_for_set_loss=bool(
                    joint_exact_set_energy_detach_association_for_set_loss
                ),
                sampling_backend=str(joint_exact_set_energy_sampling_backend),
            )
            if self.joint_exact_set_energy_enabled
            else None
        )
        # V19 leaves the complete V7 detector, activity and bounded refiner in
        # evaluation mode.  It learns only the counterfactual final-geometry
        # fidelity of each slot/proposal pair; candidates never interact.
        self.counterfactual_fidelity = (
            FourSlotCounterfactualProposalFidelity(
                self.dim,
                feature_dim=self.dim,
                slot_dim=self.hidden_dim,
                hidden_dim=int(
                    counterfactual_fidelity_hidden_dim or self.hidden_dim
                ),
                input_w=self.input_w,
                num_slots=self.num_slots,
                num_heads=int(counterfactual_fidelity_num_heads),
                ff_dim=int(
                    counterfactual_fidelity_ff_dim
                    or 2
                    * int(
                        counterfactual_fidelity_hidden_dim or self.hidden_dim
                    )
                ),
                vertical_layers=int(counterfactual_fidelity_vertical_layers),
                dropout=float(counterfactual_fidelity_dropout),
                scale_names=tuple(counterfactual_fidelity_scale_names),
                evidence_offsets_px=tuple(
                    counterfactual_fidelity_evidence_offsets_px
                ),
                sampling_backend=str(
                    counterfactual_fidelity_sampling_backend
                ),
            )
            if self.counterfactual_fidelity_enabled
            else None
        )
        # V20 keeps V7 and the learned V19 representation immutable.  It can
        # make at most one active slot/proposal replacement and explicitly
        # defaults to exact KEEP.
        self.slot_owned_safe_replacement = (
            SlotOwnedSafeReplacementHead(
                int(counterfactual_fidelity_hidden_dim or self.hidden_dim),
                hidden_dim=int(
                    slot_owned_safe_replacement_hidden_dim or self.hidden_dim
                ),
                ff_dim=int(
                    slot_owned_safe_replacement_ff_dim
                    or 2
                    * int(
                        slot_owned_safe_replacement_hidden_dim or self.hidden_dim
                    )
                ),
                input_w=self.input_w,
                context_mode=str(slot_owned_safe_replacement_context_mode),
            )
            if self.slot_owned_safe_replacement_enabled
            else None
        )
        # V13 is not another proposal selector.  It consumes the proven V12
        # visual lane state, row-wise proposal feature memory and precise local
        # P2 samples, then owns final x/range directly.  V7 remains only the
        # parity anchor and activity/score source in the causal gate.
        self.visual_precision_geometry = (
            FourSlotVisualPrecisionGeometry(
                self.dim,
                input_w=self.input_w,
                visual_dim=int(
                    visual_first_association_hidden_dim or self.hidden_dim
                ),
                num_slots=self.num_slots,
                hidden_dim=int(
                    visual_precision_geometry_hidden_dim or self.hidden_dim
                ),
                num_heads=int(visual_precision_geometry_num_heads),
                ff_dim=int(
                    visual_precision_geometry_ff_dim
                    or 2
                    * int(
                        visual_precision_geometry_hidden_dim or self.hidden_dim
                    )
                ),
                vertical_layers=int(
                    visual_precision_geometry_vertical_layers
                ),
                dropout=float(visual_precision_geometry_dropout),
                local_offsets_px=tuple(
                    visual_precision_geometry_local_offsets_px
                ),
                delta_offsets_px=tuple(
                    visual_precision_geometry_delta_offsets_px
                ),
                range_delta_offsets_norm=tuple(
                    visual_precision_geometry_range_offsets_norm
                ),
                proposal_distance_scale=float(
                    visual_precision_geometry_proposal_distance_scale
                ),
                invisible_row_logit_bias=float(
                    visual_precision_geometry_invisible_row_logit_bias
                ),
                gradient_only_candidate_scale=float(
                    visual_precision_geometry_gradient_only_candidate_scale
                ),
            )
            if self.visual_precision_geometry_enabled
            else None
        )
        # V30 is deliberately attached to the mature V7 selector rather than
        # implemented as another frozen sidecar.  Its slot-conditioned field
        # loss therefore reaches the shared image representation, while
        # detached proposal coordinates preserve the existing support bank.
        self.joint_slot_field = (
            FourSlotJointBeliefField(
                feature_dim=self.dim,
                slot_dim=self.hidden_dim,
                num_rows=int(joint_slot_field_num_rows),
                input_w=self.input_w,
                hidden_dim=int(joint_slot_field_hidden_dim),
                route_residual_scale=float(
                    joint_slot_field_route_residual_scale
                ),
            )
            if self.joint_slot_field_enabled
            else None
        )
        nn.init.normal_(self.slot_tokens.weight, std=0.02)
        if self.active is not None:
            nn.init.zeros_(self.active.weight)
            nn.init.constant_(
                self.active.bias,
                math.log(float(active_prior_prob) / (1.0 - float(active_prior_prob))),
            )

    def _proposal_features(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        row_tokens = outputs["structured_row_tokens"].detach().float()
        queries = outputs["queries"].detach().float()
        ownership = outputs.get("ownership_state")
        if not isinstance(ownership, torch.Tensor):
            raise ValueError("four-slot selection requires V5 ownership_state")
        ownership = ownership.detach().float()
        ranges = sort_range_norm(outputs["range_norm"].detach().float())
        pred_x = outputs["pred_x_rows"].detach().float()
        row_logits = outputs["row_x_logits"].detach().float()
        exist_logits = outputs["exist_logits"].detach().float()
        batch, candidates, rows, _channels = row_tokens.shape
        expected = (batch, candidates, self.dim)
        if queries.shape != expected or ownership.shape != expected:
            raise ValueError("four-slot query/ownership feature shape mismatch")

        y_norm = fixed_row_fractions(
            rows,
            device=row_tokens.device,
            dtype=row_tokens.dtype,
        ).view(1, 1, rows)
        temperature = max(self.range_temperature, 1.0e-4)
        row_weight = torch.sigmoid((y_norm - ranges[..., :1]) / temperature)
        row_weight = row_weight * torch.sigmoid(
            (ranges[..., 1:] - y_norm) / temperature
        )
        denominator = row_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-4)
        masked_mean = (
            row_tokens * row_weight.unsqueeze(-1)
        ).sum(dim=2) / denominator
        centered = row_tokens - masked_mean.unsqueeze(2)
        state_variance = (
            centered.square().mean(dim=-1) * row_weight
        ).sum(dim=-1, keepdim=True) / denominator

        log_max_probability = row_logits.amax(dim=-1) - torch.logsumexp(
            row_logits,
            dim=-1,
        )
        row_confidence = log_max_probability.exp()
        confidence_mean = (
            row_confidence * row_weight
        ).sum(dim=-1, keepdim=True) / denominator
        confidence_max = row_confidence.amax(dim=-1, keepdim=True)

        pred_x_norm = pred_x / float(max(self.input_w - 1, 1))
        first_difference = (
            pred_x_norm[..., 1:] - pred_x_norm[..., :-1]
        ).abs()
        first_weight = row_weight[..., 1:] * row_weight[..., :-1]
        slope = (
            first_difference * first_weight
        ).sum(dim=-1, keepdim=True) / first_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-4)
        second_difference = (
            pred_x_norm[..., 2:]
            - 2.0 * pred_x_norm[..., 1:-1]
            + pred_x_norm[..., :-2]
        ).abs()
        second_weight = (
            row_weight[..., 2:]
            * row_weight[..., 1:-1]
            * row_weight[..., :-2]
        )
        curvature = (
            second_difference * second_weight
        ).sum(dim=-1, keepdim=True) / second_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-4)

        reference = outputs.get("input_reference_x_rows")
        if isinstance(reference, torch.Tensor):
            reference_delta = (
                pred_x - reference.detach().float()
            ).abs() / float(max(self.input_w - 1, 1))
            reference_mean = (
                reference_delta * row_weight
            ).sum(dim=-1, keepdim=True) / denominator
            reference_max = reference_delta.amax(dim=-1, keepdim=True)
        else:
            reference_mean = pred_x.new_zeros((batch, candidates, 1))
            reference_max = pred_x.new_zeros((batch, candidates, 1))

        sample_count = min(self.curve_samples, rows)
        sample_ids = fixed_sample_indices(
            rows,
            sample_count,
            device=pred_x.device,
        )
        sampled_x = pred_x_norm.index_select(-1, sample_ids)
        sampled_confidence = row_confidence.index_select(-1, sample_ids)
        if sample_count < self.curve_samples:
            padding = self.curve_samples - sample_count
            sampled_x = torch.nn.functional.pad(sampled_x, (0, padding))
            sampled_confidence = torch.nn.functional.pad(
                sampled_confidence,
                (0, padding),
            )

        geometry_scalars = torch.cat(
            (
                ranges,
                ranges[..., 1:] - ranges[..., :1],
                confidence_mean,
                confidence_max,
                slope,
                curvature,
                reference_mean,
                reference_max,
                state_variance,
                sampled_x,
                sampled_confidence,
            ),
            dim=-1,
        )
        foreground_margin = exist_logits[..., :1] - exist_logits[..., 1:2]
        return torch.cat(
            (
                queries,
                masked_mean,
                geometry_scalars,
                ownership,
                foreground_margin,
            ),
            dim=-1,
        )

    def _candidate_valid(
        self,
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred_x = outputs["pred_x_rows"].detach()
        ranges = sort_range_norm(outputs["range_norm"].detach().float())
        rows = int(pred_x.shape[-1])
        y_norm = fixed_row_fractions(
            rows,
            device=pred_x.device,
            dtype=ranges.dtype,
        ).view(1, 1, rows)
        visible = (
            (y_norm >= ranges[..., :1])
            & (y_norm <= ranges[..., 1:])
            & torch.isfinite(pred_x)
        )
        return visible.sum(dim=-1) >= self.min_valid_rows

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        row_value_features: torch.Tensor | None = None,
        multi_scale_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self._proposal_features(outputs)
        candidate_valid = self._candidate_valid(outputs)
        memory = self.input_projection(self.input_norm(features))

        # Multi-head attention cannot consume an entirely masked memory row.
        # Such an image can only emit dustbins, but candidate zero is exposed
        # internally as a finite attention placeholder and remains masked in
        # the public routing logits below.
        attention_valid = candidate_valid.clone()
        all_invalid = ~attention_valid.any(dim=1)
        # Avoid a device-to-host boolean synchronization in every forward.
        attention_valid[:, 0] |= all_invalid
        memory = self.proposal_encoder(
            memory,
            src_key_padding_mask=~attention_valid,
        )
        slots = self.slot_tokens.weight.unsqueeze(0).expand(
            features.shape[0],
            -1,
            -1,
        )
        slots = self.slot_decoder(
            slots,
            memory,
            memory_key_padding_mask=~attention_valid,
        )
        slots = self.slot_norm(slots)
        candidates = self.candidate_norm(memory)
        real_route_logits = torch.einsum(
            "bsd,bnd->bsn",
            self.slot_query(slots),
            self.candidate_key(candidates),
        ) / math.sqrt(float(self.hidden_dim))
        joint_field_result: dict[str, torch.Tensor] | None = None
        joint_field_route_residual: torch.Tensor | None = None
        if self.joint_slot_field is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V30 joint field requires live P2 row features")
            joint_field_result = self.joint_slot_field(
                slot_states=slots,
                row_value_features=row_value_features,
                proposal_x_rows=outputs["pred_x_rows"],
                proposal_range_norm=outputs["range_norm"],
                candidate_valid=candidate_valid,
            )
            joint_field_route_residual = joint_field_result[
                "route_residual"
            ].to(dtype=real_route_logits.dtype)
            real_route_logits = real_route_logits + joint_field_route_residual
            # Preserve V7's historical backward graph everywhere else.  The
            # live tensor has already entered the new field; the mature
            # bounded refiner and all legacy optional consumers still receive
            # the same detached P2 view they received before V30.
            row_value_features = row_value_features.detach()
        real_route_logits = real_route_logits.masked_fill(
            ~candidate_valid[:, None, :],
            -1.0e4,
        )
        candidate_count = int(candidate_valid.shape[1])
        if self.factorized_routing:
            if self.active is None:
                raise RuntimeError("factorized four-slot head has no active head")
            active_logits = self.active(slots).squeeze(-1)
            real_log_probability = F.log_softmax(
                real_route_logits.float(),
                dim=-1,
            )
            # This normalized joint view is retained for evaluator/checkpoint
            # compatibility.  Cardinality and conditional real routing remain
            # separate tensors and separate gradient paths internally.
            route_logits = torch.cat(
                (
                    F.logsigmoid(active_logits.float()).unsqueeze(-1)
                    + real_log_probability,
                    F.logsigmoid(-active_logits.float()).unsqueeze(-1),
                ),
                dim=-1,
            )
            real_decoded = decode_unique_real_slot_routes(
                real_route_logits,
                candidate_valid,
                self._real_route_combinations,
            )
            geometry_indices = real_decoded["indices"]
            slot_active = (active_logits >= 0.0) & (geometry_indices >= 0)
            selected_indices = torch.where(
                slot_active,
                geometry_indices,
                geometry_indices.new_full(geometry_indices.shape, -1),
            )
            raw_real = real_decoded["raw_indices"]
            raw_indices = torch.where(
                slot_active,
                raw_real,
                raw_real.new_full(raw_real.shape, -1),
            )
            real_probability = real_log_probability.exp()
            safe = geometry_indices.clamp(min=0)
            selected_scores = real_probability.gather(
                -1,
                safe.unsqueeze(-1),
            ).squeeze(-1) * torch.sigmoid(active_logits.float())
            selected_scores = torch.where(
                slot_active,
                selected_scores,
                torch.sigmoid(-active_logits.float()),
            )
            collision_count = _count_repeated_real_indices(raw_indices)
            repair_count = (selected_indices != raw_indices).sum(dim=-1)
            route_entropy = -(
                real_probability
                * real_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            # The protected V7 contract detaches the normalized states so
            # geometry reaches only the shared route projections.  V8.1 can
            # open this exact backward edge without changing any forward
            # value, allowing a paired causal test of geometry supervision on
            # the global proposal encoder and slot decoder.
            if self.geometry_detach_router_states:
                geometry_slots = slots.detach()
                geometry_candidates = candidates.detach()
            else:
                state_scale = self.geometry_router_state_gradient_scale
                geometry_slots = slots.detach() + state_scale * (
                    slots - slots.detach()
                )
                geometry_candidates = candidates.detach() + state_scale * (
                    candidates - candidates.detach()
                )
            geometry_route_logits = torch.einsum(
                "bsd,bnd->bsn",
                self.slot_query(geometry_slots),
                self.candidate_key(geometry_candidates),
            ) / math.sqrt(float(self.hidden_dim))
            if joint_field_route_residual is not None:
                geometry_route_logits = (
                    geometry_route_logits + joint_field_route_residual
                )
            geometry_route_logits = geometry_route_logits.masked_fill(
                ~candidate_valid[:, None, :],
                -1.0e4,
            )
            result = {
                "selection_slot_logits": route_logits,
                "selection_slot_active_logits": active_logits,
                "selection_slot_real_route_logits": real_route_logits,
                "selection_slot_geometry_route_indices": geometry_indices,
                "selection_slot_candidate_valid": candidate_valid,
                "selection_slot_raw_indices": raw_indices,
                "selection_slot_raw_collision_count": collision_count,
                "selection_slot_route_entropy": route_entropy,
                "selection_slot_indices": selected_indices,
                "selection_slot_scores": selected_scores,
                "selection_slot_global_repair_count": repair_count,
            }
        else:
            if self.dustbin is None:
                raise RuntimeError("legacy four-slot head has no dustbin head")
            route_logits = torch.cat(
                (real_route_logits, self.dustbin(slots)),
                dim=-1,
            )
            probability = torch.softmax(route_logits.float(), dim=-1)
            raw_class = route_logits.argmax(dim=-1)
            raw_indices = torch.where(
                raw_class < candidate_count,
                raw_class,
                raw_class.new_full(raw_class.shape, -1),
            )
            decoded = decode_unique_four_slot_routes(
                route_logits,
                candidate_valid,
                self._route_combinations,
            )
            collision_count = decoded["raw_collision_count"]
            route_entropy = -(
                probability * probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            geometry_indices = decoded["indices"]
            slot_active = geometry_indices >= 0
            geometry_route_logits = route_logits
            result = {
                "selection_slot_logits": route_logits,
                "selection_slot_candidate_valid": candidate_valid,
                "selection_slot_raw_indices": raw_indices,
                "selection_slot_raw_collision_count": collision_count,
                "selection_slot_route_entropy": route_entropy,
                "selection_slot_indices": decoded["indices"],
                "selection_slot_scores": decoded["scores"],
                "selection_slot_global_repair_count": decoded["repair_count"],
            }
        if joint_field_result is not None:
            result.update(
                {
                    "selection_slot_joint_field_logits": joint_field_result[
                        "field_logits"
                    ],
                    "selection_slot_joint_field_candidate_score": (
                        joint_field_result["candidate_score"]
                    ),
                    "selection_slot_joint_field_route_residual": (
                        joint_field_result["route_residual"]
                    ),
                    "selection_slot_joint_field_route_gate": (
                        joint_field_result["route_gate"]
                    ),
                }
            )
        v18_route_result: dict[str, torch.Tensor] | None = None
        v18_image_features: dict[str, torch.Tensor] | None = None
        # Preserve the already-computed V7 deployment tensors from this exact
        # forward.  Gate 0 compares against these values rather than against a
        # second numerically non-deterministic CPU/GPU replay.
        v18_v7_real_route_logits = result.get(
            "selection_slot_real_route_logits"
        )
        v18_v7_geometry_indices = geometry_indices
        v18_v7_selected_indices = result.get("selection_slot_indices")
        v18_v7_selected_scores = result.get("selection_slot_scores")
        v18_v7_active_logits = result.get("selection_slot_active_logits")
        v18_v7_active = slot_active
        v18_v7_selection_logits = result.get("selection_slot_logits")
        v18_v7_raw_indices = result.get("selection_slot_raw_indices")
        v18_v7_route_entropy = result.get("selection_slot_route_entropy")
        if self.counterfactual_fidelity is not None:
            if self.slot_refinement is None:
                raise RuntimeError("V19 requires the frozen V7 refiner")
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V19 fidelity requires frozen P2 rows")
            if not isinstance(multi_scale_features, dict):
                raise ValueError("V19 fidelity requires frozen P2/P3/P4")
            image_features: dict[str, torch.Tensor] = {
                "p2": row_value_features.detach()
            }
            for scale_name in self.counterfactual_fidelity.scale_names:
                if scale_name == "p2":
                    continue
                scale_value = multi_scale_features.get(scale_name)
                if not isinstance(scale_value, torch.Tensor):
                    raise ValueError(
                        f"V19 fidelity requires projected {scale_name}"
                    )
                image_features[scale_name] = scale_value.detach()
            counterfactual = frozen_v7_counterfactual_anchors(
                self.slot_refinement,
                slot_states=slots,
                proposal_row_tokens=outputs["structured_row_tokens"],
                proposal_x_rows=outputs["pred_x_rows"],
                proposal_range_norm=outputs["range_norm"],
                candidate_valid=candidate_valid,
                row_value_features=row_value_features,
            )
            v19_result = self.counterfactual_fidelity(
                slot_states=slots,
                legacy_route_logits=real_route_logits,
                proposal_rows=outputs["structured_row_tokens"],
                proposal_x=outputs["pred_x_rows"],
                proposal_range=outputs["range_norm"],
                candidate_valid=candidate_valid,
                counterfactual_x=counterfactual["x_rows"],
                counterfactual_range=counterfactual["range_norm"],
                counterfactual_valid=counterfactual["valid"],
                image_features=image_features,
            )
            calibrated_logits = v19_result["calibrated_route_logits"]
            calibrated_decoded = decode_unique_real_slot_routes(
                calibrated_logits,
                candidate_valid,
                self._real_route_combinations,
            )
            geometry_indices = calibrated_decoded["indices"]
            # V19 is selection-only.  Cardinality is the exact frozen V7
            # decision even when the calibrated member identity changes.
            slot_active = v18_v7_active & (geometry_indices >= 0)
            selected_indices = torch.where(
                slot_active,
                geometry_indices,
                geometry_indices.new_full(geometry_indices.shape, -1),
            )
            calibrated_log_probability = F.log_softmax(
                calibrated_logits.float(), dim=-1
            )
            calibrated_probability = calibrated_log_probability.exp()
            safe_route = geometry_indices.clamp(min=0)
            selected_scores = calibrated_probability.gather(
                -1, safe_route.unsqueeze(-1)
            ).squeeze(-1) * torch.sigmoid(active_logits.float())
            selected_scores = torch.where(
                slot_active,
                selected_scores,
                torch.sigmoid(-active_logits.float()),
            )
            raw_indices = calibrated_decoded["raw_indices"]
            raw_indices = torch.where(
                slot_active,
                raw_indices,
                raw_indices.new_full(raw_indices.shape, -1),
            )
            route_entropy = -(
                calibrated_probability
                * calibrated_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            geometry_route_logits = calibrated_logits
            result.update(
                {
                    "selection_slot_logits": torch.cat(
                        (
                            F.logsigmoid(active_logits.float()).unsqueeze(-1)
                            + calibrated_log_probability,
                            F.logsigmoid(-active_logits.float()).unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    "selection_slot_real_route_logits": calibrated_logits,
                    "selection_slot_geometry_route_indices": geometry_indices,
                    "selection_slot_indices": selected_indices,
                    "selection_slot_scores": selected_scores,
                    "selection_slot_raw_indices": raw_indices,
                    "selection_slot_raw_collision_count": (
                        _count_repeated_real_indices(raw_indices)
                    ),
                    "selection_slot_route_entropy": route_entropy,
                    "selection_slot_global_repair_count": (
                        (selected_indices != raw_indices).sum(dim=-1)
                    ),
                    "selection_slot_v19_quality_logits": v19_result[
                        "quality_logits"
                    ],
                    "selection_slot_v19_p50": v19_result["p50"],
                    "selection_slot_v19_p75": v19_result["p75"],
                    "selection_slot_v19_expected_iou": v19_result[
                        "expected_iou"
                    ],
                    "selection_slot_v19_fidelity_delta": v19_result[
                        "fidelity_delta"
                    ],
                    "selection_slot_v19_calibrated_route_logits": (
                        calibrated_logits
                    ),
                    "selection_slot_v19_counterfactual_x_rows": (
                        counterfactual["x_rows"]
                    ),
                    "selection_slot_v19_counterfactual_range_norm": (
                        counterfactual["range_norm"]
                    ),
                    "selection_slot_v19_counterfactual_valid": (
                        counterfactual["valid"]
                    ),
                    "selection_slot_v19_visual_attention": v19_result[
                        "visual_attention"
                    ],
                    "selection_slot_v19_v7_real_route_logits": (
                        v18_v7_real_route_logits
                    ),
                    "selection_slot_v19_v7_geometry_route_indices": (
                        v18_v7_geometry_indices
                    ),
                    "selection_slot_v19_v7_indices": v18_v7_selected_indices,
                    "selection_slot_v19_v7_scores": v18_v7_selected_scores,
                    "selection_slot_v19_v7_active_logits": (
                        v18_v7_active_logits
                    ),
                    "selection_slot_v19_v7_active": v18_v7_active,
                    "selection_slot_v19_candidate_state": v19_result[
                        "candidate_state"
                    ],
                }
            )
            if self.slot_owned_safe_replacement is not None:
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (
                        v18_v7_real_route_logits,
                        v18_v7_geometry_indices,
                        v18_v7_selected_indices,
                        v18_v7_selected_scores,
                        v18_v7_active_logits,
                        v18_v7_active,
                        v18_v7_selection_logits,
                    )
                ):
                    raise RuntimeError("V20 requires complete exact V7 state")
                v20_result = self.slot_owned_safe_replacement(
                    candidate_state=v19_result["candidate_state"],
                    p50=v19_result["p50"],
                    p75=v19_result["p75"],
                    expected_iou=v19_result["expected_iou"],
                    legacy_route_logits=v18_v7_real_route_logits,
                    counterfactual_x=counterfactual["x_rows"],
                    counterfactual_range=counterfactual["range_norm"],
                    counterfactual_valid=counterfactual["valid"],
                    source_route=v18_v7_geometry_indices,
                    source_active=v18_v7_active,
                )
                geometry_indices = v20_result["selected_route"]
                slot_active = v18_v7_active
                selected_indices = torch.where(
                    slot_active,
                    geometry_indices,
                    geometry_indices.new_full(geometry_indices.shape, -1),
                )
                # The replacement decision must not change V7 cardinality or
                # writer confidence.  Only the selected proposal identity and
                # resulting frozen bounded geometry may change.
                geometry_route_logits = v18_v7_real_route_logits
                result.update(
                    {
                        "selection_slot_logits": v18_v7_selection_logits,
                        "selection_slot_real_route_logits": (
                            v18_v7_real_route_logits
                        ),
                        "selection_slot_geometry_route_indices": (
                            geometry_indices
                        ),
                        "selection_slot_indices": selected_indices,
                        "selection_slot_scores": v18_v7_selected_scores,
                        "selection_slot_raw_indices": (
                            selected_indices
                            if v18_v7_raw_indices is None
                            else torch.where(
                                v20_result["edit_count"].unsqueeze(-1) > 0,
                                selected_indices,
                                v18_v7_raw_indices,
                            )
                        ),
                        "selection_slot_raw_collision_count": (
                            _count_repeated_real_indices(selected_indices)
                        ),
                        "selection_slot_route_entropy": (
                            v18_v7_route_entropy
                            if v18_v7_route_entropy is not None
                            else result["selection_slot_route_entropy"]
                        ),
                        "selection_slot_global_repair_count": v20_result[
                            "edit_count"
                        ],
                        "selection_slot_v20_action_valid": v20_result[
                            "action_valid"
                        ],
                        "selection_slot_v20_policy_logits": v20_result[
                            "policy_logits"
                        ],
                        "selection_slot_v20_delta50_logits": v20_result[
                            "delta50_logits"
                        ],
                        "selection_slot_v20_delta75_logits": v20_result[
                            "delta75_logits"
                        ],
                        "selection_slot_v20_duplicate_logits": v20_result[
                            "duplicate_logits"
                        ],
                        "selection_slot_v20_abandon_logits": v20_result[
                            "abandon_logits"
                        ],
                        "selection_slot_v20_delta_iou": v20_result[
                            "delta_iou"
                        ],
                        "selection_slot_v20_expected_delta50": v20_result[
                            "expected_delta50"
                        ],
                        "selection_slot_v20_expected_delta75": v20_result[
                            "expected_delta75"
                        ],
                        "selection_slot_v20_set_attention": v20_result[
                            "set_attention"
                        ],
                        "selection_slot_v20_replace": v20_result["replace"],
                        "selection_slot_v20_replace_slot": v20_result[
                            "replace_slot"
                        ],
                        "selection_slot_v20_replace_candidate": v20_result[
                            "replace_candidate"
                        ],
                        "selection_slot_v20_edit_count": v20_result[
                            "edit_count"
                        ],
                        "selection_slot_v20_v7_geometry_route_indices": (
                            v18_v7_geometry_indices
                        ),
                        "selection_slot_v20_v7_indices": v18_v7_selected_indices,
                        "selection_slot_v20_v7_scores": v18_v7_selected_scores,
                        "selection_slot_v20_v7_active_logits": (
                            v18_v7_active_logits
                        ),
                        "selection_slot_v20_v7_active": v18_v7_active,
                    }
                )
        if self.joint_exact_set_energy is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V18 exact set energy requires live P2 rows")
            if not isinstance(multi_scale_features, dict):
                raise ValueError("V18 exact set energy requires P2/P3/P4")
            v18_image_features = {"p2": row_value_features}
            for scale_name in self.joint_exact_set_energy.association.scale_names:
                if scale_name == "p2":
                    continue
                scale_value = multi_scale_features.get(scale_name)
                if not isinstance(scale_value, torch.Tensor):
                    raise ValueError(
                        f"V18 exact set energy requires projected {scale_name}"
                    )
                v18_image_features[scale_name] = scale_value
            v18_route_result = self.joint_exact_set_energy.route(
                slot_states=slots,
                legacy_route_logits=real_route_logits,
                legacy_active_logits=active_logits,
                proposal_rows=outputs["structured_row_tokens"],
                proposal_x=outputs["pred_x_rows"],
                proposal_range=outputs["range_norm"],
                candidate_valid=candidate_valid,
                image_features=v18_image_features,
            )
            geometry_indices = v18_route_result["indices"]
            # Geometry through a hard exact route cannot train the discrete
            # set choice.  Detaching this surrogate makes that contract
            # explicit: only the unordered listwise set loss trains routing.
            # The mature bounded V7 refiner keeps its original geometry-logit
            # tensor; its hard forward depends on V18's indices, while the
            # zero-scaled soft gradient surrogate must not introduce a second
            # floating-point implementation at initialization.
            slot_active = (active_logits >= 0.0) & (geometry_indices >= 0)
            selected_indices = torch.where(
                slot_active,
                geometry_indices,
                geometry_indices.new_full(geometry_indices.shape, -1),
            )
            v18_probability = F.log_softmax(
                v18_route_result["unary"].float(), dim=-1
            ).exp()
            safe_route = geometry_indices.clamp(min=0)
            selected_scores = v18_probability.gather(
                -1, safe_route.unsqueeze(-1)
            ).squeeze(-1) * torch.sigmoid(active_logits.float())
            selected_scores = torch.where(
                slot_active,
                selected_scores,
                torch.sigmoid(-active_logits.float()),
            )
            v18_log_probability = F.log_softmax(
                v18_route_result["unary"].float(), dim=-1
            )
            result.update(
                {
                    "selection_slot_logits": torch.cat(
                        (
                            F.logsigmoid(active_logits.float()).unsqueeze(-1)
                            + v18_log_probability,
                            F.logsigmoid(-active_logits.float()).unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    # This public legacy tensor keeps its historical meaning.
                    # The trainable V18 unary is exported separately below.
                    "selection_slot_real_route_logits": real_route_logits,
                    "selection_slot_geometry_route_indices": geometry_indices,
                    "selection_slot_indices": selected_indices,
                    "selection_slot_scores": selected_scores,
                    "selection_slot_v18_unary_residual": v18_route_result[
                        "unary_residual"
                    ],
                    "selection_slot_v18_unary": v18_route_result["unary"],
                    "selection_slot_v18_pair_energy": v18_route_result[
                        "pair_energy"
                    ],
                    "selection_slot_v18_unordered_set_scores": v18_route_result[
                        "set_scores"
                    ],
                    "selection_slot_v18_valid_set": v18_route_result[
                        "valid_set"
                    ],
                    "selection_slot_v18_combination_indices": (
                        self.joint_exact_set_energy.combination_table
                    ),
                    "selection_slot_v18_set_index": v18_route_result[
                        "set_index"
                    ],
                    "selection_slot_v18_permutation_index": v18_route_result[
                        "permutation_index"
                    ],
                    "selection_slot_v18_set_margin": v18_route_result[
                        "set_margin"
                    ],
                    "selection_slot_v18_candidate_state": v18_route_result[
                        "candidate_state"
                    ],
                    "selection_slot_v18_association_visual_attention": (
                        v18_route_result["association_visual_attention"]
                    ),
                    "selection_slot_v18_v7_real_route_logits": (
                        v18_v7_real_route_logits
                    ),
                    "selection_slot_v18_v7_geometry_route_indices": (
                        v18_v7_geometry_indices
                    ),
                    "selection_slot_v18_v7_indices": v18_v7_selected_indices,
                    "selection_slot_v18_v7_scores": v18_v7_selected_scores,
                    "selection_slot_v18_v7_active_logits": (
                        v18_v7_active_logits
                    ),
                    "selection_slot_v18_v7_active": v18_v7_active,
                }
            )
        if self.slot_refinement is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "four-slot refinement requires projected P2 row features"
                )
            result.update(
                self.slot_refinement(
                    slot_states=slots,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    route_indices=geometry_indices,
                    route_logits=geometry_route_logits,
                    candidate_valid=candidate_valid,
                    slot_active=slot_active,
                    row_value_features=row_value_features,
                )
            )
        if self.joint_exact_set_energy is not None:
            if v18_route_result is None or v18_image_features is None:
                raise RuntimeError("V18 route state was not constructed")
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            geometry_valid = result.get("selection_slot_geometry_valid")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (anchor_x, anchor_range, geometry_valid)
            ):
                raise ValueError("V18 requires the complete V7 bounded anchor")
            v18_refine = self.joint_exact_set_energy.refine(
                route_result=v18_route_result,
                slot_states=slots,
                anchor_x=anchor_x,
                anchor_range=anchor_range,
                geometry_valid=geometry_valid,
                proposal_rows=outputs["structured_row_tokens"],
                proposal_x=outputs["pred_x_rows"],
                proposal_range=outputs["range_norm"],
                candidate_valid=candidate_valid,
                image_features=v18_image_features,
                legacy_active_logits=active_logits,
            )
            final_active_logits = v18_refine["active_logits"]
            final_active = (final_active_logits >= 0.0) & (
                geometry_indices >= 0
            )
            final_indices = torch.where(
                final_active,
                geometry_indices,
                geometry_indices.new_full(geometry_indices.shape, -1),
            )
            v18_probability = F.log_softmax(
                v18_route_result["unary"].float(), dim=-1
            ).exp()
            safe_route = geometry_indices.clamp(min=0)
            final_scores = v18_probability.gather(
                -1, safe_route.unsqueeze(-1)
            ).squeeze(-1) * torch.sigmoid(final_active_logits)
            final_scores = torch.where(
                final_active,
                final_scores,
                torch.sigmoid(-final_active_logits),
            )
            v18_log_probability = F.log_softmax(
                v18_route_result["unary"].float(), dim=-1
            )
            result.update(
                {
                    "selection_slot_logits": torch.cat(
                        (
                            F.logsigmoid(final_active_logits).unsqueeze(-1)
                            + v18_log_probability,
                            F.logsigmoid(-final_active_logits).unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    "selection_slot_active_logits": final_active_logits,
                    "selection_slot_indices": final_indices,
                    "selection_slot_scores": final_scores,
                    "selection_slot_active": final_active,
                    "selection_slot_pred_x_rows": v18_refine["final_x"],
                    "selection_slot_range_norm": v18_refine["final_range"],
                    "selection_slot_v18_anchor_x_rows": anchor_x,
                    "selection_slot_v18_anchor_range_norm": anchor_range,
                    "selection_slot_v18_refined_x_rows": v18_refine[
                        "refined_x"
                    ],
                    "selection_slot_v18_refined_range_norm": v18_refine[
                        "refined_range"
                    ],
                    "selection_slot_v18_delta_x_rows": v18_refine["delta"],
                    "selection_slot_v18_delta_logits": v18_refine[
                        "delta_logits"
                    ],
                    "selection_slot_v18_delta_offsets_px": (
                        self.joint_exact_set_energy.refiner.delta_offsets_px
                    ),
                    "selection_slot_v18_range_logits": v18_refine[
                        "range_logits"
                    ],
                    "selection_slot_v18_range_offsets_norm": (
                        self.joint_exact_set_energy.refiner.range_offsets_norm
                    ),
                    "selection_slot_v18_log_sigma": v18_refine["log_sigma"],
                    "selection_slot_v18_policy_logits": v18_refine[
                        "policy_logits"
                    ],
                    "selection_slot_v18_policy": v18_refine["policy"],
                    "selection_slot_v18_visual_logits": v18_refine[
                        "visual_logits"
                    ],
                    "selection_slot_v18_visual_offsets_px": (
                        self.joint_exact_set_energy.refiner.visual_offsets_px
                    ),
                    "selection_slot_v18_visual_attention": v18_refine[
                        "visual_attention"
                    ],
                    "selection_slot_v18_proposal_attention": v18_refine[
                        "proposal_attention"
                    ],
                }
            )
        if self.slot_owned_geometry is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "slot-owned geometry requires projected P2 row features"
                )
            result.update(
                self.slot_owned_geometry(
                    slot_states=slots,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    route_indices=geometry_indices,
                    route_logits=geometry_route_logits,
                    candidate_valid=candidate_valid,
                    slot_active=slot_active,
                    row_value_features=row_value_features,
                )
            )
        if self.global_visual_geometry is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "global visual slot geometry requires projected P2 rows"
                )
            result.update(
                self.global_visual_geometry(
                    slot_states=slots,
                    slot_active=slot_active,
                    row_value_features=row_value_features,
                )
            )
        if self.visual_first_association is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "visual-first association requires projected P2 rows"
                )
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            anchor_valid = result.get("selection_slot_geometry_valid")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (anchor_x, anchor_range, anchor_valid)
            ):
                raise ValueError(
                    "visual-first Stage A requires exact V7 refined anchors"
                )
            result.update(
                self.visual_first_association(
                    slot_states=slots,
                    anchor_x_rows=anchor_x,
                    anchor_range_norm=anchor_range,
                    anchor_geometry_valid=anchor_valid,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        if self.corrected_visual_first_association is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "V14 corrected association requires projected P2 rows"
                )
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            anchor_valid = result.get("selection_slot_geometry_valid")
            anchor_active = result.get("selection_slot_active")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    anchor_x,
                    anchor_range,
                    anchor_valid,
                    anchor_active,
                )
            ):
                raise ValueError(
                    "V14 Stage A requires the complete exact-V7 anchor"
                )
            result.update(
                self.corrected_visual_first_association(
                    slot_states=slots,
                    anchor_x_rows=anchor_x,
                    anchor_range_norm=anchor_range,
                    anchor_geometry_valid=anchor_valid,
                    anchor_active=anchor_active,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        if self.corrected_visual_first_geometry is not None:
            required_v14 = (
                result.get("selection_slot_v14_visual_state"),
                result.get("selection_slot_v14_proposal_attention"),
                result.get("selection_slot_v14_anchor_x_rows"),
                result.get("selection_slot_v14_anchor_range_norm"),
                result.get("selection_slot_v14_geometry_valid"),
            )
            if not all(isinstance(value, torch.Tensor) for value in required_v14):
                raise ValueError(
                    "V14 Stage B requires the complete frozen Stage-A state"
                )
            result.update(
                self.corrected_visual_first_geometry(
                    visual_state=required_v14[0],
                    proposal_attention=required_v14[1],
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    anchor_x_rows=required_v14[2],
                    anchor_range_norm=required_v14[3],
                    anchor_geometry_valid=required_v14[4],
                )
            )
        if self.bottom_aware_relational_geometry is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V15 relational geometry requires P2 rows")
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            anchor_valid = result.get("selection_slot_geometry_valid")
            anchor_active = result.get("selection_slot_active")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    anchor_x,
                    anchor_range,
                    anchor_valid,
                    anchor_active,
                )
            ):
                raise ValueError("V15 requires the complete exact-V7 anchor")
            result.update(
                self.bottom_aware_relational_geometry(
                    slot_states=slots,
                    anchor_x_rows=anchor_x,
                    anchor_range_norm=anchor_range,
                    anchor_geometry_valid=anchor_valid,
                    anchor_active=anchor_active,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        if self.candidate_aligned_reranker is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V16 candidate reranker requires P2 rows")
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            anchor_valid = result.get("selection_slot_geometry_valid")
            anchor_active = result.get("selection_slot_active")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    anchor_x,
                    anchor_range,
                    anchor_valid,
                    anchor_active,
                )
            ):
                raise ValueError("V16 requires the complete exact-V7 anchor")
            result.update(
                self.candidate_aligned_reranker(
                    slot_states=slots,
                    anchor_indices=geometry_indices,
                    anchor_x_rows=anchor_x,
                    anchor_range_norm=anchor_range,
                    anchor_geometry_valid=anchor_valid,
                    anchor_active=anchor_active,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        if self.iterative_slot_geometry is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V17 iterative geometry requires P2 rows")
            if not isinstance(multi_scale_features, dict):
                raise ValueError("V17 iterative geometry requires P2/P3/P4")
            anchor_x = result.get("selection_slot_pred_x_rows")
            anchor_range = result.get("selection_slot_range_norm")
            anchor_valid = result.get("selection_slot_geometry_valid")
            anchor_active = result.get("selection_slot_active")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    anchor_x,
                    anchor_range,
                    anchor_valid,
                    anchor_active,
                )
            ):
                raise ValueError("V17 requires the complete exact-V7 anchor")
            result.update(
                self.iterative_slot_geometry(
                    slot_states=slots,
                    anchor_x_rows=anchor_x,
                    anchor_range_norm=anchor_range,
                    anchor_geometry_valid=anchor_valid,
                    anchor_active=anchor_active,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                    multi_scale_features=multi_scale_features,
                )
            )
        if self.visual_precision_geometry is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError("V13 precision geometry requires P2 rows")
            required_v13 = (
                result.get("selection_slot_v12_visual_state"),
                result.get("selection_slot_v12_visual_x_rows"),
                result.get("selection_slot_v12_anchor_x_rows"),
                result.get("selection_slot_v12_anchor_range_norm"),
                result.get("selection_slot_v12_geometry_valid"),
            )
            if not all(isinstance(value, torch.Tensor) for value in required_v13):
                raise ValueError(
                    "V13 requires the complete V12 visual-first state"
                )
            result.update(
                self.visual_precision_geometry(
                    visual_state=required_v13[0],
                    visual_x_rows=required_v13[1],
                    anchor_x_rows=required_v13[2],
                    anchor_range_norm=required_v13[3],
                    anchor_geometry_valid=required_v13[4],
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        if self.unified_slot_decoder is not None:
            if not isinstance(row_value_features, torch.Tensor):
                raise ValueError(
                    "unified slot decoder requires projected P2 row features"
                )
            result.update(
                self.unified_slot_decoder(
                    slot_states=slots,
                    legacy_active_logits=active_logits,
                    proposal_row_tokens=outputs["structured_row_tokens"],
                    proposal_x_rows=outputs["pred_x_rows"],
                    proposal_range_norm=outputs["range_norm"],
                    route_indices=geometry_indices,
                    legacy_route_logits=geometry_route_logits,
                    candidate_valid=candidate_valid,
                    row_value_features=row_value_features,
                )
            )
        return result
