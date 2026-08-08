from __future__ import annotations

import math
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .common import sort_range_norm


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
    batch_ids = torch.arange(batch, device=logits.device)
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
    raw_collision_count = raw_indices.new_zeros((batch,))
    for batch_index in range(batch):
        active = raw_indices[batch_index][raw_indices[batch_index] >= 0]
        raw_collision_count[batch_index] = int(active.numel()) - int(
            active.unique().numel()
        )
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

    combinations = torch.tensor(
        tuple(product(range(int(slots)), repeat=int(slots))),
        dtype=torch.long,
        device=logits.device,
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
    batch_ids = torch.arange(batch, device=logits.device)
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

    outputs: list[torch.Tensor] = []
    for batch_index in range(int(batch)):
        valid_ids = torch.nonzero(
            candidate_valid[batch_index].bool(),
            as_tuple=False,
        ).flatten()
        if int(valid_ids.numel()) < int(slots):
            hard = decode_unique_real_slot_routes(
                logits[batch_index],
                candidate_valid[batch_index],
            )["indices"]
            fallback = logits.new_zeros((slots, candidates), dtype=torch.float32)
            for slot, candidate in enumerate(hard.tolist()):
                if int(candidate) >= 0:
                    fallback[slot, int(candidate)] = 1.0
            outputs.append(fallback)
            continue

        valid_count = int(valid_ids.numel())
        scores = logits[batch_index, :, valid_ids].float() / tau
        dummy = scores.new_zeros((valid_count - int(slots), valid_count))
        log_transport = torch.cat((scores, dummy), dim=0)
        # Twenty alternating projections are enough for the 32x32 matrices in
        # this model while remaining cheap relative to the image decoder.
        for _ in range(20):
            log_transport = log_transport - torch.logsumexp(
                log_transport,
                dim=1,
                keepdim=True,
            )
            log_transport = log_transport - torch.logsumexp(
                log_transport,
                dim=0,
                keepdim=True,
            )
        local = log_transport[:slots].exp()
        marginal = logits.new_zeros((slots, candidates), dtype=torch.float32)
        marginal[:, valid_ids] = local
        outputs.append(marginal)
    return torch.stack(outputs)


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
        straight_through_routing: bool = False,
        detach_slot_states: bool = True,
        route_temperature: float = 1.0,
        structured_unique_routing: bool = False,
        route_gradient_scale: float = 1.0,
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
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.hidden_dim = int(hidden_dim)
        self.straight_through_routing = bool(straight_through_routing)
        self.detach_slot_states = bool(detach_slot_states)
        self.route_temperature = float(route_temperature)
        self.structured_unique_routing = bool(structured_unique_routing)
        self.route_gradient_scale = float(route_gradient_scale)
        self.range_refinement = bool(range_refinement)
        if self.route_temperature <= 0.0:
            raise ValueError("slot refinement route_temperature must be positive")
        if not 0.0 <= self.route_gradient_scale <= 1.0:
            raise ValueError(
                "slot refinement route_gradient_scale must be in [0, 1]"
            )
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
            torch.empty(len(offsets), self.hidden_dim)
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
        self.delta_norm = nn.LayerNorm(
            self.hidden_dim,
            elementwise_affine=False,
        )
        self.delta_head = nn.Linear(
            self.hidden_dim,
            len(offsets),
            bias=False,
        )
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
        nn.init.zeros_(self.delta_head.weight)
        if self.range_delta_head is not None:
            nn.init.zeros_(self.range_delta_head.weight)
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
        )
        self.register_buffer(
            "range_delta_offsets_norm",
            torch.tensor(range_offsets, dtype=torch.float32),
        )

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
        offsets = self.delta_offsets_px.to(
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
        row_index = torch.arange(
            batch * rows,
            device=row_value_features.device,
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
        if (
            self.straight_through_routing
            and self.training
            and torch.is_grad_enabled()
        ):
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
            # Use the exact hard gather in forward, with gradients from the
            # soft route.  Computing a one-hot gather through GEMM changes the
            # verified curve by several pixels when TF32 is enabled.
            reference_x = hard_reference_x.float() + (
                soft_reference_x - soft_reference_x.detach()
            ) * self.route_gradient_scale
            slot_range = hard_slot_range + (
                soft_slot_range - soft_slot_range.detach()
            ) * self.route_gradient_scale
            routed_rows = hard_routed_rows.float() + (
                soft_routed_rows - soft_routed_rows.detach()
            ) * self.route_gradient_scale
        else:
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

        slot_input = (
            slot_states.detach() if self.detach_slot_states else slot_states
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
        return result


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
        refinement_structured_unique_routing: bool = False,
        refinement_route_gradient_scale: float = 1.0,
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
        self.factorized_routing = bool(factorized_routing)
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

        # Compatibility attributes consumed by StructuredLaneQueryHead's
        # shared set-selection integration path.
        self.use_curve_evidence = False
        self.use_semantic_decision = False
        self.detach_geometry_features = True
        self.retain_pointer_diagnostic_tensors = False
        self.requires_row_value_features = self.refinement_enabled

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
                range_refinement=bool(range_refinement_enabled),
                range_delta_offsets_norm=tuple(range_delta_offsets_norm),
            )
            if self.refinement_enabled
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

        y_norm = (
            torch.arange(rows, device=row_tokens.device, dtype=row_tokens.dtype)
            / float(max(rows, 1))
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
        sample_ids = torch.linspace(
            0,
            rows - 1,
            sample_count,
            device=pred_x.device,
        ).round().long()
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
        y_norm = (
            torch.arange(rows, device=pred_x.device, dtype=ranges.dtype)
            / float(max(rows, 1))
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
        if bool(all_invalid.any()):
            attention_valid[all_invalid, 0] = True
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
            collision_count = raw_indices.new_zeros((features.shape[0],))
            for batch_index in range(int(features.shape[0])):
                active_raw = raw_indices[batch_index][raw_indices[batch_index] >= 0]
                collision_count[batch_index] = int(active_raw.numel()) - int(
                    active_raw.unique().numel()
                )
            repair_count = (selected_indices != raw_indices).sum(dim=-1)
            route_entropy = -(
                real_probability
                * real_probability.clamp_min(1.0e-12).log()
            ).sum(dim=-1)
            # Detaching the normalized states blocks geometry loss from the
            # active head and shared router trunk, while the shared route
            # projection weights still receive the controlled route gradient.
            geometry_route_logits = torch.einsum(
                "bsd,bnd->bsn",
                self.slot_query(slots.detach()),
                self.candidate_key(candidates.detach()),
            ) / math.sqrt(float(self.hidden_dim))
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
        return result
