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


class FourSlotBoundedRefinement(nn.Module):
    """Locally refine routed proposal curves without touching the detector.

    The discrete route and every proposal/P2 input are detached.  Dense slot
    geometry losses can therefore train this module while the proven V5.1
    proposal generator and the successful V6-A router remain immutable.
    """

    def __init__(
        self,
        dim: int,
        *,
        input_w: int,
        slot_dim: int,
        hidden_dim: int,
        delta_offsets_px: tuple[float, ...],
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
        nn.init.normal_(self.offset_embedding, std=0.02)
        # Uniform probability over symmetric offsets has exactly zero expected
        # displacement, so the new graph starts as the verified V6-A model.
        nn.init.zeros_(self.delta_head.weight)
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
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
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, slots = route_indices.shape
        candidates = int(proposal_x_rows.shape[1])
        rows = int(proposal_x_rows.shape[-1])
        safe = route_indices.clamp(min=0, max=max(candidates - 1, 0))
        active = route_indices >= 0
        x_index = safe.unsqueeze(-1).expand(-1, -1, rows)
        reference_x = proposal_x_rows.detach().gather(1, x_index)
        range_index = safe.unsqueeze(-1).expand(-1, -1, 2)
        slot_range = sort_range_norm(
            proposal_range_norm.detach().gather(1, range_index).float()
        )
        token_index = safe.unsqueeze(-1).unsqueeze(-1).expand(
            -1,
            -1,
            rows,
            int(proposal_row_tokens.shape[-1]),
        )
        routed_rows = proposal_row_tokens.detach().gather(1, token_index)
        reference_x = torch.where(
            active.unsqueeze(-1),
            reference_x,
            torch.zeros_like(reference_x),
        )
        slot_range = torch.where(
            active.unsqueeze(-1),
            slot_range,
            torch.zeros_like(slot_range),
        )

        query_state = self.row_projection(self.row_norm(routed_rows.float()))
        query_state = query_state + self.slot_projection(
            self.slot_norm(slot_states.detach().float())
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
        delta = torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))
        refined_x = (reference_x.float() + delta).clamp(
            0.0,
            float(max(self.input_w - 1, 1)),
        )
        active_rows = active.unsqueeze(-1).expand(-1, -1, rows)
        active_count = active_rows.float().sum(dim=(1, 2)).clamp_min(1.0)
        mean_abs = (delta.abs() * active_rows.float()).sum(dim=(1, 2))
        mean_abs = mean_abs / active_count
        max_abs = delta.abs().masked_fill(~active_rows, 0.0).amax(dim=(1, 2))
        boundary_mass = probability[..., (0, -1)].sum(dim=-1)
        boundary_mass = (
            boundary_mass * active_rows.float()
        ).sum(dim=(1, 2)) / active_count
        return {
            "selection_slot_pred_x_rows": refined_x,
            "selection_slot_range_norm": slot_range,
            "selection_slot_active": active,
            "selection_slot_input_reference_x_rows": reference_x,
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": self.delta_offsets_px,
            "selection_slot_delta_mean_abs": mean_abs,
            "selection_slot_delta_max_abs": max_abs,
            "selection_slot_delta_boundary_mass": boundary_mass,
        }


class FourSlotLaneSelectionHead(nn.Module):
    """Route four persistent lane-object slots over frozen V5 proposals.

    V6-A deliberately reproduces the successful frozen diagnostic contract:
    the 32-query detector remains a proposal memory, while four explicit
    object slots retain the lane/GT axis until the final routing decision.
    This head only selects existing proposal geometry; bounded slot-owned
    geometry refinement belongs to the subsequent V6-B experiment.
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

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.curve_samples = int(curve_samples)
        self.range_temperature = float(range_temperature)
        self.min_valid_rows = int(min_valid_rows)
        self.refinement_enabled = bool(refinement_enabled)
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
        self.dustbin = nn.Linear(self.hidden_dim, 1)
        self.slot_refinement = (
            FourSlotBoundedRefinement(
                self.dim,
                input_w=self.input_w,
                slot_dim=self.hidden_dim,
                hidden_dim=int(refinement_hidden_dim or self.hidden_dim),
                delta_offsets_px=tuple(refinement_delta_offsets_px),
            )
            if self.refinement_enabled
            else None
        )
        nn.init.normal_(self.slot_tokens.weight, std=0.02)

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
        route_logits = torch.einsum(
            "bsd,bnd->bsn",
            self.slot_query(slots),
            self.candidate_key(candidates),
        ) / math.sqrt(float(self.hidden_dim))
        route_logits = route_logits.masked_fill(
            ~candidate_valid[:, None, :],
            -1.0e4,
        )
        route_logits = torch.cat((route_logits, self.dustbin(slots)), dim=-1)

        probability = torch.softmax(route_logits.float(), dim=-1)
        raw_class = route_logits.argmax(dim=-1)
        candidate_count = int(candidate_valid.shape[1])
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
                    route_indices=decoded["indices"],
                    row_value_features=row_value_features,
                )
            )
        return result
