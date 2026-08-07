from __future__ import annotations

import math

import torch
from torch import nn

from .common import sort_range_norm


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

        # Compatibility attributes consumed by StructuredLaneQueryHead's
        # shared set-selection integration path.
        self.use_curve_evidence = False
        self.use_semantic_decision = False
        self.detach_geometry_features = True
        self.retain_pointer_diagnostic_tensors = False

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
        collision_count = raw_indices.new_zeros((raw_indices.shape[0],))
        for batch_index in range(int(raw_indices.shape[0])):
            active = raw_indices[batch_index][raw_indices[batch_index] >= 0]
            collision_count[batch_index] = int(active.numel()) - int(
                active.unique().numel()
            )
        route_entropy = -(
            probability * probability.clamp_min(1.0e-12).log()
        ).sum(dim=-1)
        return {
            "selection_slot_logits": route_logits,
            "selection_slot_candidate_valid": candidate_valid,
            "selection_slot_raw_indices": raw_indices,
            "selection_slot_raw_collision_count": collision_count,
            "selection_slot_route_entropy": route_entropy,
        }
