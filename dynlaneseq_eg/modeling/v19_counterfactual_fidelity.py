from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_row_fractions, sort_range_norm
from .v18_joint_exact_set_energy import (
    _CandidateAssociationEncoder,
    _position_basis,
    _row_slope,
)


@torch.no_grad()
def frozen_v7_counterfactual_anchors(
    refiner: nn.Module,
    *,
    slot_states: torch.Tensor,
    proposal_row_tokens: torch.Tensor,
    proposal_x_rows: torch.Tensor,
    proposal_range_norm: torch.Tensor,
    candidate_valid: torch.Tensor,
    row_value_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the frozen V7 bounded refiner for every slot/proposal pair.

    The mature refiner is lane-wise: it never mixes one slot's row state with
    another slot.  We can therefore expose the complete ``S x N``
    counterfactual population as one enlarged private slot axis without
    copying the 32-proposal memory along a batch dimension.  No target or
    trainable V19 tensor participates in this computation.
    """

    if slot_states.ndim != 3:
        raise ValueError("V19 slot states must have shape [B,S,D]")
    if proposal_row_tokens.ndim != 4 or proposal_x_rows.ndim != 3:
        raise ValueError("V19 proposal tensors have invalid rank")
    batch, slots, slot_dim = slot_states.shape
    candidates = int(proposal_x_rows.shape[1])
    rows = int(proposal_x_rows.shape[-1])
    if proposal_row_tokens.shape[:3] != (batch, candidates, rows):
        raise ValueError("V19 proposal row memory and geometry do not match")
    if proposal_range_norm.shape != (batch, candidates, 2):
        raise ValueError("V19 proposal range shape mismatch")
    if candidate_valid.shape != (batch, candidates):
        raise ValueError("V19 candidate validity shape mismatch")

    expanded_slots = (
        slot_states.detach()
        .unsqueeze(2)
        .expand(batch, slots, candidates, slot_dim)
        .reshape(batch, slots * candidates, slot_dim)
    )
    route_indices = (
        torch.arange(candidates, device=slot_states.device, dtype=torch.long)
        .view(1, 1, candidates)
        .expand(batch, slots, candidates)
        .reshape(batch, slots * candidates)
    )
    expanded_active = (
        candidate_valid.detach()
        .bool()
        .unsqueeze(1)
        .expand(batch, slots, candidates)
        .reshape(batch, slots * candidates)
    )
    result = refiner(
        slot_states=expanded_slots,
        proposal_row_tokens=proposal_row_tokens.detach(),
        proposal_x_rows=proposal_x_rows.detach(),
        proposal_range_norm=proposal_range_norm.detach(),
        route_indices=route_indices,
        route_logits=None,
        candidate_valid=candidate_valid.detach(),
        slot_active=expanded_active,
        row_value_features=row_value_features.detach(),
    )
    x_rows = result["selection_slot_pred_x_rows"].reshape(
        batch, slots, candidates, rows
    )
    ranges = result["selection_slot_range_norm"].reshape(
        batch, slots, candidates, 2
    )
    geometry_valid = result["selection_slot_geometry_valid"].reshape(
        batch, slots, candidates
    )
    return {
        "x_rows": x_rows,
        "range_norm": ranges,
        "valid": geometry_valid,
    }


class FourSlotCounterfactualProposalFidelity(nn.Module):
    """Predict official-threshold fidelity for every frozen V7 alternative.

    Proposal candidates never attend to one another.  Each slot/proposal
    counterfactual reads its own proposal memory, signed multi-scale image
    profile and final frozen-V7 curve, then reasons only along that curve's
    row axis.  The output is an additive calibration of the immutable V7
    unary route score.
    """

    def __init__(
        self,
        proposal_dim: int,
        *,
        feature_dim: int,
        slot_dim: int,
        hidden_dim: int,
        input_w: int,
        num_slots: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        scale_names: tuple[str, ...],
        evidence_offsets_px: tuple[float, ...],
        sampling_backend: str = "linear_gather",
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1 or int(hidden_dim) % int(num_heads):
            raise ValueError("V19 attention heads must divide hidden_dim")
        if int(vertical_layers) < 1:
            raise ValueError("V19 requires at least one intra/vertical layer")
        if not scale_names or "p2" not in scale_names:
            raise ValueError("V19 image scales must include P2")
        offsets = tuple(float(value) for value in evidence_offsets_px)
        if len(offsets) < 3 or len(offsets) % 2 != 1:
            raise ValueError("V19 evidence offsets must be odd and nontrivial")
        if offsets[len(offsets) // 2] != 0.0:
            raise ValueError("V19 evidence offsets require a zero center")
        backend = str(sampling_backend).strip().lower()
        if backend not in {"grid_sample", "linear_gather"}:
            raise ValueError("V19 sampling backend is invalid")

        self.hidden_dim = int(hidden_dim)
        self.input_w = int(input_w)
        self.num_slots = int(num_slots)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.sampling_backend = backend
        self.register_buffer(
            "evidence_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
        )

        self.proposal_norm = nn.LayerNorm(int(proposal_dim))
        self.proposal_projection = nn.Linear(
            int(proposal_dim), self.hidden_dim, bias=False
        )
        self.slot_norm = nn.LayerNorm(int(slot_dim))
        self.slot_projection = nn.Linear(
            int(slot_dim), self.hidden_dim, bias=False
        )
        self.geometry_projection = nn.Linear(10, self.hidden_dim, bias=False)
        self.row_position_projection = nn.Linear(4, self.hidden_dim, bias=False)
        self.pre_visual_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.scale_norms = nn.ModuleDict(
            {name: nn.LayerNorm(int(feature_dim)) for name in self.scale_names}
        )
        self.scale_keys = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.scale_values = nn.ModuleDict(
            {
                name: nn.Linear(int(feature_dim), self.hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.scale_embedding = nn.Parameter(
            torch.empty(len(self.scale_names), self.hidden_dim)
        )
        self.offset_key = nn.Linear(4, self.hidden_dim, bias=False)
        self.offset_state = nn.Linear(4, self.hidden_dim, bias=False)
        self.visual_context = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
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
        self.intra = nn.TransformerEncoder(
            vertical_layer,
            num_layers=int(vertical_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.quality_output = nn.Linear(self.hidden_dim, 3)

        nn.init.normal_(self.scale_embedding, std=0.02)
        # Exactly equal candidate fidelity at step zero.  The route tensor is
        # therefore bit-identical to V7, not merely close to it.
        nn.init.zeros_(self.quality_output.weight)
        nn.init.zeros_(self.quality_output.bias)

    def _sample_scale(
        self,
        feature: torch.Tensor,
        x_rows: torch.Tensor,
        row_fraction: torch.Tensor,
    ) -> torch.Tensor:
        batch, slots, candidates, rows = x_rows.shape
        sampled = _CandidateAssociationEncoder.sample_feature(
            feature.detach(),
            x_rows.reshape(batch, slots * candidates, rows),
            row_fraction,
            self.evidence_offsets_px.to(x_rows),
            input_w=self.input_w,
            sampling_backend=self.sampling_backend,
        )
        return sampled.reshape(
            batch,
            slots,
            candidates,
            rows,
            int(self.evidence_offsets_px.numel()),
            int(sampled.shape[-1]),
        )

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        legacy_route_logits: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        counterfactual_x: torch.Tensor,
        counterfactual_range: torch.Tensor,
        counterfactual_valid: torch.Tensor,
        image_features: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, slots, candidates, rows = counterfactual_x.shape
        if slots != self.num_slots:
            raise ValueError("V19 slot count mismatch")
        if legacy_route_logits.shape != (batch, slots, candidates):
            raise ValueError("V19 legacy route shape mismatch")
        if proposal_rows.shape[:3] != (batch, candidates, rows):
            raise ValueError("V19 proposal memory shape mismatch")
        if proposal_x.shape != (batch, candidates, rows):
            raise ValueError("V19 proposal x shape mismatch")
        if proposal_range.shape != (batch, candidates, 2):
            raise ValueError("V19 proposal range shape mismatch")
        if counterfactual_range.shape != (batch, slots, candidates, 2):
            raise ValueError("V19 counterfactual range shape mismatch")

        x = counterfactual_x.detach().float()
        ranges = sort_range_norm(counterfactual_range.detach().float())
        proposal_x = proposal_x.detach().float()
        proposal_range = sort_range_norm(proposal_range.detach().float())
        finite = torch.isfinite(x)
        safe_x = torch.where(finite, x, torch.zeros_like(x))
        row_fraction = fixed_row_fractions(
            rows, device=x.device, dtype=torch.float32
        )
        valid = (
            candidate_valid.detach().bool().unsqueeze(1)
            & counterfactual_valid.detach().bool()
        )
        visible = (
            (row_fraction.view(1, 1, 1, rows) >= ranges[..., :1])
            & (row_fraction.view(1, 1, 1, rows) <= ranges[..., 1:])
            & finite
            & valid.unsqueeze(-1)
        )

        proposal_state = self.proposal_projection(
            self.proposal_norm(proposal_rows.detach().float())
        ).unsqueeze(1).expand(batch, slots, candidates, rows, self.hidden_dim)
        slot_state = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2).unsqueeze(3)

        width = float(max(self.input_w - 1, 1))
        cf_slope = _row_slope(safe_x) / width
        raw_x = proposal_x.unsqueeze(1).expand_as(safe_x)
        raw_slope = _row_slope(proposal_x).unsqueeze(1).expand_as(safe_x) / width
        raw_range = proposal_range.unsqueeze(1).expand(
            batch, slots, candidates, 2
        )
        geometry = torch.stack(
            (
                safe_x / width,
                raw_x / width,
                (safe_x - raw_x) / width,
                cf_slope,
                raw_slope,
                ranges[..., 0].unsqueeze(-1).expand_as(safe_x),
                ranges[..., 1].unsqueeze(-1).expand_as(safe_x),
                raw_range[..., 0].unsqueeze(-1).expand_as(safe_x),
                raw_range[..., 1].unsqueeze(-1).expand_as(safe_x),
                visible.float(),
            ),
            dim=-1,
        )
        row_position = self.row_position_projection(
            _position_basis(row_fraction)
        ).view(1, 1, 1, rows, self.hidden_dim)
        hidden = proposal_state + slot_state + self.geometry_projection(geometry)
        hidden = hidden + row_position

        offset = self.evidence_offsets_px.to(safe_x)
        offset_basis = _position_basis(offset / width)
        visual_context: torch.Tensor | None = None
        attention_sum: torch.Tensor | None = None
        query = self.visual_query(self.pre_visual_norm(hidden))
        for scale_index, name in enumerate(self.scale_names):
            feature = image_features.get(name)
            if not isinstance(feature, torch.Tensor):
                raise ValueError(f"V19 is missing frozen image scale {name!r}")
            sampled = self._sample_scale(feature, safe_x, row_fraction)
            normalized = self.scale_norms[name](sampled)
            keys = self.scale_keys[name](normalized)
            keys = keys + self.offset_key(offset_basis).view(
                1, 1, 1, 1, -1, self.hidden_dim
            )
            keys = keys + self.scale_embedding[scale_index].view(
                1, 1, 1, 1, 1, self.hidden_dim
            )
            values = self.scale_values[name](normalized)
            logits = torch.einsum(
                "bsnrh,bsnrkh->bsnrk", query, keys
            ) / math.sqrt(float(self.hidden_dim))
            probability = torch.softmax(logits.float(), dim=-1)
            context = torch.einsum(
                "bsnrk,bsnrkh->bsnrh", probability, values.float()
            )
            expected_offset = torch.einsum(
                "bsnrk,k->bsnr", probability, offset.float()
            )
            context = self.visual_context(context) + self.offset_state(
                _position_basis(expected_offset / width)
            )
            visual_context = context if visual_context is None else visual_context + context
            attention_sum = probability if attention_sum is None else attention_sum + probability
        if visual_context is None or attention_sum is None:
            raise RuntimeError("V19 has no visual evidence")
        hidden = hidden + visual_context / float(len(self.scale_names))
        hidden = hidden + self.fusion(self.fusion_norm(hidden))

        flat_hidden = hidden.reshape(
            batch * slots * candidates, rows, self.hidden_dim
        )
        flat_visible = visible.reshape(batch * slots * candidates, rows)
        safe_visible = flat_visible.clone()
        no_visible = ~safe_visible.any(dim=-1)
        safe_visible[:, 0] |= no_visible
        flat_hidden = self.intra(
            flat_hidden,
            src_key_padding_mask=~safe_visible,
        )
        hidden = self.output_norm(flat_hidden).reshape(
            batch, slots, candidates, rows, self.hidden_dim
        )

        bottom_weight = 0.10 + 0.90 * row_fraction.pow(3)
        pool_weight = visible.float() * bottom_weight.view(1, 1, 1, rows)
        pool_weight = pool_weight / pool_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        pooled = torch.einsum("bsnr,bsnrh->bsnh", pool_weight, hidden)
        logits = self.quality_output(pooled.float())
        logits = torch.where(valid.unsqueeze(-1), logits, logits.new_zeros(()))

        # Subtract the exact neutral log-probability computed through the same
        # operation.  Zero-initialized heads therefore add exact floating-point
        # zeros to the immutable V7 route logits.
        log_probability = F.logsigmoid(logits)
        neutral = F.logsigmoid(torch.zeros_like(logits))
        weights = logits.new_tensor((1.0, 0.5, 0.1))
        fidelity_delta = ((log_probability - neutral) * weights).sum(dim=-1)
        fidelity_delta = fidelity_delta.masked_fill(~valid, -1.0e4)
        calibrated = legacy_route_logits.detach().float() + fidelity_delta
        calibrated = calibrated.masked_fill(
            ~candidate_valid.detach().bool().unsqueeze(1), -1.0e4
        )
        return {
            # V20 consumes this complete-curve representation while keeping
            # every V19 parameter frozen.  Exporting it does not change the
            # V19 deployment score or checkpoint state.
            "candidate_state": pooled,
            "quality_logits": logits,
            "p50": torch.sigmoid(logits[..., 0]),
            "p75": torch.sigmoid(logits[..., 1]),
            "expected_iou": torch.sigmoid(logits[..., 2]),
            "fidelity_delta": fidelity_delta,
            "calibrated_route_logits": calibrated,
            "visual_attention": attention_sum / float(len(self.scale_names)),
            "visible": visible,
        }
