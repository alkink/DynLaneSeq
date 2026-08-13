from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import fixed_row_fractions, sort_range_norm


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


def _symmetric_expectation(
    probability: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Symmetric expectation with an exactly-zero uniform result."""

    midpoint = int(probability.shape[-1]) // 2
    positive = probability[..., midpoint + 1 :]
    negative = probability[..., :midpoint].flip(-1)
    return torch.einsum(
        "...k,k->...",
        positive - negative,
        offsets[midpoint + 1 :].to(probability),
    )


def _row_slope(x_rows: torch.Tensor) -> torch.Tensor:
    if int(x_rows.shape[-1]) <= 1:
        return torch.zeros_like(x_rows)
    delta = x_rows[..., 1:] - x_rows[..., :-1]
    return torch.cat((delta[..., :1], delta), dim=-1)


class _GeometryAwareSlotInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if int(hidden_dim) % int(num_heads):
            raise ValueError("V17 slot heads must divide hidden_dim")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.relative_bias = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, self.num_heads, bias=False),
        )
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        current_x: torch.Tensor,
        current_range: torch.Tensor,
        *,
        width: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, rows, _channels = hidden.shape
        normalized = self.norm(hidden).permute(0, 2, 1, 3)
        query = self.query(normalized).view(
            batch, rows, slots, self.num_heads, self.head_dim
        )
        key = self.key(normalized).view(
            batch, rows, slots, self.num_heads, self.head_dim
        )
        value = self.value(normalized).view(
            batch, rows, slots, self.num_heads, self.head_dim
        )
        logits = torch.einsum("brshd,brthd->brhst", query, key)
        logits = logits / math.sqrt(float(self.head_dim))

        x = current_x.permute(0, 2, 1)
        signed_dx = (x.unsqueeze(-1) - x.unsqueeze(-2)) / width
        start = current_range[..., 0]
        end = current_range[..., 1]
        start_delta = start.unsqueeze(-1) - start.unsqueeze(-2)
        end_delta = end.unsqueeze(-1) - end.unsqueeze(-2)
        overlap = (
            torch.minimum(end.unsqueeze(-1), end.unsqueeze(-2))
            - torch.maximum(start.unsqueeze(-1), start.unsqueeze(-2))
        ).clamp_min(0.0)
        pair = torch.stack(
            (
                signed_dx,
                signed_dx.abs(),
                start_delta[:, None].expand(-1, rows, -1, -1),
                end_delta[:, None].expand(-1, rows, -1, -1),
                overlap[:, None].expand(-1, rows, -1, -1),
            ),
            dim=-1,
        )
        bias = self.relative_bias(pair).permute(0, 1, 4, 2, 3)
        probability = torch.softmax((logits + bias).float(), dim=-1)
        context = torch.einsum("brhst,brthd->brshd", probability, value)
        context = context.reshape(batch, rows, slots, self.hidden_dim)
        context = self.output(context).permute(0, 2, 1, 3)
        return hidden + context, probability


class _IterativeGeometryStage(nn.Module):
    def __init__(
        self,
        *,
        proposal_dim: int,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        vertical_layers: int,
        dropout: float,
        scale_names: tuple[str, ...],
        visual_offsets_px: tuple[float, ...],
        delta_offsets_px: tuple[float, ...],
        range_offsets_norm: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.scale_names = tuple(scale_names)
        self.register_buffer(
            "visual_offsets_px",
            torch.tensor(visual_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "delta_offsets_px",
            torch.tensor(delta_offsets_px, dtype=torch.float32),
        )
        self.register_buffer(
            "range_offsets_norm",
            torch.tensor(range_offsets_norm, dtype=torch.float32),
        )

        self.geometry_projection = nn.Linear(4, hidden_dim, bias=False)
        self.slot_interaction = _GeometryAwareSlotInteraction(
            hidden_dim, num_heads
        )

        self.scale_norms = nn.ModuleDict(
            {name: nn.LayerNorm(feature_dim) for name in self.scale_names}
        )
        self.scale_keys = nn.ModuleDict(
            {
                name: nn.Linear(feature_dim, hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.scale_values = nn.ModuleDict(
            {
                name: nn.Linear(feature_dim, hidden_dim, bias=False)
                for name in self.scale_names
            }
        )
        self.visual_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.visual_offset_key = nn.Linear(4, hidden_dim, bias=False)
        self.visual_context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.visual_offset_state = nn.Linear(4, hidden_dim, bias=False)

        self.proposal_norm = nn.LayerNorm(proposal_dim)
        self.proposal_key = nn.Linear(proposal_dim, hidden_dim, bias=False)
        self.proposal_value = nn.Linear(proposal_dim, hidden_dim, bias=False)
        self.proposal_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proposal_relative_bias = nn.Sequential(
            nn.Linear(9, 64),
            nn.GELU(),
            nn.Linear(64, 1, bias=False),
        )
        self.proposal_relative_value = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.proposal_context = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        vertical_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vertical = nn.TransformerEncoder(
            vertical_layer,
            num_layers=vertical_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(
            hidden_dim, len(delta_offsets_px), bias=False
        )
        self.range_head = nn.Linear(
            hidden_dim, 2 * len(range_offsets_norm), bias=False
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.range_head.weight)

    @staticmethod
    def _sample_feature(
        feature: torch.Tensor,
        current_x: torch.Tensor,
        row_fraction: torch.Tensor,
        offsets_px: torch.Tensor,
        *,
        input_w: int,
    ) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("V17 image features must be rank four")
        # P2 row evidence arrives as BHWC; projected P3/P4 arrive as BCHW.
        if int(feature.shape[-1]) <= 512:
            # The row grid is [B,R,X,C].  NCHW pyramid tensors have channel
            # dimension at index one and a spatial width in the last index.
            if int(feature.shape[1]) == int(current_x.shape[-1]):
                feature_nchw = feature.permute(0, 3, 1, 2).contiguous()
            else:
                feature_nchw = feature
        else:
            feature_nchw = feature
        batch, channels, _height, _width = feature_nchw.shape
        slots, rows = int(current_x.shape[1]), int(current_x.shape[2])
        sample_x = current_x.unsqueeze(-1) + offsets_px.view(1, 1, 1, -1)
        sample_x = sample_x.clamp(0.0, float(max(input_w - 1, 1)))
        grid_x = sample_x / float(max(input_w - 1, 1)) * 2.0 - 1.0
        grid_y = row_fraction.view(1, 1, rows, 1).expand(
            batch, slots, rows, int(offsets_px.numel())
        )
        grid_y = grid_y * 2.0 - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch, slots * rows, int(offsets_px.numel()), 2
        )
        sampled = F.grid_sample(
            feature_nchw.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(
            batch, channels, slots, rows, int(offsets_px.numel())
        ).permute(0, 2, 3, 4, 1).contiguous()

    def _visual_evidence(
        self,
        hidden: torch.Tensor,
        current_x: torch.Tensor,
        row_fraction: torch.Tensor,
        features: dict[str, torch.Tensor],
        *,
        input_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_sum: torch.Tensor | None = None
        value_sum: torch.Tensor | None = None
        offsets = self.visual_offsets_px.to(current_x)
        for name in self.scale_names:
            if name not in features:
                raise ValueError(f"V17 missing image scale {name!r}")
            sampled = self._sample_feature(
                features[name],
                current_x,
                row_fraction,
                offsets,
                input_w=input_w,
            )
            sampled = self.scale_norms[name](sampled)
            key = self.scale_keys[name](sampled)
            value = self.scale_values[name](sampled)
            key_sum = key if key_sum is None else key_sum + key
            value_sum = value if value_sum is None else value_sum + value
        if key_sum is None or value_sum is None:
            raise RuntimeError("V17 has no image evidence scales")
        scale = 1.0 / float(len(self.scale_names))
        offset_fraction = offsets / float(max(input_w - 1, 1))
        offset_basis = _position_basis(offset_fraction)
        keys = key_sum * scale + self.visual_offset_key(offset_basis).view(
            1, 1, 1, -1, self.hidden_dim
        )
        values = value_sum * scale
        logits = torch.einsum(
            "bsrh,bsrkh->bsrk",
            self.visual_query(hidden),
            keys,
        ) / math.sqrt(float(self.hidden_dim))
        probability = torch.softmax(logits.float(), dim=-1)
        context = torch.einsum("bsrk,bsrkh->bsrh", probability, values.float())
        expected_offset = torch.einsum(
            "bsrk,k->bsr", probability, offsets.float()
        )
        hidden = hidden + self.visual_context(context)
        hidden = hidden + self.visual_offset_state(
            _position_basis(
                expected_offset / float(max(input_w - 1, 1))
            )
        )
        return hidden, logits, probability

    def _proposal_evidence(
        self,
        hidden: torch.Tensor,
        current_x: torch.Tensor,
        current_range: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_fraction: torch.Tensor,
        *,
        input_w: int,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slots, rows, _channels = hidden.shape
        candidates = int(proposal_x.shape[1])
        if not enabled:
            return hidden, hidden.new_zeros((batch, slots, rows, candidates))
        finite = torch.isfinite(proposal_x)
        safe_x = torch.where(finite, proposal_x, torch.zeros_like(proposal_x))
        proposal_slope = _row_slope(safe_x)
        current_slope = _row_slope(current_x)
        width = float(max(input_w - 1, 1))
        dx = (safe_x[:, None] - current_x.unsqueeze(2)) / width
        slope_delta = (
            proposal_slope[:, None] - current_slope.unsqueeze(2)
        ) / width
        row_y = row_fraction.view(1, 1, rows)
        visible = (
            (row_y >= proposal_range[..., :1])
            & (row_y <= proposal_range[..., 1:])
            & finite
        )
        range_start = proposal_range[:, None, :, None, 0] - current_range[
            :, :, None, None, 0
        ]
        range_end = proposal_range[:, None, :, None, 1] - current_range[
            :, :, None, None, 1
        ]
        range_start = range_start.expand(-1, -1, -1, rows)
        range_end = range_end.expand(-1, -1, -1, rows)
        relative = torch.stack(
            (
                dx,
                dx.abs(),
                dx.square(),
                slope_delta,
                slope_delta.abs(),
                visible[:, None].expand(-1, slots, -1, -1).float(),
                range_start,
                range_end,
                (range_end - range_start).abs(),
            ),
            dim=-1,
        ).permute(0, 1, 3, 2, 4)

        normalized_rows = self.proposal_norm(proposal_rows)
        keys = self.proposal_key(normalized_rows)
        values = self.proposal_value(normalized_rows)
        logits = torch.einsum(
            "bsrh,bnrh->bsrn",
            self.proposal_query(hidden),
            keys,
        ) / math.sqrt(float(self.hidden_dim))
        logits = logits + self.proposal_relative_bias(relative).squeeze(-1)
        valid = candidate_valid[:, None, None, :].bool() & visible.permute(
            0, 2, 1
        )[:, None]
        probability = torch.softmax(
            logits.masked_fill(~valid, -1.0e4).float(), dim=-1
        )
        probability = probability * valid.float()
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        value = values.permute(0, 2, 1, 3)[:, None]
        value = value + self.proposal_relative_value(relative)
        context = torch.einsum("bsrn,bsrnh->bsrh", probability, value)
        return hidden + self.proposal_context(context), probability

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        current_x: torch.Tensor,
        current_range: torch.Tensor,
        proposal_rows: torch.Tensor,
        proposal_x: torch.Tensor,
        proposal_range: torch.Tensor,
        candidate_valid: torch.Tensor,
        row_fraction: torch.Tensor,
        image_features: dict[str, torch.Tensor],
        input_w: int,
        proposal_context_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        width = float(max(input_w - 1, 1))
        geometry = torch.stack(
            (
                current_x / width,
                current_range[..., 0].unsqueeze(-1).expand_as(current_x),
                current_range[..., 1].unsqueeze(-1).expand_as(current_x),
                _row_slope(current_x) / width,
            ),
            dim=-1,
        )
        hidden = hidden + self.geometry_projection(geometry)
        hidden, slot_probability = self.slot_interaction(
            hidden, current_x, current_range, width=width
        )
        hidden, visual_logits, visual_probability = self._visual_evidence(
            hidden,
            current_x,
            row_fraction,
            image_features,
            input_w=input_w,
        )
        hidden, proposal_probability = self._proposal_evidence(
            hidden,
            current_x,
            current_range,
            proposal_rows,
            proposal_x,
            proposal_range,
            candidate_valid,
            row_fraction,
            input_w=input_w,
            enabled=proposal_context_enabled,
        )
        hidden = hidden + self.fusion_ffn(self.fusion_norm(hidden))
        batch, slots, rows, _channels = hidden.shape
        hidden = self.vertical(
            hidden.reshape(batch * slots, rows, self.hidden_dim)
        ).reshape(batch, slots, rows, self.hidden_dim)
        normalized = self.output_norm(hidden)
        delta_logits = self.delta_head(normalized)
        delta_probability = torch.softmax(delta_logits.float(), dim=-1)
        delta = _symmetric_expectation(
            delta_probability, self.delta_offsets_px
        )
        next_x = (current_x + delta).clamp(0.0, width)

        pooled = normalized.mean(dim=2)
        range_logits = self.range_head(pooled).view(
            batch, slots, 2, int(self.range_offsets_norm.numel())
        )
        range_probability = torch.softmax(range_logits.float(), dim=-1)
        range_delta = _symmetric_expectation(
            range_probability, self.range_offsets_norm
        )
        next_range = sort_range_norm(
            (current_range + range_delta).clamp(0.0, 1.0)
        )
        return {
            "hidden": hidden,
            "input_x": current_x,
            "input_range": current_range,
            "x": next_x,
            "range": next_range,
            "delta": delta,
            "delta_logits": delta_logits,
            "range_logits": range_logits,
            "visual_logits": visual_logits,
            "visual_probability": visual_probability,
            "proposal_probability": proposal_probability,
            "slot_probability": slot_probability,
        }


class FourSlotIterativeMultiScaleGeometry(nn.Module):
    """Three-stage V7-anchored lane geometry with re-centered evidence."""

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
        num_stages: int,
        vertical_layers_per_stage: int,
        dropout: float,
        min_valid_rows: int,
        scale_names: tuple[str, ...],
        visual_offsets_px: tuple[float, ...],
        delta_offsets_px: tuple[float, ...],
        range_offsets_norm: tuple[float, ...],
    ) -> None:
        super().__init__()
        if int(num_stages) != 3:
            raise ValueError("V17 causal contract requires exactly three stages")
        if int(num_slots) < 1:
            raise ValueError("V17 requires at least one slot")
        if not scale_names or "p2" not in scale_names:
            raise ValueError("V17 scale set must contain P2")
        for label, values in (
            ("visual", visual_offsets_px),
            ("delta", delta_offsets_px),
            ("range", range_offsets_norm),
        ):
            if len(values) < 3 or len(values) % 2 != 1:
                raise ValueError(f"V17 {label} offsets must be odd and nontrivial")
            if float(values[len(values) // 2]) != 0.0:
                raise ValueError(f"V17 {label} offsets require zero center")
            if any(
                abs(float(values[i]) + float(values[-1 - i])) > 1.0e-8
                for i in range(len(values) // 2)
            ):
                raise ValueError(f"V17 {label} offsets must be symmetric")
        self.dim = int(dim)
        self.input_w = int(input_w)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.min_valid_rows = int(min_valid_rows)
        self.scale_names = tuple(str(name) for name in scale_names)
        self.slot_norm = nn.LayerNorm(int(slot_dim))
        self.slot_projection = nn.Linear(int(slot_dim), self.hidden_dim)
        self.slot_tokens = nn.Embedding(self.num_slots, self.hidden_dim)
        self.row_position = nn.Linear(4, self.hidden_dim, bias=False)
        self.anchor_geometry = nn.Linear(3, self.hidden_dim, bias=False)
        self.initial_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.normal_(self.slot_tokens.weight, std=0.02)
        self.stages = nn.ModuleList(
            [
                _IterativeGeometryStage(
                    proposal_dim=self.dim,
                    feature_dim=self.dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    vertical_layers=int(vertical_layers_per_stage),
                    dropout=float(dropout),
                    scale_names=self.scale_names,
                    visual_offsets_px=tuple(visual_offsets_px),
                    delta_offsets_px=tuple(delta_offsets_px),
                    range_offsets_norm=tuple(range_offsets_norm),
                )
                for _ in range(int(num_stages))
            ]
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
        multi_scale_features: dict[str, torch.Tensor],
        proposal_context_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch, slots, rows = anchor_x_rows.shape
        if slots != self.num_slots:
            raise ValueError("V17 anchor slot count mismatch")
        anchor_x = anchor_x_rows.detach().float()
        anchor_range = sort_range_norm(anchor_range_norm.detach().float())
        geometry_valid = anchor_geometry_valid.detach().bool()
        source_active = anchor_active.detach().bool()
        proposal_rows = proposal_row_tokens.detach().float()
        proposal_x = proposal_x_rows.detach().float()
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        candidate_valid = candidate_valid.detach().bool()
        if tuple(slot_states.shape[:2]) != (batch, slots):
            raise ValueError("V17 slot-state shape mismatch")
        if int(proposal_rows.shape[2]) != rows:
            raise ValueError("V17 proposal row count mismatch")

        row_fraction = fixed_row_fractions(
            rows, device=anchor_x.device, dtype=torch.float32
        )
        slot = self.slot_projection(
            self.slot_norm(slot_states.detach().float())
        ).unsqueeze(2)
        slot = slot + self.slot_tokens.weight.view(
            1, slots, 1, self.hidden_dim
        )
        row = self.row_position(_position_basis(row_fraction)).view(
            1, 1, rows, self.hidden_dim
        )
        anchor_geometry = torch.stack(
            (
                anchor_x / float(max(self.input_w - 1, 1)),
                anchor_range[..., 0].unsqueeze(-1).expand_as(anchor_x),
                anchor_range[..., 1].unsqueeze(-1).expand_as(anchor_x),
            ),
            dim=-1,
        )
        hidden = self.initial_norm(
            slot + row + self.anchor_geometry(anchor_geometry)
        )

        image_features: dict[str, torch.Tensor] = {"p2": row_value_features}
        for name in self.scale_names:
            if name == "p2":
                continue
            value = multi_scale_features.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"V17 requires projected {name} features")
            image_features[name] = value

        current_x = anchor_x
        current_range = anchor_range
        stage_outputs: list[dict[str, torch.Tensor]] = []
        for stage in self.stages:
            result = stage(
                hidden=hidden,
                current_x=current_x,
                current_range=current_range,
                proposal_rows=proposal_rows,
                proposal_x=proposal_x,
                proposal_range=proposal_range,
                candidate_valid=candidate_valid,
                row_fraction=row_fraction,
                image_features=image_features,
                input_w=self.input_w,
                proposal_context_enabled=bool(proposal_context_enabled),
            )
            hidden = result["hidden"]
            current_x = result["x"]
            current_range = result["range"]
            stage_outputs.append(result)

        final_x = torch.where(
            geometry_valid.unsqueeze(-1), current_x, anchor_x
        )
        final_range = torch.where(
            geometry_valid.unsqueeze(-1), current_range, anchor_range
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

        def stack(name: str) -> torch.Tensor:
            return torch.stack([item[name] for item in stage_outputs], dim=1)

        delta_stack = stack("delta")
        return {
            "selection_slot_v17_anchor_x_rows": anchor_x,
            "selection_slot_v17_anchor_range_norm": anchor_range,
            "selection_slot_v17_geometry_valid": geometry_valid,
            "selection_slot_v17_source_active": source_active,
            "selection_slot_v17_writer_valid": writer_valid,
            "selection_slot_v17_stage_input_x_rows": stack("input_x"),
            "selection_slot_v17_stage_input_range_norm": stack("input_range"),
            "selection_slot_v17_stage_x_rows": stack("x"),
            "selection_slot_v17_stage_range_norm": stack("range"),
            "selection_slot_v17_stage_delta_x_rows": delta_stack,
            "selection_slot_v17_stage_delta_logits": stack("delta_logits"),
            "selection_slot_v17_stage_range_logits": stack("range_logits"),
            "selection_slot_v17_stage_visual_logits": stack("visual_logits"),
            "selection_slot_v17_stage_visual_attention": stack(
                "visual_probability"
            ),
            "selection_slot_v17_stage_proposal_attention": stack(
                "proposal_probability"
            ),
            "selection_slot_v17_stage_slot_attention": stack(
                "slot_probability"
            ),
            "selection_slot_v17_visual_offsets_px": self.stages[
                0
            ].visual_offsets_px,
            "selection_slot_v17_delta_offsets_px": self.stages[
                0
            ].delta_offsets_px,
            "selection_slot_v17_range_offsets_norm": self.stages[
                0
            ].range_offsets_norm,
            "selection_slot_v17_mean_abs_delta_by_stage": delta_stack.abs().mean(
                dim=(2, 3)
            ),
            "selection_slot_v17_max_abs_delta_by_stage": delta_stack.abs().amax(
                dim=(2, 3)
            ),
            "selection_slot_pred_x_rows": final_x,
            "selection_slot_range_norm": final_range,
            "selection_slot_input_reference_x_rows": anchor_x,
            "selection_slot_input_range_norm": anchor_range,
            "selection_slot_row_delta_logits": stage_outputs[-1][
                "delta_logits"
            ],
            "selection_slot_row_delta_offsets_px": self.stages[
                -1
            ].delta_offsets_px,
            "selection_slot_range_delta_logits": stage_outputs[-1][
                "range_logits"
            ],
            "selection_slot_range_delta_offsets_norm": self.stages[
                -1
            ].range_offsets_norm,
        }
