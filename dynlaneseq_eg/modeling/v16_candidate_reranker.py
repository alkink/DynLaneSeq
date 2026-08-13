from __future__ import annotations

import math

import torch
from torch import nn

from .common import fixed_indices, fixed_row_fractions, sort_range_norm
from .v16_candidate_groups import (
    V16CandidateGroupConfig,
    build_v16_anchor_candidate_groups,
)


class _MaskedDilatedRowBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2 * int(dilation),
            dilation=int(dilation),
            groups=hidden_dim,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(value * mask)
        update = self.pointwise(self.norm(update))
        return (value + update) * mask


class FourSlotCandidateAlignedReranker(nn.Module):
    """Select one coherent proposal from each V16 anchor-owned group.

    Geometry creates a variable-size, disjoint competition set.  Each member
    is then judged from its complete proposal row state and P2 evidence sampled
    along that member's own curve.  The public deployment remains exact V7;
    hard selected proposal coordinates are exported only as a Stage-A sidecar.
    No proposal coordinates are averaged anywhere in this module.
    """

    FEATURE_POLICIES = {
        "correct",
        "zero_content",
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
        hidden_dim: int = 128,
        row_dilations: tuple[int, ...] = (1, 2, 4),
        evidence_offsets_px: tuple[float, ...] = (-24.0, 0.0, 24.0),
        dropout: float = 0.0,
        min_valid_rows: int = 5,
        corridor_fraction: float = 0.60,
        min_corridor_px: float = 72.0,
        max_corridor_px: float = 256.0,
        min_overlap_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 16:
            raise ValueError("V16 hidden_dim must be at least 16")
        if int(num_slots) < 1:
            raise ValueError("V16 requires at least one slot")
        if not row_dilations or any(int(value) < 1 for value in row_dilations):
            raise ValueError("V16 row dilations must be positive")
        offsets = tuple(float(value) for value in evidence_offsets_px)
        if not offsets or tuple(sorted(offsets)) != offsets:
            raise ValueError("V16 evidence offsets must be sorted and nonempty")
        if not any(abs(value) < 1.0e-8 for value in offsets):
            raise ValueError("V16 evidence offsets must include zero")

        self.dim = int(dim)
        self.input_w = int(input_w)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.min_valid_rows = int(min_valid_rows)
        self.group_config = V16CandidateGroupConfig(
            corridor_fraction=float(corridor_fraction),
            min_corridor_px=float(min_corridor_px),
            max_corridor_px=float(max_corridor_px),
            min_common_rows=int(min_valid_rows),
            min_overlap_fraction=float(min_overlap_fraction),
        )
        self.group_config.validate()
        self.register_buffer(
            "evidence_offsets_px",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=True,
        )

        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.slot_projection = nn.Linear(self.slot_dim, self.hidden_dim)
        self.proposal_norm = nn.LayerNorm(self.dim)
        self.proposal_projection = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.geometry_projection = nn.Linear(9, self.hidden_dim, bias=False)
        self.relative_geometry_projection = nn.Linear(
            9, self.hidden_dim, bias=False
        )
        self.feature_norm = nn.LayerNorm(self.dim)
        self.feature_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.feature_value = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.evidence_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.offset_embedding = nn.Parameter(
            torch.empty(len(offsets), self.hidden_dim)
        )
        self.evidence_context = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.row_blocks = nn.ModuleList(
            _MaskedDilatedRowBlock(
                self.hidden_dim,
                int(dilation),
                float(dropout),
            )
            for dilation in row_dilations
        )
        self.row_pool_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.score_norm = nn.LayerNorm(self.hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.normal_(self.offset_embedding, std=0.02)

    def _sample_candidate_evidence(
        self,
        row_value_features: torch.Tensor,
        proposal_x_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample P2 at J offsets along all N complete curves."""

        batch, rows, x_bins, channels = row_value_features.shape
        if tuple(proposal_x_rows.shape[:1]) != (batch,):
            raise ValueError("V16 proposal/P2 batch mismatch")
        candidates = int(proposal_x_rows.shape[1])
        if int(proposal_x_rows.shape[-1]) != rows:
            raise ValueError("V16 proposal and P2 row grids must match")
        offsets = self.evidence_offsets_px.to(
            device=proposal_x_rows.device,
            dtype=torch.float32,
        ) * (float(self.input_w) / 1600.0)
        sample_count = int(offsets.numel())
        with torch.autocast(
            device_type=row_value_features.device.type,
            enabled=False,
        ):
            sample_x = proposal_x_rows.detach().float().unsqueeze(-1)
            sample_x = (sample_x + offsets.view(1, 1, 1, -1)).clamp(
                0.0,
                float(max(self.input_w - 1, 1)),
            )
            feature_x = sample_x * float(max(x_bins - 1, 0)) / float(
                max(self.input_w - 1, 1)
            )
            left = feature_x.floor().long()
            right = (left + 1).clamp(max=max(x_bins - 1, 0))
            alpha = feature_x - left.to(feature_x.dtype)
        flat = row_value_features.reshape(batch * rows, x_bins, channels)
        left = left.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * sample_count
        )
        right = right.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * sample_count
        )
        alpha = alpha.permute(0, 2, 1, 3).reshape(
            batch * rows, candidates * sample_count, 1
        )
        row_index = fixed_indices(
            batch * rows,
            device=row_value_features.device,
            dtype=torch.long,
        ).view(-1, 1)
        paired = torch.stack((left, right), dim=-1).reshape(
            batch * rows, candidates * sample_count * 2
        )
        sampled = flat[row_index, paired].view(
            batch * rows,
            candidates * sample_count,
            2,
            channels,
        )
        sampled = torch.lerp(
            sampled[:, :, 0],
            sampled[:, :, 1],
            alpha.to(sampled.dtype),
        )
        return sampled.view(
            batch, rows, candidates, sample_count, channels
        ).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _gather_curves(
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(values.shape[-1])
        return values.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, rows),
        )

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        anchor_indices: torch.Tensor,
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
        feature_policy = str(feature_policy).strip().lower()
        if feature_policy not in self.FEATURE_POLICIES:
            raise ValueError(f"unsupported V16 feature policy: {feature_policy}")
        if row_value_features.ndim != 4:
            raise ValueError("V16 P2 rows must have shape [B,R,X,C]")
        batch, rows, _x_bins, channels = row_value_features.shape
        candidates = int(proposal_x_rows.shape[1])
        slots = int(anchor_indices.shape[1])
        if slots != self.num_slots:
            raise ValueError("V16 slot count mismatch")
        if int(channels) != self.dim:
            raise ValueError("V16 P2 feature dimension mismatch")
        if tuple(slot_states.shape[:2]) != (batch, slots):
            raise ValueError("V16 slot state shape mismatch")
        if tuple(anchor_x_rows.shape) != (batch, slots, rows):
            raise ValueError("V16 anchor x shape mismatch")
        if tuple(anchor_range_norm.shape) != (batch, slots, 2):
            raise ValueError("V16 anchor range shape mismatch")
        if tuple(anchor_geometry_valid.shape) != (batch, slots):
            raise ValueError("V16 anchor validity shape mismatch")
        if tuple(anchor_active.shape) != (batch, slots):
            raise ValueError("V16 anchor activity shape mismatch")
        if tuple(proposal_row_tokens.shape[:3]) != (batch, candidates, rows):
            raise ValueError("V16 proposal row-token shape mismatch")

        proposal_x_raw = proposal_x_rows.detach().float()
        proposal_x = torch.nan_to_num(
            proposal_x_raw,
            nan=0.0,
            posinf=float(max(self.input_w - 1, 1)),
            neginf=0.0,
        )
        proposal_range = sort_range_norm(proposal_range_norm.detach().float())
        candidate_valid = candidate_valid.detach().bool()
        geometry_valid = anchor_geometry_valid.detach().bool()
        source_active = anchor_active.detach().bool()
        safe_anchor = anchor_indices.detach().long().clamp(
            min=0, max=max(candidates - 1, 0)
        )
        anchor_reference_x = self._gather_curves(proposal_x, safe_anchor)
        anchor_reference_range = proposal_range.gather(
            1, safe_anchor.unsqueeze(-1).expand(-1, -1, 2)
        )
        row_fraction = fixed_row_fractions(
            rows, device=proposal_x.device, dtype=torch.float32
        )
        writer_rows = (
            (row_fraction.view(1, 1, rows) >= anchor_range_norm[..., :1])
            & (row_fraction.view(1, 1, rows) <= anchor_range_norm[..., 1:])
            & torch.isfinite(anchor_x_rows)
        )
        writer_valid = (
            source_active
            & geometry_valid
            & (writer_rows.sum(dim=-1) >= self.min_valid_rows)
        )
        groups = build_v16_anchor_candidate_groups(
            proposal_x_rows=proposal_x_raw,
            proposal_range_norm=proposal_range,
            candidate_valid=candidate_valid,
            anchor_indices=safe_anchor,
            anchor_active=writer_valid,
            input_w=self.input_w,
            config=self.group_config,
        )
        group_mask = groups["group_mask"]
        proposal_visible = groups["proposal_visible"]

        raw_features = row_value_features.detach()
        if feature_policy == "zero_content":
            raw_features = torch.zeros_like(raw_features)
        elif feature_policy == "x_reversed":
            raw_features = raw_features.flip(dims=(2,))
        elif feature_policy == "row_reversed":
            raw_features = raw_features.flip(dims=(1,))
        sampled = self._sample_candidate_evidence(raw_features, proposal_x)

        proposal_rows = proposal_row_tokens.detach()
        proposal_base = self.proposal_projection(
            self.proposal_norm(proposal_rows)
        )
        scale = float(max(self.input_w - 1, 1))
        candidate_geometry = torch.stack(
            (
                proposal_x / scale,
                proposal_range[..., 0].unsqueeze(-1).expand(-1, -1, rows),
                proposal_range[..., 1].unsqueeze(-1).expand(-1, -1, rows),
                proposal_visible.to(torch.float32),
                row_fraction.view(1, 1, rows).expand(batch, candidates, rows),
                torch.isfinite(proposal_x_raw).to(torch.float32),
                (
                    proposal_x[:, None, :, :]
                    - anchor_reference_x[:, :, None, :]
                )
                .abs()
                .amin(dim=1)
                .div(scale),
                proposal_x.diff(dim=-1, prepend=proposal_x[..., :1]).div(scale),
                proposal_x.diff(dim=-1, prepend=proposal_x[..., :1])
                .abs()
                .div(scale),
            ),
            dim=-1,
        )
        proposal_base = proposal_base + self.geometry_projection(candidate_geometry)
        normalized_evidence = self.feature_norm(sampled)
        evidence_keys = self.feature_key(normalized_evidence)
        evidence_values = self.feature_value(normalized_evidence)
        evidence_keys = evidence_keys + self.offset_embedding.view(
            1, 1, 1, -1, self.hidden_dim
        )
        query = self.evidence_query(proposal_base).unsqueeze(-2)
        evidence_attention = torch.softmax(
            (query * evidence_keys).sum(dim=-1)
            / math.sqrt(float(self.hidden_dim)),
            dim=-1,
        )
        evidence_context = (
            evidence_attention.unsqueeze(-1) * evidence_values
        ).sum(dim=-2)
        candidate_base = proposal_base + self.evidence_context(evidence_context)

        slot_state = self.slot_projection(
            self.slot_norm(slot_states.detach())
        ).view(batch, slots, 1, 1, self.hidden_dim)
        anchor_x = anchor_reference_x[:, :, None, :]
        candidate_x = proposal_x[:, None, :, :]
        distance = torch.nan_to_num(
            groups["anchor_distance_px"], posinf=scale, neginf=scale
        )
        corridor = groups["corridor_px"].clamp_min(1.0)
        relative_geometry = torch.stack(
            (
                anchor_x.expand(-1, -1, candidates, -1) / scale,
                candidate_x.expand(-1, slots, -1, -1) / scale,
                (candidate_x - anchor_x).expand(-1, -1, -1, -1) / scale,
                (candidate_x - anchor_x).abs().expand(-1, -1, -1, -1)
                / scale,
                row_fraction.view(1, 1, 1, rows).expand(
                    batch, slots, candidates, rows
                ),
                proposal_visible[:, None].expand(-1, slots, -1, -1).float(),
                proposal_range[:, None, :, 0:1].expand(-1, slots, -1, rows),
                proposal_range[:, None, :, 1:2].expand(-1, slots, -1, rows),
                (distance / corridor.unsqueeze(-1)).clamp(max=4.0)
                .unsqueeze(-1)
                .expand(-1, -1, -1, rows),
            ),
            dim=-1,
        )
        hidden = candidate_base[:, None] + slot_state
        hidden = hidden + self.relative_geometry_projection(relative_geometry)
        hidden = hidden + self.fusion_ffn(self.fusion_norm(hidden))
        visible = (
            proposal_visible[:, None]
            & group_mask.unsqueeze(-1)
            & writer_valid.unsqueeze(-1).unsqueeze(-1)
        )
        flat_hidden = hidden.reshape(
            batch * slots * candidates, rows, self.hidden_dim
        ).transpose(1, 2)
        flat_mask = visible.reshape(
            batch * slots * candidates, 1, rows
        ).to(flat_hidden.dtype)
        for block in self.row_blocks:
            flat_hidden = block(flat_hidden, flat_mask)
        hidden = flat_hidden.transpose(1, 2).reshape(
            batch, slots, candidates, rows, self.hidden_dim
        )

        perspective = (0.10 + 0.90 * row_fraction.pow(3.0)).view(
            1, 1, 1, rows
        )
        pool_logits = self.row_pool_score(hidden).squeeze(-1)
        pool_logits = pool_logits + perspective.clamp_min(1.0e-6).log()
        pool_logits = pool_logits.masked_fill(~visible, -1.0e4)
        row_probability = torch.softmax(pool_logits.float(), dim=-1)
        row_probability = row_probability * visible.to(row_probability.dtype)
        row_probability = row_probability / row_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        pooled = torch.einsum("bsnr,bsnrh->bsnh", row_probability, hidden.float())
        score_summary = torch.stack(
            (
                (distance / scale).clamp(max=2.0),
                (distance / corridor.unsqueeze(-1)).clamp(max=4.0),
                groups["group_size"].to(torch.float32).unsqueeze(-1)
                .expand(-1, -1, candidates)
                / float(max(candidates, 1)),
                proposal_visible.sum(dim=-1)[:, None].to(torch.float32)
                .expand(-1, slots, -1)
                / float(max(rows, 1)),
            ),
            dim=-1,
        )
        candidate_scores = self.score_head(
            torch.cat((self.score_norm(pooled), score_summary), dim=-1)
        ).squeeze(-1)
        candidate_scores = candidate_scores.masked_fill(~group_mask, -1.0e4)
        candidate_probability = torch.softmax(candidate_scores.float(), dim=-1)
        candidate_probability = candidate_probability * group_mask.to(
            candidate_probability.dtype
        )
        candidate_probability = candidate_probability / candidate_probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        selected_indices = candidate_scores.argmax(dim=-1)
        selected_indices = torch.where(
            writer_valid, selected_indices, safe_anchor
        )
        selected_x = self._gather_curves(proposal_x, selected_indices)
        selected_range = proposal_range.gather(
            1, selected_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        selected_scores = candidate_scores.gather(
            2, selected_indices.unsqueeze(-1)
        ).squeeze(-1)

        return {
            "selection_slot_v16_anchor_indices": safe_anchor,
            "selection_slot_v16_writer_valid": writer_valid,
            "selection_slot_v16_anchor_x_rows": anchor_x_rows.detach().float(),
            "selection_slot_v16_anchor_range_norm": sort_range_norm(
                anchor_range_norm.detach().float()
            ),
            "selection_slot_v16_anchor_reference_x_rows": anchor_reference_x,
            "selection_slot_v16_anchor_reference_range_norm": anchor_reference_range,
            "selection_slot_v16_group_mask": group_mask,
            "selection_slot_v16_group_size": groups["group_size"],
            "selection_slot_v16_anchor_distance_px": groups[
                "anchor_distance_px"
            ],
            "selection_slot_v16_corridor_px": groups["corridor_px"],
            "selection_slot_v16_candidate_scores": candidate_scores,
            "selection_slot_v16_candidate_probability": candidate_probability,
            "selection_slot_v16_selected_indices": selected_indices,
            "selection_slot_v16_selected_scores": selected_scores,
            "selection_slot_v16_selected_x_rows": selected_x,
            "selection_slot_v16_selected_range_norm": selected_range,
            "selection_slot_v16_evidence_attention": evidence_attention,
            "selection_slot_v16_row_probability": row_probability,
            "selection_slot_v16_feature_policy_id": candidate_scores.new_full(
                (batch,),
                float(sorted(self.FEATURE_POLICIES).index(feature_policy)),
            ),
        }
