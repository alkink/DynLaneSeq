from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x
from .dynlaneseq_s0 import DynLaneSeqEncoder, GeometryGuidedQueryRefiner
from .evidence import CurveAlignedSampler, DynamicOffsetFusion, EvidenceAdapter, MultiScaleCurveAlignedSampler, SamplerCurriculum
from .heads_s0 import ExistenceHead, RangeHead, S0Heads
from .row_token_decoder import RowTokenDecoder
from .structured_queries import build_structured_query_head


class ActiveCorridorSearch(nn.Module):
    """Supervised soft-argmax search over lateral evidence around coarse lanes."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 2.0,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px or [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0])
        if offsets.ndim != 1 or offsets.numel() < 3:
            raise ValueError("ActiveCorridorSearch expects at least three lateral offsets")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.num_offsets = int(offsets.numel())
        self.register_buffer("offsets_px", offsets.float())
        self.offset_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.num_offsets, self.dim))
        nn.init.normal_(self.offset_embedding, std=0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        bias = torch.zeros(self.num_offsets)
        center_idx = int((offsets.abs()).argmin().item())
        bias[center_idx] = float(center_init_bias)
        self.offset_bias = nn.Parameter(bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, n, p, o, c = offset_samples.shape
        if o != self.num_offsets:
            raise ValueError(f"Expected {self.num_offsets} offset samples, got {o}")
        if p != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {p}")
        query = queries.unsqueeze(2).unsqueeze(3).expand(b, n, p, o, c)
        row = row_embedding.view(1, 1, p, 1, c).expand(b, n, p, o, c)
        token = offset_samples + query + row + self.offset_embedding.to(dtype=offset_samples.dtype)
        logits = self.net(token).squeeze(-1) + self.offset_bias.to(device=token.device, dtype=token.dtype)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=offset_samples.dtype)
        evidence = (offset_samples * weights.unsqueeze(-1)).sum(dim=3)
        offsets = self.offsets_px.to(device=offset_samples.device, dtype=offset_samples.dtype)
        pred_delta = (weights * offsets.view(1, 1, 1, o)).sum(dim=-1)
        entropy = -(weights.float() * weights.float().clamp_min(1e-6).log()).sum(dim=-1)
        center_idx = int((self.offsets_px.abs()).argmin().item())
        debug = {
            "active_offset_entropy": entropy.detach().mean(),
            "active_offset_max_prob": weights.detach().float().max(dim=-1).values.mean(),
            "active_offset_center_prob": weights.detach().float()[..., center_idx].mean(),
            "active_pred_delta_abs": pred_delta.detach().abs().mean(),
        }
        return evidence, pred_delta, logits, debug


class LateralProfileBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("LateralProfileBlock expects an odd kernel_size")
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        hidden = self.norm(tokens)
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.pointwise(hidden)
        return residual + self.dropout(hidden)


class LateralProfileCorridorSearch(nn.Module):
    """Jointly scores the complete lateral feature profile for every lane row."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        profile_dim: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 0.0,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px or [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0])
        if offsets.ndim != 1 or offsets.numel() < 3:
            raise ValueError("LateralProfileCorridorSearch expects at least three lateral offsets")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.num_offsets = int(offsets.numel())
        self.profile_dim = int(profile_dim)
        self.register_buffer("offsets_px", offsets.float())
        self.sample_norm = nn.LayerNorm(self.dim)
        self.input_proj = nn.Linear(self.dim * 2, self.profile_dim)
        self.condition_norm = nn.LayerNorm(self.dim)
        self.condition_proj = nn.Linear(self.dim, self.profile_dim * 2)
        self.offset_embedding = nn.Parameter(torch.zeros(1, 1, 1, self.num_offsets, self.profile_dim))
        self.context_proj = nn.Linear(self.profile_dim * 2, self.profile_dim)
        self.blocks = nn.ModuleList(
            [
                LateralProfileBlock(self.profile_dim, kernel_size=int(kernel_size), dropout=float(dropout))
                for _ in range(int(num_blocks))
            ]
        )
        self.output_norm = nn.LayerNorm(self.profile_dim)
        self.output_head = nn.Linear(self.profile_dim, 1)
        nn.init.normal_(self.offset_embedding, std=0.02)
        if zero_init:
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
        bias = torch.zeros(self.num_offsets)
        center_idx = int(offsets.abs().argmin().item())
        bias[center_idx] = float(center_init_bias)
        self.offset_bias = nn.Parameter(bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        b, n, p, o, c = offset_samples.shape
        if o != self.num_offsets:
            raise ValueError(f"Expected {self.num_offsets} offset samples, got {o}")
        if p != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {p}")
        normalized = self.sample_norm(offset_samples)
        center_idx = int(self.offsets_px.abs().argmin().item())
        center = normalized[..., center_idx : center_idx + 1, :]
        relative = normalized - center
        tokens = self.input_proj(torch.cat([normalized, relative], dim=-1))

        row = row_embedding.view(1, 1, p, c)
        condition = self.condition_norm(queries.unsqueeze(2) + row)
        scale, bias = self.condition_proj(condition).chunk(2, dim=-1)
        tokens = tokens * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(3)) + bias.unsqueeze(3)
        tokens = tokens + self.offset_embedding.to(device=tokens.device, dtype=tokens.dtype)
        profile_context = self.context_proj(torch.cat([tokens.mean(dim=3), tokens.amax(dim=3)], dim=-1))
        tokens = tokens + profile_context.unsqueeze(3)

        hidden = tokens.reshape(b * n * p, o, self.profile_dim)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output_head(self.output_norm(hidden)).view(b, n, p, o).squeeze(-1)
        logits = logits + self.offset_bias.to(device=logits.device, dtype=logits.dtype)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=offset_samples.dtype)
        evidence = (offset_samples * weights.unsqueeze(-1)).sum(dim=3)
        offsets = self.offsets_px.to(device=offset_samples.device, dtype=offset_samples.dtype)
        pred_delta = (weights * offsets.view(1, 1, 1, o)).sum(dim=-1)
        entropy = -(weights.float() * weights.float().clamp_min(1e-6).log()).sum(dim=-1)
        debug = {
            "active_offset_entropy": entropy.detach().mean(),
            "active_offset_max_prob": weights.detach().float().max(dim=-1).values.mean(),
            "active_offset_center_prob": weights.detach().float()[..., center_idx].mean(),
            "active_pred_delta_abs": pred_delta.detach().abs().mean(),
            "active_profile_relative_abs": relative.detach().float().abs().mean(),
            "active_profile_context_abs": profile_context.detach().float().abs().mean(),
        }
        return evidence, pred_delta, logits, debug


class RowLateralProfileBlock(nn.Module):
    """Separable local reasoning over ordered lane rows and lateral samples."""

    def __init__(self, dim: int, row_kernel: int = 5, offset_kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        if row_kernel % 2 == 0 or offset_kernel % 2 == 0:
            raise ValueError("RowLateralProfileBlock expects odd kernel sizes")
        self.norm = nn.GroupNorm(1, dim)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=(row_kernel, offset_kernel),
            padding=(row_kernel // 2, offset_kernel // 2),
            groups=dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )
        self.dropout = nn.Dropout2d(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(self.norm(tokens))
        hidden = self.pointwise(hidden)
        return tokens + self.dropout(hidden)


class PixelOnlyCorridorSearch(nn.Module):
    """Single-stage lateral scorer with no query, row, or learned-position shortcut."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        profile_dim: int = 64,
        num_blocks: int = 2,
        row_kernel: int = 5,
        offset_kernel: int = 3,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 0.0,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px or [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0])
        if offsets.ndim != 1 or offsets.numel() < 3:
            raise ValueError("PixelOnlyCorridorSearch expects at least three lateral offsets")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.profile_dim = int(profile_dim)
        self.register_buffer("offsets_px", offsets.float())
        self.sample_norm = nn.LayerNorm(self.dim)
        # The fixed coordinate channel tells the network where a sample was taken
        # without providing a learnable per-slot or per-row answer prior.
        self.input_proj = nn.Linear(self.dim * 2 + 1, self.profile_dim)
        self.blocks = nn.ModuleList(
            [
                RowLateralProfileBlock(
                    self.profile_dim,
                    row_kernel=int(row_kernel),
                    offset_kernel=int(offset_kernel),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.output_norm = nn.LayerNorm(self.profile_dim)
        self.output_head = nn.Linear(self.profile_dim, 1)
        if zero_init:
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
        bias = torch.zeros(offsets.numel())
        bias[int(offsets.abs().argmin().item())] = float(center_init_bias)
        self.offset_bias = nn.Parameter(bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del queries, row_embedding
        b, n, p, o, c = offset_samples.shape
        if p != self.num_rows or o != self.offsets_px.numel() or c != self.dim:
            raise ValueError(
                f"Expected [B,N,{self.num_rows},{self.offsets_px.numel()},{self.dim}], "
                f"got {tuple(offset_samples.shape)}"
            )
        normalized = self.sample_norm(offset_samples)
        center_idx = int(self.offsets_px.abs().argmin().item())
        relative = normalized - normalized[..., center_idx : center_idx + 1, :]
        max_offset = self.offsets_px.abs().max().clamp_min(1.0)
        coordinate = (self.offsets_px / max_offset).to(device=normalized.device, dtype=normalized.dtype)
        coordinate = coordinate.view(1, 1, 1, o, 1).expand(b, n, p, o, 1)
        tokens = self.input_proj(torch.cat([normalized, relative, coordinate], dim=-1))
        hidden = tokens.permute(0, 1, 4, 2, 3).reshape(b * n, self.profile_dim, p, o)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.reshape(b, n, self.profile_dim, p, o).permute(0, 1, 3, 4, 2).contiguous()
        logits = self.output_head(self.output_norm(hidden)).squeeze(-1)
        logits = logits + self.offset_bias.to(device=logits.device, dtype=logits.dtype)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=offset_samples.dtype)
        offsets = self.offsets_px.to(device=offset_samples.device, dtype=offset_samples.dtype)
        pred_delta = (weights * offsets.view(1, 1, 1, o)).sum(dim=-1)
        evidence = (offset_samples * weights.unsqueeze(-1)).sum(dim=3)
        row_context = (hidden * weights.unsqueeze(-1)).sum(dim=3)
        entropy = -(weights.float() * weights.float().clamp_min(1e-6).log()).sum(dim=-1)
        return evidence, pred_delta, logits, {
            "active_offset_entropy": entropy.detach().mean(),
            "active_offset_max_prob": weights.detach().float().max(dim=-1).values.mean(),
            "active_offset_center_prob": weights.detach().float()[..., center_idx].mean(),
            "active_pred_delta_abs": pred_delta.detach().abs().mean(),
            "active_pixel_only_relative_abs": relative.detach().float().abs().mean(),
            "active_pixel_row_context": row_context,
        }


class EvidenceTowerBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.pointwise(self.depthwise(self.norm(features)))


class LaneEvidenceTowerCorridorSearch(nn.Module):
    """Trainable dense lane evidence plus a pixel-only signed-offset branch."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        evidence_dim: int = 64,
        tower_blocks: int = 2,
        profile_dim: int = 64,
        num_blocks: int = 2,
        row_kernel: int = 5,
        offset_kernel: int = 3,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 0.0,
        gate_hidden_dim: int = 64,
        gate_bias: float = -1.0,
        gate_enabled: bool = True,
    ):
        super().__init__()
        self.num_rows = int(num_rows)
        self.evidence_dim = int(evidence_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(int(dim), self.evidence_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.evidence_dim),
            nn.GELU(),
        )
        self.tower_blocks = nn.ModuleList(
            [EvidenceTowerBlock(self.evidence_dim, dropout=float(dropout)) for _ in range(int(tower_blocks))]
        )
        self.seg_head = nn.Conv2d(self.evidence_dim, 1, kernel_size=1)
        self.centerline_head = nn.Conv2d(self.evidence_dim, 1, kernel_size=1)
        self.scorer = PixelOnlyCorridorSearch(
            dim=self.evidence_dim,
            num_rows=self.num_rows,
            offsets_px=offsets_px,
            profile_dim=int(profile_dim),
            num_blocks=int(num_blocks),
            row_kernel=int(row_kernel),
            offset_kernel=int(offset_kernel),
            dropout=float(dropout),
            zero_init=bool(zero_init),
            center_init_bias=float(center_init_bias),
        )
        self.gate_enabled = bool(gate_enabled)
        if self.gate_enabled:
            gate_input_dim = int(profile_dim) + 3
            self.gate_head = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, int(gate_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(gate_hidden_dim), 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))
        else:
            self.gate_head = None

    @property
    def offsets_px(self) -> torch.Tensor:
        return self.scorer.offsets_px

    def build_evidence_map(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        evidence = self.stem(features)
        for block in self.tower_blocks:
            evidence = block(evidence)
        seg_logits = self.seg_head(evidence)
        centerline_logits = self.centerline_head(evidence)
        return evidence, {
            "active_tower_seg_logits": seg_logits,
            "active_tower_centerline_logits": centerline_logits,
            "active_tower_feature_abs": evidence.detach().float().abs().mean(),
        }

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        evidence, raw_delta, logits, debug = self.scorer(offset_samples, queries, row_embedding)
        row_context = debug.pop("active_pixel_row_context")
        if not self.gate_enabled:
            debug.update(
                {
                    "active_ungated_pred_delta_x_rows": raw_delta,
                    "active_gate_mean": raw_delta.new_tensor(1.0),
                    "active_gate_open_rate": raw_delta.new_tensor(1.0),
                }
            )
            return evidence, raw_delta, logits, debug
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
        max_prob = probs.max(dim=-1, keepdim=True).values
        max_offset = self.offsets_px.detach().abs().max().clamp_min(1.0)
        delta_magnitude = raw_delta.float().abs().unsqueeze(-1) / max_offset.to(
            device=raw_delta.device,
            dtype=torch.float32,
        )
        gate_input = torch.cat(
            [row_context, entropy.to(row_context.dtype), max_prob.to(row_context.dtype), delta_magnitude.to(row_context.dtype)],
            dim=-1,
        )
        gate_logits = self.gate_head(gate_input).squeeze(-1)
        gate = torch.sigmoid(gate_logits.float()).to(raw_delta.dtype)
        pred_delta = raw_delta * gate
        debug.update(
            {
                "active_ungated_pred_delta_x_rows": raw_delta,
                "active_gate_logits": gate_logits,
                "active_gate_mean": gate.detach().float().mean(),
                "active_gate_open_rate": (gate.detach().float() >= 0.5).float().mean(),
            }
        )
        return evidence, pred_delta, logits, debug


class CoarseToFineCorridorSearch(nn.Module):
    """Two-step geometry search with dense-prior and row-continuity evidence."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        fine_offsets_px: list[float] | None = None,
        profile_dim: int = 64,
        num_blocks: int = 3,
        row_kernel: int = 5,
        offset_kernel: int = 3,
        prior_channels: int = 2,
        dropout: float = 0.0,
        zero_init: bool = True,
        center_init_bias: float = 0.0,
    ):
        super().__init__()
        coarse_offsets = torch.tensor(offsets_px or [-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 32.0])
        fine_offsets = torch.tensor(fine_offsets_px or [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0])
        if coarse_offsets.ndim != 1 or coarse_offsets.numel() < 3:
            raise ValueError("CoarseToFineCorridorSearch expects at least three coarse offsets")
        if fine_offsets.ndim != 1 or fine_offsets.numel() < 3:
            raise ValueError("CoarseToFineCorridorSearch expects at least three fine offsets")
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.profile_dim = int(profile_dim)
        self.prior_channels = int(prior_channels)
        self.register_buffer("offsets_px", coarse_offsets.float())
        self.register_buffer("fine_offsets_px", fine_offsets.float())

        self.sample_norm = nn.LayerNorm(self.dim)
        input_dim = self.dim * 2 + self.prior_channels
        self.coarse_input = nn.Linear(input_dim, self.profile_dim)
        self.fine_input = nn.Linear(input_dim, self.profile_dim)
        self.condition_norm = nn.LayerNorm(self.dim)
        self.coarse_condition = nn.Linear(self.dim, self.profile_dim * 2)
        self.fine_condition = nn.Linear(self.dim, self.profile_dim * 2)
        self.coarse_offset_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, coarse_offsets.numel(), self.profile_dim)
        )
        self.fine_offset_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, fine_offsets.numel(), self.profile_dim)
        )
        self.coarse_blocks = nn.ModuleList(
            [
                RowLateralProfileBlock(
                    self.profile_dim,
                    row_kernel=int(row_kernel),
                    offset_kernel=int(offset_kernel),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.fine_blocks = nn.ModuleList(
            [
                RowLateralProfileBlock(
                    self.profile_dim,
                    row_kernel=int(row_kernel),
                    offset_kernel=int(offset_kernel),
                    dropout=float(dropout),
                )
                for _ in range(int(num_blocks))
            ]
        )
        self.coarse_norm = nn.LayerNorm(self.profile_dim)
        self.fine_norm = nn.LayerNorm(self.profile_dim)
        self.coarse_head = nn.Linear(self.profile_dim, 1)
        self.fine_head = nn.Linear(self.profile_dim, 1)
        nn.init.normal_(self.coarse_offset_embedding, std=0.02)
        nn.init.normal_(self.fine_offset_embedding, std=0.02)
        if zero_init:
            nn.init.zeros_(self.coarse_head.weight)
            nn.init.zeros_(self.coarse_head.bias)
            nn.init.zeros_(self.fine_head.weight)
            nn.init.zeros_(self.fine_head.bias)
        coarse_bias = torch.zeros(coarse_offsets.numel())
        fine_bias = torch.zeros(fine_offsets.numel())
        coarse_bias[int(coarse_offsets.abs().argmin().item())] = float(center_init_bias)
        fine_bias[int(fine_offsets.abs().argmin().item())] = float(center_init_bias)
        self.coarse_offset_bias = nn.Parameter(coarse_bias)
        self.fine_offset_bias = nn.Parameter(fine_bias)

    def _score_stage(
        self,
        samples: torch.Tensor,
        priors: torch.Tensor | None,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
        offsets: torch.Tensor,
        input_proj: nn.Linear,
        condition_proj: nn.Linear,
        offset_embedding: torch.Tensor,
        blocks: nn.ModuleList,
        output_norm: nn.LayerNorm,
        output_head: nn.Linear,
        offset_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, p, o, c = samples.shape
        normalized = self.sample_norm(samples)
        center_idx = int(offsets.abs().argmin().item())
        relative = normalized - normalized[..., center_idx : center_idx + 1, :]
        if priors is None:
            priors = normalized.new_zeros((b, n, p, o, self.prior_channels))
        if priors.shape[:-1] != normalized.shape[:-1] or priors.shape[-1] != self.prior_channels:
            raise ValueError(
                f"Expected dense priors [B,N,R,O,{self.prior_channels}], got {tuple(priors.shape)}"
            )
        tokens = input_proj(torch.cat([normalized, relative, priors.to(dtype=normalized.dtype)], dim=-1))
        condition = self.condition_norm(queries.unsqueeze(2) + row_embedding.view(1, 1, p, c))
        scale, bias = condition_proj(condition).chunk(2, dim=-1)
        tokens = tokens * (1.0 + 0.1 * torch.tanh(scale).unsqueeze(3)) + bias.unsqueeze(3)
        tokens = tokens + offset_embedding.to(device=tokens.device, dtype=tokens.dtype)
        hidden = tokens.permute(0, 1, 4, 2, 3).reshape(b * n, self.profile_dim, p, o)
        for block in blocks:
            hidden = block(hidden)
        hidden = hidden.reshape(b, n, self.profile_dim, p, o).permute(0, 1, 3, 4, 2).contiguous()
        logits = output_head(output_norm(hidden)).squeeze(-1)
        logits = logits + offset_bias.to(device=logits.device, dtype=logits.dtype)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=samples.dtype)
        stage_offsets = offsets.to(device=samples.device, dtype=samples.dtype)
        delta = (weights * stage_offsets.view(1, 1, 1, o)).sum(dim=-1)
        evidence = (samples * weights.unsqueeze(-1)).sum(dim=3)
        return evidence, delta, logits, weights

    def score_coarse(
        self,
        samples: torch.Tensor,
        priors: torch.Tensor | None,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._score_stage(
            samples,
            priors,
            queries,
            row_embedding,
            self.offsets_px,
            self.coarse_input,
            self.coarse_condition,
            self.coarse_offset_embedding,
            self.coarse_blocks,
            self.coarse_norm,
            self.coarse_head,
            self.coarse_offset_bias,
        )

    def score_fine(
        self,
        samples: torch.Tensor,
        priors: torch.Tensor | None,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._score_stage(
            samples,
            priors,
            queries,
            row_embedding,
            self.fine_offsets_px,
            self.fine_input,
            self.fine_condition,
            self.fine_offset_embedding,
            self.fine_blocks,
            self.fine_norm,
            self.fine_head,
            self.fine_offset_bias,
        )


class RowConvRefinerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("RowConvRefinerBlock expects an odd kernel_size")
        padding = (kernel_size // 2) * int(dilation)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=int(dilation), groups=dim)
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
        )
        self.drop = nn.Dropout(float(dropout))
        if zero_init:
            nn.init.zeros_(self.pointwise[-1].weight)
            nn.init.zeros_(self.pointwise[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        x = self.norm(tokens).transpose(1, 2).contiguous()
        x = self.depthwise(x)
        x = self.pointwise(x).transpose(1, 2).contiguous()
        return residual + self.drop(x)


class ContinuousSequenceRefiner(nn.Module):
    """1D row-sequence S2 refiner that preserves lateral evidence profiles."""

    def __init__(
        self,
        dim: int = 256,
        num_rows: int = 72,
        x_bins: int = 200,
        num_offsets: int = 5,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        kernel_size: int = 9,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_rows = int(num_rows)
        self.x_bins = int(x_bins)
        self.num_offsets = int(num_offsets)
        if self.num_offsets <= 0:
            raise ValueError("ContinuousSequenceRefiner expects at least one offset")
        self.offset_fuser = nn.Sequential(
            nn.LayerNorm(self.dim * self.num_offsets),
            nn.Linear(self.dim * self.num_offsets, self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.dim, self.dim),
        )
        dilation_values = list(dilations or [])
        if not dilation_values:
            dilation_values = [1] * int(num_blocks)
        if len(dilation_values) < int(num_blocks):
            dilation_values = dilation_values + [dilation_values[-1]] * (int(num_blocks) - len(dilation_values))
        self.blocks = nn.ModuleList(
            [
                RowConvRefinerBlock(
                    dim=self.dim,
                    hidden_dim=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation_values[idx]),
                    dropout=float(dropout),
                    zero_init=bool(zero_init),
                )
                for idx in range(int(num_blocks))
            ]
        )
        self.out_norm = nn.LayerNorm(self.dim)
        self.delta_head = nn.Linear(self.dim, self.x_bins)
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        if zero_init:
            nn.init.zeros_(self.offset_fuser[-1].weight)
            nn.init.zeros_(self.offset_fuser[-1].bias)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.quality_head[-1].weight)
            nn.init.zeros_(self.quality_head[-1].bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
        stage_extra: torch.Tensor | None = None,
        base_row_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, n, p, o, c = offset_samples.shape
        if p != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {p}")
        if o != self.num_offsets:
            raise ValueError(f"Expected {self.num_offsets} offsets, got {o}")
        if c != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {c}")
        fused_evidence = self.offset_fuser(offset_samples.reshape(b, n, p, o * c))
        if base_row_tokens is None:
            tokens = queries.unsqueeze(2) + row_embedding.view(1, 1, p, c)
        else:
            tokens = base_row_tokens
        tokens = tokens + fused_evidence
        if stage_extra is not None:
            tokens = tokens + stage_extra
        flat = tokens.reshape(b * n, p, c)
        for block in self.blocks:
            flat = block(flat)
        hidden = self.out_norm(flat).reshape(b, n, p, c)
        delta_logits = self.delta_head(hidden)
        pooled = torch.cat([hidden.mean(dim=2), hidden.amax(dim=2)], dim=-1)
        quality_delta = self.quality_head(pooled).squeeze(-1)
        return {
            "row_x_logits": delta_logits,
            "row_hidden": hidden,
            "quality_delta": quality_delta,
            "fused_evidence": fused_evidence,
        }


class DynLaneSeqS2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        evidence_cfg = model_cfg.get("evidence_sampler", {})
        multi_scale_cfg = model_cfg.get("multi_scale_evidence", {})
        active_cfg = model_cfg.get("active_corridor", {})
        geometry_cfg = model_cfg.get("s0_geometry_evidence", {})
        oracle_cfg = model_cfg.get("oracle_coarse", {})
        structured_cfg = model_cfg.get("structured_query", {})
        decision_cfg = model_cfg.get("s2_decision_refiner", {})
        csr_cfg = model_cfg.get("s2_csr", {})
        self.input_w = int(model_cfg.get("input_w", 800))
        self.input_h = int(model_cfg.get("input_h", 288))
        self.num_rows = int(model_cfg.get("num_rows", 72))
        self.x_bins = int(model_cfg.get("x_bins", 200))
        dim = int(model_cfg.get("dim", 256))
        self.structured_query_head = build_structured_query_head(model_cfg)
        self.active_corridor_enabled = bool(active_cfg.get("enabled", False))
        self.active_corridor_scorer_type = str(active_cfg.get("scorer_type", "independent_mlp")).lower()
        self.oracle_coarse_enabled = bool(oracle_cfg.get("enabled", False))
        self.oracle_score_logit = float(oracle_cfg.get("score_logit", 8.0))
        self.oracle_bg_logit = float(oracle_cfg.get("background_logit", 8.0))
        self.s2_decision_enabled = bool(decision_cfg.get("enabled", False))
        self.s2_decision_detach_base = bool(decision_cfg.get("detach_base", True))
        self.s2_decision_exist_delta_scale = float(decision_cfg.get("exist_delta_scale", 1.0))
        self.s2_decision_quality_base = str(decision_cfg.get("quality_base", "coarse")).lower()
        self.s2_refiner_type = str(model_cfg.get("s2_refiner_type", "row_transformer")).lower()
        self.s2_csr_enabled = self.s2_refiner_type in {"csr", "csr_conv1d", "continuous_sequence_refiner"}
        self.s2_csr_quality_base = str(csr_cfg.get("quality_base", "coarse")).lower()
        self.s2_csr_detach_quality_base = bool(csr_cfg.get("detach_quality_base", True))
        self.active_corridor_detach_center = bool(active_cfg.get("detach_center", True))
        self.active_corridor_detach_refined_x = bool(active_cfg.get("detach_refined_x_for_decoder", False))
        self.active_corridor_authoritative_geometry = bool(active_cfg.get("authoritative_geometry", False))
        self.active_corridor_detach_coarse_for_fine = bool(active_cfg.get("detach_coarse_for_fine", True))
        self.active_corridor_use_dense_priors = bool(active_cfg.get("use_dense_priors", False))
        self.active_corridor_freeze_non_active = bool(active_cfg.get("freeze_non_active_modules", False))
        self.active_corridor_skip_row_decoder = bool(active_cfg.get("skip_row_decoder", False))
        self.active_corridor_train_center_jitter_px = float(active_cfg.get("train_center_jitter_px", 0.0))
        self.active_corridor_train_center_jitter_knots = int(active_cfg.get("train_center_jitter_knots", 5))
        self.active_corridor_train_center_jitter_prob = float(active_cfg.get("train_center_jitter_prob", 1.0))
        local_window_cfg = evidence_cfg.get("local_window", {})
        self.dynamic_offset_enabled = bool(local_window_cfg.get("enabled", False)) and str(
            local_window_cfg.get("aggregation", "mean")
        ).lower() in {"dynamic", "learned", "token"}
        self.multi_scale_enabled = bool(multi_scale_cfg.get("enabled", False))
        self.multi_scale_return_separate = bool(multi_scale_cfg.get("return_separate", False))
        self.multi_scale_scales = list(multi_scale_cfg.get("scales", ["p2", "p3", "p4"]))
        if self.multi_scale_enabled and self.dynamic_offset_enabled:
            raise ValueError("multi_scale_evidence and dynamic local-window aggregation should be ablated separately")
        self.s2_mode = str(model_cfg.get("s2_mode", "direct")).lower()
        if self.s2_mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported S2 mode: {self.s2_mode}")
        if self.s2_refiner_type not in {"row_transformer", "transformer", "csr", "csr_conv1d", "continuous_sequence_refiner"}:
            raise ValueError(f"Unsupported s2_refiner_type: {self.s2_refiner_type}")
        if self.s2_decision_enabled and self.s2_mode != "residual":
            raise ValueError("s2_decision_refiner currently requires residual S2 mode")
        if self.s2_csr_enabled and self.s2_mode != "residual":
            raise ValueError("s2_refiner_type=csr_conv1d currently requires residual S2 mode")
        if self.s2_csr_enabled and self.s2_decision_enabled:
            raise ValueError("s2_csr and s2_decision_refiner should be ablated separately")
        if self.s2_csr_enabled and (self.multi_scale_enabled or self.dynamic_offset_enabled or self.active_corridor_enabled):
            raise ValueError("s2_csr should be isolated from multi-scale, dynamic offset fusion, and active corridor")
        if self.active_corridor_authoritative_geometry and not self.active_corridor_enabled:
            raise ValueError("active_corridor.authoritative_geometry requires active_corridor.enabled=true")
        if self.active_corridor_scorer_type not in {
            "independent_mlp",
            "mlp",
            "lateral_profile",
            "profile_conv1d",
            "coarse_to_fine",
            "coarse_to_fine_2d",
            "pixel_only",
            "pixel_only_2d",
            "lane_evidence_tower",
            "pixel_evidence_tower",
        }:
            raise ValueError(f"Unsupported active_corridor.scorer_type: {self.active_corridor_scorer_type}")
        if self.s2_decision_quality_base not in {"coarse", "none", "zero", ""}:
            raise ValueError(f"Unsupported s2_decision_refiner.quality_base: {self.s2_decision_quality_base}")
        if self.s2_csr_quality_base not in {"coarse", "none", "zero", ""}:
            raise ValueError(f"Unsupported s2_csr.quality_base: {self.s2_csr_quality_base}")
        if self.structured_query_head is not None and self.s2_mode != "residual":
            raise ValueError("structured_query currently requires residual S2/S3 mode")
        if bool(geometry_cfg.get("enabled", False)) and bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False)):
            raise ValueError("Use either dynamic_evidence v1 or s0_geometry_evidence v2, not both")
        if bool(geometry_cfg.get("enabled", False)) and self.s2_mode != "residual":
            raise ValueError("s0_geometry_evidence currently requires residual S2/S3 mode")
        if self.structured_query_head is not None and (
            bool(geometry_cfg.get("enabled", False))
            or bool(model_cfg.get("dynamic_evidence", {}).get("enabled", False))
            or bool(model_cfg.get("dynamic_proposal", {}).get("enabled", False))
            or self.oracle_coarse_enabled
        ):
            raise ValueError(
                "structured_query must be isolated from dynamic_evidence, dynamic_proposal, s0_geometry_evidence, and oracle_coarse"
            )
        if self.structured_query_head is not None and int(structured_cfg.get("num_instances", model_cfg.get("num_slots", 20))) != int(
            model_cfg.get("num_slots", 20)
        ):
            raise ValueError("structured_query.num_instances must match model.num_slots for S1/S2/S3 stages")
        self.encoder = DynLaneSeqEncoder(cfg)
        self.freeze_s0_frontend = bool(model_cfg.get("freeze_s0_frontend", False))
        if self.s2_mode == "residual":
            needs_residual_heads = self.structured_query_head is None and (
                not self.oracle_coarse_enabled or bool(geometry_cfg.get("enabled", False))
            )
            self.heads = (
                S0Heads(
                    dim=dim,
                    num_rows=self.num_rows,
                    x_bins=self.x_bins,
                    input_w=self.input_w,
                )
                if needs_residual_heads
                else None
            )
            self.coarse_x_embed = nn.Linear(1, dim)
            self.residual_logit_scale = float(model_cfg.get("residual_logit_scale", 1.0))
            self.detach_coarse_x = bool(model_cfg.get("detach_coarse_x", False))
        else:
            self.exist_head = ExistenceHead(dim)
            self.range_head = RangeHead(dim)
        self.sampler = CurveAlignedSampler(
            input_w=self.input_w,
            input_h=self.input_h,
            num_rows=self.num_rows,
            local_window_enabled=bool(local_window_cfg.get("enabled", False)) and not self.dynamic_offset_enabled,
            offsets_px=local_window_cfg.get("offsets_px", [-8, -4, 0, 4, 8]),
        )
        self.active_corridor_sampler = (
            CurveAlignedSampler(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                local_window_enabled=False,
                offsets_px=active_cfg.get("offsets_px", [-32, -24, -16, -8, 0, 8, 16, 24, 32]),
            )
            if self.active_corridor_enabled
            else None
        )
        self.active_corridor_fine_sampler = (
            CurveAlignedSampler(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                local_window_enabled=False,
                offsets_px=active_cfg.get("fine_offsets_px", [-8, -4, -2, 0, 2, 4, 8]),
            )
            if self.active_corridor_enabled and self.active_corridor_scorer_type in {"coarse_to_fine", "coarse_to_fine_2d"}
            else None
        )
        self.active_corridor = None
        if self.active_corridor_enabled:
            active_kwargs = {
                "dim": dim,
                "num_rows": self.num_rows,
                "offsets_px": active_cfg.get("offsets_px", [-32, -24, -16, -8, 0, 8, 16, 24, 32]),
                "dropout": float(active_cfg.get("dropout", 0.0)),
                "zero_init": bool(active_cfg.get("zero_init", True)),
                "center_init_bias": float(active_cfg.get("center_init_bias", 2.0)),
            }
            if self.active_corridor_scorer_type in {"coarse_to_fine", "coarse_to_fine_2d"}:
                self.active_corridor = CoarseToFineCorridorSearch(
                    **active_kwargs,
                    fine_offsets_px=active_cfg.get("fine_offsets_px", [-8, -4, -2, 0, 2, 4, 8]),
                    profile_dim=int(active_cfg.get("profile_dim", 64)),
                    num_blocks=int(active_cfg.get("profile_blocks", 3)),
                    row_kernel=int(active_cfg.get("row_kernel_size", 5)),
                    offset_kernel=int(active_cfg.get("profile_kernel_size", 3)),
                    prior_channels=int(active_cfg.get("prior_channels", 2)),
                )
            elif self.active_corridor_scorer_type in {"pixel_only", "pixel_only_2d"}:
                self.active_corridor = PixelOnlyCorridorSearch(
                    **active_kwargs,
                    profile_dim=int(active_cfg.get("profile_dim", 64)),
                    num_blocks=int(active_cfg.get("profile_blocks", 2)),
                    row_kernel=int(active_cfg.get("row_kernel_size", 5)),
                    offset_kernel=int(active_cfg.get("profile_kernel_size", 3)),
                )
            elif self.active_corridor_scorer_type in {"lane_evidence_tower", "pixel_evidence_tower"}:
                self.active_corridor = LaneEvidenceTowerCorridorSearch(
                    **active_kwargs,
                    evidence_dim=int(active_cfg.get("evidence_dim", 64)),
                    tower_blocks=int(active_cfg.get("tower_blocks", 2)),
                    profile_dim=int(active_cfg.get("profile_dim", 64)),
                    num_blocks=int(active_cfg.get("profile_blocks", 2)),
                    row_kernel=int(active_cfg.get("row_kernel_size", 5)),
                    offset_kernel=int(active_cfg.get("profile_kernel_size", 3)),
                    gate_hidden_dim=int(active_cfg.get("gate_hidden_dim", 64)),
                    gate_bias=float(active_cfg.get("gate_bias", -1.0)),
                    gate_enabled=bool(active_cfg.get("gate_enabled", True)),
                )
            elif self.active_corridor_scorer_type in {"lateral_profile", "profile_conv1d"}:
                self.active_corridor = LateralProfileCorridorSearch(
                    **active_kwargs,
                    profile_dim=int(active_cfg.get("profile_dim", 64)),
                    num_blocks=int(active_cfg.get("profile_blocks", 2)),
                    kernel_size=int(active_cfg.get("profile_kernel_size", 3)),
                )
            else:
                self.active_corridor = ActiveCorridorSearch(
                    **active_kwargs,
                    hidden_dim=int(active_cfg.get("hidden_dim", dim)),
                )
        self.offset_fusion = (
            DynamicOffsetFusion(
                dim=dim,
                num_offsets=len(self.sampler.offsets_px),
                hidden_dim=int(local_window_cfg.get("hidden_dim", dim)),
                dropout=float(local_window_cfg.get("dropout", 0.0)),
                zero_init=bool(local_window_cfg.get("zero_init", True)),
            )
            if self.dynamic_offset_enabled
            else None
        )
        self.multi_scale_sampler = (
            MultiScaleCurveAlignedSampler(
                input_w=self.input_w,
                input_h=self.input_h,
                num_rows=self.num_rows,
                dim=dim,
                scales=self.multi_scale_scales,
                gate_hidden_dim=int(multi_scale_cfg.get("gate_hidden_dim", dim)),
                dropout=float(multi_scale_cfg.get("dropout", 0.0)),
                zero_init_gate=bool(multi_scale_cfg.get("zero_init_gate", True)),
                fusion_mode=str(multi_scale_cfg.get("fusion_mode", "weighted_sum")),
                base_scale=str(multi_scale_cfg.get("base_scale", "p2")),
                residual_scale_init=float(multi_scale_cfg.get("residual_scale_init", 0.0)),
                initial_gate_bias=multi_scale_cfg.get("initial_gate_bias"),
            )
            if self.multi_scale_enabled and not self.multi_scale_return_separate
            else None
        )
        self.curriculum = SamplerCurriculum(
            noise_std=float(evidence_cfg.get("noise_std", 3.0)),
            detach_sample_coords=bool(evidence_cfg.get("detach_sample_coords", True)),
            input_w=self.input_w,
        )
        self.s0_geometry_detach_draft = bool(geometry_cfg.get("detach_draft_x", True))
        self.s0_geometry_refiner = (
            GeometryGuidedQueryRefiner(
                dim=dim,
                input_h=self.input_h,
                input_w=self.input_w,
                num_rows=self.num_rows,
                hidden_dim=int(geometry_cfg.get("hidden_dim", dim)),
                dropout=float(geometry_cfg.get("dropout", 0.0)),
                pooling=str(geometry_cfg.get("pooling", "mean")),
                local_window_enabled=bool(geometry_cfg.get("local_window_enabled", False)),
                offsets_px=geometry_cfg.get("offsets_px"),
                local_reduce=str(geometry_cfg.get("local_reduce", "max")),
            )
            if bool(geometry_cfg.get("enabled", False))
            else None
        )
        self.adapter = None if self.s2_csr_enabled else EvidenceAdapter(dim=dim, gamma_init=float(model_cfg.get("evidence_gamma_init", 0.1)))
        self.csr_refiner = (
            ContinuousSequenceRefiner(
                dim=dim,
                num_rows=self.num_rows,
                x_bins=self.x_bins,
                num_offsets=len(self.sampler.offsets_px),
                hidden_dim=int(csr_cfg.get("hidden_dim", dim * 2)),
                num_blocks=int(csr_cfg.get("num_blocks", 3)),
                kernel_size=int(csr_cfg.get("kernel_size", 9)),
                dilations=list(csr_cfg.get("dilations", [])),
                dropout=float(csr_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
                zero_init=bool(csr_cfg.get("zero_init", True)),
            )
            if self.s2_csr_enabled
            else None
        )
        if self.s2_decision_enabled:
            decision_hidden_dim = int(decision_cfg.get("hidden_dim", dim))
            decision_dropout = float(decision_cfg.get("dropout", 0.0))
            self.s2_decision_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, decision_hidden_dim),
                nn.GELU(),
                nn.Dropout(decision_dropout),
                nn.Linear(decision_hidden_dim, 3),
            )
            if bool(decision_cfg.get("zero_init", True)):
                nn.init.zeros_(self.s2_decision_head[-1].weight)
                nn.init.zeros_(self.s2_decision_head[-1].bias)
        else:
            self.s2_decision_head = None
        self.row_embedding = nn.Embedding(self.num_rows, dim)
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        self.row_decoder = (
            RowTokenDecoder(
                num_rows=self.num_rows,
                dim=dim,
                x_bins=self.x_bins,
                num_layers=int(model_cfg.get("row_decoder_layers", 2)),
                num_heads=int(model_cfg.get("num_heads", 8)),
                ff_dim=int(model_cfg.get("row_decoder_ff_dim", 512)),
                dropout=float(model_cfg.get("dropout", 0.1)),
                zero_init_head=bool(model_cfg.get("zero_init_residual_head", self.s2_mode == "residual")),
                local_attn_window=int(model_cfg.get("row_local_attn_window", 0)),
                visibility_head=bool(model_cfg.get("row_visibility", {}).get("enabled", False)),
            )
            if not self.s2_csr_enabled
            else None
        )
        if self.active_corridor_enabled and (self.multi_scale_enabled or self.dynamic_offset_enabled):
            raise ValueError("active_corridor should be tested separately from multi-scale and dynamic offset fusion")
        if self.freeze_s0_frontend:
            self._freeze_s0_frontend()
        if self.active_corridor_freeze_non_active:
            self._freeze_non_active_corridor()

    def _freeze_s0_frontend(self) -> None:
        for module in [self.encoder, self.structured_query_head, getattr(self, "heads", None)]:
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def _freeze_non_active_corridor(self) -> None:
        if self.active_corridor is None:
            raise ValueError("freeze_non_active_modules requires active_corridor.enabled=true")
        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith("active_corridor."))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.active_corridor_freeze_non_active:
            super().train(False)
            if self.active_corridor is not None:
                self.active_corridor.train(mode)
            return self
        if mode and self.freeze_s0_frontend:
            self.encoder.eval()
            if self.structured_query_head is not None:
                self.structured_query_head.eval()
            if getattr(self, "heads", None) is not None:
                self.heads.eval()
        return self

    def apply_s2_decision_refiner(
        self,
        coarse: dict[str, torch.Tensor],
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.s2_decision_head is None:
            return coarse["exist_logits"], coarse["quality_logits"], {}
        evidence_pooled = evidence.mean(dim=2)
        decision = self.s2_decision_head(evidence_pooled)
        exist_delta = decision[..., :2] * self.s2_decision_exist_delta_scale
        quality_delta = decision[..., 2]
        base_exist = coarse["exist_logits"].detach() if self.s2_decision_detach_base else coarse["exist_logits"]
        exist_logits = base_exist + exist_delta
        if self.s2_decision_quality_base == "coarse":
            base_quality = coarse["quality_logits"].detach() if self.s2_decision_detach_base else coarse["quality_logits"]
            quality_logits = base_quality + quality_delta
        else:
            quality_logits = quality_delta
        debug = {
            "s2_decision_evidence_abs": evidence_pooled.detach().abs().mean(),
            "s2_decision_delta_exist_abs": exist_delta.detach().abs().mean(),
            "s2_decision_delta_quality_abs": quality_delta.detach().abs().mean(),
        }
        return exist_logits, quality_logits, debug

    def build_csr_quality_logits(
        self,
        coarse: dict[str, torch.Tensor],
        quality_delta: torch.Tensor,
    ) -> torch.Tensor:
        if self.s2_csr_quality_base == "coarse":
            base_quality = coarse["quality_logits"].detach() if self.s2_csr_detach_quality_base else coarse["quality_logits"]
            return base_quality + quality_delta
        return quality_delta

    def build_coarse_tokens(
        self,
        queries: torch.Tensor,
        base_row_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_row_tokens is not None:
            return base_row_tokens
        b, n, d = queries.shape
        row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
        return queries.unsqueeze(2) + row_emb

    def build_oracle_coarse(
        self,
        queries: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        b, n, _ = queries.shape
        device = queries.device
        dtype = queries.dtype
        p = self.num_rows
        x_bins = self.x_bins
        lane_logit = float(self.oracle_score_logit)
        bg_logit = float(self.oracle_bg_logit)
        exist_logits = queries.new_empty((b, n, 2))
        exist_logits[..., 0] = -bg_logit
        exist_logits[..., 1] = bg_logit
        pred_x_rows = queries.new_zeros((b, n, p))
        range_norm = queries.new_zeros((b, n, 2))
        row_x_logits = queries.new_full((b, n, p, x_bins), -lane_logit)
        quality_logits = queries.new_full((b, n), -bg_logit)
        for bi, target in enumerate(targets):
            x_rows = target["x_rows"].to(device=device, dtype=dtype)
            valid = target["valid_mask"].to(device=device).bool()
            x_bin_targets = target["x_bins"].to(device=device).long()
            range_y = target["range_y"].to(device=device, dtype=dtype)
            lanes = min(int(x_rows.shape[0]), n)
            if lanes <= 0:
                continue
            exist_logits[bi, :lanes, 0] = lane_logit
            exist_logits[bi, :lanes, 1] = -lane_logit
            pred_x_rows[bi, :lanes] = x_rows[:lanes].clamp(0, self.input_w - 1)
            range_norm[bi, :lanes] = (range_y[:lanes] / float(self.input_h)).clamp(0.0, 1.0)
            quality_logits[bi, :lanes] = lane_logit
            bins = x_bin_targets[:lanes].clamp(0, x_bins - 1)
            row_x_logits[bi, :lanes].scatter_(-1, bins.unsqueeze(-1), lane_logit)
            row_x_logits[bi, :lanes] = row_x_logits[bi, :lanes].masked_fill(~valid[:lanes].unsqueeze(-1), 0.0)
        return {
            "exist_logits": exist_logits,
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_norm,
            "range_norm": range_norm,
            "quality_logits": quality_logits,
            "quality_pred_x_rows": pred_x_rows,
        }

    def bridge_evidence(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return evidence, {}

    def sample_evidence(
        self,
        features: torch.Tensor | dict[str, torch.Tensor],
        sample_x: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.multi_scale_enabled and self.multi_scale_return_separate:
            if not isinstance(features, dict):
                raise TypeError("multi_scale_evidence.return_separate requires encoder multi_scale_features")
            evidence = {}
            debug = {}
            for scale_name in self.multi_scale_scales:
                if scale_name not in features:
                    raise KeyError(f"Missing multi-scale feature: {scale_name}")
                evidence[scale_name] = self.sampler(features[scale_name], sample_x)
                debug[f"ms_raw_{scale_name}_abs"] = evidence[scale_name].abs().mean().detach()
            return evidence, debug
        if self.multi_scale_sampler is not None:
            if not isinstance(features, dict):
                raise TypeError("multi_scale_evidence requires encoder multi_scale_features")
            return self.multi_scale_sampler(features, sample_x, queries, self.row_embedding.weight)
        if self.offset_fusion is None:
            if not isinstance(features, torch.Tensor):
                raise TypeError("single-scale evidence expects a feature tensor")
            return self.sampler(features, sample_x), {}
        if not isinstance(features, torch.Tensor):
            raise TypeError("dynamic offset fusion expects a single-scale feature tensor")
        offset_samples = self.sampler.sample_local_window(features, sample_x)
        evidence, offset_debug = self.offset_fusion(offset_samples, queries, self.row_embedding.weight)
        return evidence, offset_debug

    def sample_active_corridor(
        self,
        features: torch.Tensor,
        coarse_x: torch.Tensor,
        queries: torch.Tensor,
        seg_logits: torch.Tensor | None = None,
        centerline_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.active_corridor is None or self.active_corridor_sampler is None:
            raise RuntimeError("active_corridor is not enabled")
        base_center_x = coarse_x.detach() if self.active_corridor_detach_center else coarse_x
        center_jitter = torch.zeros_like(base_center_x)
        # The parent model is intentionally kept in eval mode when every module
        # except Active Corridor is frozen. Use the head's mode for train-only jitter.
        if self.active_corridor.training and self.active_corridor_train_center_jitter_px > 0:
            b, n, p = base_center_x.shape
            knots = max(int(self.active_corridor_train_center_jitter_knots), 2)
            random_knots = base_center_x.new_empty((b * n, 1, knots)).uniform_(-1.0, 1.0)
            center_jitter = F.interpolate(random_knots, size=p, mode="linear", align_corners=True).view(b, n, p)
            center_jitter = center_jitter * float(self.active_corridor_train_center_jitter_px)
            jitter_prob = min(max(float(self.active_corridor_train_center_jitter_prob), 0.0), 1.0)
            if jitter_prob < 1.0:
                jitter_mask = (torch.rand((b, n, 1), device=base_center_x.device) < jitter_prob).to(base_center_x.dtype)
                center_jitter = center_jitter * jitter_mask
        center_x = (base_center_x + center_jitter).clamp(0.0, float(self.input_w - 1))
        center_jitter = center_x - base_center_x
        sampling_features = features
        tower_debug: dict[str, torch.Tensor] = {}
        if hasattr(self.active_corridor, "build_evidence_map"):
            sampling_features, tower_debug = self.active_corridor.build_evidence_map(features)
        offset_samples = self.active_corridor_sampler.sample_local_window(sampling_features, center_x)
        if isinstance(self.active_corridor, CoarseToFineCorridorSearch):
            if self.active_corridor_fine_sampler is None:
                raise RuntimeError("coarse-to-fine active corridor requires a fine sampler")

            def sample_priors(
                sampler: CurveAlignedSampler,
                sample_x: torch.Tensor,
                reference_samples: torch.Tensor,
            ) -> torch.Tensor | None:
                if not self.active_corridor_use_dense_priors:
                    return None
                prior_items = []
                for dense_logits in (seg_logits, centerline_logits):
                    if dense_logits is None:
                        prior_items.append(reference_samples.new_zeros((*reference_samples.shape[:-1], 1)))
                    else:
                        sampled = sampler.sample_local_window(dense_logits, sample_x)
                        prior_items.append(torch.tanh(sampled.float() / 4.0).to(dtype=reference_samples.dtype))
                return torch.cat(prior_items, dim=-1)

            coarse_priors = sample_priors(self.active_corridor_sampler, center_x, offset_samples)
            _, coarse_delta, coarse_logits, coarse_weights = self.active_corridor.score_coarse(
                offset_samples,
                coarse_priors,
                queries,
                self.row_embedding.weight,
            )
            fine_center = center_x + coarse_delta
            if self.active_corridor_detach_coarse_for_fine:
                fine_center = fine_center.detach()
            fine_samples = self.active_corridor_fine_sampler.sample_local_window(features, fine_center)
            fine_priors = sample_priors(self.active_corridor_fine_sampler, fine_center, fine_samples)
            evidence, fine_delta, fine_logits, fine_weights = self.active_corridor.score_fine(
                fine_samples,
                fine_priors,
                queries,
                self.row_embedding.weight,
            )
            pred_delta = coarse_delta + fine_delta
            combined_logits = coarse_logits.unsqueeze(-1) + fine_logits.unsqueeze(-2)
            logits = combined_logits.flatten(-2)
            coarse_offsets = self.active_corridor.offsets_px.to(device=sampling_features.device, dtype=sampling_features.dtype)
            fine_offsets = self.active_corridor.fine_offsets_px.to(device=sampling_features.device, dtype=sampling_features.dtype)
            combined_offsets = (coarse_offsets.unsqueeze(-1) + fine_offsets.unsqueeze(0)).flatten()
            combined_weights = (coarse_weights.unsqueeze(-1) * fine_weights.unsqueeze(-2)).flatten(-2)
            entropy = -(
                combined_weights.float() * combined_weights.float().clamp_min(1e-6).log()
            ).sum(dim=-1)
            center_idx = int(combined_offsets.abs().argmin().item())
            debug = {
                "active_offset_entropy": entropy.detach().mean(),
                "active_offset_max_prob": combined_weights.detach().float().max(dim=-1).values.mean(),
                "active_offset_center_prob": combined_weights.detach().float()[..., center_idx].mean(),
                "active_pred_delta_abs": pred_delta.detach().abs().mean(),
                "active_coarse_offset_logits": coarse_logits,
                "active_coarse_offsets_px": coarse_offsets,
                "active_coarse_pred_delta_x_rows": coarse_delta,
                "active_fine_center_x_rows": fine_center,
                "active_fine_offset_logits": fine_logits,
                "active_fine_offsets_px": fine_offsets,
                "active_fine_pred_delta_x_rows": fine_delta,
                "active_dense_prior_abs": (
                    coarse_priors.detach().float().abs().mean()
                    if coarse_priors is not None
                    else features.new_tensor(0.0)
                ),
            }
            offsets = combined_offsets
        else:
            evidence, pred_delta, logits, debug = self.active_corridor(
                offset_samples,
                queries,
                self.row_embedding.weight,
            )
            offsets = self.active_corridor.offsets_px.to(device=sampling_features.device, dtype=sampling_features.dtype)
        refined_x = center_x + pred_delta
        debug = {
            **debug,
            **tower_debug,
            "active_base_center_x_rows": base_center_x,
            "active_center_x_rows": center_x,
            "active_center_jitter_x_rows": center_jitter,
            "active_center_jitter_abs": center_jitter.detach().abs().mean(),
            "active_refined_x_rows": refined_x,
            "active_pred_delta_x_rows": pred_delta,
            "active_offset_logits": logits,
            "active_offsets_px": offsets,
            "active_refined_x_mean": refined_x.detach().mean(),
        }
        return evidence, refined_x, debug

    def build_final_tokens(
        self,
        queries: torch.Tensor,
        evidence: torch.Tensor,
        stage_extra: torch.Tensor | None = None,
        base_row_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _, d = evidence.shape
        if base_row_tokens is None:
            row_emb = self.row_embedding.weight.view(1, 1, self.num_rows, d)
            tokens = queries.unsqueeze(2) + row_emb
        else:
            tokens = base_row_tokens
        if self.adapter is None:
            raise RuntimeError("build_final_tokens requires adapter, but adapter is disabled for s2_csr")
        tokens = tokens + self.adapter(evidence)
        if stage_extra is not None:
            tokens = tokens + stage_extra
        return tokens

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
        matches: list[dict[str, torch.Tensor]] | None = None,
        sampler_alpha: float = 0.0,
        return_features: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if self.freeze_s0_frontend:
            with torch.no_grad():
                enc = self.encoder.forward_features(images)
                structured = self.structured_query_head(enc["features"]) if self.structured_query_head is not None else None
        else:
            enc = self.encoder.forward_features(images)
            structured = self.structured_query_head(enc["features"]) if self.structured_query_head is not None else None
        q = structured["queries"] if structured is not None else enc["queries"]
        structured_row_tokens = structured["structured_row_tokens"] if structured is not None else None
        geometry_debug = None
        q_pre_geometry = None
        geometry_draft = None
        if self.s2_mode == "residual":
            if self.s0_geometry_refiner is not None:
                if self.heads is None:
                    raise RuntimeError("s0_geometry_refiner requires residual S0Heads")
                q_pre_geometry = q
                geometry_draft = self.heads(q)
                geometry_x = (
                    geometry_draft["pred_x_rows"].detach()
                    if self.s0_geometry_detach_draft
                    else geometry_draft["pred_x_rows"]
                )
                q, geometry_debug = self.s0_geometry_refiner(q, enc["features"], geometry_x)
            if self.oracle_coarse_enabled:
                if targets is None:
                    raise ValueError("oracle_coarse.enabled requires targets in forward")
                coarse = self.build_oracle_coarse(q, targets)
            else:
                if structured is not None:
                    coarse = structured
                else:
                    if self.heads is None:
                        raise RuntimeError("Residual S2 without structured_query/oracle_coarse requires S0Heads")
                    coarse = self.heads(q)
            coarse_x = coarse["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            features_for_sampling = enc["multi_scale_features"] if self.multi_scale_enabled else enc["features"]
            if self.active_corridor_enabled:
                if not isinstance(features_for_sampling, torch.Tensor):
                    raise TypeError("active_corridor currently expects a single feature tensor")
                evidence, refined_x, offset_debug = self.sample_active_corridor(
                    features_for_sampling,
                    coarse_x,
                    q,
                    seg_logits=enc.get("seg_logits"),
                    centerline_logits=enc.get("centerline_logits"),
                )
                sample_x_for_log = refined_x.detach()
                stage_x = refined_x.detach() if self.active_corridor_detach_refined_x else refined_x
            else:
                evidence, offset_debug = self.sample_evidence(features_for_sampling, sample_x, q)
                sample_x_for_log = sample_x
                stage_x = coarse_x.detach() if self.detach_coarse_x else coarse_x
            stage_x_norm = (stage_x / float(self.input_w)).unsqueeze(-1)
            if self.active_corridor_enabled and self.active_corridor_skip_row_decoder:
                if not self.active_corridor_authoritative_geometry:
                    raise ValueError("active_corridor.skip_row_decoder requires authoritative_geometry=true")
                if structured_row_tokens is None:
                    row_hidden = self.build_coarse_tokens(q)
                else:
                    row_hidden = structured_row_tokens
                row = {
                    "row_x_logits": torch.zeros_like(coarse["row_x_logits"]),
                    "row_hidden": row_hidden,
                }
                bridge_debug = {
                    "active_skipped_row_decoder": evidence.new_tensor(1.0),
                }
            elif self.csr_refiner is not None:
                if not isinstance(features_for_sampling, torch.Tensor):
                    raise TypeError("s2_csr currently expects a single feature tensor")
                offset_samples = self.sampler.sample_local_window(features_for_sampling, sample_x)
                row = self.csr_refiner(
                    offset_samples,
                    q,
                    self.row_embedding.weight,
                    stage_extra=self.coarse_x_embed(stage_x_norm),
                    base_row_tokens=structured_row_tokens,
                )
                evidence = row["fused_evidence"]
                bridge_debug = {
                    "csr_offset_profile_abs": offset_samples.detach().abs().mean(),
                    "csr_fused_evidence_abs": evidence.detach().abs().mean(),
                    "csr_row_hidden_abs": row["row_hidden"].detach().abs().mean(),
                    "csr_delta_logits_abs": row["row_x_logits"].detach().abs().mean(),
                    "csr_quality_delta_abs": row["quality_delta"].detach().abs().mean(),
                }
            else:
                evidence, bridge_debug = self.bridge_evidence(evidence, q)
                row = self.row_decoder(
                    self.build_final_tokens(
                        q,
                        evidence,
                        stage_extra=self.coarse_x_embed(stage_x_norm),
                        base_row_tokens=structured_row_tokens,
                    ),
                    input_w=self.input_w,
                )
            base_logits = coarse["row_x_logits"].detach() if self.detach_coarse_x else coarse["row_x_logits"]
            row_x_logits = base_logits + self.residual_logit_scale * row["row_x_logits"]
            decoder_pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
            pred_x_rows = (
                refined_x.clamp(0.0, float(self.input_w - 1))
                if self.active_corridor_enabled and self.active_corridor_authoritative_geometry
                else decoder_pred_x_rows
            )
            if self.csr_refiner is not None:
                final_exist_logits = coarse["exist_logits"]
                final_quality_logits = self.build_csr_quality_logits(coarse, row["quality_delta"])
                quality_pred_x_rows = pred_x_rows
                decision_debug = {}
            else:
                final_exist_logits, final_quality_logits, decision_debug = self.apply_s2_decision_refiner(coarse, evidence)
                quality_pred_x_rows = pred_x_rows if self.active_corridor_authoritative_geometry else coarse["pred_x_rows"]
            out = {
                "coarse": {
                    **coarse,
                    "row_hidden": row["row_hidden"],
                },
                "final": {
                    "exist_logits": final_exist_logits,
                    "row_x_logits": row_x_logits,
                    "pred_x_rows": pred_x_rows,
                    "range_raw": coarse["range_raw"],
                    "range_norm": coarse["range_norm"],
                    "quality_logits": final_quality_logits,
                    "quality_pred_x_rows": quality_pred_x_rows,
                    "row_hidden": row["row_hidden"],
                },
                "evidence": {
                    "sample_x_rows": sample_x_for_log,
                    "E_seq": evidence,
                    "evidence_scale": self.adapter.gamma if self.adapter is not None else evidence.new_tensor(0.0),
                    "decoder_pred_x_rows": decoder_pred_x_rows,
                    "active_authoritative_geometry": evidence.new_tensor(
                        float(self.active_corridor_enabled and self.active_corridor_authoritative_geometry)
                    ),
                    **offset_debug,
                    **bridge_debug,
                    **decision_debug,
                    **(geometry_debug or {}),
                },
                "queries": q,
                "row_delta_logits": row["row_x_logits"],
            }
            if structured is not None:
                out["structured_row_tokens"] = structured_row_tokens
                out["structured_debug"] = structured.get("structured_debug", {})
            if geometry_debug is not None:
                out["queries_pre_geometry"] = q_pre_geometry
                out["geometry_evidence"] = geometry_debug
                out["s0_geometry_draft"] = geometry_draft
            if "row_visibility_logits" in row:
                out["final"]["row_visibility_logits"] = row["row_visibility_logits"]
        else:
            exist_logits = self.exist_head(q)
            range_raw, range_norm = self.range_head(q)
            coarse_row = self.row_decoder(self.build_coarse_tokens(q, base_row_tokens=structured_row_tokens), input_w=self.input_w)
            coarse_logits = coarse_row["row_x_logits"]
            coarse_x = coarse_row["pred_x_rows"]
            sample_x = self.curriculum.build_sample_x(coarse_x, targets, matches, alpha=float(sampler_alpha))
            features_for_sampling = enc["multi_scale_features"] if self.multi_scale_enabled else enc["features"]
            evidence, offset_debug = self.sample_evidence(features_for_sampling, sample_x, q)
            evidence, bridge_debug = self.bridge_evidence(evidence, q)
            row = self.row_decoder(
                self.build_final_tokens(q, evidence, base_row_tokens=structured_row_tokens),
                input_w=self.input_w,
            )
            out = {
                "coarse": {
                    "exist_logits": exist_logits,
                    "row_x_logits": coarse_logits,
                    "pred_x_rows": coarse_x,
                    "range_raw": range_raw,
                    "range_norm": range_norm,
                    "row_hidden": coarse_row["row_hidden"],
                },
                "final": {
                    "exist_logits": exist_logits,
                    "row_x_logits": row["row_x_logits"],
                    "pred_x_rows": row["pred_x_rows"],
                    "range_raw": range_raw,
                    "range_norm": range_norm,
                    "row_hidden": row["row_hidden"],
                },
                "evidence": {
                    "sample_x_rows": sample_x,
                    "E_seq": evidence,
                    "evidence_scale": self.adapter.gamma if self.adapter is not None else evidence.new_tensor(0.0),
                    **offset_debug,
                    **bridge_debug,
                },
                "queries": q,
            }
            if "row_visibility_logits" in row:
                out["final"]["row_visibility_logits"] = row["row_visibility_logits"]
        for key, value in enc.items():
            if key.startswith("seg_logits") or key == "centerline_logits":
                out[key] = value
                coarse_key = f"coarse_{key}"
                out[coarse_key] = value
        tower_seg = out.get("evidence", {}).get("active_tower_seg_logits")
        tower_centerline = out.get("evidence", {}).get("active_tower_centerline_logits")
        if tower_seg is not None:
            out["seg_logits"] = tower_seg
        if tower_centerline is not None:
            if tower_centerline.shape[-2:] != (self.num_rows, self.x_bins):
                tower_centerline = F.interpolate(
                    tower_centerline,
                    size=(self.num_rows, self.x_bins),
                    mode="bilinear",
                    align_corners=False,
                )
            out["centerline_logits"] = tower_centerline
        if "dynamic_evidence" in enc:
            out["dynamic_evidence"] = enc["dynamic_evidence"]
        if "dynamic_proposals" in enc:
            out["dynamic_proposals"] = enc["dynamic_proposals"]
        if return_features:
            out["features"] = enc["features"]
            if "multi_scale_features" in enc:
                out["multi_scale_features"] = enc["multi_scale_features"]
        return out
