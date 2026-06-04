from __future__ import annotations

import torch
from torch import nn

from .curve_aligned_sampler import CurveAlignedSampler


class MultiScaleCurveAlignedSampler(nn.Module):
    """Sample curve-aligned evidence from multiple FPN scales and fuse it with learned gates."""

    def __init__(
        self,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        dim: int = 256,
        scales: list[str] | None = None,
        gate_hidden_dim: int = 256,
        dropout: float = 0.0,
        zero_init_gate: bool = True,
        fusion_mode: str = "weighted_sum",
        base_scale: str = "p2",
        residual_scale_init: float = 0.0,
        initial_gate_bias: list[float] | None = None,
    ):
        super().__init__()
        self.scales = list(scales or ["p2", "p3", "p4"])
        if len(self.scales) < 2:
            raise ValueError("MultiScaleCurveAlignedSampler expects at least two scales")
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"weighted_sum", "residual"}:
            raise ValueError(f"Unsupported multi-scale fusion_mode: {fusion_mode}")
        self.base_scale = str(base_scale)
        if self.base_scale not in self.scales:
            raise ValueError(f"base_scale={self.base_scale} must be in scales={self.scales}")
        self.base_idx = self.scales.index(self.base_scale)
        self.sampler = CurveAlignedSampler(input_w=input_w, input_h=input_h, num_rows=num_rows)
        self.scale_embeddings = nn.Parameter(torch.zeros(len(self.scales), dim))
        nn.init.normal_(self.scale_embeddings, std=0.02)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden_dim), len(self.scales)),
        )
        if zero_init_gate:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        if initial_gate_bias is not None:
            if len(initial_gate_bias) != len(self.scales):
                raise ValueError("initial_gate_bias length must match scales length")
            with torch.no_grad():
                self.gate[-1].bias.copy_(torch.tensor(initial_gate_bias, dtype=self.gate[-1].bias.dtype))

    def forward(
        self,
        features: dict[str, torch.Tensor],
        sample_x_rows: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = [name for name in self.scales if name not in features]
        if missing:
            raise KeyError(f"Missing multi-scale features: {missing}")
        samples = []
        for name in self.scales:
            samples.append(self.sampler(features[name], sample_x_rows))
        stacked = torch.stack(samples, dim=3)

        token = queries.unsqueeze(2) + row_embedding.view(1, 1, row_embedding.shape[0], row_embedding.shape[1])
        gate_logits = self.gate(token)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        scale_emb = self.scale_embeddings.view(1, 1, 1, len(self.scales), -1)
        weighted = (gate_weights.unsqueeze(-1) * (stacked + scale_emb)).sum(dim=3)
        base = stacked[:, :, :, self.base_idx]
        if self.fusion_mode == "residual":
            fused = base + self.residual_scale * (weighted - base)
        else:
            fused = weighted

        entropy = -(gate_weights * gate_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        debug = {
            "ms_gate_entropy": entropy.detach(),
            "ms_residual_scale": self.residual_scale.detach(),
        }
        for idx, name in enumerate(self.scales):
            debug[f"ms_gate_{name}"] = gate_weights[..., idx].mean().detach()
        return fused, debug
