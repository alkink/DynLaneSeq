from __future__ import annotations

import torch
from torch import nn


class DynamicOffsetFusion(nn.Module):
    """Token-conditioned fusion over curve-aligned lateral offset samples."""

    def __init__(
        self,
        dim: int = 256,
        num_offsets: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.num_offsets = int(num_offsets)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_offsets),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        offset_samples: torch.Tensor,
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, n, p, o, c = offset_samples.shape
        if o != self.num_offsets:
            raise ValueError(f"Expected {self.num_offsets} offset samples, got {o}")
        tokens = queries.unsqueeze(2) + row_embedding.view(1, 1, p, c)
        logits = self.net(tokens)
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=offset_samples.dtype)
        fused = (offset_samples * weights.unsqueeze(-1)).sum(dim=3)
        entropy = -(weights.float() * weights.float().clamp_min(1e-6).log()).sum(dim=-1)
        center_idx = o // 2
        return fused, {
            "offset_weight_entropy": entropy.detach().mean(),
            "offset_weight_max": weights.detach().float().max(dim=-1).values.mean(),
            "offset_weight_center": weights.detach().float()[..., center_idx].mean(),
        }
