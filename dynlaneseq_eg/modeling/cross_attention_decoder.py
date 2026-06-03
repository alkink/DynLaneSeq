from __future__ import annotations

import torch
from torch import nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q_content: torch.Tensor,
        q_pos: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
    ) -> torch.Tensor:
        q_with_pos = q_content + q_pos
        q_content = self.norm1(
            q_content
            + self.drop(
                self.self_attn(
                    query=q_with_pos,
                    key=q_with_pos,
                    value=q_content,
                    need_weights=False,
                )[0]
            )
        )
        q_with_pos = q_content + q_pos
        q_content = self.norm2(
            q_content
            + self.drop(
                self.cross_attn(
                    query=q_with_pos,
                    key=memory_key,
                    value=memory_value,
                    need_weights=False,
                )[0]
            )
        )
        q_content = self.norm3(q_content + self.drop(self.ffn(q_content)))
        return q_content


class LaneCrossAttentionDecoder(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = 256, num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(dim=dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        q_content: torch.Tensor,
        q_pos: torch.Tensor,
        memory_key: torch.Tensor,
        memory_value: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            q_content = layer(q_content, q_pos, memory_key, memory_value)
        return q_content
