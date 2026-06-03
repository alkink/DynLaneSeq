from __future__ import annotations

import torch
from torch import nn

from .common import soft_expected_x


class RowTokenDecoder(nn.Module):
    def __init__(
        self,
        num_rows: int = 72,
        dim: int = 256,
        x_bins: int = 200,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        zero_init_head: bool = False,
        local_attn_window: int = 0,
        visibility_head: bool = False,
    ):
        super().__init__()
        self.num_rows = num_rows
        self.x_bins = x_bins
        self.local_attn_window = int(local_attn_window)
        self.visibility_head_enabled = bool(visibility_head)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(dim, x_bins)
        if zero_init_head:
            nn.init.constant_(self.head.weight, 0.0)
            nn.init.constant_(self.head.bias, 0.0)
        self.visibility_head = nn.Linear(dim, 1) if self.visibility_head_enabled else None

    def _local_attention_mask(self, num_rows: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.local_attn_window <= 0:
            return None
        idx = torch.arange(num_rows, device=device)
        local = (idx[:, None] - idx[None, :]).abs() <= self.local_attn_window
        mask = torch.zeros((num_rows, num_rows), device=device, dtype=dtype)
        return mask.masked_fill(~local, float("-inf"))

    def forward(self, row_tokens: torch.Tensor, input_w: int = 800, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        b, n, p, d = row_tokens.shape
        flat = row_tokens.view(b * n, p, d)
        attn_mask = self._local_attention_mask(p, row_tokens.device, row_tokens.dtype)
        hidden = self.encoder(flat, mask=attn_mask)
        logits = self.head(hidden).view(b, n, p, self.x_bins)
        pred_x = soft_expected_x(logits, input_w=input_w, x_bins=self.x_bins, temperature=temperature)
        out = {"row_x_logits": logits, "pred_x_rows": pred_x, "row_hidden": hidden.view(b, n, p, d)}
        if self.visibility_head is not None:
            out["row_visibility_logits"] = self.visibility_head(hidden).view(b, n, p)
        return out
