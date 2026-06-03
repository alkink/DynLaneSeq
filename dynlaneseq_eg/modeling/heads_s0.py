from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import soft_expected_x, sort_range_norm


class ExistenceHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q)


class RowXHead(nn.Module):
    def __init__(self, dim: int = 256, num_rows: int = 72, x_bins: int = 200, hidden_dim: int = 256):
        super().__init__()
        self.num_rows = num_rows
        self.x_bins = x_bins
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_rows * x_bins))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        b, n, _ = q.shape
        return self.net(q).view(b, n, self.num_rows, self.x_bins)


class RangeHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.constant_(self.net[-1].weight, 0.0)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-2.0, 2.0]))

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(q)
        norm = sort_range_norm(torch.sigmoid(raw))
        return raw, norm


class QualityHead(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.net(q).squeeze(-1)


class S0Heads(nn.Module):
    def __init__(self, dim: int = 256, num_rows: int = 72, x_bins: int = 200, input_w: int = 800):
        super().__init__()
        self.input_w = input_w
        self.x_bins = x_bins
        self.exist = ExistenceHead(dim)
        self.row_x = RowXHead(dim, num_rows=num_rows, x_bins=x_bins)
        self.range = RangeHead(dim)
        self.quality = QualityHead(dim)

    def forward(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        exist_logits = self.exist(q)
        row_x_logits = self.row_x(q)
        pred_x_rows = soft_expected_x(row_x_logits, input_w=self.input_w, x_bins=self.x_bins)
        range_raw, range_norm = self.range(q)
        return {
            "exist_logits": exist_logits,
            "row_x_logits": row_x_logits,
            "pred_x_rows": pred_x_rows,
            "range_raw": range_raw,
            "range_norm": range_norm,
            "quality_logits": self.quality(q),
        }


class SegAuxHead(nn.Module):
    """Dense lane-mask supervision head used only as an auxiliary training signal."""

    def __init__(self, dim: int = 256, input_h: int = 288, input_w: int = 800, dropout: float = 0.1):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.net = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return F.interpolate(logits, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
