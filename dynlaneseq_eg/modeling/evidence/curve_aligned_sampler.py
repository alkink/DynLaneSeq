from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..common import fixed_y_rows, input_to_grid


class CurveAlignedSampler(nn.Module):
    def __init__(
        self,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        local_window_enabled: bool = False,
        offsets_px: list[float] | None = None,
    ):
        super().__init__()
        self.input_w = input_w
        self.input_h = input_h
        self.num_rows = num_rows
        self.local_window_enabled = local_window_enabled
        self.offsets_px = offsets_px or [-8.0, -4.0, 0.0, 4.0, 8.0]

    def forward(self, features: torch.Tensor, sample_x_rows: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = features.shape
        _, n, p = sample_x_rows.shape
        x = sample_x_rows.clamp(0, self.input_w - 1)
        y = fixed_y_rows(p, self.input_h, device=x.device, dtype=x.dtype).view(1, 1, p).expand(b, n, p)

        if self.local_window_enabled:
            return self.sample_local_window(features, sample_x_rows).mean(dim=3)
        return self._sample_once(features, x, y)

    def sample_local_window(self, features: torch.Tensor, sample_x_rows: torch.Tensor) -> torch.Tensor:
        x = sample_x_rows.clamp(0, self.input_w - 1)
        y = fixed_y_rows(x.shape[-1], self.input_h, device=x.device, dtype=x.dtype).view(1, 1, x.shape[-1]).expand_as(x)
        samples = []
        for offset in self.offsets_px:
            samples.append(self._sample_once(features, (x + float(offset)).clamp(0, self.input_w - 1), y))
        return torch.stack(samples, dim=3)

    def _sample_once(self, features: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = features.shape
        _, n, p = x.shape
        grid = input_to_grid(x, y, self.input_w, self.input_h).view(b, n * p, 1, 2)
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).permute(0, 2, 1).contiguous().view(b, n, p, c)
