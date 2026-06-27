from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..common import fixed_y_rows, input_to_grid, sort_range_norm


class OrthogonalEvidenceSampler(nn.Module):
    """Sample feature profiles along the normal of a predicted lane polyline.

    The lane representation is fixed-row ``x(y)``.  For each row, this sampler
    estimates a pixel-space tangent from neighbouring rows, rotates it into a
    normal, and samples a small set of offsets around the current hypothesis.
    It returns both sampled features and an explicit validity mask so callers do
    not learn from border-padded evidence outside the image/range.
    """

    def __init__(
        self,
        input_w: int = 800,
        input_h: int = 288,
        num_rows: int = 72,
        offsets_px: list[float] | None = None,
        detach_sample_x: bool = True,
        range_pad: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.num_rows = int(num_rows)
        self.detach_sample_x = bool(detach_sample_x)
        self.range_pad = float(range_pad)
        self.eps = float(eps)
        offsets = offsets_px if offsets_px is not None else [-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0]
        if len(offsets) < 1:
            raise ValueError("orthogonal sampler needs at least one offset")
        self.register_buffer("offsets_px", torch.tensor(offsets, dtype=torch.float32))

    def tangent_normal(self, x_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pixel-space tangent and normal components with full row length."""
        if x_rows.ndim != 3:
            raise ValueError(f"x_rows must be [B,N,R], got shape={tuple(x_rows.shape)}")
        b, n, r = x_rows.shape
        if r != self.num_rows:
            raise ValueError(f"Expected {self.num_rows} rows, got {r}")
        if r < 2:
            raise ValueError("At least two rows are required to estimate tangent")
        y = fixed_y_rows(r, self.input_h, device=x_rows.device, dtype=x_rows.dtype).view(1, 1, r).expand(b, n, r)

        dx = torch.empty_like(x_rows)
        dy = torch.empty_like(x_rows)
        dx[..., 0] = x_rows[..., 1] - x_rows[..., 0]
        dy[..., 0] = y[..., 1] - y[..., 0]
        dx[..., -1] = x_rows[..., -1] - x_rows[..., -2]
        dy[..., -1] = y[..., -1] - y[..., -2]
        if r > 2:
            dx[..., 1:-1] = x_rows[..., 2:] - x_rows[..., :-2]
            dy[..., 1:-1] = y[..., 2:] - y[..., :-2]

        denom = torch.sqrt(dx.square() + dy.square()).clamp_min(self.eps)
        tx = dx / denom
        ty = dy / denom
        # Rotate tangent by +90 degrees. Offset signs are symmetric, so the
        # left/right naming is not semantically important.
        nx = -ty
        ny = tx
        return tx, ty, nx, ny

    def _range_valid(self, y: torch.Tensor, range_norm: torch.Tensor | None) -> torch.Tensor:
        if range_norm is None:
            return torch.ones_like(y, dtype=torch.bool)
        if range_norm.ndim != 3 or range_norm.shape[-1] != 2:
            raise ValueError(f"range_norm must be [B,N,2], got shape={tuple(range_norm.shape)}")
        ranges = sort_range_norm(range_norm.to(device=y.device, dtype=y.dtype))
        y_norm = y / float(self.input_h)
        return (y_norm >= (ranges[..., :1] - self.range_pad)) & (y_norm <= (ranges[..., 1:] + self.range_pad))

    def forward(
        self,
        features: torch.Tensor,
        sample_x_rows: torch.Tensor,
        range_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got shape={tuple(features.shape)}")
        x = sample_x_rows.detach() if self.detach_sample_x else sample_x_rows
        x = x.to(dtype=features.dtype)
        b, c, _, _ = features.shape
        xb, n, r = x.shape
        if xb != b:
            raise ValueError(f"Batch mismatch: features B={b}, x_rows B={xb}")

        y = fixed_y_rows(r, self.input_h, device=x.device, dtype=x.dtype).view(1, 1, r).expand(b, n, r)
        _, _, nx, ny = self.tangent_normal(x)
        offsets = self.offsets_px.to(device=x.device, dtype=x.dtype).view(1, 1, 1, -1)

        sx = x.unsqueeze(-1) + offsets * nx.unsqueeze(-1)
        sy = y.unsqueeze(-1) + offsets * ny.unsqueeze(-1)
        image_valid = (sx >= 0.0) & (sx <= float(self.input_w - 1)) & (sy >= 0.0) & (sy <= float(self.input_h - 1))
        range_valid = self._range_valid(y, range_norm).unsqueeze(-1)
        valid = image_valid & range_valid

        # Clamp only for stable interpolation coordinates; invalid samples are
        # masked to zero after sampling and should not contribute to pooling.
        sx_safe = sx.clamp(0.0, float(self.input_w - 1))
        sy_safe = sy.clamp(0.0, float(self.input_h - 1))
        grid = input_to_grid(sx_safe, sy_safe, self.input_w, self.input_h).view(b, n * r * offsets.shape[-1], 1, 2)
        sampled = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous().view(b, n, r, offsets.shape[-1], c)
        sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)
        debug = {
            "orthogonal_valid_frac": valid.float().mean().detach(),
            "orthogonal_normal_x_abs": nx.detach().abs().mean(),
            "orthogonal_normal_y_abs": ny.detach().abs().mean(),
        }
        return sampled, valid, debug

