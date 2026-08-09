from __future__ import annotations

import math

import torch
from torch import nn


_SINE_POSITION_CACHE: dict[
    tuple[int, int, int, int, str, int | None, torch.dtype],
    torch.Tensor,
] = {}


class SinePositionEncoding2D(nn.Module):
    def __init__(self, dim: int = 256, temperature: int = 10000):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("2D sine position encoding dim must be divisible by 4.")
        self.dim = dim
        self.temperature = temperature

    # PyTorch 2.1 Inductor's BF16 value-range analysis raises an
    # ``Invalid NaN comparison`` while lowering the tensor power used for the
    # frequency schedule below.  Keep this tiny, parameter-free construction
    # eager so compiling the enclosing detector preserves the exact eager
    # positional encoding instead of changing its numerical formula.
    @torch._dynamo.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        device_index = device.index
        if device.type == "cuda" and device_index is None:
            device_index = torch.cuda.current_device()
        key = (
            int(self.dim),
            int(self.temperature),
            int(h),
            int(w),
            device.type,
            device_index,
            dtype,
        )
        cached = _SINE_POSITION_CACHE.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode(False), torch.no_grad():
            y = torch.linspace(0, 1, h, device=device, dtype=dtype)
            xcoord = torch.linspace(0, 1, w, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, xcoord, indexing="ij")
            num_feats = self.dim // 4
            omega = torch.arange(num_feats, device=device, dtype=dtype)
            omega = 1.0 / (self.temperature ** (omega / max(1, num_feats - 1)))
            out_x = xx[..., None] * omega * 2 * math.pi
            out_y = yy[..., None] * omega * 2 * math.pi
            pe = torch.cat(
                (out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()),
                dim=-1,
            )
            pe = pe.permute(2, 0, 1).unsqueeze(0)
        _SINE_POSITION_CACHE[key] = pe
        return pe
