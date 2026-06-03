from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SequenceLowRankBridge(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        rank: int = 16,
        kernel_size: int = 3,
        bridge_scale_init: float = 0.1,
        force_fp32: bool = True,
        param_clip: float | None = None,
        delta_clip: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.kernel_size = kernel_size
        self.force_fp32 = force_fp32
        self.param_clip = param_clip
        self.delta_clip = delta_clip
        self.param_dim = dim * rank + rank * dim + rank * kernel_size
        self.generator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, self.param_dim),
        )
        nn.init.normal_(self.generator[-1].weight, std=1e-3)
        nn.init.zeros_(self.generator[-1].bias)
        self.bridge_scale = nn.Parameter(torch.tensor(float(bridge_scale_init)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.force_fp32:
            with torch.autocast(device_type=evidence.device.type, enabled=False):
                return self._forward_impl(evidence.float(), queries.float(), output_dtype=evidence.dtype)
        return self._forward_impl(evidence, queries, output_dtype=evidence.dtype)

    def _forward_impl(
        self,
        evidence: torch.Tensor,
        queries: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, n, p, c = evidence.shape
        params = self.generator(queries)
        if self.param_clip is not None:
            params = params.clamp(min=-self.param_clip, max=self.param_clip)
        out = torch.empty_like(evidence)
        cursor_u = self.dim * self.rank
        cursor_v = cursor_u + self.rank * self.dim
        for bi in range(b):
            for ni in range(n):
                vec = params[bi, ni]
                u = vec[:cursor_u].view(c, self.rank)
                v = vec[cursor_u:cursor_v].view(self.rank, c)
                s = vec[cursor_v:].view(self.rank, 1, self.kernel_size)
                e = evidence[bi, ni]
                e_norm = self.norm(e)
                z = e_norm @ u
                z_conv = F.conv1d(
                    z.transpose(0, 1).unsqueeze(0),
                    s,
                    padding=self.kernel_size // 2,
                    groups=self.rank,
                ).squeeze(0).transpose(0, 1)
                delta = z_conv @ v
                if self.delta_clip is not None:
                    delta = delta.clamp(min=-self.delta_clip, max=self.delta_clip)
                out[bi, ni] = e + self.bridge_scale * delta
        delta_all = out - evidence
        mean_e = evidence.abs().mean()
        mean_delta = delta_all.abs().mean()
        return out.to(dtype=output_dtype), {
            "bridge_scale": self.bridge_scale.detach(),
            "mean_abs_delta_E": mean_delta,
            "mean_abs_E_seq": mean_e,
            "delta_ratio": mean_delta / (mean_e + 1e-6),
        }
