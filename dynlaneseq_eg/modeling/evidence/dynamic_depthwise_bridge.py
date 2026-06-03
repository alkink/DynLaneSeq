from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DynamicDepthwiseBridge(nn.Module):
    """Query-conditioned depthwise 1D filtering over curve-aligned evidence."""

    def __init__(
        self,
        dim: int = 256,
        kernel_size: int = 3,
        bridge_scale_init: float = 0.0,
        force_fp32: bool = True,
        param_clip: float | None = None,
        delta_clip: float | None = None,
        generator_init_std: float = 1e-3,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("DynamicDepthwiseBridge expects an odd kernel_size")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.force_fp32 = bool(force_fp32)
        self.param_clip = param_clip
        self.delta_clip = delta_clip
        self.kernel_dim = self.dim * self.kernel_size
        self.param_dim = self.kernel_dim + 2 * self.dim
        self.generator = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 512),
            nn.GELU(),
            nn.Linear(512, self.param_dim),
        )
        nn.init.normal_(self.generator[-1].weight, std=float(generator_init_std))
        nn.init.zeros_(self.generator[-1].bias)
        self.bridge_scale = nn.Parameter(torch.tensor(float(bridge_scale_init)))
        self.norm = nn.LayerNorm(self.dim)

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
        cursor_gamma = self.kernel_dim
        cursor_beta = cursor_gamma + c
        for bi in range(b):
            for ni in range(n):
                vec = params[bi, ni]
                kernel = vec[: self.kernel_dim].view(c, 1, self.kernel_size)
                gamma = vec[cursor_gamma:cursor_beta].view(1, c)
                beta = vec[cursor_beta:].view(1, c)
                e = evidence[bi, ni]
                e_norm = self.norm(e)
                filtered = F.conv1d(
                    e_norm.transpose(0, 1).unsqueeze(0),
                    kernel,
                    padding=self.kernel_size // 2,
                    groups=c,
                ).squeeze(0).transpose(0, 1)
                delta = filtered * (1.0 + gamma) + beta
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
