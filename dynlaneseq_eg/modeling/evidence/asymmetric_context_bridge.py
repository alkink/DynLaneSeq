from __future__ import annotations

import torch
from torch import nn

from .dynamic_depthwise_bridge import DynamicDepthwiseBridge


class AsymmetricContextModulationBridge(nn.Module):
    """Use low-resolution context evidence to modulate high-resolution lane evidence."""

    def __init__(
        self,
        dim: int = 256,
        kernel_size: int = 3,
        base_scale: str = "p2",
        context_scale: str = "p3",
        bridge_scale_init: float = 0.0,
        acm_scale_init: float = 0.01,
        use_acm_scale: bool = True,
        context_hidden_dim: int = 512,
        context_dropout: float = 0.0,
        force_fp32: bool = True,
        param_clip: float | None = None,
        delta_clip: float | None = None,
        generator_init_std: float = 1e-3,
    ):
        super().__init__()
        self.base_scale = str(base_scale)
        self.context_scale = str(context_scale)
        self.force_fp32 = bool(force_fp32)
        self.dim = int(dim)
        self.use_acm_scale = bool(use_acm_scale)
        self.p2_filter = DynamicDepthwiseBridge(
            dim=dim,
            kernel_size=kernel_size,
            bridge_scale_init=bridge_scale_init,
            force_fp32=False,
            param_clip=param_clip,
            delta_clip=delta_clip,
            generator_init_std=generator_init_std,
        )
        self.context = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, int(context_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(context_dropout)),
            nn.Linear(int(context_hidden_dim), 2 * dim),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        if self.use_acm_scale:
            self.acm_scale = nn.Parameter(torch.tensor(float(acm_scale_init)))
        else:
            self.register_parameter("acm_scale", None)

    def forward(
        self,
        evidence: dict[str, torch.Tensor],
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.base_scale not in evidence or self.context_scale not in evidence:
            raise KeyError(
                f"ACM expects evidence scales {self.base_scale!r} and {self.context_scale!r}; "
                f"got {sorted(evidence)}"
            )
        if self.force_fp32:
            with torch.autocast(device_type=queries.device.type, enabled=False):
                return self._forward_impl(
                    {key: value.float() for key, value in evidence.items()},
                    queries.float(),
                    row_embedding.float(),
                    output_dtype=evidence[self.base_scale].dtype,
                )
        return self._forward_impl(evidence, queries, row_embedding, output_dtype=evidence[self.base_scale].dtype)

    def _forward_impl(
        self,
        evidence: dict[str, torch.Tensor],
        queries: torch.Tensor,
        row_embedding: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base = evidence[self.base_scale]
        context = evidence[self.context_scale]
        filtered, depthwise_debug = self.p2_filter(base, queries)
        row = row_embedding.view(1, 1, row_embedding.shape[0], row_embedding.shape[1]).expand_as(context)
        query = queries.unsqueeze(2).expand_as(context)
        token = torch.cat([context, query, row], dim=-1)
        gamma_beta = self.context(token)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        if self.use_acm_scale:
            scale = self.acm_scale
            out = filtered * (1.0 + scale * torch.tanh(gamma)) + scale * beta
        else:
            out = filtered * (1.0 + torch.tanh(gamma)) + beta
        delta = out - base
        mean_base = base.abs().mean()
        mean_context = context.abs().mean()
        mean_delta = delta.abs().mean()
        debug = {
            **depthwise_debug,
            "acm_mean_abs_context": mean_context.detach(),
            "acm_mean_abs_delta": mean_delta.detach(),
            "acm_delta_ratio": (mean_delta / (mean_base + 1e-6)).detach(),
            "acm_gamma_abs": gamma.abs().mean().detach(),
            "acm_beta_abs": beta.abs().mean().detach(),
        }
        if self.use_acm_scale:
            debug["acm_scale"] = self.acm_scale.detach()
        return out.to(dtype=output_dtype), debug
