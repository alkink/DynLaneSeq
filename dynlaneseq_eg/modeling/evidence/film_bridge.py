from __future__ import annotations

import torch
from torch import nn


class FiLMBridge(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, evidence: torch.Tensor, queries: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gamma, beta = self.net(queries).chunk(2, dim=-1)
        out = evidence * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        delta = out - evidence
        return out, {"mean_abs_delta_E": delta.abs().mean(), "mean_abs_E_seq": evidence.abs().mean()}

