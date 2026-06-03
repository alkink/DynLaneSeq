from __future__ import annotations

import torch
from torch import nn


class EvidenceAdapter(nn.Module):
    def __init__(self, dim: int = 256, gamma_init: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.net(evidence)

