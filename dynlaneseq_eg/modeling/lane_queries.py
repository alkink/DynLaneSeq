from __future__ import annotations

import torch
from torch import nn


class LaneQueries(nn.Module):
    def __init__(self, num_slots: int = 20, dim: int = 256):
        super().__init__()
        self.query = nn.Embedding(num_slots, dim)
        nn.init.normal_(self.query.weight, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        q = self.query.weight.unsqueeze(0)
        return q.expand(batch_size, -1, -1).contiguous()

