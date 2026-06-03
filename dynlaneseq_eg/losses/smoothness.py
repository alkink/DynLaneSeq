from __future__ import annotations

import torch


def second_difference_loss(x: torch.Tensor, input_w: int = 800) -> torch.Tensor:
    if x.numel() < 3:
        return x.sum() * 0.0
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return (d2 / float(input_w)).abs().mean()

