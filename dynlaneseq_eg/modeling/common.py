from __future__ import annotations

import torch


def soft_expected_x(
    row_x_logits: torch.Tensor,
    input_w: int = 800,
    x_bins: int = 200,
    temperature: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(row_x_logits / temperature, dim=-1)
    centers = torch.arange(x_bins, device=row_x_logits.device, dtype=row_x_logits.dtype)
    expected = (probs * centers).sum(dim=-1)
    return expected * (float(input_w) / float(x_bins))


def soft_expected_x_with_log_probs(
    row_x_logits: torch.Tensor,
    input_w: int = 800,
    x_bins: int = 200,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute localization and its log distribution with one normalization.

    Structured S0 needs the same categorical distribution for expected-X
    decoding and DFL supervision.  Computing softmax in the head and log-softmax
    again in the criterion duplicates a large reduction over
    ``B x lanes x rows x bins``.  Log-softmax plus exp is mathematically the
    same distribution and lets both objectives share one autograd graph.
    """

    log_probs = torch.log_softmax(row_x_logits / temperature, dim=-1)
    centers = torch.arange(x_bins, device=row_x_logits.device, dtype=log_probs.dtype)
    expected = (log_probs.exp() * centers).sum(dim=-1)
    return expected * (float(input_w) / float(x_bins)), log_probs


def sort_range_norm(range_norm: torch.Tensor) -> torch.Tensor:
    y_min = torch.minimum(range_norm[..., 0], range_norm[..., 1])
    y_max = torch.maximum(range_norm[..., 0], range_norm[..., 1])
    return torch.stack([y_min, y_max], dim=-1)


def fixed_y_rows(num_rows: int = 72, input_h: int = 288, device=None, dtype=None) -> torch.Tensor:
    return torch.arange(num_rows, device=device, dtype=dtype or torch.float32) * (float(input_h) / float(num_rows))


def input_to_grid(x: torch.Tensor, y: torch.Tensor, input_w: int = 800, input_h: int = 288) -> torch.Tensor:
    x_grid = 2.0 * x / float(input_w - 1) - 1.0
    y_grid = 2.0 * y / float(input_h - 1) - 1.0
    return torch.stack([x_grid, y_grid], dim=-1)


def nested_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: nested_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nested_to_device(v, device) for v in obj]
    return obj
