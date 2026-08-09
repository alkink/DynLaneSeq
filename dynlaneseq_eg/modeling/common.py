from __future__ import annotations

import torch


_CONSTANT_TENSOR_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _canonical_device(device: object | None) -> torch.device:
    value = torch.device("cpu" if device is None else device)
    if value.type == "cuda" and value.index is None:
        value = torch.device("cuda", torch.cuda.current_device())
    return value


def _cached_row_constant(
    kind: str,
    count: int,
    scale: int,
    *,
    device: object | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    resolved_device = _canonical_device(device)
    resolved_dtype = dtype or torch.float32
    key = (
        resolved_device.type,
        resolved_device.index,
        resolved_dtype,
        str(kind),
        int(count),
        int(scale),
    )
    value = _CONSTANT_TENSOR_CACHE.get(key)
    if value is None:
        # Audits often populate caches under inference_mode before training.
        # Disable it while constructing constants so the cached tensors remain
        # legal operands in autograd graphs on subsequent calls.
        with torch.inference_mode(False), torch.no_grad():
            value = torch.arange(
                int(count),
                device=resolved_device,
                dtype=resolved_dtype,
            )
            if kind == "fixed_y":
                value = value * (float(scale) / float(count))
            elif kind != "index":
                raise ValueError(f"unknown cached row constant: {kind}")
        _CONSTANT_TENSOR_CACHE[key] = value
    return value


def fixed_indices(
    count: int,
    *,
    device: object | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return _cached_row_constant(
        "index",
        int(count),
        1,
        device=device,
        dtype=dtype,
    )


def fixed_linspace(
    start: float,
    end: float,
    steps: int,
    *,
    device: object | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_device = _canonical_device(device)
    resolved_dtype = dtype or torch.float32
    key = (
        resolved_device.type,
        resolved_device.index,
        resolved_dtype,
        "linspace",
        float(start),
        float(end),
        int(steps),
    )
    value = _CONSTANT_TENSOR_CACHE.get(key)
    if value is None:
        with torch.inference_mode(False), torch.no_grad():
            value = torch.linspace(
                float(start),
                float(end),
                int(steps),
                device=resolved_device,
                dtype=resolved_dtype,
            )
        _CONSTANT_TENSOR_CACHE[key] = value
    return value


def fixed_row_fractions(
    rows: int,
    *,
    device: object | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_device = _canonical_device(device)
    resolved_dtype = dtype or torch.float32
    key = (
        resolved_device.type,
        resolved_device.index,
        resolved_dtype,
        "row_fraction_exclusive",
        int(rows),
    )
    value = _CONSTANT_TENSOR_CACHE.get(key)
    if value is None:
        with torch.inference_mode(False), torch.no_grad():
            value = fixed_indices(
                int(rows),
                device=resolved_device,
                dtype=resolved_dtype,
            ) / float(rows)
        _CONSTANT_TENSOR_CACHE[key] = value
    return value


def fixed_sample_indices(
    rows: int,
    samples: int,
    *,
    device: object | None = None,
) -> torch.Tensor:
    resolved_device = _canonical_device(device)
    key = (
        resolved_device.type,
        resolved_device.index,
        torch.long,
        "rounded_sample_indices",
        int(rows),
        int(samples),
    )
    value = _CONSTANT_TENSOR_CACHE.get(key)
    if value is None:
        with torch.inference_mode(False), torch.no_grad():
            value = fixed_linspace(
                0.0,
                float(int(rows) - 1),
                int(samples),
                device=resolved_device,
                dtype=torch.float32,
            ).round().long()
        _CONSTANT_TENSOR_CACHE[key] = value
    return value


def soft_expected_x(
    row_x_logits: torch.Tensor,
    input_w: int = 800,
    x_bins: int = 200,
    temperature: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(row_x_logits / temperature, dim=-1)
    centers = fixed_indices(
        int(x_bins),
        device=row_x_logits.device,
        dtype=row_x_logits.dtype,
    )
    expected = (probs * centers).sum(dim=-1)
    return expected * (float(input_w) / float(x_bins))


def sort_range_norm(range_norm: torch.Tensor) -> torch.Tensor:
    y_min = torch.minimum(range_norm[..., 0], range_norm[..., 1])
    y_max = torch.maximum(range_norm[..., 0], range_norm[..., 1])
    return torch.stack([y_min, y_max], dim=-1)


def fixed_y_rows(num_rows: int = 72, input_h: int = 288, device=None, dtype=None) -> torch.Tensor:
    return _cached_row_constant(
        "fixed_y",
        int(num_rows),
        int(input_h),
        device=device,
        dtype=dtype,
    )


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
