from __future__ import annotations

from dataclasses import dataclass

import torch


def row_lane_iou(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    line_width: float = 30.0,
) -> torch.Tensor:
    """Analytic row-space ribbon IoU with broadcastable leading dimensions."""

    pred_x = pred_x.float()
    gt_x = gt_x.to(device=pred_x.device, dtype=pred_x.dtype)
    pred_mask = pred_mask.to(device=pred_x.device).bool()
    gt_mask = gt_mask.to(device=pred_x.device).bool()
    both = pred_mask & gt_mask
    either = pred_mask | gt_mask
    overlap = (float(line_width) - (pred_x - gt_x).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(either, torch.full_like(overlap, float(line_width)), torch.zeros_like(overlap)),
    )
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)


@dataclass(frozen=True)
class OracleGeometry:
    pred_x: torch.Tensor
    pred_mask: torch.Tensor
    parameter: float | None = None


def best_constant_shift(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    shifts_px: torch.Tensor,
    *,
    input_w: int = 800,
    line_width: float = 30.0,
) -> OracleGeometry:
    shifts = shifts_px.to(device=pred_x.device, dtype=pred_x.dtype).flatten()
    if shifts.numel() == 0:
        raise ValueError("shifts_px must not be empty")
    candidates = (pred_x.unsqueeze(0) + shifts[:, None]).clamp(0.0, float(input_w - 1))
    pred_masks = pred_mask.unsqueeze(0).expand_as(candidates)
    gt_values = gt_x.unsqueeze(0).expand_as(candidates)
    gt_masks = gt_mask.unsqueeze(0).expand_as(candidates)
    ious = row_lane_iou(
        candidates,
        pred_masks,
        gt_values,
        gt_masks,
        line_width=line_width,
    )
    best_idx = int(ious.argmax().item())
    return OracleGeometry(candidates[best_idx], pred_mask.clone(), float(shifts[best_idx]))


def bounded_row_correction(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    bound_px: float,
    *,
    input_w: int = 800,
    use_gt_range: bool = False,
) -> OracleGeometry:
    bound = max(float(bound_px), 0.0)
    delta = (gt_x - pred_x).clamp(min=-bound, max=bound)
    corrected = torch.where(gt_mask.bool(), pred_x + delta, pred_x).clamp(0.0, float(input_w - 1))
    mask = gt_mask.clone().bool() if use_gt_range else pred_mask.clone().bool()
    return OracleGeometry(corrected, mask, bound)


def polynomial_row_correction(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    degree: int,
    *,
    input_w: int = 800,
    max_displacement_px: float = 64.0,
    use_gt_range: bool = False,
) -> OracleGeometry:
    """Least-squares low-rank oracle for translation/slope/curvature errors."""

    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    rows = int(pred_x.shape[-1])
    valid = pred_mask.bool() & gt_mask.bool()
    output_mask = gt_mask.clone().bool() if use_gt_range else pred_mask.clone().bool()
    if int(valid.sum()) < degree + 1:
        return OracleGeometry(pred_x.clone(), output_mask, float(degree))
    y = torch.linspace(-1.0, 1.0, rows, device=pred_x.device, dtype=pred_x.dtype)
    design = torch.stack([y.pow(power) for power in range(degree + 1)], dim=-1)
    target_delta = gt_x - pred_x
    coefficients = torch.linalg.lstsq(design[valid], target_delta[valid].unsqueeze(-1)).solution.squeeze(-1)
    correction = design @ coefficients
    correction = correction.clamp(-float(max_displacement_px), float(max_displacement_px))
    corrected = torch.where(gt_mask.bool(), pred_x + correction, pred_x).clamp(0.0, float(input_w - 1))
    return OracleGeometry(corrected, output_mask, float(degree))


def fit_polynomial_delta(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    degree: int,
) -> torch.Tensor | None:
    """Return least-squares delta coefficients over normalized row position.

    Coefficients are ordered from degree zero upward.  For the affine case,
    ``delta(y) = a0 + a1*y`` and ``y`` spans [-1, 1].  This helper exposes the
    exact target family used by ``polynomial_row_correction`` without changing
    oracle behavior.
    """

    degree = int(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    rows = int(pred_x.shape[-1])
    valid = pred_mask.bool() & gt_mask.bool()
    valid &= torch.isfinite(pred_x) & torch.isfinite(gt_x)
    if int(valid.sum()) < degree + 1:
        return None
    y = torch.linspace(-1.0, 1.0, rows, device=pred_x.device, dtype=pred_x.dtype)
    design = torch.stack([y.pow(power) for power in range(degree + 1)], dim=-1)
    target_delta = gt_x.to(pred_x) - pred_x
    return torch.linalg.lstsq(
        design[valid], target_delta[valid].unsqueeze(-1)
    ).solution.squeeze(-1)


def near_miss_oracle_variants(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_x: torch.Tensor,
    gt_mask: torch.Tensor,
    shifts_px: torch.Tensor,
    row_bounds_px: list[float] | tuple[float, ...],
    *,
    input_w: int = 800,
    line_width: float = 30.0,
) -> dict[str, OracleGeometry]:
    shifted = best_constant_shift(
        pred_x,
        pred_mask,
        gt_x,
        gt_mask,
        shifts_px,
        input_w=input_w,
        line_width=line_width,
    )
    shifted_range = best_constant_shift(
        pred_x,
        gt_mask,
        gt_x,
        gt_mask,
        shifts_px,
        input_w=input_w,
        line_width=line_width,
    )
    variants: dict[str, OracleGeometry] = {
        "constant_shift": shifted,
        "range_only": OracleGeometry(pred_x.clone(), gt_mask.clone().bool()),
        "constant_shift_plus_range": shifted_range,
    }
    polynomial_names = {0: "translation_lstsq", 1: "affine_deformation", 2: "quadratic_deformation"}
    for degree, name in polynomial_names.items():
        variants[name] = polynomial_row_correction(
            pred_x,
            pred_mask,
            gt_x,
            gt_mask,
            degree,
            input_w=input_w,
            use_gt_range=False,
        )
        variants[f"{name}_plus_range"] = polynomial_row_correction(
            pred_x,
            pred_mask,
            gt_x,
            gt_mask,
            degree,
            input_w=input_w,
            use_gt_range=True,
        )
    for value in row_bounds_px:
        label = f"row_bound_{float(value):g}px"
        variants[label] = bounded_row_correction(
            pred_x,
            pred_mask,
            gt_x,
            gt_mask,
            value,
            input_w=input_w,
            use_gt_range=False,
        )
        variants[f"{label}_plus_range"] = bounded_row_correction(
            pred_x,
            pred_mask,
            gt_x,
            gt_mask,
            value,
            input_w=input_w,
            use_gt_range=True,
        )
        shifted_label = f"constant_shift_then_{label}"
        variants[shifted_label] = bounded_row_correction(
            shifted.pred_x,
            pred_mask,
            gt_x,
            gt_mask,
            value,
            input_w=input_w,
            use_gt_range=False,
        )
        variants[f"{shifted_label}_plus_range"] = bounded_row_correction(
            shifted_range.pred_x,
            gt_mask,
            gt_x,
            gt_mask,
            value,
            input_w=input_w,
            use_gt_range=True,
        )
    return variants
