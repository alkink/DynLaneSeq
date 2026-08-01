from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


def pairwise_range_aware_row_strip_iou(
    pred_x_rows: torch.Tensor,
    pred_range_norm: torch.Tensor,
    gt_x_rows: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    *,
    input_h: int,
    line_width: float = 30.0,
    min_valid_rows: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a differentiable CULane-aligned row-strip IoU surrogate.

    The matrix is prediction-major with shape ``[num_predictions, num_gt]``.
    A predicted row contributes only inside the prediction's visible range;
    GT rows use the dataset validity mask.  Rows present on only one side add
    one strip width to the union, which makes over-extended ranges costly.

    This is deliberately the single implementation shared by the redesigned
    matcher and unified selection target.  Keeping those two consumers on the
    same geometry definition prevents the assignment/selection conflict found
    in the 70k gradient audit.
    """

    if pred_x_rows.ndim != 2 or gt_x_rows.ndim != 2:
        raise ValueError("pred_x_rows and gt_x_rows must have shape [lanes, rows]")
    if pred_range_norm.shape != (pred_x_rows.shape[0], 2):
        raise ValueError(
            "pred_range_norm must have shape [num_predictions, 2], got "
            f"{tuple(pred_range_norm.shape)}"
        )
    if gt_valid_mask.shape != gt_x_rows.shape:
        raise ValueError("gt_valid_mask must match gt_x_rows")
    if int(pred_x_rows.shape[-1]) != int(gt_x_rows.shape[-1]):
        raise ValueError("prediction and GT row counts must match")
    if float(line_width) <= 0.0:
        raise ValueError("line_width must be positive")

    pred_x = pred_x_rows.float()
    pred_range = sort_range_norm(pred_range_norm.float())
    gt_x = gt_x_rows.to(device=pred_x.device, dtype=pred_x.dtype)
    gt_valid = gt_valid_mask.to(device=pred_x.device).bool()
    gt_valid = gt_valid & torch.isfinite(gt_x)
    num_predictions, rows = pred_x.shape
    num_gt = int(gt_x.shape[0])

    if num_predictions == 0 or num_gt == 0:
        return (
            pred_x.new_zeros((num_predictions, num_gt)),
            torch.zeros(num_predictions, dtype=torch.bool, device=pred_x.device),
            torch.zeros(num_gt, dtype=torch.bool, device=pred_x.device),
        )

    y_rows = fixed_y_rows(
        rows,
        int(input_h),
        device=pred_x.device,
        dtype=pred_x.dtype,
    )
    pred_valid = (
        (y_rows.view(1, -1) >= pred_range[:, :1] * float(input_h))
        & (y_rows.view(1, -1) <= pred_range[:, 1:] * float(input_h))
        & torch.isfinite(pred_x)
    )
    candidate_valid = pred_valid.sum(dim=-1) >= int(min_valid_rows)
    gt_lane_valid = gt_valid.sum(dim=-1) >= int(min_valid_rows)

    pred = pred_x[:, None, :]
    gt = gt_x[None, :, :]
    pred_rows = pred_valid[:, None, :]
    gt_rows = gt_valid[None, :, :]
    both = pred_rows & gt_rows
    either = pred_rows | gt_rows

    width = float(line_width)
    overlap = (width - (pred - gt).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * width - overlap,
        torch.where(
            either,
            torch.full_like(overlap, width),
            torch.zeros_like(overlap),
        ),
    )
    pairwise_iou = overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)
    pairwise_iou = torch.where(
        candidate_valid[:, None] & gt_lane_valid[None, :],
        pairwise_iou,
        torch.zeros_like(pairwise_iou),
    )
    return pairwise_iou, candidate_valid, gt_lane_valid
