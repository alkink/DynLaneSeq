from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline

from dynlaneseq_eg.evaluation.candidate_diagnostics import stage_scores
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


Lane = List[Tuple[float, float]]


@dataclass(frozen=True)
class DecoderVariant:
    """One fixed decoder/writer treatment in the parity audit.

    ``clr_nms_reference_px`` is expressed in CLRNet's 800-pixel input space
    and is scaled to the evaluated model's input width.  A value of ``None``
    keeps DynLaneSeq's configured NMS distance.
    """

    name: str
    description: str
    range_mode: str = "current"
    writer_mode: str = "current"
    clr_nms_reference_px: float | None = None


DECODER_VARIANTS: tuple[DecoderVariant, ...] = (
    DecoderVariant(
        "A_current",
        "Exact DynLaneSeq range mask, NMS, and writer.",
    ),
    DecoderVariant(
        "B_clr_writer",
        "Only CLRNet's cubic-spline CULane writer/resampling is enabled.",
        writer_mode="clr_spline",
    ),
    DecoderVariant(
        "C_clr_range",
        "Only CLRNet-style bottom extension from the predicted upper endpoint is enabled.",
        range_mode="clr_bottom_extend",
    ),
    DecoderVariant(
        "D_clr_nms",
        "Only CLRNet's 50px-at-width-800 lane NMS distance is enabled.",
        clr_nms_reference_px=50.0,
    ),
    DecoderVariant(
        "E_clr_combined",
        "CLRNet-style range extension, NMS distance, and spline writer together.",
        range_mode="clr_bottom_extend",
        writer_mode="clr_spline",
        clr_nms_reference_px=50.0,
    ),
)


def variant_by_name(name: str) -> DecoderVariant:
    for variant in DECODER_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(f"Unknown decoder variant: {name}")


def candidate_geometry(
    stage: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    range_mode: str,
    min_valid_rows: int = 5,
    row_visibility_thresh: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Lane]]:
    """Build fixed-row candidate geometry for a decoder treatment.

    CLRNet predicts an upper endpoint and permits the curve to extend toward
    the image bottom while its sampled x values remain in frame.  DynLaneSeq
    predicts both endpoints.  ``clr_bottom_extend`` is the smallest faithful
    adaptation: preserve DynLaneSeq's predicted upper endpoint and ignore its
    lower endpoint, retaining the contiguous in-frame bottom extension.
    """

    raw_x = stage["pred_x_rows"].float()
    pred_x = raw_x.clamp(0.0, float(input_w - 1))
    proposal_count, row_count = pred_x.shape
    y_rows = fixed_y_rows(row_count, input_h, device=pred_x.device, dtype=pred_x.dtype)
    ranges = stage.get("range_norm")

    if ranges is None:
        masks = torch.ones((proposal_count, row_count), dtype=torch.bool, device=pred_x.device)
    else:
        ranges = sort_range_norm(ranges.float())
        y_min = ranges[:, 0:1] * float(input_h)
        y_max = ranges[:, 1:2] * float(input_h)
        if range_mode == "current":
            masks = (y_rows.view(1, -1) >= y_min) & (y_rows.view(1, -1) <= y_max)
        elif range_mode == "clr_bottom_extend":
            masks = (y_rows.view(1, -1) >= y_min) & (y_rows.view(1, -1) <= y_max)
            in_frame = torch.isfinite(raw_x) & (raw_x >= 0.0) & (raw_x < float(input_w))
            # Mirror CLRNet's contiguous extension from the predicted start
            # toward the image bottom.  Once an out-of-frame row is met, rows
            # farther toward the bottom are excluded as well.
            for proposal_idx in range(proposal_count):
                extension = y_rows > y_max[proposal_idx, 0]
                if bool(extension.any()):
                    contiguous = torch.cumprod(in_frame[proposal_idx, extension].to(torch.int64), dim=0).bool()
                    masks[proposal_idx, extension] = contiguous
        else:
            raise ValueError(f"Unsupported range mode: {range_mode}")

    visibility = stage.get("row_visibility_logits")
    if row_visibility_thresh > 0 and visibility is not None:
        masks &= torch.sigmoid(visibility.float()) >= float(row_visibility_thresh)
    masks &= torch.isfinite(raw_x)
    candidate_valid = masks.sum(dim=-1) >= int(min_valid_rows)

    y_cpu = y_rows.detach().cpu()
    x_cpu = pred_x.detach().cpu()
    masks_cpu = masks.detach().cpu()
    lanes: list[Lane] = []
    for proposal_idx in range(proposal_count):
        mask = masks_cpu[proposal_idx]
        lanes.append([(float(x), float(y)) for x, y in zip(x_cpu[proposal_idx, mask], y_cpu[mask])])
    return pred_x, masks, candidate_valid, lanes


def select_candidate_ids(
    stage: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    variant: DecoderVariant,
    score_thresh: float,
    quality_power: float,
    current_nms_distance_px: float,
    nms_min_overlap_points: int,
    min_valid_rows: int,
    top_k: int,
    row_visibility_thresh: float = 0.0,
) -> tuple[list[int], list[Lane]]:
    pred_x, masks, candidate_valid, lanes = candidate_geometry(
        stage,
        input_h=input_h,
        input_w=input_w,
        range_mode=variant.range_mode,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    scores = stage_scores(stage, quality_power=quality_power)
    eligible = [
        proposal_idx
        for proposal_idx in range(pred_x.shape[0])
        if bool(candidate_valid[proposal_idx]) and float(scores[proposal_idx]) >= float(score_thresh)
    ]
    eligible.sort(key=lambda proposal_idx: float(scores[proposal_idx]), reverse=True)

    if variant.clr_nms_reference_px is None:
        nms_distance = float(current_nms_distance_px)
    else:
        nms_distance = float(variant.clr_nms_reference_px) * float(input_w) / 800.0

    if nms_distance <= 0.0 or len(eligible) <= 1:
        kept = eligible
    else:
        eligible_x = pred_x[eligible]
        eligible_masks = masks[eligible]
        overlap_masks = eligible_masks[:, None, :] & eligible_masks[None, :, :]
        overlaps = overlap_masks.sum(dim=-1)
        differences = (eligible_x[:, None, :] - eligible_x[None, :, :]).abs()
        differences = torch.where(overlap_masks, differences, torch.zeros_like(differences))
        distances = differences.sum(dim=-1) / overlaps.clamp_min(1)
        close = (overlaps >= int(nms_min_overlap_points)) & (distances < nms_distance)
        close = close.detach().cpu().numpy()
        kept: list[int] = []
        kept_local: list[int] = []
        for local_idx, proposal_idx in enumerate(eligible):
            if any(bool(close[local_idx, kept_idx]) for kept_idx in kept_local):
                continue
            kept.append(proposal_idx)
            kept_local.append(local_idx)

    selected = kept[: int(top_k)] if top_k > 0 else kept
    return selected, lanes


def lane_to_original_current(lane: Lane, meta: dict[str, Any]) -> Lane:
    sx = float(meta.get("scale_x", 1.0))
    sy = float(meta.get("scale_y", 1.0))
    crop_x = float(meta.get("crop_x", 0.0))
    crop_y = float(meta.get("crop_y", 0.0))
    return [
        (round(float(x) / sx + crop_x, 3), round(float(y) / sy + crop_y, 3))
        for x, y in lane
    ]


def lane_to_original_clr_spline(lane: Lane, meta: dict[str, Any]) -> Lane:
    """Reproduce CLRNet's CULane spline writer (y=270..589, step 8)."""

    original = lane_to_original_current(lane, meta)
    if len(original) <= 1:
        return []
    points = np.asarray(original, dtype=np.float64)
    order = np.argsort(points[:, 1], kind="stable")
    points = points[order]
    _, unique_indices = np.unique(points[:, 1], return_index=True)
    points = points[np.sort(unique_indices)]
    if len(points) <= 1:
        return []

    spline = InterpolatedUnivariateSpline(
        points[:, 1],
        points[:, 0],
        k=min(3, len(points) - 1),
    )
    sample_y = np.arange(270.0, 590.0, 8.0, dtype=np.float64)
    in_domain = (sample_y >= points[:, 1].min() - 0.01) & (sample_y <= points[:, 1].max() + 0.01)
    sample_x = spline(sample_y)
    orig_w = float(meta.get("orig_w", 1640))
    valid = in_domain & np.isfinite(sample_x) & (sample_x >= 0.0) & (sample_x < orig_w)
    result = [
        (round(float(x), 5), round(float(y), 5))
        for x, y in zip(sample_x[valid], sample_y[valid])
    ]
    # CLRNet writes CULane points from bottom to top.
    return result[::-1]


def lane_to_original(lane: Lane, meta: dict[str, Any], writer_mode: str) -> Lane:
    if writer_mode == "current":
        return lane_to_original_current(lane, meta)
    if writer_mode == "clr_spline":
        return lane_to_original_clr_spline(lane, meta)
    raise ValueError(f"Unsupported writer mode: {writer_mode}")


def prediction_relative_path(meta: dict[str, Any]) -> Path:
    image_path = Path(str(meta["image_path"]))
    return Path(*image_path.parts[-3:]).with_suffix(".lines.txt")


def write_prediction_file(path: str | Path, lanes: Iterable[Lane], writer_mode: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    precision = 5 if writer_mode == "clr_spline" else 3
    with output.open("w", encoding="utf-8") as handle:
        for lane in lanes:
            if len(lane) <= 1:
                continue
            values: list[str] = []
            for x, y in lane:
                values.extend((f"{x:.{precision}f}", f"{y:.{precision}f}"))
            handle.write(" ".join(values) + "\n")
