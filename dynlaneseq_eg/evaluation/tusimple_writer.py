from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .postprocess import predictions_to_lanes


def lane_to_tusimple_samples(
    lane: list[tuple[float, float]],
    meta: dict[str, Any],
) -> list[int]:
    h_samples = [int(y) for y in meta["h_samples"]]
    if len(lane) < 2:
        return [-2] * len(h_samples)

    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])
    crop_x = float(meta.get("crop_x", 0.0))
    crop_y = float(meta.get("crop_y", 0.0))
    orig_w = int(meta["orig_w"])
    points = sorted(
        ((float(x) / scale_x + crop_x, float(y) / scale_y + crop_y) for x, y in lane),
        key=lambda point: point[1],
    )
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    unique_y, inverse = np.unique(ys, return_inverse=True)
    if unique_y.size != ys.size:
        x_sum = np.zeros_like(unique_y)
        x_count = np.zeros_like(unique_y)
        np.add.at(x_sum, inverse, xs)
        np.add.at(x_count, inverse, 1.0)
        xs = x_sum / np.maximum(x_count, 1.0)
        ys = unique_y
    if ys.size < 2:
        return [-2] * len(h_samples)

    sampled: list[int] = []
    y_min = float(ys[0])
    y_max = float(ys[-1])
    for y in h_samples:
        if float(y) < y_min - 1e-6 or float(y) > y_max + 1e-6:
            sampled.append(-2)
            continue
        x = float(np.interp(float(y), ys, xs))
        if not np.isfinite(x) or x < 0.0 or x >= float(orig_w):
            sampled.append(-2)
        else:
            sampled.append(min(max(int(np.rint(x)), 0), orig_w - 1))
    return sampled


@torch.no_grad()
def outputs_to_tusimple_records(
    outputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    score_thresh: float,
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    top_k: int,
    row_visibility_thresh: float,
    quality_score_power: float,
    run_time_ms: float = 0.0,
) -> list[dict[str, Any]]:
    if not metas:
        return []
    input_w = int(metas[0]["input_w"])
    input_h = int(metas[0]["input_h"])
    batch_lanes = predictions_to_lanes(
        outputs,
        score_thresh=score_thresh,
        min_pred_points=min_pred_points,
        input_w=input_w,
        input_h=input_h,
        nms_distance_thresh_px=nms_distance_thresh_px,
        nms_min_overlap_points=nms_min_overlap_points,
        top_k=top_k,
        row_visibility_thresh=row_visibility_thresh,
        quality_score_power=quality_score_power,
    )
    if len(batch_lanes) != len(metas):
        raise ValueError("Prediction batch size does not match metadata batch size")
    records = []
    for lanes, meta in zip(batch_lanes, metas):
        records.append(
            {
                "raw_file": str(meta["raw_file"]),
                "lanes": [lane_to_tusimple_samples(lane, meta) for lane in lanes],
                "run_time": float(run_time_ms),
            }
        )
    return records


def write_tusimple_json_lines(records: list[dict[str, Any]], output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
