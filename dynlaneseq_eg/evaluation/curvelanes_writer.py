from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .postprocess import predictions_to_lanes


def lane_to_curvelanes_points(
    lane: list[tuple[float, float]],
    meta: dict[str, Any],
) -> list[dict[str, float]]:
    """Map an input-space lane back to CurveLanes' native image geometry."""
    if len(lane) < 2:
        return []
    scale_x = float(meta["scale_x"])
    scale_y = float(meta["scale_y"])
    crop_x = float(meta.get("crop_x", 0.0))
    crop_y = float(meta.get("crop_y", 0.0))
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])
    points: list[dict[str, float]] = []
    for x, y in lane:
        x_native = float(x) / scale_x + crop_x
        y_native = float(y) / scale_y + crop_y
        if not (0.0 <= x_native < orig_w and 0.0 <= y_native < orig_h):
            continue
        points.append({"x": float(x_native), "y": float(y_native)})
    return points if len(points) >= 2 else []


@torch.no_grad()
def outputs_to_curvelanes_records(
    outputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    score_thresh: float,
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    top_k: int,
    row_visibility_thresh: float,
    quality_score_power: float,
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
    records: list[dict[str, Any]] = []
    for lanes, meta in zip(batch_lanes, metas):
        records.append(
            {
                "raw_file": str(meta["raw_file"]),
                "Lines": [
                    points
                    for lane in lanes
                    if (points := lane_to_curvelanes_points(lane, meta))
                ],
                "Shape": {"width": int(meta["orig_w"]), "height": int(meta["orig_h"])},
            }
        )
    return records


def prediction_path_for_raw_file(prediction_dir: str | Path, raw_file: str) -> Path:
    raw_path = Path(str(raw_file).lstrip("/"))
    return Path(prediction_dir) / raw_path.with_suffix(".lines.json")


def write_curvelanes_predictions(records: list[dict[str, Any]], prediction_dir: str | Path) -> list[Path]:
    """Write one native-format prediction JSON per list entry.

    Keeping the ``images/...`` relative layout makes every prediction directly
    traceable to the corresponding line in ``valid/valid.txt``.
    """
    prediction_dir = Path(prediction_dir)
    written: list[Path] = []
    seen: set[str] = set()
    for record in records:
        raw_file = str(record["raw_file"])
        if raw_file in seen:
            raise ValueError(f"Duplicate CurveLanes prediction for {raw_file!r}")
        seen.add(raw_file)
        path = prediction_path_for_raw_file(prediction_dir, raw_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"Lines": record["Lines"], "Shape": record["Shape"]}
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        written.append(path)
    return written
