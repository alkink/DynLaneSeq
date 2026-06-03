from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .postprocess import predictions_to_lanes


@torch.no_grad()
def write_culane_predictions(
    outputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
    out_dir: str | Path,
    score_thresh: float = 0.5,
    min_pred_points: int = 5,
    nms_distance_thresh_px: float = 0.0,
    nms_min_overlap_points: int = 5,
    top_k: int = 0,
    row_visibility_thresh: float = 0.0,
    quality_score_power: float = 0.0,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes_batch = predictions_to_lanes(
        outputs,
        score_thresh=score_thresh,
        min_pred_points=min_pred_points,
        nms_distance_thresh_px=nms_distance_thresh_px,
        nms_min_overlap_points=nms_min_overlap_points,
        top_k=top_k,
        row_visibility_thresh=row_visibility_thresh,
        quality_score_power=quality_score_power,
    )
    written: list[Path] = []
    for lanes, meta in zip(lanes_batch, metas):
        image_path = Path(meta["image_path"])
        rel = Path(*image_path.parts[-3:]).with_suffix(".lines.txt")
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sx = float(meta.get("scale_x", 1.0))
        sy = float(meta.get("scale_y", 1.0))
        crop_x = float(meta.get("crop_x", 0.0))
        crop_y = float(meta.get("crop_y", 0.0))
        with out_path.open("w", encoding="utf-8") as f:
            for lane in lanes:
                vals = []
                for x_in, y_in in lane:
                    vals.extend([x_in / sx + crop_x, y_in / sy + crop_y])
                f.write(" ".join(f"{v:.3f}" for v in vals) + "\n")
        written.append(out_path)
    return written
