from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from dynlaneseq_eg.tools.sweep_curvelanes_postprocess import evaluate_grid, load_ground_truth


LANE = [
    {"x": 50.0, "y": 0.0},
    {"x": 50.0, "y": 25.0},
    {"x": 50.0, "y": 50.0},
    {"x": 50.0, "y": 75.0},
]


def _write_validation_sample(root: Path) -> None:
    image_path = root / "valid" / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(20, 20, 20)).save(image_path)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "valid" / "valid.txt").write_text("images/sample.jpg\n", encoding="utf-8")
    (root / "valid" / "labels" / "sample.lines.json").write_text(
        json.dumps({"Lines": [LANE]}), encoding="utf-8"
    )


def test_curvelanes_cached_sweep_changes_only_top_k(tmp_path: Path) -> None:
    _write_validation_sample(tmp_path)
    cache = {
        "metadata": {"checkpoint": str(tmp_path / "iter_0000001.pt")},
        "batches": [
            {
                "outputs": {
                    "pred_x_rows": torch.tensor([[[50.0, 50.0, 50.0, 50.0], [80.0, 80.0, 80.0, 80.0]]]),
                    "exist_logits": torch.tensor([[[10.0, -10.0], [9.0, -9.0]]]),
                    "range_norm": torch.tensor([[[0.0, 0.99], [0.0, 0.99]]]),
                },
                "metas": [
                    {
                        "raw_file": "images/sample.jpg",
                        "input_w": 100,
                        "input_h": 100,
                        "orig_w": 100,
                        "orig_h": 100,
                        "scale_x": 1.0,
                        "scale_y": 1.0,
                        "crop_x": 0.0,
                        "crop_y": 0.0,
                    }
                ],
            }
        ],
    }

    rows = evaluate_grid(
        cache=cache,
        ground_truth=load_ground_truth(tmp_path, "val"),
        score_thresholds=[0.50],
        quality_powers=[0.0],
        top_ks=[1, 0],
        min_pred_points=2,
        nms_distance_thresh_px=0.0,
        nms_min_overlap_points=2,
        row_visibility_thresh=0.0,
    )
    by_top_k = {row["top_k"]: row for row in rows}

    assert by_top_k[1]["F1"] == 1.0
    assert by_top_k[1]["TP"] == 1
    assert by_top_k[1]["FP"] == 0
    assert by_top_k[0]["TP"] == 1
    assert by_top_k[0]["FP"] == 1
    assert by_top_k[0]["F1"] == pytest.approx(2.0 / 3.0)
