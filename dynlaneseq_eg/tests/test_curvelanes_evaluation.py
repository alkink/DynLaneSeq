from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dynlaneseq_eg.evaluation.curvelanes_metric import evaluate_curvelanes, lane_iou
from dynlaneseq_eg.evaluation.curvelanes_writer import lane_to_curvelanes_points, write_curvelanes_predictions


LANE = [{"x": 20.0, "y": 95.0}, {"x": 30.0, "y": 65.0}, {"x": 40.0, "y": 35.0}]


def _write_validation_sample(root: Path) -> None:
    image_path = root / "valid" / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(20, 20, 20)).save(image_path)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "valid" / "valid.txt").write_text("images/sample.jpg\n", encoding="utf-8")
    (root / "valid" / "labels" / "sample.lines.json").write_text(json.dumps({"Lines": [LANE]}), encoding="utf-8")


def test_curvelanes_metric_is_perfect_for_exact_prediction(tmp_path: Path) -> None:
    _write_validation_sample(tmp_path)
    prediction_dir = tmp_path / "predictions"
    write_curvelanes_predictions(
        [{"raw_file": "images/sample.jpg", "Lines": [LANE], "Shape": {"width": 100, "height": 100}}],
        prediction_dir,
    )

    results = evaluate_curvelanes(tmp_path, prediction_dir, progress=False)

    assert results["F1"] == 1.0
    assert results["Precision"] == 1.0
    assert results["Recall"] == 1.0
    assert results["TP"] == 1
    assert results["FP"] == 0
    assert results["FN"] == 0
    assert lane_iou(LANE, LANE) == 1.0


def test_curvelanes_writer_inverts_crop_and_resize() -> None:
    scale_x = 1600.0 / 2560.0
    scale_y = 640.0 / 800.0
    lane = [(1280.0 * scale_x, (float(y) - 640.0) * scale_y) for y in (700, 900, 1200)]
    meta = {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "crop_x": 0.0,
        "crop_y": 640.0,
        "orig_w": 2560,
        "orig_h": 1440,
    }
    points = lane_to_curvelanes_points(lane, meta)

    assert [(round(point["x"]), round(point["y"])) for point in points] == [
        (1280, 700),
        (1280, 900),
        (1280, 1200),
    ]
