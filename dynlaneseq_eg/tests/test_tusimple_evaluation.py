from __future__ import annotations

import json
from pathlib import Path

from dynlaneseq_eg.evaluation.tusimple_metric import TuSimpleLaneEval
from dynlaneseq_eg.evaluation.tusimple_writer import lane_to_tusimple_samples


def test_lane_to_tusimple_samples_inverts_crop_and_resize() -> None:
    h_samples = tuple(range(160, 720, 10))
    scale_x = 1600.0 / 1280.0
    scale_y = 640.0 / (720.0 - 160.0)
    lane = [(640.0 * scale_x, (float(y) - 160.0) * scale_y) for y in h_samples]
    meta = {
        "h_samples": h_samples,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "crop_x": 0.0,
        "crop_y": 160.0,
        "orig_w": 1280,
    }
    assert lane_to_tusimple_samples(lane, meta) == [640] * len(h_samples)


def test_lane_to_tusimple_samples_clamps_rounded_right_edge() -> None:
    meta = {
        "h_samples": (160, 170),
        "scale_x": 1.0,
        "scale_y": 1.0,
        "crop_x": 0.0,
        "crop_y": 0.0,
        "orig_w": 1280,
    }
    lane = [(1279.8, 160.0), (1279.8, 170.0)]
    assert lane_to_tusimple_samples(lane, meta) == [1279, 1279]


def test_tusimple_metric_is_perfect_for_exact_prediction(tmp_path: Path) -> None:
    gt = {
        "raw_file": "clips/example/20.jpg",
        "h_samples": [160, 170, 180, 190],
        "lanes": [[300, 302, 304, 306], [900, 898, 896, 894]],
    }
    pred = {"raw_file": gt["raw_file"], "lanes": gt["lanes"], "run_time": 0.0}
    gt_file = tmp_path / "gt.jsonl"
    pred_file = tmp_path / "pred.jsonl"
    gt_file.write_text(json.dumps(gt) + "\n", encoding="utf-8")
    pred_file.write_text(json.dumps(pred) + "\n", encoding="utf-8")

    results = TuSimpleLaneEval.evaluate(pred_file, gt_file)
    assert results == {
        "Accuracy": 1.0,
        "F1_score": 1.0,
        "FP": 0.0,
        "FN": 0.0,
        "num_samples": 1,
    }
