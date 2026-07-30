from __future__ import annotations

import json

import pytest

from dynlaneseq_eg.tools.summarize_row_reference_gate_trajectory import summarize


def _payload(iteration: int, *, gain: float = 0.2) -> dict:
    lanes = 10
    control_recall = 0.4
    candidate_recall = control_recall + gain
    threshold = {
        "control_all_recall": control_recall,
        "candidate_all_recall": candidate_recall,
        "candidate_wrong_image_all_recall": 0.1,
        "candidate_gain_points": 100.0 * gain,
        "candidate_image_specificity_points": 100.0 * (candidate_recall - 0.1),
        "unique_control_misses_recovered": 3,
        "control_hits_lost": 1,
        "net_unique_hits": 2,
        "control_scored_topk_recall": 0.2,
        "candidate_scored_topk_recall": 0.4,
    }
    return {
        "diagnostic_only": True,
        "split": "val",
        "sample_strategy": "uniform",
        "images": 4,
        "control": {
            "iteration": iteration,
            "sampled_dataset_indices": [0, 1, 2, 3],
        },
        "candidate": {
            "iteration": iteration,
            "sampled_dataset_indices": [0, 1, 2, 3],
        },
        "summary": {
            "lanes": lanes,
            "thresholds": {"0.50": threshold, "0.70": threshold},
            "control_mean_best_iou": 0.4,
            "candidate_mean_best_iou": 0.5,
            "mean_best_iou_gain": 0.1,
        },
    }


def test_summarize_orders_checkpoints_and_detects_stable_trend(tmp_path):
    paths = []
    for iteration in (10000, 2500, 7500, 5000):
        path = tmp_path / f"{iteration}.json"
        path.write_text(json.dumps(_payload(iteration)), encoding="utf-8")
        paths.append(str(path))

    result = summarize(paths)

    assert [row["iteration"] for row in result["trajectory"]] == [
        2500,
        5000,
        7500,
        10000,
    ]
    assert result["stable_positive_trajectory"] is True
    assert result["trajectory"][0]["control_hit_retention_050"] == 0.75


def test_summarize_fixed_control_uses_candidate_iteration(tmp_path):
    paths = []
    for candidate_iteration in (55000, 52500):
        payload = _payload(50000)
        payload["candidate"]["iteration"] = candidate_iteration
        path = tmp_path / f"{candidate_iteration}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))

    with pytest.raises(ValueError, match="iterations differ"):
        summarize(paths)

    result = summarize(paths, allow_fixed_control=True)

    assert result["fixed_control"] is True
    assert [row["iteration"] for row in result["trajectory"]] == [52500, 55000]
    assert {row["control_iteration"] for row in result["trajectory"]} == {50000}
    assert [row["candidate_iteration"] for row in result["trajectory"]] == [
        52500,
        55000,
    ]
