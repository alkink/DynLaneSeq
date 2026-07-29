from __future__ import annotations

from dynlaneseq_eg.tools.evaluate_row_reference_gate import _summarize_pair


def _records(values: list[float], wrong: list[float] | None = None):
    if wrong is None:
        wrong = [0.0 for _ in values]
    return [
        {
            "dataset_index": index,
            "lane_index": 0,
            "all_iou": value,
            "top_iou": value,
            "wrong_image_all_iou": wrong[index],
        }
        for index, value in enumerate(values)
    ]


def test_gate_requires_unique_recovery_hit_retention_and_image_specificity() -> None:
    control = {"records": _records([0.2] * 5 + [0.8] * 5)}
    candidate = {
        "records": _records(
            [0.6] * 5 + [0.8] * 5,
            wrong=[0.1] * 10,
        )
    }
    summary = _summarize_pair(
        control,
        candidate,
        min_recovered_lanes=5,
        max_lost_lanes=2,
        min_image_specificity_points=5.0,
    )
    assert summary["positive_gate"] is True
    assert (
        summary["thresholds"]["0.50"]["unique_control_misses_recovered"]
        == 5
    )
    assert summary["thresholds"]["0.50"]["control_hits_lost"] == 0


def test_gate_rejects_capacity_that_destroys_existing_hits() -> None:
    control = {"records": _records([0.2] * 5 + [0.8] * 5)}
    candidate = {
        "records": _records(
            [0.6] * 5 + [0.2] * 3 + [0.8] * 2,
            wrong=[0.1] * 10,
        )
    }
    summary = _summarize_pair(
        control,
        candidate,
        min_recovered_lanes=5,
        max_lost_lanes=2,
        min_image_specificity_points=5.0,
    )
    assert summary["positive_gate"] is False
    assert summary["thresholds"]["0.50"]["control_hits_lost"] == 3

