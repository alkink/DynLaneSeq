from __future__ import annotations

from copy import deepcopy

import pytest

from dynlaneseq_eg.tools.analyze_oracle_topk import _official_metric_key
from dynlaneseq_eg.tools.summarize_official_pr_frontier_pair import (
    summarize_pair,
)


def _metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _row(
    iou_threshold: float,
    quality_power: float,
    score_threshold: float,
    *,
    tp: int,
    fp: int,
    fn: int,
) -> dict:
    metrics = _metrics(tp, fp, fn)
    return {
        "stage": "main",
        "strategy": "model_topk_nms",
        "top_k": 4,
        "iou_threshold": iou_threshold,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "hits": tp,
        "gt": tp + fn,
        "recall": metrics["recall"],
        "mean_best_iou": 0.75,
        "images": 4,
        _official_metric_key(iou_threshold): metrics,
    }


def _report(candidate: bool = False) -> dict:
    rows: list[dict] = []
    for iou_threshold in (0.5, 0.75):
        if candidate:
            outcomes = (
                (0.0, 0.10, 82, 18, 18),
                (0.5, 0.15, 79, 7, 21),
            )
        else:
            outcomes = (
                (0.0, 0.10, 80, 20, 20),
                (0.5, 0.15, 76, 8, 24),
            )
        rows.extend(
            _row(
                iou_threshold,
                quality_power,
                score_threshold,
                tp=tp,
                fp=fp,
                fn=fn,
            )
            for quality_power, score_threshold, tp, fp, fn in outcomes
        )
    return {
        "metadata": {
            "split": "val",
            "list_sha256": "same",
            "max_batches": 1,
            "num_records": 4,
            "iou_space": "official_raster",
            "sample_strategy": "uniform",
            "sampled_dataset_indices": [0, 3, 6, 9],
            "top_k_values": [4],
            "iou_thresholds": [0.5, 0.75],
            "quality_powers": [0.0, 0.5],
            "score_thresholds": [0.1, 0.15],
            "line_width": 30.0,
            "nms_distance_thresh_px": 20.0,
            "nms_min_overlap_points": 5,
        },
        "rows": rows,
    }


def test_official_metric_key_preserves_requested_strict_iou() -> None:
    assert _official_metric_key(0.5) == "official_iou_0.5"
    assert _official_metric_key(0.75) == "official_iou_0.75"


def test_pr_frontier_summary_finds_candidate_advantage_at_both_ious() -> None:
    payload = summarize_pair(_report(), _report(candidate=True), top_k=4)
    assert payload["comparability"]["all_checks_pass"]
    assert set(payload["iou"]) == {"0.5", "0.75"}
    for summary in payload["iou"].values():
        assert summary["candidate_has_higher_observed_max_f1"]
        assert summary["best_f1_delta_points"] > 0.0
        assert summary["base_frontier_points_covered_by_candidate"] > 0
        assert len(summary["paired_grid"]) == 2


def test_pr_frontier_summary_rejects_mismatched_grid() -> None:
    candidate = _report(candidate=True)
    candidate = deepcopy(candidate)
    candidate["metadata"]["score_thresholds"] = [0.1]
    with pytest.raises(ValueError, match="not directly comparable"):
        summarize_pair(_report(), candidate, top_k=4)
