from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch

from dynlaneseq_eg.evaluation import candidate_diagnostics
from dynlaneseq_eg.tools.analyze_oracle_topk import _resolve_operating_points
from dynlaneseq_eg.tools.summarize_hybrid_primary_auxiliary_full_validation import (
    summarize,
)


def _row(
    strategy: str,
    iou: float,
    *,
    recall: float,
    quality_power: float | None = None,
    score_threshold: float | None = None,
    f1: float | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    row = {
        "stage": "main",
        "strategy": strategy,
        "top_k": top_k,
        "iou_threshold": iou,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "recall": recall,
    }
    if f1 is not None:
        total = 1000
        tp = int(round(total * f1))
        fp = total - tp
        fn = total - tp
        row[f"official_iou_{iou:g}"] = {
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return row


def _metadata() -> dict[str, Any]:
    return {
        "split": "val",
        "list_sha256": "same",
        "max_batches": 0,
        "num_records": 8,
        "iou_space": "official_raster",
        "sample_strategy": "sequential",
        "sampled_dataset_indices": list(range(8)),
        "candidate_counts_by_stage": {"main": 32},
    }


def _q025_report(*, candidate: bool) -> dict[str, Any]:
    rows = []
    for iou, raw, quality, oracle in (
        (0.5, 0.95, 0.65, 0.95),
        (0.75, 0.74, 0.50, 0.74),
    ):
        if candidate:
            raw += -0.005 if iou == 0.5 else 0.02
            quality += 0.05 if iou == 0.5 else 0.03
            oracle = raw
        rows.extend(
            [
                _row("all_raw", iou, recall=raw, top_k=0),
                _row("oracle_topk", iou, recall=oracle),
                _row("quality_topk", iou, recall=quality),
            ]
        )
        control_f1 = {
            (0.5, 0.15): 0.770,
            (0.5, 0.20): 0.740,
            (0.75, 0.15): 0.540,
            (0.75, 0.20): 0.530,
        }
        candidate_f1 = {
            (0.5, 0.15): 0.772,
            (0.5, 0.20): 0.771,
            (0.75, 0.15): 0.535,
            (0.75, 0.20): 0.560,
        }
        table = candidate_f1 if candidate else control_f1
        for threshold in (0.15, 0.20):
            rows.append(
                _row(
                    "model_topk_nms",
                    iou,
                    recall=table[(iou, threshold)],
                    quality_power=0.25,
                    score_threshold=threshold,
                    f1=table[(iou, threshold)],
                )
            )
    return {"metadata": _metadata(), "rows": rows}


def _historical_report(*, candidate: bool) -> dict[str, Any]:
    rows = []
    for iou, control_f1, candidate_f1 in (
        (0.5, 0.70, 0.72),
        (0.75, 0.49, 0.51),
    ):
        value = candidate_f1 if candidate else control_f1
        rows.append(
            _row(
                "model_topk_nms",
                iou,
                recall=value,
                quality_power=0.50,
                score_threshold=0.30,
                f1=value,
            )
        )
    return {"metadata": _metadata(), "rows": rows}


def test_full_validation_summary_uses_frozen_and_historical_points() -> None:
    payload = summarize(
        _q025_report(candidate=False),
        _q025_report(candidate=True),
        _historical_report(candidate=False),
        _historical_report(candidate=True),
    )

    assert payload["protocol"]["images"] == 8
    assert payload["gate"]["geometry_preserved"] is True
    assert payload["gate"]["primary_f1_preserved"] is True
    assert payload["gate"]["strict_f1_improved"] is True
    assert (
        payload["gate"]["verdict"]
        == "confirmed_continue_exact_checkpoint_to_50k"
    )
    selected = payload["selected_operating_points"]
    assert selected["control"]["score_threshold"] == 0.15
    assert selected["candidate"]["score_threshold"] == 0.20
    assert selected["candidate_minus_control"]["0.75"]["f1_points"] == pytest.approx(
        2.0
    )
    historical = payload["historical_q0.50_score0.30"]
    assert historical["candidate_minus_control"]["0.50"]["f1_points"] == pytest.approx(
        2.0
    )


def test_full_validation_summary_rejects_subset_reports() -> None:
    control = _q025_report(candidate=False)
    candidate = _q025_report(candidate=True)
    control_historical = _historical_report(candidate=False)
    candidate_historical = _historical_report(candidate=True)
    for report in (
        control,
        candidate,
        control_historical,
        candidate_historical,
    ):
        report["metadata"]["max_batches"] = 64

    with pytest.raises(ValueError, match="refuses a max-batches subset"):
        summarize(control, candidate, control_historical, candidate_historical)


def test_full_validation_summary_accepts_combined_fixed_point_reports() -> None:
    control = _q025_report(candidate=False)
    control["rows"].extend(_historical_report(candidate=False)["rows"])
    candidate = _q025_report(candidate=True)
    candidate["rows"].extend(_historical_report(candidate=True)["rows"])

    payload = summarize(control, candidate, control, candidate)

    assert payload["selected_operating_points"]["candidate"]["score_threshold"] == 0.20
    assert payload["historical_q0.50_score0.30"]["candidate"]["quality_power"] == 0.50


def test_explicit_operating_points_are_paired_not_cartesian() -> None:
    points = _resolve_operating_points(
        ["0.25:0.15", "0.25:0.20", "0.50:0.30", "0.25:0.15"],
        [0.25, 0.50],
        [0.15, 0.20, 0.30],
    )
    assert points == [(0.25, 0.15), (0.25, 0.20), (0.50, 0.30)]


def test_operating_points_keep_legacy_cartesian_default() -> None:
    assert _resolve_operating_points([], [0.25, 0.50], [0.15, 0.30]) == [
        (0.25, 0.15),
        (0.25, 0.30),
        (0.50, 0.15),
        (0.50, 0.30),
    ]


@pytest.mark.parametrize("encoded", [["bad"], ["0.25:1.1"], ["-0.1:0.2"]])
def test_operating_points_reject_invalid_values(encoded: list[str]) -> None:
    with pytest.raises(ValueError):
        _resolve_operating_points(encoded, [0.25], [0.15])


def test_parallel_official_iou_cache_matches_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_iou(record, stage_name, **_kwargs):
        value = float(record["value"])
        return torch.tensor([[value]]), torch.tensor([True])

    monkeypatch.setattr(
        candidate_diagnostics,
        "official_proposal_gt_iou_matrix",
        fake_iou,
    )
    template = {
        "metadata": {},
        "records": [
            {"value": index, "stages": {"main": {}}}
            for index in range(6)
        ],
    }
    sequential = deepcopy(template)
    sequential["metadata"]["cache_path"] = str(tmp_path / "sequential.pt")
    parallel = deepcopy(template)
    parallel["metadata"]["cache_path"] = str(tmp_path / "parallel.pt")

    candidate_diagnostics.ensure_official_iou_cache(sequential, workers=1)
    candidate_diagnostics.ensure_official_iou_cache(parallel, workers=3)

    for expected, actual in zip(sequential["records"], parallel["records"]):
        assert torch.equal(
            expected["stages"]["main"]["official_iou"],
            actual["stages"]["main"]["official_iou"],
        )
        assert torch.equal(
            expected["stages"]["main"]["official_candidate_valid"],
            actual["stages"]["main"]["official_candidate_valid"],
        )
