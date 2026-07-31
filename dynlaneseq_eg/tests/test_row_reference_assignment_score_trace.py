from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.analyze_row_reference_assignment_score_trace import (
    CandidateTraceAccumulator,
    _provisional_diagnosis,
    _training_quality_targets_for_image,
)


def test_candidate_trace_summarizes_assignment_loss_and_churn() -> None:
    stats = CandidateTraceAccumulator(num_layers=4)
    stats.update(
        score=0.2,
        exist=0.4,
        quality=0.25,
        exist_target=0.0,
        quality_target=0.0,
        best_iou=0.8,
        status="below_threshold",
        assignments=[1, 1, None, None],
    )
    stats.update(
        score=0.3,
        exist=0.5,
        quality=0.36,
        exist_target=1.0,
        quality_target=0.8,
        best_iou=0.9,
        status="below_threshold",
        assignments=[None, 2, 3, 3],
    )
    summary = stats.summary()
    assert summary["count"] == 2
    assert summary["matched_fraction_by_decoder_layer"] == [0.5, 1.0, 0.5, 0.5]
    assert summary["matched_earlier_but_not_final_fraction"] == pytest.approx(0.5)
    assert summary["final_layer_matched_fraction"] == pytest.approx(0.5)
    assert summary["gt_identity_change_rate_when_consecutively_matched"] == pytest.approx(1.0 / 3.0)
    assert summary["mean_training_exist_target"] == pytest.approx(0.5)
    assert summary["mean_training_quality_target"] == pytest.approx(0.4)


def test_provisional_diagnosis_detects_missing_assignment_coverage() -> None:
    row = {
        "scopes": {
            "oracle_missed_gt_rescue_candidate": {
                "count": 10,
                "never_matched_fraction": 0.7,
                "final_layer_matched_fraction": 0.2,
                "matched_earlier_but_not_final_fraction": 0.1,
            }
        }
    }
    result = _provisional_diagnosis(row)
    assert result["primary_signal"] == "assignment_coverage_failure"


def test_provisional_diagnosis_detects_post_assignment_score_failure() -> None:
    row = {
        "scopes": {
            "oracle_missed_gt_rescue_candidate": {
                "count": 10,
                "never_matched_fraction": 0.1,
                "final_layer_matched_fraction": 0.8,
                "matched_earlier_but_not_final_fraction": 0.1,
            }
        }
    }
    result = _provisional_diagnosis(row)
    assert result["primary_signal"] == "score_learning_failure_after_positive_assignment"


def test_training_quality_target_matches_row_strip_iou_and_zeros_unmatched() -> None:
    pred = torch.tensor([[10.0, 10.0], [50.0, 50.0]])
    target = {
        "x_rows": torch.tensor([[10.0, 10.0]]),
        "valid_mask": torch.tensor([[True, True]]),
    }
    quality = _training_quality_targets_for_image(
        pred,
        target,
        {0: 0},
        radius=15.0,
    )
    assert quality.tolist() == pytest.approx([1.0, 0.0])
