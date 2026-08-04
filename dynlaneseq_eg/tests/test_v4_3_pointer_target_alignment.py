from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.analyze_v4_3_pointer_target_alignment import (
    compare_target_alignment_image,
    summarize_alignment,
)


def test_alignment_separates_learning_gap_when_target_is_correct() -> None:
    official = torch.tensor(
        [
            [0.90, 0.80, 0.05, 0.00],
            [0.05, 0.00, 0.85, 0.70],
        ]
    )
    row = official.clone()
    result = compare_target_alignment_image(
        row,
        official,
        torch.ones(4, dtype=torch.bool),
        torch.tensor([1, 3, -1, -1]),
        torch.tensor([0, 2, 4, -100]),
    )
    report = summarize_alignment([result], (0.50, 0.75))
    strict = report["official_iou_gap_decomposition"]["0.75"]
    assert strict["official_oracle_tp"] == 2
    assert strict["training_target_set_tp"] == 2
    assert strict["pointer_tp"] == 1
    assert strict["training_target_definition_gap"] == 0
    assert strict["pointer_learning_or_exposure_gap"] == 1
    assert report["verdict"] == "pointer_learning_or_exposure_is_primary"


def test_alignment_separates_surrogate_target_mismatch() -> None:
    official = torch.tensor(
        [
            [0.90, 0.70, 0.05, 0.00],
            [0.05, 0.00, 0.85, 0.70],
        ]
    )
    # The row surrogate incorrectly prefers candidates 1 and 3.
    row = torch.tensor(
        [
            [0.60, 0.95, 0.05, 0.00],
            [0.05, 0.00, 0.60, 0.95],
        ]
    )
    result = compare_target_alignment_image(
        row,
        official,
        torch.ones(4, dtype=torch.bool),
        torch.tensor([1, 3, -1, -1]),
        torch.tensor([1, 3, 4, -100]),
    )
    report = summarize_alignment([result], (0.50, 0.75))
    strict = report["official_iou_gap_decomposition"]["0.75"]
    assert strict["official_oracle_tp"] == 2
    assert strict["training_target_set_tp"] == 0
    assert strict["pointer_tp"] == 0
    assert strict["training_target_definition_gap"] == 2
    assert strict["pointer_learning_or_exposure_gap"] == 0
    assert report["verdict"] == "row_strip_target_misalignment_is_primary"
    alignment = report["row_strip_vs_official"]
    assert alignment["best_candidate_top1_agreement"] == 0.0
    assert alignment["official_regret_when_using_row_strip_top1"]["mean"] == pytest.approx(
        0.175
    )


def test_alignment_reports_pointer_target_set_overlap() -> None:
    matrix = torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.1, 0.85]])
    result = compare_target_alignment_image(
        matrix,
        matrix,
        torch.ones(3, dtype=torch.bool),
        torch.tensor([0, 1, -1, -1]),
        torch.tensor([0, 2, 3, -100]),
    )
    # Target set is {0,2}; pointer set is {0,1}.
    assert result["target_pointer_overlap"] == 1
    assert result["target_pointer_union"] == 3
    assert result["exact_target_set"] is False
