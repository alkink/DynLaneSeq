from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.analyze_v4_selection_coverage import (
    _build_verdict,
    _curve_distance_matrix,
    _finish_counter,
    _mmr_ids,
    _new_counter,
    _update_counter,
)


def _stage() -> dict[str, torch.Tensor]:
    rows = 8
    return {
        "pred_x_rows": torch.tensor(
            [
                [100.0] * rows,
                [104.0] * rows,
                [300.0] * rows,
            ]
        ),
        "range_norm": torch.tensor([[0.0, 1.0]] * 3),
        "exist_logits": torch.tensor(
            [[2.0, 0.0], [1.8, 0.0], [1.5, 0.0]]
        ),
    }


def test_curve_distance_and_mmr_prefer_a_distinct_lane() -> None:
    distance = _curve_distance_matrix(
        _stage(),
        input_h=640,
        input_w=1600,
        min_valid_rows=5,
        row_visibility_thresh=0.0,
        min_overlap_points=5,
    )
    assert float(distance[0, 1]) == pytest.approx(4.0)
    assert float(distance[0, 2]) == pytest.approx(200.0)
    selected = _mmr_ids(
        torch.tensor([0.90, 0.85, 0.80]),
        distance,
        torch.ones(3, dtype=torch.bool),
        penalty=0.50,
        sigma=10.0,
        top_k=2,
    )
    assert selected == [0, 2]


def test_counter_marks_redundant_high_iou_selection_as_duplicate() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.80, 0.05],
            [0.05, 0.10, 0.90],
        ]
    )
    distance = torch.tensor(
        [
            [0.0, 4.0, 200.0],
            [4.0, 0.0, 196.0],
            [200.0, 196.0, 0.0],
        ]
    )
    redundant = _new_counter()
    _update_counter(
        redundant,
        iou,
        [0, 1],
        distance,
        threshold=0.50,
        near_min_iou=0.30,
    )
    redundant_result = _finish_counter(redundant)
    assert redundant_result["tp"] == 1
    assert redundant_result["false_positive_breakdown"]["duplicate_fp"][
        "count"
    ] == 1
    assert redundant_result["selected_curve_diversity"][
        "close_pair_fraction_below_20px"
    ] == pytest.approx(1.0)

    diverse = _new_counter()
    _update_counter(
        diverse,
        iou,
        [0, 2],
        distance,
        threshold=0.50,
        near_min_iou=0.30,
    )
    diverse_result = _finish_counter(diverse)
    assert diverse_result["tp"] == 2
    assert diverse_result["recall"] == pytest.approx(1.0)


def test_verdict_requires_material_recovery_of_oracle_gap() -> None:
    base = {
        "precision": 0.50,
        "recall": 0.50,
        "false_positive_breakdown": {
            "duplicate_fp": {"fraction_of_fp": 0.75}
        },
    }
    diverse = {
        "precision": 0.75,
        "recall": 0.80,
        "false_positive_breakdown": {
            "duplicate_fp": {"fraction_of_fp": 0.10}
        },
    }
    methods = {
        "score_top4": {"0.50": base},
        "hard_diverse_20px": {"0.50": diverse},
    }
    capacity = {
        "0.50": {"all_candidate_oracle": {"recall": 0.90}}
    }
    verdict = _build_verdict(
        methods,
        capacity,
        primary_threshold=0.50,
    )
    assert verdict["best_diversity_method"] == "hard_diverse_20px"
    assert verdict["best_diversity_gain_points"] == pytest.approx(30.0)
    assert verdict["fraction_of_oracle_gap_recovered"] == pytest.approx(0.75)
    assert (
        verdict["interpretation"]
        == "duplicate_coverage_is_a_primary_selection_bottleneck"
    )
