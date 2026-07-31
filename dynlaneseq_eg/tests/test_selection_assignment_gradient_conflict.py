from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_selection_assignment_gradient_conflict import (
    GradientPairStats,
    _selection_assignment_map,
    compare_assignment_maps,
)


def test_assignment_comparison_detects_opposite_official_winners() -> None:
    official_iou = torch.tensor(
        [
            [0.80, 0.55, 0.10, 0.05],
            [0.05, 0.10, 0.45, 0.78],
        ]
    )
    result = compare_assignment_maps(
        main_by_gt={0: 0, 1: 2},
        selection_by_gt={0: 1, 1: 3},
        official_iou=official_iou,
        target_to_official={0: 0, 1: 1},
        win_margin=0.02,
    )

    assert result["agreement"] == 0
    assert result["disagreement"] == 2
    assert result["main_wins"] == 1
    assert result["selection_wins"] == 1
    assert result["transitions"]["0.50"] == {
        "main_miss_selection_hit": 1,
        "main_hit_selection_miss": 0,
    }


def test_gradient_pair_stats_reports_negative_alignment() -> None:
    stats = GradientPairStats()
    stats.update(
        torch.tensor([1.0, 0.0]),
        torch.tensor([-2.0, 0.0]),
    )
    summary = stats.summary()

    assert summary["cosine"]["mean"] == -1.0
    assert summary["negative_fraction"] == 1.0
    assert summary["right_to_left_norm_ratio"]["mean"] == 2.0


def test_zero_quality_hungarian_pair_is_not_a_positive_teacher() -> None:
    assignment = _selection_assignment_map(
        torch.tensor(
            [
                [0.8, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        torch.tensor([True, True]),
    )

    assert assignment == {0: 0}
