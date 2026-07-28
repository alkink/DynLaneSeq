from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.analyze_cross_backbone_error_overlap import (
    QuerySpecializationStats,
    _lane_shape_stats,
    _pearson,
    summarize_records,
)
from dynlaneseq_eg.tools.diagnostic_sampling import uniformly_spaced_indices


def _record(r_iou: float, d_iou: float) -> dict[str, float | int]:
    return {
        "valid_rows": 80,
        "row_span": 80,
        "curvature_px": 1.0,
        "lanes_in_image": 4,
        "mean_x_norm": 0.5,
        "r34_l4_group0_iou": r_iou,
        "dla34_l4_group0_iou": d_iou,
        "r34_layer_union_group0_iou": min(r_iou + 0.1, 1.0),
        "dla34_layer_union_group0_iou": min(d_iou + 0.1, 1.0),
        "cross_backbone_l4_all32_union_iou": max(r_iou, d_iou),
        "r34_l4_all32_iou": r_iou,
        "dla34_l4_all32_iou": d_iou,
        "r34_l4_model_top4_iou": r_iou,
        "dla34_l4_model_top4_iou": d_iou,
    }


def test_pearson_detects_aligned_and_opposed_errors() -> None:
    assert _pearson([0.1, 0.2, 0.3], [0.2, 0.4, 0.6]) == pytest.approx(1.0)
    assert _pearson([0.1, 0.2, 0.3], [0.6, 0.4, 0.2]) == pytest.approx(-1.0)


def test_overlap_summary_counts_complementary_hits() -> None:
    records = [
        _record(0.8, 0.8),
        _record(0.8, 0.2),
        _record(0.2, 0.8),
        _record(0.2, 0.2),
    ]
    summary = summarize_records(records, (0.5,))
    threshold = summary["thresholds"]["0.50"]
    assert threshold["r34_recall"] == pytest.approx(0.5)
    assert threshold["dla34_recall"] == pytest.approx(0.5)
    assert threshold["cross_backbone_union_recall"] == pytest.approx(0.75)
    assert threshold["status_counts"] == {
        "common_hit": 1,
        "r34_only": 1,
        "dla34_only": 1,
        "common_miss": 1,
    }


def test_lane_shape_stats_use_only_consecutive_valid_rows() -> None:
    x = torch.tensor([0.0, 1.0, 4.0, 20.0, 25.0])
    valid = torch.tensor([True, True, True, False, True])
    stats = _lane_shape_stats(x, valid, input_w=100)
    assert stats["valid_rows"] == 4
    assert stats["row_span"] == 5
    assert stats["mean_x_norm"] == pytest.approx(0.075)
    assert stats["curvature_px"] == pytest.approx(2.0)


def test_uniform_indices_span_the_complete_split_without_repeats() -> None:
    indices = uniformly_spaced_indices(9675, 64)
    assert len(indices) == 64
    assert len(set(indices)) == 64
    assert indices[0] == 0
    assert indices[-1] == 9674
    assert indices[1] > 64


def test_query_specialization_separates_assigned_and_useful_queries() -> None:
    stats = QuerySpecializationStats(group_size=3)
    matrix = torch.tensor(
        [
            [0.8, 0.1],
            [0.2, 0.7],
            [0.4, 0.1],
        ]
    )
    stats.update(matrix, gt_ranks={0: 0, 1: 1}, assignments={0: 0, 1: 1})
    summary = stats.summary()
    assert summary["mean_active_queries_iou030_per_image"] == pytest.approx(3.0)
    assert summary["mean_active_queries_iou050_per_image"] == pytest.approx(2.0)
    assert summary["mean_unassigned_useful_iou030_per_image"] == pytest.approx(1.0)
    assert summary["mean_unassigned_useful_iou050_per_image"] == pytest.approx(0.0)
