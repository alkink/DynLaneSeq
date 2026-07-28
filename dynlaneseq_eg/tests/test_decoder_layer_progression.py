from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.analyze_decoder_layer_progression import (
    LayerTransitionStats,
    RecallTransitionStats,
)


def test_layer_transition_stats_uses_only_group_zero_and_tracks_no_harm() -> None:
    current = torch.zeros(1, 8, 4)
    following = torch.zeros_like(current)
    current[0, 0] = torch.tensor([0.0, 6.0, 9.0, 10.0])
    following[0, 0] = torch.tensor([5.0, 12.0, 10.0, 15.0])

    # This deliberately bad duplicate belongs to group one and must not affect
    # the group-zero deployment diagnostic.
    current[0, 4] = -100.0
    following[0, 4] = 100.0

    targets = [
        {
            "x_rows": torch.tensor([[10.0, 10.0, 10.0, 10.0]]),
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([0, 4]),
            "gt_indices": torch.tensor([0, 0]),
        }
    ]

    stats = LayerTransitionStats()
    stats.update(current, following, targets, matches, group_size=4)
    summary = stats.summary()

    all_rows = summary["all"]
    assert all_rows["rows"] == 4
    assert all_rows["mae_before_px"] == pytest.approx(3.75)
    assert all_rows["mae_after_px"] == pytest.approx(3.0)
    assert all_rows["mean_error_reduction_px"] == pytest.approx(0.75)
    assert all_rows["improved_fraction"] == pytest.approx(0.75)
    assert all_rows["worsened_fraction"] == pytest.approx(0.25)
    assert all_rows["direction_accuracy_over_4px"] == pytest.approx(1.0)
    assert all_rows["harm_from_4px_fraction"] == pytest.approx(1.0 / 3.0)
    assert all_rows["rescue_to_8px_fraction"] == pytest.approx(1.0)

    assert summary["error_8_16px"]["rows"] == 1
    assert summary["error_16px_plus"]["rows"] == 0


def test_recall_transition_stats_separates_retained_lost_and_gained_lanes() -> None:
    stats = RecallTransitionStats((0.5,))
    stats.update(
        torch.tensor([0.60, 0.40, 0.80, 0.20]),
        torch.tensor([0.70, 0.55, 0.30, 0.10]),
    )

    row = stats.summary()["0.50"]
    assert row["gt"] == 4
    assert row["retained"] == 1
    assert row["lost"] == 1
    assert row["gained"] == 1
    assert row["remained_miss"] == 1
    assert row["before_recall"] == pytest.approx(0.5)
    assert row["after_recall"] == pytest.approx(0.5)
    assert row["retention_fraction"] == pytest.approx(0.5)
    assert row["recovery_fraction"] == pytest.approx(0.5)
    assert row["mean_best_iou_delta"] == pytest.approx(-0.0875)
    assert row["iou_improved_fraction"] == pytest.approx(0.5)
