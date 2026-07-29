from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_row_distribution_decoding import (
    DecodeStats,
    DistributionStats,
    decode_argmax,
    decode_local_mode_expectation,
)


def test_argmax_decode_uses_model_bin_coordinate_contract() -> None:
    logits = torch.tensor([[[0.0, 1.0, 3.0, 2.0]]])
    decoded = decode_argmax(logits, input_w=16, x_bins=4)
    torch.testing.assert_close(decoded, torch.tensor([[8.0]]))


def test_local_mode_ignores_distant_secondary_peak() -> None:
    logits = torch.full((1, 1, 8), -20.0)
    logits[..., 1] = 3.0
    logits[..., 2] = 2.0
    logits[..., 7] = 2.9
    decoded = decode_local_mode_expectation(
        logits,
        radius_bins=1,
        input_w=80,
        x_bins=8,
    )
    assert 10.0 <= float(decoded) < 20.0


def test_decode_stats_reports_recall_and_assigned_mae() -> None:
    stats = DecodeStats(thresholds=(0.5,))
    gt = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
    valid = torch.ones(5, dtype=torch.bool)
    candidates = torch.stack((gt, gt + 100.0))
    stats.update_lane(
        candidates,
        gt_x=gt,
        valid=valid,
        assigned_candidate=gt + 2.0,
        line_width=30.0,
    )
    summary = stats.summary()
    assert summary["raw_recall@0.50"] == 1.0
    assert summary["assigned_row_mae_px"] == 2.0


def test_distribution_stats_locates_gt_bin_and_probability_mass() -> None:
    stats = DistributionStats()
    logits = torch.tensor(
        [
            [0.0, 1.0, 8.0, 0.0],
            [0.0, 7.0, 1.0, 0.0],
        ]
    )
    valid = torch.ones(2, dtype=torch.bool)
    gt_x = torch.tensor([8.0, 4.0])
    stats.update(
        logits,
        valid_rows=valid,
        input_w=16,
        x_bins=4,
        gt_x=gt_x,
    )
    summary = stats.summary()
    assert summary["valid_gt_rows"] == 2
    assert summary["median_gt_bin_rank"] == 1
    assert summary["p90_gt_bin_rank"] == 1
    assert summary["fraction_gt_bin_in_top1"] == 1.0
    assert summary["mean_mode_abs_error_px"] == 0.0
    assert summary["mean_probability_at_nearest_gt_bin"] > 0.99
    assert summary["mean_gt_probability_mass_within_1_bins"] > 0.99
