from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_single_group_bottleneck import (
    _counterfactual_matrices,
    _derive_recall_gains,
    _slice_group,
    _summary,
)


def test_slice_group_uses_contiguous_assignment_block() -> None:
    stage = {
        "pred_x_rows": torch.arange(32 * 3).view(32, 3).float(),
        "exist_logits": torch.zeros(32, 2),
        "range_norm": torch.zeros(32, 2),
        "debug_scalar": torch.tensor(1.0),
    }
    group, ids = _slice_group(
        stage,
        num_query_groups=4,
        query_group_index=2,
    )
    assert ids == list(range(16, 24))
    assert group["pred_x_rows"].shape == (8, 3)
    assert torch.equal(group["pred_x_rows"][0], stage["pred_x_rows"][16])
    assert "debug_scalar" not in group


def test_counterfactuals_separate_range_and_x_errors() -> None:
    stage = {
        "pred_x_rows": torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],
                [30.0, 30.0, 30.0, 30.0],
            ]
        ),
        # Candidate 0 has correct x but covers only the last half. Candidate 1
        # has the correct full range but wrong x.
        "range_norm": torch.tensor(
            [
                [0.50, 0.99],
                [0.00, 0.99],
            ]
        ),
    }
    target = {
        "x_rows": torch.tensor([[10.0, 10.0, 10.0, 10.0]]),
        "valid_mask": torch.tensor([[True, True, True, True]]),
    }
    matrices = _counterfactual_matrices(
        stage,
        target,
        input_h=4,
        input_w=64,
        line_width=10.0,
        min_valid_rows=2,
    )
    assert matrices["predicted"].shape == (1, 2)
    assert float(matrices["predicted"][0, 0]) < 1.0
    assert torch.isclose(
        matrices["oracle_gt_range"][0, 0],
        torch.tensor(1.0),
    )
    assert torch.isclose(
        matrices["oracle_x_pred_range"][0, 1],
        torch.tensor(1.0),
    )
    assert torch.allclose(
        matrices["oracle_x_gt_range"],
        torch.ones(1, 2),
    )
    assert torch.isclose(
        matrices["best_x_mae_px"][0],
        torch.tensor(0.0),
    )


def test_recall_gain_summary_uses_percentage_points() -> None:
    official = {
        "0.50": {
            "model_top4": {"recall": 0.60},
            "score_threshold_all": {"recall": 0.62},
            "all_group_candidates": {"recall": 0.65},
            "oracle_top4_predicted": {"recall": 0.64},
        }
    }
    row_space = {
        "0.50": {
            "oracle_top4__predicted": {"recall": 0.63},
            "oracle_top4__oracle_gt_range": {"recall": 0.73},
            "oracle_top4__oracle_x_pred_range": {"recall": 0.83},
            "oracle_top4__oracle_x_gt_range": {"recall": 0.93},
        }
    }
    gains = _derive_recall_gains(official, row_space)["0.50"]
    assert abs(gains["official_oracle_top4_vs_model_top4_points"] - 4.0) < 1e-8
    assert abs(gains["row_oracle_gt_range_vs_predicted_points"] - 10.0) < 1e-8
    assert abs(gains["row_oracle_x_vs_predicted_points"] - 20.0) < 1e-8


def test_x_error_summary_reports_pixel_cdf() -> None:
    summary = _summary([1.0, 3.0, 7.0, 20.0])
    assert summary["median"] == 7.0
    assert summary["fraction_le_2px"] == 0.25
    assert summary["fraction_le_4px"] == 0.50
    assert summary["fraction_le_8px"] == 0.75
    assert summary["fraction_le_16px"] == 0.75
