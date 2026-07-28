from __future__ import annotations

import torch

from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion


def _targets_and_matches():
    targets = [
        {
            "x_rows": torch.tensor(
                [
                    [20.0, 20.0, 20.0, 20.0],
                    [80.0, 80.0, 80.0, 80.0],
                ]
            ),
            "valid_mask": torch.tensor(
                [
                    [True, True, True, True],
                    [True, False, False, False],
                ]
            ),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([0, 1]),
            "gt_indices": torch.tensor([0, 1]),
        }
    ]
    return targets, matches


def test_lane_mean_point_loss_gives_short_lane_equal_total_weight() -> None:
    targets, matches = _targets_and_matches()
    outputs = {
        "pred_x_rows": torch.tensor(
            [
                [
                    [30.0, 30.0, 30.0, 30.0],
                    [120.0, 120.0, 120.0, 120.0],
                ]
            ]
        )
    }
    global_loss = S0Criterion(
        LossConfig(input_w=100, smooth_l1_beta=0.0, geometry_reduction="global_rows")
    ).compute_point_loss(outputs, targets, matches)
    lane_loss = S0Criterion(
        LossConfig(input_w=100, smooth_l1_beta=0.0, geometry_reduction="lane_mean")
    ).compute_point_loss(outputs, targets, matches)
    # Global rows: (4 * 0.1 + 1 * 0.4) / 5 = 0.16.
    # Lane mean:   (0.1 + 0.4) / 2 = 0.25.
    assert torch.allclose(global_loss, torch.tensor(0.16), atol=1e-6)
    assert torch.allclose(lane_loss, torch.tensor(0.25), atol=1e-6)


def test_lane_mean_dfl_differs_only_when_lane_losses_differ() -> None:
    targets, matches = _targets_and_matches()
    logits = torch.zeros(1, 2, 4, 5)
    # Long lane target is bin 1; make it easy. Short lane target is bin 4;
    # make it deliberately difficult.
    logits[0, 0, :, 1] = 5.0
    logits[0, 1, :, 0] = 5.0
    outputs = {
        "row_x_logits": logits,
        "pred_x_rows": torch.zeros(1, 2, 4),
    }
    global_loss = S0Criterion(
        LossConfig(input_w=100, geometry_reduction="global_rows")
    ).compute_row_dfl_loss(outputs, targets, matches)
    lane_loss = S0Criterion(
        LossConfig(input_w=100, geometry_reduction="lane_mean")
    ).compute_row_dfl_loss(outputs, targets, matches)
    assert lane_loss > global_loss


def test_default_geometry_reduction_preserves_historical_behavior() -> None:
    assert LossConfig().geometry_reduction == "global_rows"


def test_unknown_geometry_reduction_fails_loudly() -> None:
    try:
        S0Criterion(LossConfig(geometry_reduction="mystery"))
    except ValueError as error:
        assert "geometry_reduction" in str(error)
    else:
        raise AssertionError("Unknown geometry reduction should fail")
