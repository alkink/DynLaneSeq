from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_geometry_loss_gradients import (
    lane_balanced_dfl_loss,
    lane_balanced_point_loss,
)


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "x_rows": torch.tensor(
                [
                    [2.0, 4.0, 6.0, 8.0],
                    [8.0, 6.0, 4.0, 2.0],
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


def _matches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "pred_indices": torch.tensor([0, 1]),
            "gt_indices": torch.tensor([0, 1]),
        }
    ]


def test_lane_balanced_point_gives_each_lane_equal_outer_weight() -> None:
    pred = torch.tensor(
        [[[3.0, 5.0, 7.0, 9.0], [9.0, 0.0, 0.0, 0.0]]]
    )
    loss = lane_balanced_point_loss(
        pred,
        _targets(),
        _matches(),
        input_w=10,
        beta=0.01,
    )
    # Both lanes have exactly one-pixel error on their valid rows.
    assert torch.isclose(loss, torch.tensor(0.095))


def test_lane_balanced_dfl_is_finite_and_differentiable() -> None:
    logits = torch.zeros((1, 2, 4, 5), requires_grad=True)
    loss = lane_balanced_dfl_loss(
        logits,
        _targets(),
        _matches(),
        input_w=10,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
