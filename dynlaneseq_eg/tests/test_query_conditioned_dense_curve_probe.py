from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_query_conditioned_dense_curve import (
    QueryConditionedDenseCurveProbe,
    _group_zero_matches,
    matched_dense_curve_loss,
)


def _probe() -> QueryConditionedDenseCurveProbe:
    return QueryConditionedDenseCurveProbe(
        in_dim=16,
        state_dim=12,
        hidden_dim=8,
        num_rows=5,
        evidence_width=7,
        input_w=70,
    )


def test_dense_probe_shape_and_decode_range() -> None:
    probe = _probe()
    p2 = torch.randn(2, 16, 3, 4)
    states = torch.randn(2, 3, 5, 12)
    logits = probe(p2, states)
    curves = probe.decode(logits)
    assert logits.shape == (2, 3, 5, 7)
    assert curves.shape == (2, 3, 5)
    assert bool((curves > 0.0).all())
    assert bool((curves < 70.0).all())


def test_group_zero_matches_filters_repeated_training_groups() -> None:
    match = {
        "pred_indices": torch.tensor([0, 3, 8, 11, 16, 19]),
        "gt_indices": torch.tensor([0, 1, 0, 1, 0, 1]),
    }
    assert _group_zero_matches(match, group_size=8) == [(0, 0), (3, 1)]


def test_dense_curve_loss_is_lane_balanced_and_differentiable() -> None:
    logits = torch.zeros(1, 4, 5, 7, requires_grad=True)
    target = {
        "x_rows": torch.tensor(
            [
                [5.0, 15.0, 25.0, 35.0, 45.0],
                [0.0, 0.0, 25.0, 35.0, 45.0],
            ]
        ),
        "valid_mask": torch.tensor(
            [
                [True, True, True, True, True],
                [False, False, True, True, True],
            ]
        ),
    }
    match = {
        "pred_indices": torch.tensor([0, 1, 2, 3]),
        "gt_indices": torch.tensor([0, 1, 0, 1]),
    }
    loss, lanes, rows = matched_dense_curve_loss(
        logits,
        [target],
        [match],
        group_size=2,
        input_w=70,
    )
    assert lanes == 2
    assert rows == 8
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, :2].abs().sum()) > 0.0
    assert float(logits.grad[0, 2:].abs().sum()) == 0.0
