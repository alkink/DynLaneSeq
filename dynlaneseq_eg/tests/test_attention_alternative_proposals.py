from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_attention_alternative_proposals import (
    _attention_curves,
    _select_curves,
)


def test_attention_curves_recover_expected_and_peak_positions() -> None:
    # [B, R, H, N, E] with one head and one lane candidate.
    attention = torch.zeros(1, 3, 1, 1, 4)
    attention[:, 0, :, :, 0] = 1.0
    attention[:, 1, :, :, 1] = 1.0
    attention[:, 2, :, :, 3] = 1.0

    curves = _attention_curves(attention, input_w=80.0)

    expected_centers = torch.tensor([10.0, 30.0, 70.0])
    assert set(curves) == {
        "attention_expected",
        "attention_peak",
        "attention_expected_v5",
        "attention_peak_v5",
        "attention_expected_v11",
        "attention_peak_v11",
    }
    assert curves["attention_expected"].shape == (1, 1, 3)
    assert torch.allclose(
        curves["attention_expected"][0, 0],
        expected_centers,
    )
    assert torch.equal(
        curves["attention_peak"][0, 0],
        expected_centers,
    )
    for values in curves.values():
        assert bool(torch.isfinite(values).all())


def test_select_curves_preserves_requested_order() -> None:
    curves = torch.tensor(
        [
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            [
                [30.0, 31.0],
                [40.0, 41.0],
                [50.0, 51.0],
            ],
        ]
    )
    indices = torch.tensor([[2, 0], [1, 2]])

    selected = _select_curves(curves, indices)

    assert selected.tolist() == [
        [[20.0, 21.0], [0.0, 1.0]],
        [[40.0, 41.0], [50.0, 51.0]],
    ]


def test_select_curves_rejects_invalid_rank() -> None:
    try:
        _select_curves(torch.zeros(2, 3), torch.zeros(2, 1, dtype=torch.long))
    except ValueError as error:
        assert "must be [B,N,R]" in str(error)
    else:
        raise AssertionError("invalid curve rank should raise ValueError")
