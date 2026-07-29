from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_frozen_p2_centerline_separability import (
    FrozenP2CenterlineProbe,
    build_centerline_target,
    lane_center_nll,
)


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "x_rows": torch.tensor(
                [
                    [4.0, 6.0, 8.0, 10.0],
                    [12.0, 11.0, 10.0, 9.0],
                ]
            ),
            "valid_mask": torch.tensor(
                [
                    [True, True, True, True],
                    [False, True, True, True],
                ]
            ),
        }
    ]


def test_centerline_target_contains_every_visible_lane_center() -> None:
    target = build_centerline_target(
        _targets(),
        num_rows=4,
        x_bins=8,
        input_w=16.0,
        sigma_bins=1.0,
        device=torch.device("cpu"),
    )
    assert target.shape == (1, 1, 4, 8)
    assert float(target[0, 0, 0, 2]) == 1.0
    assert float(target[0, 0, 0, 6]) < 1e-3
    assert float(target[0, 0, 1, 3]) == 1.0
    assert float(target[0, 0, 1, 5]) > 0.8


def test_lane_center_nll_prefers_logits_at_gt_centers() -> None:
    targets = _targets()
    aligned = torch.zeros(1, 1, 4, 8)
    shifted = torch.zeros_like(aligned)
    for row, bins in enumerate(((2,), (3, 5), (4, 5), (5, 4))):
        for center in bins:
            aligned[0, 0, row, center] = 6.0
            shifted[0, 0, row, (center + 2) % 8] = 6.0
    aligned_loss = lane_center_nll(aligned, targets, input_w=16.0)
    shifted_loss = lane_center_nll(shifted, targets, input_w=16.0)
    assert float(aligned_loss) < float(shifted_loss)


def test_probe_returns_requested_dense_shape() -> None:
    probe = FrozenP2CenterlineProbe(
        in_dim=16,
        hidden_dim=16,
        num_rows=5,
        x_bins=13,
    )
    output = probe(torch.randn(2, 16, 3, 7))
    assert output.shape == (2, 1, 5, 13)
