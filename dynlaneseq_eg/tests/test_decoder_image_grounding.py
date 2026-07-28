from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_decoder_image_grounding import (
    ShiftResponseStats,
    shift_feature_x,
    shift_targets_x,
)


def test_shift_feature_x_uses_zero_fill_without_wraparound() -> None:
    source = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    right = shift_feature_x(source, 2)
    left = shift_feature_x(source, -2)
    torch.testing.assert_close(
        right,
        torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 2.0]]]]),
    )
    torch.testing.assert_close(
        left,
        torch.tensor([[[[2.0, 3.0, 4.0, 0.0, 0.0]]]]),
    )


def test_shift_response_ratio_tracks_expected_direction_and_magnitude() -> None:
    stats = ShiftResponseStats(expected_shift_px=32.0)
    stats.update(torch.tensor([32.0, 16.0, -4.0]))
    summary = stats.summary()
    assert summary["count"] == 3
    assert abs(float(summary["response_ratio"]) - (44.0 / 3.0 / 32.0)) < 1e-6
    assert abs(float(summary["correct_direction_fraction"]) - (2.0 / 3.0)) < 1e-6
    assert abs(float(summary["within_half_expected_shift_fraction"]) - (2.0 / 3.0)) < 1e-6


def test_shift_targets_masks_rows_that_leave_the_image() -> None:
    target = {
        "x_rows": torch.tensor([[1.0, 7.0, 14.0]]),
        "valid_mask": torch.tensor([[True, True, True]]),
    }
    shifted = shift_targets_x([target], 4.0, input_w=16)[0]
    torch.testing.assert_close(
        shifted["x_rows"],
        torch.tensor([[5.0, 11.0, 18.0]]),
    )
    assert shifted["valid_mask"].tolist() == [[True, True, False]]
