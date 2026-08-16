from __future__ import annotations

import torch

from dynlaneseq_eg.evaluation.decoder_parity import (
    candidate_geometry,
    lane_to_original_clr_spline,
)


def test_clr_range_preserves_top_and_extends_bottom() -> None:
    stage = {
        "pred_x_rows": torch.full((1, 8), 50.0),
        "range_norm": torch.tensor([[0.25, 0.625]]),
    }
    _, current_mask, _, _ = candidate_geometry(
        stage,
        input_h=80,
        input_w=100,
        range_mode="current",
    )
    _, clr_mask, _, _ = candidate_geometry(
        stage,
        input_h=80,
        input_w=100,
        range_mode="clr_bottom_extend",
    )
    assert current_mask[0].tolist() == [False, False, True, True, True, True, False, False]
    assert clr_mask[0].tolist() == [False, False, True, True, True, True, True, True]


def test_clr_spline_writer_uses_official_y_grid_and_bottom_first() -> None:
    lane = [(100.0, float(y)) for y in range(0, 640, 4)]
    meta = {
        "scale_x": 1600.0 / 1640.0,
        "scale_y": 2.0,
        "crop_x": 0.0,
        "crop_y": 270.0,
        "orig_w": 1640,
    }
    written = lane_to_original_clr_spline(lane, meta)
    assert len(written) == 40
    assert written[0][1] == 582.0
    assert written[-1][1] == 270.0
    assert all(abs(x - 102.5) < 1e-4 for x, _ in written)
