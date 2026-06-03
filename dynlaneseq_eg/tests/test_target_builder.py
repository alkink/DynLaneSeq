from __future__ import annotations

import numpy as np

from dynlaneseq_eg.data.lane_target_builder import LaneTargetBuilder, TargetBuilderConfig


def test_target_builder_shapes_and_ignore_index():
    builder = LaneTargetBuilder(TargetBuilderConfig(input_w=800, input_h=288, num_rows=72, x_bins=200))
    lanes = [[(100.0, 100.0), (200.0, 200.0), (300.0, 300.0), (400.0, 400.0)]]
    target = builder.build(lanes, orig_w=800, orig_h=590)
    assert target["x_rows"].shape == (1, 72)
    assert target["valid_mask"].shape == (1, 72)
    assert target["x_bins"].shape == (1, 72)
    assert np.all(target["x_rows"][~target["valid_mask"]] == -1)
    assert np.all(target["x_bins"][~target["valid_mask"]] == -100)
    assert target["valid_mask"].sum() >= 5


def test_duplicate_y_is_averaged_and_range_is_valid():
    builder = LaneTargetBuilder(TargetBuilderConfig(input_w=800, input_h=288, num_rows=72, x_bins=200))
    lanes = [[(100.0, 100.0), (120.0, 100.0), (200.0, 200.0), (300.0, 300.0)]]
    target = builder.build(lanes, orig_w=800, orig_h=590)
    assert target["range_y"].shape == (1, 2)
    assert target["range_y"][0, 0] <= target["range_y"][0, 1]
    assert target["valid_mask"].sum() >= 5

