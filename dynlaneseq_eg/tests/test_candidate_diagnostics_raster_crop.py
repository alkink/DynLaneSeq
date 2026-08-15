from __future__ import annotations

import numpy as np

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    _crop_intersection_count,
    _raster_lane_crop,
    _raster_lane_mask,
)


def _restore_crop(
    crop: tuple[np.ndarray, int, int], image_h: int, image_w: int
) -> np.ndarray:
    mask, top, left = crop
    restored = np.zeros((image_h, image_w), dtype=np.uint8)
    if mask.size:
        restored[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return restored


def test_tight_lane_raster_has_full_canvas_pixel_parity() -> None:
    rng = np.random.default_rng(3407)
    image_h, image_w, width = 590, 1640, 30
    lanes: list[list[tuple[float, float]]] = [
        [(-80.0, 580.0), (200.0, 400.0), (820.0, 200.0)],
        [(1600.0, 589.0), (1645.0, 300.0), (1700.0, -40.0)],
        [(-100.0, 300.0), (820.0, 280.0), (1740.0, 260.0)],
    ]
    for _ in range(24):
        ys = np.sort(rng.uniform(-80.0, image_h + 80.0, size=6))[::-1]
        xs = rng.uniform(-120.0, image_w + 120.0, size=6)
        lanes.append([(float(x), float(y)) for x, y in zip(xs, ys)])

    for lane in lanes:
        expected = _raster_lane_mask(lane, image_h, image_w, width)
        actual = _restore_crop(
            _raster_lane_crop(lane, image_h, image_w, width), image_h, image_w
        )
        np.testing.assert_array_equal(actual, expected)


def test_tight_lane_raster_intersection_matches_full_canvas() -> None:
    image_h, image_w, width = 590, 1640, 30
    lanes = [
        [(100.0, 589.0), (300.0, 360.0), (520.0, 120.0)],
        [(120.0, 589.0), (320.0, 360.0), (540.0, 120.0)],
        [(1500.0, 589.0), (1100.0, 330.0), (820.0, -20.0)],
        [(-40.0, 500.0), (800.0, 300.0), (1680.0, 100.0)],
    ]
    full = [_raster_lane_mask(lane, image_h, image_w, width) for lane in lanes]
    cropped = [
        _raster_lane_crop(lane, image_h, image_w, width) for lane in lanes
    ]
    for first_index in range(len(lanes)):
        for second_index in range(len(lanes)):
            expected = int(
                np.count_nonzero(full[first_index] & full[second_index])
            )
            actual = _crop_intersection_count(
                cropped[first_index], cropped[second_index]
            )
            assert actual == expected
