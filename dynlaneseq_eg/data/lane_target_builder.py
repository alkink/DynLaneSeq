from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TargetBuilderConfig:
    input_w: int = 800
    input_h: int = 288
    num_rows: int = 72
    x_bins: int = 200
    min_valid_rows: int = 5
    eps: float = 1e-6
    token_ignore_index: int = -100

    @property
    def row_stride(self) -> float:
        return self.input_h / self.num_rows

    @property
    def y_rows(self) -> np.ndarray:
        return np.arange(self.num_rows, dtype=np.float32) * self.row_stride

    @property
    def bin_width(self) -> float:
        return self.input_w / self.x_bins


class LaneTargetBuilder:
    """Build PRD fixed-row lane targets from raw CULane polyline points."""

    def __init__(self, cfg: TargetBuilderConfig | None = None):
        self.cfg = cfg or TargetBuilderConfig()

    def build(
        self,
        lanes_orig: Iterable[Iterable[tuple[float, float]]],
        orig_w: int,
        orig_h: int,
    ) -> dict[str, np.ndarray]:
        lanes_in = [
            self._clean_and_resize_lane(lane, orig_w=orig_w, orig_h=orig_h)
            for lane in lanes_orig
        ]
        built = [self._interpolate_lane(lane) for lane in lanes_in if len(lane) >= 2]
        built = [item for item in built if int(item["valid_mask"].sum()) >= self.cfg.min_valid_rows]

        if not built:
            return {
                "x_rows": np.zeros((0, self.cfg.num_rows), dtype=np.float32),
                "x_bins": np.zeros((0, self.cfg.num_rows), dtype=np.int64),
                "valid_mask": np.zeros((0, self.cfg.num_rows), dtype=np.bool_),
                "range_y": np.zeros((0, 2), dtype=np.float32),
                "exist": np.zeros((0,), dtype=np.int64),
            }

        x_rows = np.stack([item["x_rows"] for item in built]).astype(np.float32)
        valid_mask = np.stack([item["valid_mask"] for item in built]).astype(np.bool_)
        range_y = np.stack([item["range_y"] for item in built]).astype(np.float32)
        x_bins = self._build_x_bins(x_rows, valid_mask)
        exist = np.zeros((len(built),), dtype=np.int64)
        return {
            "x_rows": x_rows,
            "x_bins": x_bins,
            "valid_mask": valid_mask,
            "range_y": range_y,
            "exist": exist,
        }

    def _clean_and_resize_lane(
        self,
        lane: Iterable[tuple[float, float]],
        orig_w: int,
        orig_h: int,
    ) -> list[tuple[float, float]]:
        sx = self.cfg.input_w / float(orig_w)
        sy = self.cfg.input_h / float(orig_h)
        grouped: dict[float, list[float]] = {}
        for x_raw, y_raw in lane:
            x = float(x_raw)
            y = float(y_raw)
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0 or x >= orig_w or y < 0 or y >= orig_h:
                continue
            x_in = x * sx
            y_in = y * sy
            key = round(y_in, 4)
            grouped.setdefault(key, []).append(x_in)
        points = [(float(np.mean(xs)), float(y)) for y, xs in grouped.items()]
        points.sort(key=lambda p: p[1])
        return points

    def _interpolate_lane(self, lane: list[tuple[float, float]]) -> dict[str, np.ndarray]:
        x_rows = np.full((self.cfg.num_rows,), -1.0, dtype=np.float32)
        valid = np.zeros((self.cfg.num_rows,), dtype=np.bool_)
        pts = lane
        for idx, y_row in enumerate(self.cfg.y_rows):
            xs: list[float] = []
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                if abs(yb - ya) < self.cfg.eps:
                    continue
                lo = min(ya, yb) - self.cfg.eps
                hi = max(ya, yb) + self.cfg.eps
                if lo <= y_row <= hi:
                    t = (y_row - ya) / (yb - ya)
                    x = xa + t * (xb - xa)
                    if 0.0 <= x < self.cfg.input_w:
                        xs.append(float(x))
            if xs:
                x_rows[idx] = float(np.mean(xs))
                valid[idx] = True

        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            range_y = np.array([0.0, 0.0], dtype=np.float32)
        else:
            range_y = np.array(
                [self.cfg.y_rows[valid_idx[0]], self.cfg.y_rows[valid_idx[-1]]],
                dtype=np.float32,
            )
        return {"x_rows": x_rows, "valid_mask": valid, "range_y": range_y}

    def _build_x_bins(self, x_rows: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        bins = np.full(x_rows.shape, self.cfg.token_ignore_index, dtype=np.int64)
        valid = valid_mask.astype(bool)
        raw = np.floor(x_rows[valid] / self.cfg.bin_width).astype(np.int64)
        bins[valid] = np.clip(raw, 0, self.cfg.x_bins - 1)
        return bins


def decode_targets_to_points(
    x_rows: np.ndarray,
    valid_mask: np.ndarray,
    input_h: int = 288,
) -> list[list[tuple[float, float]]]:
    num_rows = x_rows.shape[-1]
    y_rows = np.arange(num_rows, dtype=np.float32) * (input_h / num_rows)
    lanes: list[list[tuple[float, float]]] = []
    for xs, mask in zip(x_rows, valid_mask):
        lanes.append([(float(x), float(y)) for x, y, m in zip(xs, y_rows, mask) if bool(m)])
    return lanes

