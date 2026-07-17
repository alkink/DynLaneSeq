from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .curvelanes_writer import prediction_path_for_raw_file


DEFAULT_EVAL_WIDTH = 224
DEFAULT_EVAL_HEIGHT = 224
DEFAULT_LANE_WIDTH = 5
DEFAULT_IOU_THRESHOLD = 0.5


def _as_points(lane: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    """Remove malformed and duplicate adjacent points before spline fitting."""
    points: list[dict[str, float]] = []
    for point in lane:
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if points and math.hypot(x - points[-1]["x"], y - points[-1]["y"]) <= 1e-6:
            continue
        points.append({"x": x, "y": y})
    return points


def _calc_params(lane: list[dict[str, float]]) -> list[dict[str, float]]:
    """Natural cubic spline parameters, matching the public CondLSTR metric."""
    params: list[dict[str, float]] = []
    n_pt = len(lane)
    if n_pt < 2:
        return params
    distances = [
        math.hypot(lane[index]["x"] - lane[index + 1]["x"], lane[index]["y"] - lane[index + 1]["y"])
        for index in range(n_pt - 1)
    ]
    if any(distance <= 1e-6 for distance in distances):
        return params
    if n_pt == 2:
        h = distances[0]
        return [
            {
                "a_x": lane[0]["x"],
                "b_x": (lane[1]["x"] - lane[0]["x"]) / h,
                "c_x": 0.0,
                "d_x": 0.0,
                "a_y": lane[0]["y"],
                "b_y": (lane[1]["y"] - lane[0]["y"]) / h,
                "c_y": 0.0,
                "d_y": 0.0,
                "h": h,
            }
        ]

    a: list[float] = []
    b: list[float] = []
    c: list[float] = []
    d_x: list[float] = []
    d_y: list[float] = []
    for index in range(n_pt - 2):
        h0, h1 = distances[index], distances[index + 1]
        a.append(h0)
        b_value = 2.0 * (h0 + h1)
        c_value = h1
        delta_x = 6.0 * ((lane[index + 2]["x"] - lane[index + 1]["x"]) / h1 - (lane[index + 1]["x"] - lane[index]["x"]) / h0)
        delta_y = 6.0 * ((lane[index + 2]["y"] - lane[index + 1]["y"]) / h1 - (lane[index + 1]["y"] - lane[index]["y"]) / h0)
        if index == 0:
            c.append(c_value / b_value)
            d_x.append(delta_x / b_value)
            d_y.append(delta_y / b_value)
        else:
            base = b_value - a[index] * c[index - 1]
            c.append(c_value / base)
            d_x.append((delta_x - a[index] * d_x[index - 1]) / base)
            d_y.append((delta_y - a[index] * d_y[index - 1]) / base)
        b.append(b_value)

    m_x = np.zeros((n_pt,), dtype=np.float64)
    m_y = np.zeros((n_pt,), dtype=np.float64)
    m_x[n_pt - 2] = d_x[n_pt - 3]
    m_y[n_pt - 2] = d_y[n_pt - 3]
    for index in range(n_pt - 4, -1, -1):
        m_x[index + 1] = d_x[index] - c[index] * m_x[index + 2]
        m_y[index + 1] = d_y[index] - c[index] * m_y[index + 2]

    for index in range(n_pt - 1):
        h = distances[index]
        params.append(
            {
                "a_x": lane[index]["x"],
                "b_x": (lane[index + 1]["x"] - lane[index]["x"]) / h - (2.0 * h * m_x[index] + h * m_x[index + 1]) / 6.0,
                "c_x": m_x[index] / 2.0,
                "d_x": (m_x[index + 1] - m_x[index]) / (6.0 * h),
                "a_y": lane[index]["y"],
                "b_y": (lane[index + 1]["y"] - lane[index]["y"]) / h - (2.0 * h * m_y[index] + h * m_y[index + 1]) / 6.0,
                "c_y": m_y[index] / 2.0,
                "d_y": (m_y[index + 1] - m_y[index]) / (6.0 * h),
                "h": h,
            }
        )
    return params


def _spline_interp(lane: Iterable[dict[str, Any]], step_t: int = 1) -> list[dict[str, float]]:
    points = _as_points(lane)
    if len(points) < 2:
        return points
    params = _calc_params(points)
    if not params:
        return points
    result: list[dict[str, float]] = []
    for param in params:
        t = 0
        while t < param["h"]:
            result.append(
                {
                    "x": param["a_x"] + param["b_x"] * t + param["c_x"] * t * t + param["d_x"] * t * t * t,
                    "y": param["a_y"] + param["b_y"] * t + param["c_y"] * t * t + param["d_y"] * t * t * t,
                }
            )
            t += step_t
    result.append(points[-1])
    return result


def _resize_lane(lane: Iterable[dict[str, Any]], native_width: float, native_height: float, eval_width: int, eval_height: int) -> list[dict[str, float]]:
    if native_width <= 0.0 or native_height <= 0.0:
        raise ValueError(f"Invalid CurveLanes image size {native_width}x{native_height}")
    x_ratio = native_width / float(eval_width)
    y_ratio = native_height / float(eval_height)
    return [{"x": float(point["x"]) / x_ratio, "y": float(point["y"]) / y_ratio} for point in _as_points(lane)]


def lane_iou(
    lane_a: Iterable[dict[str, Any]],
    lane_b: Iterable[dict[str, Any]],
    eval_width: int = DEFAULT_EVAL_WIDTH,
    eval_height: int = DEFAULT_EVAL_HEIGHT,
    lane_width: int = DEFAULT_LANE_WIDTH,
) -> float:
    """Raster IoU used by the public CurveLanes evaluator."""
    canvas_a = np.zeros((eval_height, eval_width), dtype=np.uint8)
    canvas_b = np.zeros((eval_height, eval_width), dtype=np.uint8)
    for canvas, lane in ((canvas_a, lane_a), (canvas_b, lane_b)):
        sampled = _spline_interp(lane)
        for point_a, point_b in zip(sampled[:-1], sampled[1:]):
            cv2.line(
                canvas,
                (int(point_a["x"]), int(point_a["y"])),
                (int(point_b["x"]), int(point_b["y"])),
                255,
                int(lane_width),
            )
    union = cv2.bitwise_or(canvas_a, canvas_b)
    union_sum = int(union.sum())
    if union_sum == 0:
        return 0.0
    return float((int(canvas_a.sum()) + int(canvas_b.sum()) - union_sum) / union_sum)


def evaluate_image(
    gt_data: dict[str, Any],
    pred_data: dict[str, Any],
    gt_size: tuple[int, int],
    eval_width: int = DEFAULT_EVAL_WIDTH,
    eval_height: int = DEFAULT_EVAL_HEIGHT,
    lane_width: int = DEFAULT_LANE_WIDTH,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Return ``(hits, predicted_count, ground_truth_count)`` for one image."""
    gt_lanes = [lane for lane in gt_data.get("Lines", []) if len(_as_points(lane)) >= 2]
    pred_lanes = [lane for lane in pred_data.get("Lines", []) if len(_as_points(lane)) >= 2]
    gt_width, gt_height = gt_size
    pred_shape = pred_data.get("Shape", {})
    pred_width = float(pred_shape.get("width", gt_width))
    pred_height = float(pred_shape.get("height", gt_height))
    gt_scaled = [_resize_lane(lane, gt_width, gt_height, eval_width, eval_height) for lane in gt_lanes]
    pred_scaled = [_resize_lane(lane, pred_width, pred_height, eval_width, eval_height) for lane in pred_lanes]
    if not gt_scaled or not pred_scaled:
        return 0, len(pred_scaled), len(gt_scaled)
    ious = np.zeros((len(gt_scaled), len(pred_scaled)), dtype=np.float64)
    for gt_index, gt_lane in enumerate(gt_scaled):
        for pred_index, pred_lane in enumerate(pred_scaled):
            ious[gt_index, pred_index] = lane_iou(
                gt_lane,
                pred_lane,
                eval_width=eval_width,
                eval_height=eval_height,
                lane_width=lane_width,
            )
    gt_indices, pred_indices = linear_sum_assignment(1.0 - ious)
    hits = int(sum(ious[gt_index, pred_index] > float(iou_threshold) for gt_index, pred_index in zip(gt_indices, pred_indices)))
    return hits, len(pred_scaled), len(gt_scaled)


def evaluate_curvelanes(
    dataset_root: str | Path,
    prediction_dir: str | Path,
    split: str = "val",
    eval_width: int = DEFAULT_EVAL_WIDTH,
    eval_height: int = DEFAULT_EVAL_HEIGHT,
    lane_width: int = DEFAULT_LANE_WIDTH,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    progress: bool = True,
) -> dict[str, float | int]:
    """Evaluate a complete CurveLanes split with one prediction per list item."""
    dataset_root = Path(dataset_root)
    split_dir = {"val": "valid", "train": "train", "test": "test"}.get(str(split), str(split))
    list_path = dataset_root / split_dir / f"{split_dir}.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"CurveLanes split list not found: {list_path}")
    if split_dir == "test":
        raise ValueError("CurveLanes test/images is unlabeled; use split=val for the public F1 protocol")
    raw_files = [line.strip().lstrip("/") for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_files:
        raise ValueError(f"CurveLanes split list is empty: {list_path}")

    missing = [raw_file for raw_file in raw_files if not prediction_path_for_raw_file(prediction_dir, raw_file).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Missing {len(missing)} CurveLanes predictions under {prediction_dir}; first entries: {preview}"
        )

    true_positive = false_positive = false_negative = 0
    iterator: Iterable[str] = raw_files
    if progress:
        iterator = tqdm(raw_files, ncols=88, desc=f"evaluating CurveLanes {split_dir}")
    for raw_file in iterator:
        image_path = dataset_root / split_dir / raw_file
        annotation_path = dataset_root / split_dir / "labels" / f"{Path(raw_file).stem}.lines.json"
        prediction_path = prediction_path_for_raw_file(prediction_dir, raw_file)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"CurveLanes annotation not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            gt_data = json.load(handle)
        with prediction_path.open("r", encoding="utf-8") as handle:
            pred_data = json.load(handle)
        with Image.open(image_path) as image:
            image_size = image.size
        hits, pred_count, gt_count = evaluate_image(
            gt_data,
            pred_data,
            image_size,
            eval_width=eval_width,
            eval_height=eval_height,
            lane_width=lane_width,
            iou_threshold=iou_threshold,
        )
        true_positive += hits
        false_positive += pred_count - hits
        false_negative += gt_count - hits

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP": int(true_positive),
        "FP": int(false_positive),
        "FN": int(false_negative),
        "samples": int(len(raw_files)),
        "eval_width": int(eval_width),
        "eval_height": int(eval_height),
        "lane_width": int(lane_width),
        "iou_threshold": float(iou_threshold),
    }


def format_curvelanes_results(results: dict[str, float | int]) -> str:
    return (
        f"F1={float(results['F1']):.6f} "
        f"P={float(results['Precision']):.6f} "
        f"R={float(results['Recall']):.6f} "
        f"TP={int(results['TP'])} FP={int(results['FP'])} FN={int(results['FN'])} "
        f"samples={int(results['samples'])}"
    )
