from __future__ import annotations

from functools import partial
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from tqdm import tqdm


Lane = List[Tuple[float, float]]


def draw_lane(lane: np.ndarray, img_shape: tuple[int, int, int] = (590, 1640, 3), width: int = 30) -> np.ndarray:
    img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(255, 255, 255), thickness=width)
    return img > 0


def discrete_cross_iou(
    xs: Iterable[np.ndarray],
    ys: Iterable[np.ndarray],
    width: int = 30,
    img_shape: tuple[int, int, int] = (590, 1640, 3),
) -> np.ndarray:
    xs_masks = [draw_lane(lane, img_shape=img_shape, width=width) for lane in xs]
    ys_masks = [draw_lane(lane, img_shape=img_shape, width=width) for lane in ys]
    ious = np.zeros((len(xs_masks), len(ys_masks)), dtype=np.float32)
    for i, x in enumerate(xs_masks):
        for j, y in enumerate(ys_masks):
            union = (x | y).sum()
            ious[i, j] = 0.0 if union == 0 else float((x & y).sum()) / float(union)
    return ious


def continuous_cross_iou(
    xs: Iterable[np.ndarray],
    ys: Iterable[np.ndarray],
    width: int = 30,
    img_shape: tuple[int, int, int] = (590, 1640, 3),
) -> np.ndarray:
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs_poly = [LineString(lane).buffer(width / 2.0, cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys_poly = [LineString(lane).buffer(width / 2.0, cap_style=1, join_style=2).intersection(image) for lane in ys]
    ious = np.zeros((len(xs_poly), len(ys_poly)), dtype=np.float32)
    for i, x in enumerate(xs_poly):
        for j, y in enumerate(ys_poly):
            union = x.union(y).area
            ious[i, j] = 0.0 if union == 0 else float(x.intersection(y).area) / float(union)
    return ious


def interp(points: Lane, n: int = 5) -> np.ndarray:
    if len(points) <= 1:
        return np.asarray(points, dtype=np.float32)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    k = min(3, len(points) - 1)
    try:
        tck, u = splprep([x, y], s=0, k=k)
        u_new = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
        return np.asarray(splev(u_new, tck), dtype=np.float32).T
    except Exception:
        return np.asarray(points, dtype=np.float32)


def culane_metric(
    pred: list[Lane],
    anno: list[Lane],
    width: int = 30,
    iou_thresholds: list[float] | tuple[float, ...] = (0.5,),
    official: bool = True,
    img_shape: tuple[int, int, int] = (590, 1640, 3),
) -> dict[float, list[int]]:
    if len(pred) == 0 or len(anno) == 0:
        return {float(thr): [0, len(pred), len(anno)] for thr in iou_thresholds}

    interp_pred = [interp(lane, n=5) for lane in pred if len(lane) >= 2]
    interp_anno = [interp(lane, n=5) for lane in anno if len(lane) >= 2]
    if len(interp_pred) == 0 or len(interp_anno) == 0:
        return {float(thr): [0, len(interp_pred), len(interp_anno)] for thr in iou_thresholds}

    ious = (
        discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
        if official
        else continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    )
    row_ind, col_ind = linear_sum_assignment(1.0 - ious)
    return {
        float(thr): [
            int((ious[row_ind, col_ind] > float(thr)).sum()),
            len(interp_pred) - int((ious[row_ind, col_ind] > float(thr)).sum()),
            len(interp_anno) - int((ious[row_ind, col_ind] > float(thr)).sum()),
        ]
        for thr in iou_thresholds
    }


def load_culane_img_data(path: str | Path) -> list[Lane]:
    path = Path(path)
    if not path.exists():
        return []
    lanes: list[Lane] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) < 4:
                continue
            lane = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
            if len(lane) >= 2:
                lanes.append(lane)
    return lanes


def list_image_rel_paths(list_path: str | Path) -> list[str]:
    rels: list[str] = []
    with Path(list_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0]
            rels.append(rel[1:] if rel.startswith("/") else rel)
    return rels


def load_culane_data(data_dir: str | Path, list_path: str | Path) -> list[list[Lane]]:
    data_dir = Path(data_dir)
    data = []
    for rel in list_image_rel_paths(list_path):
        data.append(load_culane_img_data(data_dir / rel.replace(".jpg", ".lines.txt")))
    return data


def _metric_one(args) -> dict[float, list[int]]:
    pred, anno, width, iou_thresholds, official, img_shape = args
    return culane_metric(pred, anno, width=width, iou_thresholds=iou_thresholds, official=official, img_shape=img_shape)


def eval_predictions(
    pred_dir: str | Path,
    anno_dir: str | Path,
    list_path: str | Path,
    iou_thresholds: list[float] | tuple[float, ...] = (0.5,),
    width: int = 30,
    official: bool = True,
    sequential: bool = False,
    img_shape: tuple[int, int, int] = (590, 1640, 3),
) -> dict[float | str, dict[str, float | int]]:
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    tasks = list(zip(predictions, annotations, repeat(width), repeat(tuple(iou_thresholds)), repeat(official), repeat(img_shape)))
    desc = f"evaluating {Path(list_path).name}"
    if sequential:
        results = [_metric_one(task) for task in tqdm(tasks, desc=desc, ncols=80)]
    else:
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(_metric_one, tasks), total=len(tasks), desc=desc, ncols=80))

    ret: dict[float | str, dict[str, float | int]] = {}
    mean_f1 = 0.0
    mean_precision = 0.0
    mean_recall = 0.0
    total_tp = total_fp = total_fn = 0
    for thr in iou_thresholds:
        thr = float(thr)
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        ret[thr] = {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}
        mean_f1 += f1 / len(iou_thresholds)
        mean_precision += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    if len(iou_thresholds) > 1:
        ret["mean"] = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": mean_precision,
            "Recall": mean_recall,
            "F1": mean_f1,
        }
    return ret


def format_results(results: dict[float | str, dict[str, float | int]]) -> str:
    lines = []
    for key, value in results.items():
        label = f"IoU {key:.2f}" if isinstance(key, float) else str(key)
        lines.append(
            f"{label}: TP={value['TP']} FP={value['FP']} FN={value['FN']} "
            f"P={value['Precision']:.4f} R={value['Recall']:.4f} F1={value['F1']:.4f}"
        )
    return "\n".join(lines)


def format_category_results(category_results: dict[str, dict[float | str, dict[str, float | int]]]) -> str:
    lines = []
    for category, results in category_results.items():
        for key, value in results.items():
            if key == "mean":
                continue
            label = f"{category}@{key:.2f}" if isinstance(key, float) else f"{category}@{key}"
            lines.append(
                f"{label}: TP={value['TP']} FP={value['FP']} FN={value['FN']} "
                f"P={value['Precision']:.4f} R={value['Recall']:.4f} F1={value['F1']:.4f}"
            )
    return "\n".join(lines)
