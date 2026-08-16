from __future__ import annotations

import argparse
from collections import Counter
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    interp,
    list_image_rel_paths,
    load_culane_img_data,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


THRESHOLDS = (0.50, 0.75)
LINE_WIDTH = 30
IMAGE_SHAPE = (590, 1640)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Training-free exhaustive V23 source-versus-raw per-lane oracle."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=32)
    return parser.parse_args()


def _lane_distance(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    right_by_y = {round(y, 3): x for x, y in right}
    differences = [
        abs(x - right_by_y[round(y, 3)])
        for x, y in left
        if round(y, 3) in right_by_y
    ]
    if differences:
        return float(sum(differences) / len(differences))
    if not left or not right:
        return 1.0e6
    left_y = (left[0][1], left[-1][1])
    right_y = (right[0][1], right[-1][1])
    return 1.0e4 + abs(left_y[0] - right_y[0]) + abs(
        left_y[1] - right_y[1]
    )


def align_raw_to_source(
    source: list[list[tuple[float, float]]],
    raw: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    if len(source) != len(raw):
        raise ValueError("source/raw prediction count mismatch")
    if not source:
        return []
    cost = np.asarray(
        [[_lane_distance(left, right) for right in raw] for left in source],
        dtype=np.float64,
    )
    source_ids, raw_ids = linear_sum_assignment(cost)
    aligned: list[list[tuple[float, float]] | None] = [None] * len(source)
    for source_id, raw_id in zip(source_ids.tolist(), raw_ids.tolist()):
        aligned[source_id] = raw[raw_id]
    if any(lane is None for lane in aligned):
        raise RuntimeError("source/raw alignment is incomplete")
    return [lane for lane in aligned if lane is not None]


def _quality_matrix(
    source: list[list[tuple[float, float]]],
    raw: list[list[tuple[float, float]]],
    annotation: list[list[tuple[float, float]]],
) -> np.ndarray:
    candidates = source + raw
    candidate_interp = [interp(lane, n=5) for lane in candidates]
    annotation_interp = [
        interp(lane, n=5) for lane in annotation if len(lane) >= 2
    ]
    if not candidates or not annotation_interp:
        return np.zeros((len(candidates), len(annotation_interp)), np.float32)
    return discrete_cross_iou(
        candidate_interp,
        annotation_interp,
        width=LINE_WIDTH,
        img_shape=IMAGE_SHAPE,
    )


def _score_selection(
    quality: np.ndarray,
    selection: tuple[int, ...],
    *,
    prediction_count: int,
    gt_count: int,
) -> dict[str, Any]:
    if prediction_count == 0 or gt_count == 0:
        matched = np.zeros((0,), dtype=np.float32)
    else:
        selected = quality[np.asarray(selection, dtype=np.int64)]
        pred_ids, gt_ids = linear_sum_assignment(1.0 - selected)
        matched = selected[pred_ids, gt_ids]
    result: dict[str, Any] = {
        "total_iou": float(matched.sum()),
        "matched_iou": matched.tolist(),
    }
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        tp = int((matched > threshold).sum())
        result[key] = {
            "TP": tp,
            "FP": prediction_count - tp,
            "FN": gt_count - tp,
        }
    return result


def _key(score: dict[str, Any], edits: int) -> tuple[int, int, float, int]:
    return (
        int(score["0.50"]["TP"]),
        int(score["0.75"]["TP"]),
        float(score["total_iou"]),
        -int(edits),
    )


def _popcount(value: int) -> int:
    return bin(int(value)).count("1")


def _evaluate_image(
    rel: str,
    dataset_root: str,
    source_root: str,
    raw_root: str,
) -> dict[str, Any]:
    annotation = load_culane_img_data(
        Path(dataset_root) / rel.replace(".jpg", ".lines.txt")
    )
    source = load_culane_img_data(
        Path(source_root) / rel.replace(".jpg", ".lines.txt")
    )
    raw_unaligned = load_culane_img_data(
        Path(raw_root) / rel.replace(".jpg", ".lines.txt")
    )
    raw = align_raw_to_source(source, raw_unaligned)
    lanes = len(source)
    gt_count = len([lane for lane in annotation if len(lane) >= 2])
    quality = _quality_matrix(source, raw, annotation)
    source_selection = tuple(range(lanes))
    raw_selection = tuple(range(lanes, 2 * lanes))
    source_score = _score_selection(
        quality,
        source_selection,
        prediction_count=lanes,
        gt_count=gt_count,
    )
    raw_score = _score_selection(
        quality,
        raw_selection,
        prediction_count=lanes,
        gt_count=gt_count,
    )

    best_score = source_score
    best_mask = 0
    for mask in range(1, 1 << lanes):
        selection = tuple(
            slot + lanes if mask & (1 << slot) else slot
            for slot in range(lanes)
        )
        score = _score_selection(
            quality,
            selection,
            prediction_count=lanes,
            gt_count=gt_count,
        )
        edits = _popcount(mask)
        if _key(score, edits) > _key(best_score, _popcount(best_mask)):
            best_score = score
            best_mask = mask

    image_candidates = ((source_score, 0), (raw_score, lanes))
    image_score, image_edits = max(
        image_candidates, key=lambda value: _key(value[0], value[1])
    )
    return {
        "source": source_score,
        "raw": raw_score,
        "whole_image_binary_oracle": image_score,
        "per_lane_binary_oracle": best_score,
        "per_lane_edit_count": _popcount(best_mask),
        "whole_image_edit_count": int(image_edits),
        "prediction_count": lanes,
        "gt_count": gt_count,
    }


def _finish(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policies = (
        "source",
        "raw",
        "whole_image_binary_oracle",
        "per_lane_binary_oracle",
    )
    totals = {
        policy: {
            f"{threshold:.2f}": {"TP": 0, "FP": 0, "FN": 0}
            for threshold in THRESHOLDS
        }
        for policy in policies
    }
    lane_edits: Counter[int] = Counter()
    image_edits: Counter[int] = Counter()
    for row in rows:
        lane_edits[int(row["per_lane_edit_count"])] += 1
        image_edits[int(row["whole_image_edit_count"])] += 1
        for policy in policies:
            for threshold in THRESHOLDS:
                key = f"{threshold:.2f}"
                for name in ("TP", "FP", "FN"):
                    totals[policy][key][name] += int(row[policy][key][name])
    metrics = {
        policy: {key: _finish(value) for key, value in threshold.items()}
        for policy, threshold in totals.items()
    }
    source = metrics["source"]
    return {
        "metrics": metrics,
        "f1_gain_points_vs_source": {
            policy: {
                key: 100.0
                * (float(value["F1"]) - float(source[key]["F1"]))
                for key, value in threshold.items()
            }
            for policy, threshold in metrics.items()
            if policy != "source"
        },
        "tp_gain_vs_source": {
            policy: {
                key: int(value["TP"]) - int(source[key]["TP"])
                for key, value in threshold.items()
            }
            for policy, threshold in metrics.items()
            if policy != "source"
        },
        "per_lane_edit_histogram": dict(sorted(lane_edits.items())),
        "whole_image_edit_histogram": dict(sorted(image_edits.items())),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(dataset_root, split="val")
    rels = list_image_rel_paths(Path(population["list_path"]))
    tasks = zip(
        rels,
        repeat(str(dataset_root)),
        repeat(str(Path(args.source_predictions).expanduser().resolve())),
        repeat(str(Path(args.raw_predictions).expanduser().resolve())),
    )
    workers = int(args.workers) if int(args.workers) > 0 else cpu_count()
    with Pool(workers) as pool:
        rows = list(
            tqdm(
                pool.starmap(
                    _evaluate_image,
                    tasks,
                    chunksize=max(int(args.chunksize), 1),
                ),
                total=len(rels),
                desc="V23 source/raw binary oracle",
                ncols=80,
            )
        )
    report = {
        "experiment": "V23 fixed-endpoint source-versus-raw binary oracle",
        "diagnostic_only": True,
        "images": len(rows),
        **summarize(rows),
        "contract": {
            "optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "gate_or_interpolation_sweep_performed": False,
            "candidate_choices_per_lane": ["source_v7", "raw_student"],
            "joint_objective": "TP@.50 then TP@.75 then total IoU then minimum edits",
            "official_val_rows": 9_675,
            "validation_subset_used": False,
            "test_set_used": False,
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
