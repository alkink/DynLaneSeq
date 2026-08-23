from __future__ import annotations

import argparse
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    interp,
    list_image_rel_paths,
    load_culane_img_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two CULane prediction directories GT by GT using the exact "
            "official-raster Hungarian assignment."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default="paired GT transition audit",
    )
    parser.add_argument("--source-pred-dir", required=True)
    parser.add_argument("--candidate-pred-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--top-changed-images", type=int, default=50)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prediction_path(root: str | Path, rel: str) -> Path:
    return Path(root) / rel.replace(".jpg", ".lines.txt")


def _matched_gt_values(
    pred: list[list[tuple[float, float]]],
    anno: list[list[tuple[float, float]]],
    *,
    width: int,
) -> np.ndarray:
    values = np.zeros((len(anno),), dtype=np.float32)
    if not pred or not anno:
        return values
    interp_pred = [interp(lane, n=5) for lane in pred if len(lane) >= 2]
    interp_anno = [interp(lane, n=5) for lane in anno if len(lane) >= 2]
    if not interp_pred or not interp_anno:
        return values
    ious = discrete_cross_iou(interp_pred, interp_anno, width=width)
    pred_ids, gt_ids = linear_sum_assignment(1.0 - ious)
    values[gt_ids] = ious[pred_ids, gt_ids]
    return values


def _audit_one(args: tuple[Any, ...]) -> dict[str, Any]:
    rel, source_dir, candidate_dir, dataset_root, width, thresholds = args
    anno = load_culane_img_data(_prediction_path(dataset_root, rel))
    source = load_culane_img_data(_prediction_path(source_dir, rel))
    candidate = load_culane_img_data(_prediction_path(candidate_dir, rel))
    source_values = _matched_gt_values(source, anno, width=int(width))
    candidate_values = _matched_gt_values(candidate, anno, width=int(width))
    transitions: dict[str, list[int]] = {}
    for threshold in thresholds:
        source_tp = source_values > float(threshold)
        candidate_tp = candidate_values > float(threshold)
        transitions[str(float(threshold))] = [
            int((source_tp & candidate_tp).sum()),
            int((~source_tp & candidate_tp).sum()),
            int((source_tp & ~candidate_tp).sum()),
            int((~source_tp & ~candidate_tp).sum()),
        ]
    return {
        "rel": rel,
        "gt": len(anno),
        "source_predictions": len(source),
        "candidate_predictions": len(candidate),
        "transitions": transitions,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def main() -> None:
    args = parse_args()
    rels = list_image_rel_paths(args.list_path)
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    tasks: Iterable[tuple[Any, ...]] = zip(
        rels,
        repeat(args.source_pred_dir),
        repeat(args.candidate_pred_dir),
        repeat(args.dataset_root),
        repeat(args.width),
        repeat(thresholds),
    )
    workers = int(args.workers) if int(args.workers) > 0 else cpu_count()
    with Pool(workers) as pool:
        rows = list(
            tqdm(
                pool.imap(_audit_one, tasks, chunksize=max(1, int(args.chunksize))),
                total=len(rels),
                desc="paired GT transition audit",
                ncols=80,
            )
        )

    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        key = str(float(threshold))
        counts = np.asarray([row["transitions"][key] for row in rows], dtype=np.int64)
        both_tp, recovered, lost, both_fn = counts.sum(axis=0).tolist()
        source_tp = int(both_tp + lost)
        candidate_tp = int(both_tp + recovered)
        gt_total = int(counts.sum())
        image_net = counts[:, 1] - counts[:, 2]
        order_improved = np.argsort(-image_net, kind="stable")
        order_worsened = np.argsort(image_net, kind="stable")

        def top_records(order: np.ndarray, positive: bool) -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            for index in order.tolist():
                net = int(image_net[index])
                if (positive and net <= 0) or (not positive and net >= 0):
                    break
                output.append(
                    {
                        "rel": rows[index]["rel"],
                        "net_tp_delta": net,
                        "recovered": int(counts[index, 1]),
                        "lost": int(counts[index, 2]),
                    }
                )
                if len(output) >= int(args.top_changed_images):
                    break
            return output

        by_threshold[key] = {
            "gt_total": gt_total,
            "both_tp": int(both_tp),
            "recovered_source_fn_to_candidate_tp": int(recovered),
            "lost_source_tp_to_candidate_fn": int(lost),
            "both_fn": int(both_fn),
            "source_tp": source_tp,
            "candidate_tp": candidate_tp,
            "net_tp_delta": int(candidate_tp - source_tp),
            "source_tp_retention": _safe_ratio(int(both_tp), source_tp),
            "source_fn_recovery_rate": _safe_ratio(int(recovered), gt_total - source_tp),
            "recovered_to_lost_ratio": _safe_ratio(int(recovered), int(lost)),
            "images_net_improved": int((image_net > 0).sum()),
            "images_net_worsened": int((image_net < 0).sum()),
            "images_net_tied": int((image_net == 0).sum()),
            "top_improved_images": top_records(order_improved, True),
            "top_worsened_images": top_records(order_worsened, False),
        }

    source_prediction_count = sum(int(row["source_predictions"]) for row in rows)
    candidate_prediction_count = sum(int(row["candidate_predictions"]) for row in rows)
    payload = {
        "experiment": args.experiment_name,
        "diagnostic_only": True,
        "test_set_used": False,
        "official_raster": True,
        "images": len(rows),
        "source_prediction_count": source_prediction_count,
        "candidate_prediction_count": candidate_prediction_count,
        "prediction_count_delta": candidate_prediction_count - source_prediction_count,
        "thresholds": by_threshold,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
