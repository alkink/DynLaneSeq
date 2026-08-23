from __future__ import annotations

import argparse
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from dynlaneseq_eg.evaluation.culane_metric import (
    culane_metric,
    list_image_rel_paths,
    load_culane_img_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired image- and clip-level official-raster audit for a "
            "full-validation candidate and its source."
        )
    )
    parser.add_argument(
        "--experiment-name",
        default="paired full-validation official-raster effect audit",
    )
    parser.add_argument("--source-pred-dir", required=True)
    parser.add_argument("--candidate-pred-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--uniform-report", required=True)
    parser.add_argument("--source-metrics")
    parser.add_argument("--candidate-metrics")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _prediction_path(root: str | Path, rel: str) -> Path:
    return Path(root) / rel.replace(".jpg", ".lines.txt")


def _paired_metric_one(args: tuple[Any, ...]):
    (
        rel,
        source_pred_dir,
        candidate_pred_dir,
        dataset_root,
        width,
        thresholds,
    ) = args
    annotation = load_culane_img_data(_prediction_path(dataset_root, rel))
    source = load_culane_img_data(_prediction_path(source_pred_dir, rel))
    candidate = load_culane_img_data(_prediction_path(candidate_pred_dir, rel))
    return (
        culane_metric(
            source,
            annotation,
            width=int(width),
            iou_thresholds=thresholds,
            official=True,
        ),
        culane_metric(
            candidate,
            annotation,
            width=int(width),
            iou_thresholds=thresholds,
            official=True,
        ),
    )


def _f1(tp: int | np.ndarray, fp: int | np.ndarray, fn: int | np.ndarray):
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )


def _aggregate(rows: np.ndarray, indices: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn = rows[indices].sum(axis=0).astype(np.int64).tolist()
    prediction_count = int(tp + fp)
    precision = float(tp) / float(prediction_count) if prediction_count else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "prediction_count": prediction_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _paired_transition(
    source: np.ndarray, candidate: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    delta = candidate[indices] - source[indices]
    delta_tp = delta[:, 0]
    delta_predictions = delta[:, 0] + delta[:, 1]
    values, counts = np.unique(delta_tp, return_counts=True)
    return {
        "images": int(indices.size),
        "tp_improved_images": int((delta_tp > 0).sum()),
        "tp_worsened_images": int((delta_tp < 0).sum()),
        "tp_tied_images": int((delta_tp == 0).sum()),
        "positive_tp_changes": int(delta_tp[delta_tp > 0].sum()),
        "negative_tp_changes": int(delta_tp[delta_tp < 0].sum()),
        "prediction_count_increased_images": int((delta_predictions > 0).sum()),
        "prediction_count_decreased_images": int((delta_predictions < 0).sum()),
        "prediction_count_tied_images": int((delta_predictions == 0).sum()),
        "tp_delta_histogram": {
            str(int(value)): int(count) for value, count in zip(values, counts)
        },
    }


def _paired_bootstrap(
    source: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    deltas: list[np.ndarray] = []
    chunk_size = 256
    for start in range(0, int(samples), chunk_size):
        current = min(chunk_size, int(samples) - start)
        draws = rng.choice(indices, size=(current, indices.size), replace=True)
        source_sum = source[draws].sum(axis=1)
        candidate_sum = candidate[draws].sum(axis=1)
        deltas.append(
            _f1(candidate_sum[:, 0], candidate_sum[:, 1], candidate_sum[:, 2])
            - _f1(source_sum[:, 0], source_sum[:, 1], source_sum[:, 2])
        )
    values = np.concatenate(deltas) if deltas else np.zeros((0,), dtype=np.float64)
    return {
        "samples": int(samples),
        "seed": int(seed),
        "mean_delta_f1": float(values.mean()),
        "median_delta_f1": float(np.median(values)),
        "ci_2p5": float(np.percentile(values, 2.5)),
        "ci_97p5": float(np.percentile(values, 97.5)),
        "probability_delta_positive": float((values > 0.0).mean()),
    }


def _paired_cluster_bootstrap(
    source: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Resample complete clips so correlated frames move together."""

    labels = cluster_labels[indices]
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    cluster_count = int(unique_labels.size)
    source_by_cluster = np.zeros((cluster_count, 3), dtype=np.int64)
    candidate_by_cluster = np.zeros((cluster_count, 3), dtype=np.int64)
    np.add.at(source_by_cluster, inverse, source[indices])
    np.add.at(candidate_by_cluster, inverse, candidate[indices])

    rng = np.random.default_rng(int(seed))
    deltas: list[np.ndarray] = []
    chunk_size = 256
    cluster_indices = np.arange(cluster_count, dtype=np.int64)
    for start in range(0, int(samples), chunk_size):
        current = min(chunk_size, int(samples) - start)
        draws = rng.choice(
            cluster_indices,
            size=(current, cluster_count),
            replace=True,
        )
        source_sum = source_by_cluster[draws].sum(axis=1)
        candidate_sum = candidate_by_cluster[draws].sum(axis=1)
        deltas.append(
            _f1(candidate_sum[:, 0], candidate_sum[:, 1], candidate_sum[:, 2])
            - _f1(source_sum[:, 0], source_sum[:, 1], source_sum[:, 2])
        )
    values = (
        np.concatenate(deltas)
        if deltas
        else np.zeros((0,), dtype=np.float64)
    )
    return {
        "samples": int(samples),
        "seed": int(seed),
        "cluster_definition": "parent directory of validation image path",
        "clusters": cluster_count,
        "images": int(indices.size),
        "mean_delta_f1": float(values.mean()),
        "median_delta_f1": float(np.median(values)),
        "ci_2p5": float(np.percentile(values, 2.5)),
        "ci_97p5": float(np.percentile(values, 97.5)),
        "probability_delta_positive": float((values > 0.0).mean()),
    }


def _reported_counts(path: str | None, threshold: float) -> list[int] | None:
    if not path:
        return None
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    row = report["results"][str(float(threshold))]
    return [int(row["TP"]), int(row["FP"]), int(row["FN"])]


def _group_summary(
    source: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    source_summary = _aggregate(source, indices)
    candidate_summary = _aggregate(candidate, indices)
    return {
        "source": source_summary,
        "candidate": candidate_summary,
        "delta": {
            name: float(candidate_summary[name]) - float(source_summary[name])
            for name in ("tp", "fp", "fn", "prediction_count", "precision", "recall", "f1")
        },
        "paired_transitions": _paired_transition(source, candidate, indices),
        "paired_bootstrap_f1_delta": _paired_bootstrap(
            source,
            candidate,
            indices,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "paired_clip_bootstrap_f1_delta": _paired_cluster_bootstrap(
            source,
            candidate,
            indices,
            cluster_labels,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 5000,
        ),
    }


def main() -> None:
    args = parse_args()
    rels = list_image_rel_paths(args.list_path)
    cluster_labels = np.asarray(
        [Path(rel).parent.as_posix() for rel in rels], dtype=object
    )
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
        paired = list(
            tqdm(
                pool.imap(
                    _paired_metric_one,
                    tasks,
                    chunksize=max(1, int(args.chunksize)),
                ),
                total=len(rels),
                desc="paired official-raster audit",
                ncols=80,
            )
        )

    uniform_report = json.loads(
        Path(args.uniform_report).read_text(encoding="utf-8")
    )
    uniform_indices = np.asarray(
        uniform_report["metadata"]["sampled_dataset_indices"], dtype=np.int64
    )
    if np.unique(uniform_indices).size != uniform_indices.size:
        raise ValueError("uniform sampled indices contain duplicates")
    if uniform_indices.min() < 0 or uniform_indices.max() >= len(rels):
        raise ValueError("uniform sampled indices are outside the validation split")
    uniform_mask = np.zeros((len(rels),), dtype=bool)
    uniform_mask[uniform_indices] = True
    complement_indices = np.flatnonzero(~uniform_mask)
    all_indices = np.arange(len(rels), dtype=np.int64)

    by_threshold: dict[str, Any] = {}
    report_match: dict[str, Any] = {}
    for threshold in thresholds:
        source = np.asarray(
            [row[0][threshold] for row in paired], dtype=np.int64
        )
        candidate = np.asarray(
            [row[1][threshold] for row in paired], dtype=np.int64
        )
        key = str(float(threshold))
        by_threshold[key] = {
            "full_validation": _group_summary(
                source,
                candidate,
                all_indices,
                cluster_labels,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + int(round(threshold * 100)),
            ),
            "uniform_256": _group_summary(
                source,
                candidate,
                uniform_indices,
                cluster_labels,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 1000 + int(round(threshold * 100)),
            ),
            "complement_9419": _group_summary(
                source,
                candidate,
                complement_indices,
                cluster_labels,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 2000 + int(round(threshold * 100)),
            ),
        }
        source_reported = _reported_counts(args.source_metrics, threshold)
        candidate_reported = _reported_counts(args.candidate_metrics, threshold)
        source_recomputed = source.sum(axis=0).astype(int).tolist()
        candidate_recomputed = candidate.sum(axis=0).astype(int).tolist()
        report_match[key] = {
            "source_reported": source_reported,
            "source_recomputed": source_recomputed,
            "source_exact": source_reported in (None, source_recomputed),
            "candidate_reported": candidate_reported,
            "candidate_recomputed": candidate_recomputed,
            "candidate_exact": candidate_reported in (None, candidate_recomputed),
        }

    payload = {
        "experiment": args.experiment_name,
        "diagnostic_only": True,
        "test_set_used": False,
        "official_raster": True,
        "images": len(rels),
        "clips": int(np.unique(cluster_labels).size),
        "uniform_images": int(uniform_indices.size),
        "complement_images": int(complement_indices.size),
        "source_pred_dir": args.source_pred_dir,
        "candidate_pred_dir": args.candidate_pred_dir,
        "report_reproduction": report_match,
        "thresholds": by_threshold,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
