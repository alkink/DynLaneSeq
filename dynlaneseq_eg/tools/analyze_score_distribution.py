from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    exact_official_counts,
    load_or_collect_cache,
    metadata_for_json,
    stage_scores,
    trace_postprocess,
    unique_candidate_labels,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate-aware proposal score and calibration diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.55])
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--quality-powers", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--exact-postprocess", action="store_true")
    return parser.parse_args()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels.astype(bool)].sum())
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int) -> float:
    if len(scores) == 0:
        return 0.0
    total = float(len(scores))
    error = 0.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & (scores < upper if index + 1 < bins else scores <= upper)
        if not mask.any():
            continue
        error += float(mask.sum()) / total * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def safe_spearman(scores: np.ndarray, ious: np.ndarray) -> float:
    if len(scores) < 2 or np.all(scores == scores[0]) or np.all(ious == ious[0]):
        return 0.0
    value = float(spearmanr(scores, ious).statistic)
    return 0.0 if not math.isfinite(value) else value


def summarize_strategy(items: list[dict[str, Any]], thresholds: list[float], bins: int) -> dict[str, Any]:
    valid_items = [item for item in items if item["label"] != "invalid_range"]
    scores = np.asarray([item["score"] for item in valid_items], dtype=np.float64)
    ious = np.asarray([item["best_iou"] for item in valid_items], dtype=np.float64)
    labels = np.asarray([item["label"] == "unique_tp" for item in valid_items], dtype=np.int64)
    label_counts = Counter(item["label"] for item in items)
    result: dict[str, Any] = {
        "slots": len(items),
        "label_counts": dict(sorted(label_counts.items())),
        "spearman_score_iou": safe_spearman(scores, ious),
        "average_precision_unique_tp": average_precision(scores, labels),
        "auroc_unique_tp": binary_auroc(scores, labels),
        "ece_unique_tp": expected_calibration_error(scores, labels, bins),
    }
    threshold_rows = []
    for threshold in thresholds:
        high = scores >= float(threshold)
        unique = labels.astype(bool)
        duplicate = np.asarray([item["label"] == "duplicate" for item in valid_items], dtype=bool)
        background = np.asarray([item["label"] == "background" for item in valid_items], dtype=bool)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "unique_tp_high": int((unique & high).sum()),
                "unique_tp_low": int((unique & ~high).sum()),
                "duplicate_high": int((duplicate & high).sum()),
                "duplicate_low": int((duplicate & ~high).sum()),
                "background_high": int((background & high).sum()),
                "background_low": int((background & ~high).sum()),
                "matchable_below_threshold": int((unique & ~high).sum()),
            }
        )
    result["thresholds"] = threshold_rows

    ranks = [int(item["rank"]) for item in valid_items if item["label"] == "unique_tp"]
    if ranks:
        sorted_ranks = sorted(ranks)
        result["unique_tp_rank"] = {
            "count": len(ranks),
            "mean": float(sum(ranks) / len(ranks)),
            "median": float(sorted_ranks[len(sorted_ranks) // 2]),
            "p90": float(sorted_ranks[min(len(sorted_ranks) - 1, int(0.9 * (len(sorted_ranks) - 1)))]),
            "top4_rate": float(sum(rank <= 4 for rank in ranks) / len(ranks)),
            "top6_rate": float(sum(rank <= 6 for rank in ranks) / len(ranks)),
            "top8_rate": float(sum(rank <= 8 for rank in ranks) / len(ranks)),
        }
    else:
        result["unique_tp_rank"] = {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "top4_rate": 0.0, "top6_rate": 0.0, "top8_rate": 0.0}
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        desc="score cache",
    )
    metadata = cache["metadata"]
    if args.exact_postprocess:
        cache = ensure_official_iou_cache(
            cache,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
    input_h = int(metadata["input_h"])
    input_w = int(metadata["input_w"])
    post = metadata.get("postprocess", {})
    nms_distance = float(
        args.nms_distance_thresh_px
        if args.nms_distance_thresh_px is not None
        else post.get("lane_nms_distance_thresh_px", 20.0)
    )
    nms_overlap = int(
        args.nms_min_overlap_points
        if args.nms_min_overlap_points is not None
        else post.get("lane_nms_min_overlap_points", 5)
    )
    stage_names = sorted({name for record in cache["records"] for name in record["stages"]})
    strategy_items: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    exact_selections: dict[tuple[str, str, float | None, int, float], dict[str, list[int]]] = defaultdict(dict)

    for record in tqdm(cache["records"], ncols=80, desc="analyzing score dist"):
        for stage_name in stage_names:
            stage = record["stages"].get(stage_name)
            if stage is None:
                continue
            iou, _valid_gt, candidate_valid = diagnostic_iou_matrix(
                record,
                stage_name,
                use_official=args.exact_postprocess,
                input_h=input_h,
                input_w=input_w,
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            labels, assignment = unique_candidate_labels(iou, args.iou_thresh, candidate_valid)
            best_iou = iou.max(dim=0).values if iou.shape[0] else torch.zeros(iou.shape[1])
            strategies: list[tuple[str, float | None, torch.Tensor]] = [
                ("exist", 0.0, stage_scores(stage, quality_power=0.0)),
            ]
            quality = stage.get("quality_logits")
            if quality is not None:
                strategies.append(("quality", None, torch.sigmoid(quality.float())))
            for quality_power in args.quality_powers:
                strategies.append(("combined", float(quality_power), stage_scores(stage, quality_power=quality_power)))

            for strategy_name, quality_power, scores in strategies:
                valid_order = [idx for idx in range(len(labels)) if labels[idx] != "invalid_range"]
                valid_order.sort(key=lambda idx: float(scores[idx]), reverse=True)
                ranks = {proposal_idx: rank + 1 for rank, proposal_idx in enumerate(valid_order)}
                key = (stage_name, strategy_name, quality_power)
                for proposal_idx, label in enumerate(labels):
                    strategy_items[key].append(
                        {
                            "image_id": record["image_id"],
                            "proposal_id": proposal_idx,
                            "label": label,
                            "score": float(scores[proposal_idx]),
                            "best_iou": float(best_iou[proposal_idx]),
                            "rank": ranks.get(proposal_idx, len(valid_order) + 1),
                        }
                    )
                if args.exact_postprocess and strategy_name in {"exist", "combined"}:
                    qpower = 0.0 if strategy_name == "exist" else float(quality_power or 0.0)
                    for top_k in args.top_k_values:
                        for score_threshold in args.score_thresholds:
                            trace = trace_postprocess(
                                stage,
                                input_h=input_h,
                                input_w=input_w,
                                score_thresh=score_threshold,
                                quality_power=qpower,
                                min_valid_rows=args.min_valid_rows,
                                nms_distance_thresh_px=nms_distance,
                                nms_min_overlap_points=nms_overlap,
                                top_k=top_k,
                                row_visibility_thresh=args.row_visibility_thresh,
                            )
                            exact_selections[(stage_name, strategy_name, quality_power, top_k, score_threshold)][record["image_id"]] = list(trace["selected_ids"])

    summaries: list[dict[str, Any]] = []
    for key in sorted(strategy_items, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        stage_name, strategy_name, quality_power = key
        summary = summarize_strategy(strategy_items[key], args.score_thresholds, args.calibration_bins)
        row: dict[str, Any] = {
            "stage": stage_name,
            "strategy": strategy_name,
            "quality_power": quality_power,
            **summary,
        }
        if args.exact_postprocess:
            official_rows = []
            for top_k in args.top_k_values:
                for score_threshold in args.score_thresholds:
                    selection_key = (stage_name, strategy_name, quality_power, top_k, score_threshold)
                    if selection_key not in exact_selections:
                        continue
                    official_rows.append(
                        {
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            **exact_official_counts(
                                cache["records"],
                                stage_name,
                                exact_selections[selection_key],
                                iou_threshold=0.5,
                                width=int(round(args.line_width)),
                            ),
                        }
                    )
            row["official_postprocess"] = official_rows
        summaries.append(row)
        rank = row["unique_tp_rank"]
        print(
            f"{stage_name:>10} {strategy_name:>9} q={str(quality_power):>4} "
            f"AP={row['average_precision_unique_tp']:.4f} AUROC={row['auroc_unique_tp']:.4f} "
            f"rho={row['spearman_score_iou']:.4f} ECE={row['ece_unique_tp']:.4f} "
            f"TP@4={rank['top4_rate']:.4f}"
        )

    output = {
        "metadata": metadata_for_json(
            cache,
            tool="analyze_score_distribution",
            iou_threshold=args.iou_thresh,
            quality_powers=args.quality_powers,
            score_thresholds=args.score_thresholds,
            top_k_values=args.top_k_values,
            line_width=args.line_width,
            iou_space="official_raster" if args.exact_postprocess else "row_space",
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
        ),
        "summaries": summaries,
    }
    write_json(args.output_json or None, output)


if __name__ == "__main__":
    main()
