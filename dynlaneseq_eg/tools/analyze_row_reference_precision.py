from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    metadata_for_json,
    stage_scores,
    trace_postprocess,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose row-reference precision loss into geometry ceiling, score "
            "threshold, NMS, Top-K/ranking, duplicate, near-miss, and background "
            "components using official CULane raster IoU."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache/rowref_precision")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy", choices=("uniform", "sequential"), default="uniform"
    )
    parser.add_argument("--stage", default="main")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--quality-power", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _resolve_stage(record: dict[str, Any], requested: str) -> str:
    if requested in record["stages"]:
        return requested
    if requested == "main":
        for name in ("final", "stage2", "coarse"):
            if name in record["stages"]:
                return name
        if len(record["stages"]) == 1:
            return next(iter(record["stages"]))
    raise KeyError(f"stage {requested!r} is unavailable for {record['image_id']}")


def _valid_ids(mask: torch.Tensor) -> list[int]:
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def _oracle_hits(
    iou: torch.Tensor,
    ids: Iterable[int],
    *,
    threshold: float,
    top_k: int,
) -> int:
    allowed = torch.zeros(int(iou.shape[1]), dtype=torch.bool)
    for index in ids:
        allowed[int(index)] = True
    return int(
        cardinality_oracle_assignment(
            iou,
            threshold,
            top_k=top_k,
            candidate_valid=allowed,
        ).hit_count
    )


def classify_false_positive(
    *,
    proposal_index: int,
    assigned_proposals: set[int],
    best_iou: torch.Tensor,
    gt_count: int,
    iou_threshold: float,
    near_min_iou: float,
) -> str:
    if int(proposal_index) in assigned_proposals:
        return "true_positive"
    if int(gt_count) == 0:
        return "empty_scene_fp"
    value = float(best_iou[int(proposal_index)])
    if value > float(iou_threshold):
        return "duplicate_fp"
    if value >= float(near_min_iou):
        return "near_miss_fp"
    return "background_fp"


def _metric(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = float(tp) / max(int(tp + fp), 1)
    recall = float(tp) / max(int(tp + fn), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _score_bin_index(score: float, bins: int) -> int:
    return min(max(int(float(score) * int(bins)), 0), int(bins) - 1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.near_min_iou) < min(args.iou_thresholds):
        raise ValueError("near_min_iou must be below every evaluated IoU threshold")
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache or args.cache_only),
        require_cache=bool(args.cache_only),
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sample_strategy=args.sample_strategy,
        desc="row-reference precision cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
    )
    metadata = cache["metadata"]
    input_h = int(metadata["input_h"])
    input_w = int(metadata["input_w"])
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    counters = {
        threshold: {
            "gt": 0,
            "selected": 0,
            "current_hits": 0,
            "oracle_all_hits": 0,
            "oracle_threshold_hits": 0,
            "oracle_nms_pool_hits": 0,
            "oracle_model_topk_pool_hits": 0,
            "fp_labels": Counter(),
            "matchable_status": Counter(),
            "all_status": Counter(),
        }
        for threshold in thresholds
    }
    score_bins: dict[float, list[dict[str, float | int]]] = {
        threshold: [
            {"count": 0, "matchable": 0, "best_iou_sum": 0.0}
            for _ in range(int(args.score_bins))
        ]
        for threshold in thresholds
    }
    empty_images = 0
    empty_images_with_prediction = 0

    for record in tqdm(cache["records"], ncols=100, desc="row-reference precision audit"):
        stage_name = _resolve_stage(record, args.stage)
        stage = record["stages"][stage_name]
        iou, _valid_gt, candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
            input_h=input_h,
            input_w=input_w,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        scores = stage_scores(stage, quality_power=args.quality_power)
        trace = trace_postprocess(
            stage,
            input_h=input_h,
            input_w=input_w,
            score_thresh=args.score_threshold,
            quality_power=args.quality_power,
            min_valid_rows=args.min_valid_rows,
            nms_distance_thresh_px=args.nms_distance,
            nms_min_overlap_points=args.nms_min_overlap_points,
            top_k=args.top_k,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        selected = [int(index) for index in trace["selected_ids"]]
        eligible = [int(index) for index in trace["eligible_ids"]]
        nms_pool = [int(index) for index in trace["nms_kept_ids"]]
        valid_ids = _valid_ids(candidate_valid)
        ranked_valid = sorted(valid_ids, key=lambda index: float(scores[index]), reverse=True)
        model_topk_pool = ranked_valid[: int(args.top_k)]
        gt_count = int(iou.shape[0])
        if gt_count == 0:
            empty_images += 1
            empty_images_with_prediction += int(bool(selected))
        best_iou = (
            iou.max(dim=0).values
            if gt_count
            else torch.zeros(int(iou.shape[1]), dtype=torch.float32)
        )

        for threshold in thresholds:
            current = evaluator_hungarian_assignment(iou, selected, threshold)
            assigned_proposals = {int(index) for index in current.proposal_ids}
            row = counters[threshold]
            row["gt"] += gt_count
            row["selected"] += len(selected)
            row["current_hits"] += int(current.hit_count)
            row["oracle_all_hits"] += _oracle_hits(
                iou, valid_ids, threshold=threshold, top_k=args.top_k
            )
            row["oracle_threshold_hits"] += _oracle_hits(
                iou, eligible, threshold=threshold, top_k=args.top_k
            )
            row["oracle_nms_pool_hits"] += _oracle_hits(
                iou, nms_pool, threshold=threshold, top_k=args.top_k
            )
            row["oracle_model_topk_pool_hits"] += _oracle_hits(
                iou, model_topk_pool, threshold=threshold, top_k=args.top_k
            )

            for proposal_index in selected:
                label = classify_false_positive(
                    proposal_index=proposal_index,
                    assigned_proposals=assigned_proposals,
                    best_iou=best_iou,
                    gt_count=gt_count,
                    iou_threshold=threshold,
                    near_min_iou=args.near_min_iou,
                )
                if label != "true_positive":
                    row["fp_labels"][label] += 1

            for proposal_index, status in enumerate(trace["status"]):
                if not bool(candidate_valid[proposal_index]):
                    continue
                row["all_status"][str(status)] += 1
                if gt_count and float(best_iou[proposal_index]) > threshold:
                    row["matchable_status"][str(status)] += 1

            for proposal_index in valid_ids:
                bin_index = _score_bin_index(float(scores[proposal_index]), args.score_bins)
                bin_row = score_bins[threshold][bin_index]
                bin_row["count"] += 1
                value = float(best_iou[proposal_index]) if gt_count else 0.0
                bin_row["best_iou_sum"] += value
                if value > threshold:
                    bin_row["matchable"] += 1

    summaries: dict[str, Any] = {}
    for threshold, row in counters.items():
        gt = max(int(row["gt"]), 1)
        selected = int(row["selected"])
        current_hits = int(row["current_hits"])
        fp = selected - current_hits
        fn = int(row["gt"]) - current_hits
        oracle_all = int(row["oracle_all_hits"])
        oracle_threshold = int(row["oracle_threshold_hits"])
        oracle_nms = int(row["oracle_nms_pool_hits"])
        oracle_model_topk = int(row["oracle_model_topk_pool_hits"])
        fp_labels = dict(row["fp_labels"])
        summaries[f"{threshold:.2f}"] = {
            "metric": _metric(current_hits, fp, fn),
            "oracle_ladder": {
                "geometry_oracle_topk_hits": oracle_all,
                "threshold_eligible_oracle_topk_hits": oracle_threshold,
                "post_nms_pool_oracle_topk_hits": oracle_nms,
                "model_ranked_topk_hits": oracle_model_topk,
                "deployed_hits": current_hits,
                "geometry_oracle_topk_recall": oracle_all / gt,
                "threshold_eligible_oracle_topk_recall": oracle_threshold / gt,
                "post_nms_pool_oracle_topk_recall": oracle_nms / gt,
                "model_ranked_topk_recall": oracle_model_topk / gt,
                "deployed_recall": current_hits / gt,
                "geometry_ceiling_misses": int(row["gt"]) - oracle_all,
                "threshold_exclusion_cost_hits": max(oracle_all - oracle_threshold, 0),
                "nms_pool_cost_hits": max(oracle_threshold - oracle_nms, 0),
                "ranking_cost_vs_post_nms_oracle_hits": max(oracle_nms - current_hits, 0),
                "total_selection_headroom_hits": max(oracle_all - current_hits, 0),
                "total_selection_headroom_points": 100.0 * max(oracle_all - current_hits, 0) / gt,
            },
            "false_positive_breakdown": {
                label: {
                    "count": int(fp_labels.get(label, 0)),
                    "fraction_of_fp": float(fp_labels.get(label, 0)) / max(fp, 1),
                }
                for label in (
                    "duplicate_fp",
                    "near_miss_fp",
                    "background_fp",
                    "empty_scene_fp",
                )
            },
            "matchable_candidate_postprocess_status": dict(row["matchable_status"]),
            "all_valid_candidate_postprocess_status": dict(row["all_status"]),
            "score_calibration_bins": [],
        }
        for bin_index, bin_row in enumerate(score_bins[threshold]):
            count = int(bin_row["count"])
            summaries[f"{threshold:.2f}"]["score_calibration_bins"].append(
                {
                    "low": bin_index / int(args.score_bins),
                    "high": (bin_index + 1) / int(args.score_bins),
                    "count": count,
                    "matchable_fraction": float(bin_row["matchable"]) / max(count, 1),
                    "mean_best_iou": float(bin_row["best_iou_sum"]) / max(count, 1),
                }
            )

    payload = {
        "diagnostic_only": True,
        "warning": (
            "Oracle pools diagnose available capacity and are not deployable "
            "predictions or benchmark results."
        ),
        "metadata": metadata_for_json(
            cache,
            tool="analyze_row_reference_precision",
            iou_space="official_raster",
            score_threshold=args.score_threshold,
            quality_power=args.quality_power,
            top_k=args.top_k,
            iou_thresholds=list(thresholds),
            near_min_iou=args.near_min_iou,
            nms_distance=args.nms_distance,
            nms_min_overlap_points=args.nms_min_overlap_points,
        ),
        "thresholds": summaries,
        "empty_scene": {
            "images": empty_images,
            "images_with_prediction": empty_images_with_prediction,
            "prediction_image_fraction": empty_images_with_prediction / max(empty_images, 1),
        },
    }
    write_json(args.output_json, payload)
    compact = {
        threshold: {
            "metric": value["metric"],
            "oracle_ladder": value["oracle_ladder"],
            "false_positive_breakdown": value["false_positive_breakdown"],
        }
        for threshold, value in summaries.items()
    }
    print(json.dumps(compact, indent=2))
    print(f"output_json: {Path(args.output_json)}")


if __name__ == "__main__":
    main()
