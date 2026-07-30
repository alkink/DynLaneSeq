from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    exact_official_counts,
    load_or_collect_cache,
    metadata_for_json,
    recall_from_ids,
    stage_scores,
    trace_postprocess,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate-safe Oracle Top-K and deployed-ranking diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--top-k", type=int, default=None, help="Compatibility alias for a single Top-K value.")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--quality-powers", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.55])
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--sample-strategy",
        choices=("sequential", "uniform"),
        default="sequential",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--exact-postprocess", action="store_true")
    return parser.parse_args()


def _rank_ids(scores: torch.Tensor, candidate_valid: torch.Tensor, top_k: int) -> list[int]:
    ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
    ids.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return ids[: int(top_k)] if top_k > 0 else ids


def _quality_scores(stage: dict[str, torch.Tensor]) -> torch.Tensor:
    quality = stage.get("quality_logits")
    if quality is None:
        return torch.ones(stage["pred_x_rows"].shape[0], dtype=torch.float32)
    return torch.sigmoid(quality.float())


def _selection_scores(stage: dict[str, torch.Tensor]) -> torch.Tensor | None:
    logits = stage.get("selection_logits")
    if logits is None:
        return None
    return torch.sigmoid(logits.float())


def _new_counter() -> dict[str, Any]:
    return {"hits": 0, "gt": 0, "best_iou_sum": 0.0, "images": 0}


def _update(counter: dict[str, Any], iou: torch.Tensor, ids: list[int], threshold: float) -> None:
    hits, gt_count, best = recall_from_ids(iou, ids, threshold)
    counter["hits"] += hits
    counter["gt"] += gt_count
    counter["best_iou_sum"] += float(best.sum())
    counter["images"] += 1


def _finish(counter: dict[str, Any]) -> dict[str, Any]:
    gt = max(int(counter["gt"]), 1)
    return {
        "hits": int(counter["hits"]),
        "gt": int(counter["gt"]),
        "recall": float(counter["hits"]) / gt,
        "mean_best_iou": float(counter["best_iou_sum"]) / gt,
        "images": int(counter["images"]),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    top_k_values = [args.top_k] if args.top_k is not None else list(args.top_k_values)
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        dataset_root=args.dataset_root or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sample_strategy=args.sample_strategy,
        desc="oracle cache",
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
    counters: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(_new_counter)
    exact_selections: dict[tuple[Any, ...], dict[str, list[int]]] = defaultdict(dict)

    for record in tqdm(cache["records"], ncols=80, desc="analyzing oracle topk"):
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
            all_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
            exist_scores = stage_scores(stage, quality_power=0.0)
            quality_scores = _quality_scores(stage)
            selection_scores = _selection_scores(stage)
            for iou_threshold in args.iou_thresholds:
                _update(counters[(stage_name, "all_raw", 0, iou_threshold, None, None)], iou, all_ids, iou_threshold)
                for top_k in top_k_values:
                    exist_ids = _rank_ids(exist_scores, candidate_valid, top_k)
                    quality_ids = _rank_ids(quality_scores, candidate_valid, top_k)
                    _update(counters[(stage_name, "exist_topk", top_k, iou_threshold, 0.0, None)], iou, exist_ids, iou_threshold)
                    _update(counters[(stage_name, "quality_topk", top_k, iou_threshold, None, None)], iou, quality_ids, iou_threshold)
                    if selection_scores is not None:
                        selection_ids = _rank_ids(
                            selection_scores,
                            candidate_valid,
                            top_k,
                        )
                        _update(
                            counters[
                                (
                                    stage_name,
                                    "selection_topk",
                                    top_k,
                                    iou_threshold,
                                    None,
                                    None,
                                )
                            ],
                            iou,
                            selection_ids,
                            iou_threshold,
                        )
                        for score_threshold in args.score_thresholds:
                            selection_trace = trace_postprocess(
                                stage,
                                input_h=input_h,
                                input_w=input_w,
                                score_thresh=score_threshold,
                                quality_power=0.0,
                                min_valid_rows=args.min_valid_rows,
                                nms_distance_thresh_px=nms_distance,
                                nms_min_overlap_points=nms_overlap,
                                top_k=top_k,
                                row_visibility_thresh=args.row_visibility_thresh,
                                score_override={
                                    index: float(selection_scores[index])
                                    for index in range(
                                        int(selection_scores.shape[0])
                                    )
                                },
                            )
                            selection_nms_ids = list(
                                selection_trace["selected_ids"]
                            )
                            selection_key = (
                                stage_name,
                                "selection_topk_nms",
                                top_k,
                                iou_threshold,
                                None,
                                score_threshold,
                            )
                            _update(
                                counters[selection_key],
                                iou,
                                selection_nms_ids,
                                iou_threshold,
                            )
                            if (
                                args.exact_postprocess
                                and abs(float(iou_threshold) - 0.5) < 1e-9
                            ):
                                exact_selections[selection_key][
                                    record["image_id"]
                                ] = selection_nms_ids

                    oracle = cardinality_oracle_assignment(iou, iou_threshold, top_k, candidate_valid)
                    oracle_ids = list(oracle.proposal_ids)
                    _update(counters[(stage_name, "oracle_topk", top_k, iou_threshold, None, None)], iou, oracle_ids, iou_threshold)
                    oracle_scores = {proposal_idx: 2.0 + float(iou[gt_idx, proposal_idx]) for gt_idx, proposal_idx in oracle.pairs}
                    oracle_trace = trace_postprocess(
                        stage,
                        input_h=input_h,
                        input_w=input_w,
                        score_thresh=-1.0,
                        quality_power=0.0,
                        min_valid_rows=args.min_valid_rows,
                        nms_distance_thresh_px=nms_distance,
                        nms_min_overlap_points=nms_overlap,
                        top_k=top_k,
                        row_visibility_thresh=args.row_visibility_thresh,
                        allowed_ids=oracle_ids,
                        score_override=oracle_scores,
                    )
                    oracle_nms_ids = list(oracle_trace["selected_ids"])
                    _update(
                        counters[(stage_name, "oracle_topk_nms", top_k, iou_threshold, None, None)],
                        iou,
                        oracle_nms_ids,
                        iou_threshold,
                    )
                    if args.exact_postprocess and abs(float(iou_threshold) - 0.5) < 1e-9:
                        exact_selections[(stage_name, "oracle_topk_nms", top_k, iou_threshold, None, None)][record["image_id"]] = oracle_nms_ids

                    for quality_power in args.quality_powers:
                        scores = stage_scores(stage, quality_power=quality_power)
                        ranked_ids = _rank_ids(scores, candidate_valid, top_k)
                        _update(
                            counters[(stage_name, "model_topk", top_k, iou_threshold, quality_power, None)],
                            iou,
                            ranked_ids,
                            iou_threshold,
                        )
                        for score_threshold in args.score_thresholds:
                            trace = trace_postprocess(
                                stage,
                                input_h=input_h,
                                input_w=input_w,
                                score_thresh=score_threshold,
                                quality_power=quality_power,
                                min_valid_rows=args.min_valid_rows,
                                nms_distance_thresh_px=nms_distance,
                                nms_min_overlap_points=nms_overlap,
                                top_k=top_k,
                                row_visibility_thresh=args.row_visibility_thresh,
                            )
                            selected_ids = list(trace["selected_ids"])
                            key = (
                                stage_name,
                                "model_topk_nms",
                                top_k,
                                iou_threshold,
                                quality_power,
                                score_threshold,
                            )
                            _update(counters[key], iou, selected_ids, iou_threshold)
                            if args.exact_postprocess and abs(float(iou_threshold) - 0.5) < 1e-9:
                                exact_selections[key][record["image_id"]] = selected_ids

    rows: list[dict[str, Any]] = []
    for key in sorted(counters, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        stage_name, strategy, top_k, iou_threshold, quality_power, score_threshold = key
        row = {
            "stage": stage_name,
            "strategy": strategy,
            "top_k": int(top_k),
            "iou_threshold": float(iou_threshold),
            "quality_power": quality_power,
            "score_threshold": score_threshold,
            **_finish(counters[key]),
        }
        if args.exact_postprocess and key in exact_selections:
            row["official_iou_0.5"] = exact_official_counts(
                cache["records"],
                stage_name,
                exact_selections[key],
                iou_threshold=0.5,
                width=int(round(args.line_width)),
            )
        rows.append(row)
        print(
            f"{stage_name:>10} {strategy:>18} K={top_k:<2d} IoU={iou_threshold:.1f} "
            f"q={str(quality_power):>4} thr={str(score_threshold):>4} "
            f"R={row['recall']:.4f} meanIoU={row['mean_best_iou']:.4f}"
        )

    output = {
        "metadata": metadata_for_json(
            cache,
            tool="analyze_oracle_topk",
            top_k_values=top_k_values,
            iou_thresholds=args.iou_thresholds,
            quality_powers=args.quality_powers,
            score_thresholds=args.score_thresholds,
            line_width=args.line_width,
            iou_space="official_raster" if args.exact_postprocess else "row_space",
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
        ),
        "rows": rows,
    }
    write_json(args.output_json or None, output)


if __name__ == "__main__":
    main()
