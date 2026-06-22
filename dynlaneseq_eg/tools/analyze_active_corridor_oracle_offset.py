from __future__ import annotations

import argparse
from copy import deepcopy

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.active_corridor_diagnostics import build_discrete_offset_oracle_stage
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    exact_official_counts,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    metadata_for_json,
    official_proposal_gt_iou_matrix,
    trace_postprocess,
    write_json,
)
from dynlaneseq_eg.factory import build_matcher
from dynlaneseq_eg.modeling.common import fixed_y_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact CULane ceiling for a GT-oracle Active Corridor offset selector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="outputs/candidate_diagnostics/cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--offsets-px", type=float, nargs="+", default=None)
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.55])
    parser.add_argument("--quality-power", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _target_with_range(target: dict[str, torch.Tensor], input_h: int) -> dict[str, torch.Tensor]:
    out = {key: value for key, value in target.items()}
    mask = target["valid_mask"].bool()
    y_rows = fixed_y_rows(mask.shape[-1], input_h, device=mask.device, dtype=torch.float32)
    ranges = torch.zeros((mask.shape[0], 2), dtype=torch.float32, device=mask.device)
    for gt_idx in range(mask.shape[0]):
        valid_y = y_rows[mask[gt_idx]]
        if valid_y.numel() > 0:
            ranges[gt_idx, 0] = valid_y.min()
            ranges[gt_idx, 1] = valid_y.max()
    out["range_y"] = ranges
    return out


def _batched_stage(stage: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    required = ("exist_logits", "pred_x_rows", "range_norm")
    return {key: stage[key].unsqueeze(0) for key in required}


def _aggregate_stage(
    records: list[dict],
    stage_name: str,
    *,
    input_h: int,
    input_w: int,
    score_thresholds: list[float],
    quality_power: float,
    top_k: int,
    iou_thresholds: list[float],
    line_width: float,
    min_valid_rows: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    row_visibility_thresh: float,
) -> dict[str, object]:
    raw_hits = {threshold: 0 for threshold in iou_thresholds}
    oracle_hits = {threshold: 0 for threshold in iou_thresholds}
    total_gt = 0
    selections_by_threshold: dict[float, dict[str, list[int]]] = {threshold: {} for threshold in score_thresholds}
    for record in records:
        stage = record["stages"][stage_name]
        matrix = stage["official_iou"].float()
        candidate_valid = stage["official_candidate_valid"].bool()
        valid_ids = torch.nonzero(candidate_valid, as_tuple=False).flatten().tolist()
        total_gt += int(matrix.shape[0])
        for threshold in iou_thresholds:
            raw_hits[threshold] += evaluator_hungarian_assignment(matrix, valid_ids, threshold).hit_count
            oracle_hits[threshold] += cardinality_oracle_assignment(
                matrix,
                threshold=threshold,
                top_k=top_k,
                candidate_valid=candidate_valid,
            ).hit_count
        for score_threshold in score_thresholds:
            trace = trace_postprocess(
                stage,
                input_h=input_h,
                input_w=input_w,
                score_thresh=score_threshold,
                quality_power=quality_power,
                min_valid_rows=min_valid_rows,
                nms_distance_thresh_px=nms_distance_thresh_px,
                nms_min_overlap_points=nms_min_overlap_points,
                top_k=top_k,
                row_visibility_thresh=row_visibility_thresh,
            )
            selections_by_threshold[score_threshold][record["image_id"]] = trace["selected_ids"]

    deployed = {}
    for score_threshold, selections in selections_by_threshold.items():
        deployed[str(score_threshold)] = {
            str(iou_threshold): exact_official_counts(
                records,
                stage_name,
                selections,
                iou_threshold=iou_threshold,
                width=int(round(line_width)),
            )
            for iou_threshold in iou_thresholds
        }
    denom = max(total_gt, 1)
    return {
        "total_gt": total_gt,
        "all_raw_recall": {str(key): value / denom for key, value in raw_hits.items()},
        "oracle_topk_recall": {str(key): value / denom for key, value in oracle_hits.items()},
        "deployed": deployed,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    active_cfg = model_cfg.get("active_corridor", {})
    offsets = torch.tensor(args.offsets_px or active_cfg.get("offsets_px", [-32, -24, -16, -8, 0, 8, 16, 24, 32]))
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        desc="oracle offset cache",
    )
    working = deepcopy(cache)
    matcher = build_matcher(cfg)
    totals = {"matched_slots": 0, "valid_rows": 0, "clamped_rows": 0}

    for record in working["records"]:
        coarse_name = "coarse" if "coarse" in record["stages"] else "main"
        coarse = record["stages"][coarse_name]
        target = _target_with_range(record["target"], input_h=input_h)
        match = matcher(_batched_stage(coarse), [target])[0]
        oracle_stage, stats = build_discrete_offset_oracle_stage(coarse, target, match, offsets)
        record["stages"]["oracle_discrete_offset"] = oracle_stage
        for key in totals:
            totals[key] += int(stats[key])

    for record in tqdm(working["records"], ncols=80, desc="oracle offset official IoU"):
        matrix, candidate_valid = official_proposal_gt_iou_matrix(
            record,
            "oracle_discrete_offset",
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        record["stages"]["oracle_discrete_offset"]["official_iou"] = matrix
        record["stages"]["oracle_discrete_offset"]["official_candidate_valid"] = candidate_valid.cpu()

    post = cache["metadata"].get("postprocess", {})
    stage_names = [name for name in ("coarse", "final", "main", "oracle_discrete_offset") if any(name in r["stages"] for r in working["records"])]
    results = {}
    for stage_name in stage_names:
        stage_records = [record for record in working["records"] if stage_name in record["stages"]]
        if not all("official_iou" in record["stages"][stage_name] for record in stage_records):
            continue
        results[stage_name] = _aggregate_stage(
            stage_records,
            stage_name,
            input_h=input_h,
            input_w=input_w,
            score_thresholds=list(args.score_thresholds),
            quality_power=float(args.quality_power),
            top_k=int(args.top_k),
            iou_thresholds=list(args.iou_thresholds),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            nms_distance_thresh_px=float(post.get("lane_nms_distance_thresh_px", 20.0)),
            nms_min_overlap_points=int(post.get("lane_nms_min_overlap_points", 5)),
            row_visibility_thresh=float(args.row_visibility_thresh),
        )

    clamp_ratio = totals["clamped_rows"] / max(totals["valid_rows"], 1)
    payload = {
        "metadata": metadata_for_json(
            cache,
            offsets_px=offsets.tolist(),
            quality_power=float(args.quality_power),
            top_k=int(args.top_k),
            oracle_assignment=totals,
            oracle_clamp_ratio=clamp_ratio,
        ),
        "results": results,
    }
    for stage_name, result in results.items():
        print(f"\n{stage_name}:")
        print(f"  all_raw_recall: {result['all_raw_recall']}")
        print(f"  oracle_topk_recall: {result['oracle_topk_recall']}")
        for score_threshold, by_iou in result["deployed"].items():
            metric = by_iou.get("0.5", {})
            print(
                f"  thr={score_threshold} IoU=0.5 "
                f"P={metric.get('precision', 0.0):.4f} R={metric.get('recall', 0.0):.4f} "
                f"F1={metric.get('f1', 0.0):.4f}"
            )
    print(f"\noracle_assignment: {totals} clamp_ratio={clamp_ratio:.4f}")
    write_json(args.output_json, payload)
    if args.output_json:
        print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
