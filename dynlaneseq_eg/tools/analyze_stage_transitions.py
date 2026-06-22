from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    exact_official_counts,
    load_or_collect_cache,
    metadata_for_json,
    trace_postprocess,
    write_json,
)


STATE_PRIORITY = ("topk_removed", "nms_removed", "below_threshold", "geometry_matchable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry, score, NMS, and Top-K stage-transition accounting.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.55])
    parser.add_argument("--quality-powers", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[4])
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--exact-postprocess", action="store_true")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--stage-configs", nargs="*", default=[])
    parser.add_argument("--stage-checkpoints", nargs="*", default=[])
    parser.add_argument("--stage-names", nargs="*", default=[])
    return parser.parse_args()


def primary_stage_name(record: dict[str, Any]) -> str:
    stages = record["stages"]
    for key in ("final", "stage2", "main", "coarse"):
        if key in stages:
            return key
    if not stages:
        raise ValueError(f"No prediction stages for image {record['image_id']}")
    return next(iter(stages))


def gt_states(
    iou: torch.Tensor,
    trace: dict[str, Any],
    threshold: float,
) -> tuple[list[str], int, int]:
    gt_count = iou.shape[0]
    selected = list(trace["selected_ids"])
    selected_assignment = evaluator_hungarian_assignment(iou, selected, threshold=threshold)
    selected_gt = {gt_idx for gt_idx, _proposal_idx in selected_assignment.pairs}
    states: list[str] = []
    for gt_idx in range(gt_count):
        matchable = torch.nonzero(iou[gt_idx] >= float(threshold), as_tuple=False).flatten().tolist()
        if not matchable:
            states.append("absent_raw")
            continue
        if gt_idx in selected_gt:
            states.append("selected_tp")
            continue
        proposal_states = {trace["status"][proposal_idx] for proposal_idx in matchable}
        state = next((name for name in STATE_PRIORITY if name in proposal_states), "geometry_matchable")
        states.append(state)
    selected_fp = max(0, len(selected) - selected_assignment.hit_count)
    return states, selected_assignment.hit_count, selected_fp


def evaluate_record_stage(
    record: dict[str, Any],
    stage_name: str,
    input_h: int,
    input_w: int,
    args: argparse.Namespace,
    score_threshold: float,
    quality_power: float,
    top_k: int,
    nms_distance: float,
    nms_overlap: int,
) -> dict[str, Any]:
    stage = record["stages"][stage_name]
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
    states, selected_tp, selected_fp = gt_states(iou, trace, args.iou_thresh)
    slot_best_iou = iou.max(dim=0).values if iou.shape[0] else torch.zeros(iou.shape[1])
    return {
        "states": states,
        "raw_matchable": [bool(value >= args.iou_thresh) for value in (iou.max(dim=1).values if iou.shape[1] else torch.zeros(iou.shape[0]))],
        "selected_tp": selected_tp,
        "selected_fp": selected_fp,
        "selected_ids": list(trace["selected_ids"]),
        "slot_best_iou": slot_best_iou,
        "slot_status": list(trace["status"]),
        "candidate_valid": candidate_valid,
    }


def summarize_pair(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    same_model_slots: bool,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    state_matrix: Counter[str] = Counter()
    summary: Counter[str] = Counter()
    slot_matrix: Counter[str] = Counter()
    for item_a, item_b in zip(before, after):
        for state_a, state_b, raw_a, raw_b in zip(
            item_a["states"], item_b["states"], item_a["raw_matchable"], item_b["raw_matchable"]
        ):
            state_matrix[f"{state_a}->{state_b}"] += 1
            summary["total_gt"] += 1
            if state_a != "selected_tp" and state_b == "selected_tp":
                summary["rescued_tp"] += 1
            if raw_a and not raw_b:
                summary["killed_by_geometry"] += 1
            if state_a == "selected_tp" and state_b == "below_threshold":
                summary["killed_by_score"] += 1
            if state_a == "selected_tp" and state_b == "nms_removed":
                summary["killed_by_nms"] += 1
            if state_a == "selected_tp" and state_b == "topk_removed":
                summary["killed_by_topk"] += 1
        summary["selected_tp_before"] += int(item_a["selected_tp"])
        summary["selected_tp_after"] += int(item_b["selected_tp"])
        summary["selected_fp_before"] += int(item_a["selected_fp"])
        summary["selected_fp_after"] += int(item_b["selected_fp"])
        summary["fp_removed_net"] += max(0, int(item_a["selected_fp"]) - int(item_b["selected_fp"]))

        if same_model_slots:
            count = min(len(item_a["slot_status"]), len(item_b["slot_status"]))
            for proposal_idx in range(count):
                geom_a = bool(item_a["slot_best_iou"][proposal_idx] >= float(iou_threshold))
                geom_b = bool(item_b["slot_best_iou"][proposal_idx] >= float(iou_threshold))
                key = f"{'tp' if geom_a else 'fp'}:{item_a['slot_status'][proposal_idx]}->{'tp' if geom_b else 'fp'}:{item_b['slot_status'][proposal_idx]}"
                slot_matrix[key] += 1
    for key in (
        "total_gt",
        "rescued_tp",
        "killed_by_geometry",
        "killed_by_score",
        "killed_by_nms",
        "killed_by_topk",
        "selected_tp_before",
        "selected_tp_after",
        "selected_fp_before",
        "selected_fp_after",
        "fp_removed_net",
    ):
        summary.setdefault(key, 0)
    summary["fp_removed"] = summary["fp_removed_net"]
    out: dict[str, Any] = {
        "summary": dict(sorted(summary.items())),
        "gt_state_transitions": dict(sorted(state_matrix.items())),
        "slot_transitions": dict(sorted(slot_matrix.items())) if same_model_slots else None,
    }
    return out


def load_caches(args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]], bool]:
    if args.stage_configs and args.stage_checkpoints:
        if len(args.stage_configs) != len(args.stage_checkpoints):
            raise ValueError("--stage-configs and --stage-checkpoints must have equal length")
        if args.stage_names and len(args.stage_names) != len(args.stage_configs):
            raise ValueError("--stage-names must match --stage-configs length")
        names = list(args.stage_names) if args.stage_names else [f"stage{idx}" for idx in range(len(args.stage_configs))]
        caches = [
            load_or_collect_cache(
                config,
                checkpoint,
                split=args.split,
                list_path=args.list_path or None,
                device=args.device,
                cache_dir=args.cache_dir,
                reuse_cache=args.reuse_cache,
                max_batches=args.max_batches,
                desc=f"cache {name}",
            )
            for name, config, checkpoint in zip(names, args.stage_configs, args.stage_checkpoints)
        ]
        hashes = {cache["metadata"]["list_sha256"] for cache in caches}
        if len(hashes) != 1:
            raise ValueError("Multi-checkpoint caches do not use the same evaluation list")
        return names, caches, False
    if not args.config or not args.checkpoint:
        raise ValueError("Provide either config/checkpoint or stage-configs/stage-checkpoints")
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        desc="transition cache",
    )
    stage_order = [name for name in ("coarse", "s0_geometry_draft", "stage1", "stage2", "final", "main") if any(name in record["stages"] for record in cache["records"])]
    if len(stage_order) < 2:
        raise ValueError(f"Single checkpoint exposes fewer than two stages: {stage_order}")
    return stage_order, [cache], True


@torch.no_grad()
def main() -> None:
    args = parse_args()
    names, caches, single_cache = load_caches(args)
    if args.exact_postprocess:
        caches = [
            ensure_official_iou_cache(
                cache,
                line_width=args.line_width,
                min_valid_rows=args.min_valid_rows,
                row_visibility_thresh=args.row_visibility_thresh,
            )
            for cache in caches
        ]
    base_metadata = caches[0]["metadata"]
    post = base_metadata.get("postprocess", {})
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
    cache_maps = [{record["image_id"]: record for record in cache["records"]} for cache in caches]
    common_ids = set(cache_maps[0])
    for mapping in cache_maps[1:]:
        common_ids &= set(mapping)
    image_ids = sorted(common_ids)
    if not image_ids:
        raise ValueError("No common image IDs across stage caches")

    results: list[dict[str, Any]] = []
    for quality_power in args.quality_powers:
        for score_threshold in args.score_thresholds:
            for top_k in args.top_k_values:
                evaluated: list[list[dict[str, Any]]] = []
                exact_maps: list[dict[str, list[int]]] = []
                for stage_idx, name in enumerate(names):
                    cache = caches[0] if single_cache else caches[stage_idx]
                    mapping = cache_maps[0] if single_cache else cache_maps[stage_idx]
                    stage_results: list[dict[str, Any]] = []
                    selection_map: dict[str, list[int]] = {}
                    for image_id in image_ids:
                        record = mapping[image_id]
                        stage_name = name if single_cache else primary_stage_name(record)
                        item = evaluate_record_stage(
                            record,
                            stage_name,
                            input_h=int(cache["metadata"]["input_h"]),
                            input_w=int(cache["metadata"]["input_w"]),
                            args=args,
                            score_threshold=score_threshold,
                            quality_power=quality_power,
                            top_k=top_k,
                            nms_distance=nms_distance,
                            nms_overlap=nms_overlap,
                        )
                        stage_results.append(item)
                        selection_map[image_id] = item["selected_ids"]
                    evaluated.append(stage_results)
                    exact_maps.append(selection_map)

                pair_rows = []
                for index in range(len(names) - 1):
                    row = {
                        "from": names[index],
                        "to": names[index + 1],
                        **summarize_pair(
                            evaluated[index],
                            evaluated[index + 1],
                            same_model_slots=single_cache,
                            iou_threshold=args.iou_thresh,
                        ),
                    }
                    pair_rows.append(row)
                    summary = row["summary"]
                    print(
                        f"{names[index]} -> {names[index + 1]} q={quality_power:g} thr={score_threshold:g} K={top_k} "
                        f"rescued={summary.get('rescued_tp', 0)} geom_killed={summary.get('killed_by_geometry', 0)} "
                        f"score_killed={summary.get('killed_by_score', 0)} nms_killed={summary.get('killed_by_nms', 0)} "
                        f"topk_killed={summary.get('killed_by_topk', 0)}"
                    )

                stage_rows = []
                for stage_idx, name in enumerate(names):
                    row: dict[str, Any] = {
                        "stage": name,
                        "selected_tp": int(sum(item["selected_tp"] for item in evaluated[stage_idx])),
                        "selected_fp": int(sum(item["selected_fp"] for item in evaluated[stage_idx])),
                    }
                    if args.exact_postprocess:
                        cache = caches[0] if single_cache else caches[stage_idx]
                        mapping = cache_maps[0] if single_cache else cache_maps[stage_idx]
                        records = [mapping[image_id] for image_id in image_ids]
                        exact_stage_name = name if single_cache else primary_stage_name(records[0])
                        row["official_iou_0.5"] = exact_official_counts(
                            records,
                            exact_stage_name,
                            exact_maps[stage_idx],
                            iou_threshold=0.5,
                            width=int(round(args.line_width)),
                        )
                    stage_rows.append(row)
                results.append(
                    {
                        "quality_power": quality_power,
                        "score_threshold": score_threshold,
                        "top_k": top_k,
                        "stages": stage_rows,
                        "pairs": pair_rows,
                    }
                )

    output = {
        "metadata": metadata_for_json(
            caches[0],
            tool="analyze_stage_transitions",
            mode="single_checkpoint" if single_cache else "multi_checkpoint",
            stage_names=names,
            common_images=len(image_ids),
            iou_threshold=args.iou_thresh,
            score_thresholds=args.score_thresholds,
            quality_powers=args.quality_powers,
            top_k_values=args.top_k_values,
            line_width=args.line_width,
            iou_space="official_raster" if args.exact_postprocess else "row_space",
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
        ),
        "results": results,
    }
    write_json(args.output_json or None, output)


if __name__ == "__main__":
    main()
