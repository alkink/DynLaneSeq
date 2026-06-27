from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import cv2
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    _raster_lane_mask,
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    lanes_to_original,
    metadata_for_json,
    proposal_gt_iou_matrix,
    trace_postprocess,
    write_json,
)
from dynlaneseq_eg.evaluation.culane_metric import load_culane_img_data
from dynlaneseq_eg.evaluation.near_miss_oracles import near_miss_oracle_variants, row_lane_iou
from dynlaneseq_eg.modeling.common import fixed_y_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose selected CULane near-miss false positives with geometry/range oracles."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--list-path", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default="outputs/diagnostic_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--stage", default="main")
    parser.add_argument("--score-thresh", type=float, default=0.40)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--iou-thresh", type=float, default=0.50)
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--near-max-iou", type=float, default=0.50)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=None)
    parser.add_argument("--nms-min-overlap-points", type=int, default=None)
    parser.add_argument("--shift-max-px", type=float, default=64.0)
    parser.add_argument("--shift-step-px", type=float, default=1.0)
    parser.add_argument("--row-bounds-px", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0, 32.0])
    parser.add_argument(
        "--oracle-names",
        nargs="*",
        default=[],
        help="Optional subset of oracle variant names; empty runs every variant.",
    )
    parser.add_argument(
        "--exact-raster-oracles",
        action="store_true",
        help="Also rerasterize modified near-miss lanes in the official CULane IoU space.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "max": float(tensor.max()),
    }


def _stage_name(record: dict[str, Any], requested: str) -> str:
    if requested in record["stages"]:
        return requested
    if requested == "main":
        for candidate in ("final", "coarse", "stage2"):
            if candidate in record["stages"]:
                return candidate
    raise KeyError(f"Stage {requested!r} not found in record {record['image_id']}")


def _official_iou_vector(
    pred_x: torch.Tensor,
    pred_mask: torch.Tensor,
    record: dict[str, Any],
    gt_masks: list[Any],
    gt_counts: list[int],
    *,
    width: int,
) -> torch.Tensor:
    if int(pred_mask.sum()) < 2:
        return torch.zeros(len(gt_masks), dtype=torch.float32)
    input_h = int(record["meta"].get("input_h", 288))
    y_rows = fixed_y_rows(pred_x.shape[-1], input_h, device=pred_x.device, dtype=pred_x.dtype)
    lane = [(float(x), float(y)) for x, y in zip(pred_x[pred_mask], y_rows[pred_mask])]
    lane_original = lanes_to_original([lane], record["meta"])[0]
    image_h = int(record["meta"].get("orig_h", 590))
    image_w = int(record["meta"].get("orig_w", 1640))
    pred_raster = _raster_lane_mask(lane_original, image_h, image_w, width)
    pred_count = int(pred_raster.sum())
    values = torch.zeros(len(gt_masks), dtype=torch.float32)
    if pred_count == 0:
        return values
    for gt_id, gt_raster in enumerate(gt_masks):
        intersection = int(cv2.countNonZero(cv2.bitwise_and(pred_raster, gt_raster)))
        union = pred_count + gt_counts[gt_id] - intersection
        values[gt_id] = 0.0 if union <= 0 else float(intersection) / float(union)
    return values


def main() -> None:
    args = parse_args()
    if args.shift_step_px <= 0:
        raise ValueError("--shift-step-px must be positive")
    if args.near_min_iou >= args.near_max_iou:
        raise ValueError("--near-min-iou must be smaller than --near-max-iou")
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    input_h = int(model_cfg.get("input_h", 288))
    input_w = int(model_cfg.get("input_w", 800))
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        list_path=args.list_path or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
        max_batches=args.max_batches,
        desc="near-miss oracle cache",
    )
    # Near-miss membership and deployed TP/FP status must use the same raster
    # IoU space as the official evaluator. Existing caches make this cheap.
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
    )
    post = cache["metadata"].get("postprocess", {})
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
    shifts = torch.arange(
        -float(args.shift_max_px),
        float(args.shift_max_px) + 0.5 * float(args.shift_step_px),
        float(args.shift_step_px),
    )

    counts = defaultdict(int)
    baseline_official_ious: list[float] = []
    baseline_row_ious: list[float] = []
    oracle_ious: dict[str, list[float]] = defaultdict(list)
    oracle_candidate_rescues = defaultdict(int)
    oracle_system_rescues = defaultdict(int)
    oracle_system_kills = defaultdict(int)
    exact_oracle_ious: dict[str, list[float]] = defaultdict(list)
    exact_system_rescues = defaultdict(int)
    exact_system_kills = defaultdict(int)
    shift_values: dict[str, list[float]] = defaultdict(list)

    for record in tqdm(cache["records"], ncols=90, desc="near-miss oracles"):
        stage_name = _stage_name(record, args.stage)
        stage = record["stages"][stage_name]
        official, _valid_gt, _candidate_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
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
            score_thresh=args.score_thresh,
            quality_power=args.quality_power,
            min_valid_rows=args.min_valid_rows,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
            top_k=args.top_k,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        selected = list(trace["selected_ids"])
        official_assignment = evaluator_hungarian_assignment(official, selected, args.iou_thresh)
        official_tp_ids = set(official_assignment.proposal_ids)
        counts["images"] += 1
        counts["selected"] += len(selected)
        counts["selected_tp_official"] += official_assignment.hit_count
        counts["selected_fp_official"] += len(selected) - official_assignment.hit_count
        if official.shape[0] == 0:
            continue

        near_ids: list[int] = []
        near_gt: dict[int, int] = {}
        for proposal_id in selected:
            if proposal_id in official_tp_ids:
                continue
            best_value, best_gt = official[:, proposal_id].max(dim=0)
            value = float(best_value)
            if float(args.near_min_iou) <= value <= float(args.near_max_iou):
                near_ids.append(int(proposal_id))
                near_gt[int(proposal_id)] = int(best_gt)
                baseline_official_ious.append(value)
        counts["near_miss_fp"] += len(near_ids)
        if not near_ids:
            continue

        row_matrix, valid_gt, _row_candidate_valid = proposal_gt_iou_matrix(
            stage,
            record["target"],
            input_h=input_h,
            input_w=input_w,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )
        target_x_all = record["target"]["x_rows"].float()[valid_gt]
        target_mask_all = record["target"]["valid_mask"].bool()[valid_gt]
        if row_matrix.shape[0] != official.shape[0]:
            counts["skipped_gt_alignment_images"] += 1
            counts["skipped_gt_alignment_near_misses"] += len(near_ids)
            continue

        pred_x_all, pred_masks_all, _ = candidate_row_masks(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
        )

        row_baseline_assignment = evaluator_hungarian_assignment(row_matrix, selected, args.iou_thresh)
        joint_matrices: dict[str, torch.Tensor] = {}
        exact_joint_matrices: dict[str, torch.Tensor] = {}
        gt_rasters: list[Any] = []
        gt_counts: list[int] = []
        if args.exact_raster_oracles:
            anno_path = record["meta"].get("anno_path")
            gt_lanes = load_culane_img_data(anno_path) if anno_path else []
            if len(gt_lanes) != official.shape[0]:
                counts["skipped_exact_gt_alignment_images"] += 1
            else:
                image_h = int(record["meta"].get("orig_h", 590))
                image_w = int(record["meta"].get("orig_w", 1640))
                width = int(round(args.line_width))
                gt_rasters = [_raster_lane_mask(lane, image_h, image_w, width) for lane in gt_lanes]
                gt_counts = [int(mask.sum()) for mask in gt_rasters]
        for proposal_id in near_ids:
            gt_id = near_gt[proposal_id]
            pred_x = pred_x_all[proposal_id]
            pred_mask = pred_masks_all[proposal_id]
            gt_x = target_x_all[gt_id]
            gt_mask = target_mask_all[gt_id]
            baseline_row_ious.append(float(row_matrix[gt_id, proposal_id]))
            variants = near_miss_oracle_variants(
                pred_x,
                pred_mask,
                gt_x,
                gt_mask,
                shifts,
                args.row_bounds_px,
                input_w=input_w,
                line_width=args.line_width,
            )
            if args.oracle_names:
                variants = {name: value for name, value in variants.items() if name in set(args.oracle_names)}
                missing = set(args.oracle_names) - set(variants)
                if missing:
                    raise KeyError(f"Unknown oracle variants: {sorted(missing)}")
            for name, variant in variants.items():
                candidate_iou = float(
                    row_lane_iou(
                        variant.pred_x,
                        variant.pred_mask,
                        gt_x,
                        gt_mask,
                        line_width=args.line_width,
                    )
                )
                oracle_ious[name].append(candidate_iou)
                if candidate_iou > float(args.iou_thresh):
                    oracle_candidate_rescues[name] += 1
                if variant.parameter is not None and name in {"constant_shift", "constant_shift_plus_range"}:
                    shift_values[name].append(float(variant.parameter))

                matrix = joint_matrices.setdefault(name, row_matrix.clone())
                all_gt_iou = row_lane_iou(
                    variant.pred_x.unsqueeze(0).expand_as(target_x_all),
                    variant.pred_mask.unsqueeze(0).expand_as(target_mask_all),
                    target_x_all,
                    target_mask_all,
                    line_width=args.line_width,
                )
                matrix[:, proposal_id] = all_gt_iou
                if args.exact_raster_oracles and gt_rasters:
                    exact_vector = _official_iou_vector(
                        variant.pred_x,
                        variant.pred_mask,
                        record,
                        gt_rasters,
                        gt_counts,
                        width=int(round(args.line_width)),
                    )
                    exact_oracle_ious[name].append(float(exact_vector[gt_id]))
                    exact_matrix = exact_joint_matrices.setdefault(name, official.clone())
                    exact_matrix[:, proposal_id] = exact_vector

        for name, matrix in joint_matrices.items():
            modified = evaluator_hungarian_assignment(matrix, selected, args.iou_thresh)
            delta = modified.hit_count - row_baseline_assignment.hit_count
            if delta > 0:
                oracle_system_rescues[name] += delta
            elif delta < 0:
                oracle_system_kills[name] += -delta
        if args.exact_raster_oracles and gt_rasters:
            for name, matrix in exact_joint_matrices.items():
                modified = evaluator_hungarian_assignment(matrix, selected, args.iou_thresh)
                delta = modified.hit_count - official_assignment.hit_count
                if delta > 0:
                    exact_system_rescues[name] += delta
                elif delta < 0:
                    exact_system_kills[name] += -delta

    near_count = max(counts["near_miss_fp"], 1)
    baseline_row_mean = sum(baseline_row_ious) / max(len(baseline_row_ious), 1)
    oracle_rows = {}
    for name in sorted(oracle_ious):
        values = oracle_ious[name]
        mean_iou = sum(values) / max(len(values), 1)
        row = {
            "mean_row_iou": mean_iou,
            "mean_row_iou_gain": mean_iou - baseline_row_mean,
            "candidate_rescues": int(oracle_candidate_rescues[name]),
            "candidate_rescue_rate": oracle_candidate_rescues[name] / near_count,
            "joint_system_tp_gain": int(oracle_system_rescues[name]),
            "joint_system_tp_killed": int(oracle_system_kills[name]),
        }
        if name in shift_values:
            row["chosen_shift_px"] = _summary(shift_values[name])
            row["chosen_abs_shift_px"] = _summary([abs(value) for value in shift_values[name]])
        if args.exact_raster_oracles:
            exact_values = exact_oracle_ious.get(name, [])
            row["exact_official_iou"] = _summary(exact_values)
            row["exact_candidate_rescues"] = sum(
                value > float(args.iou_thresh) for value in exact_values
            )
            row["exact_candidate_rescue_rate"] = (
                row["exact_candidate_rescues"] / max(len(exact_values), 1)
            )
            row["exact_joint_system_tp_gain"] = int(exact_system_rescues[name])
            row["exact_joint_system_tp_killed"] = int(exact_system_kills[name])
        oracle_rows[name] = row

    payload = {
        "metadata": metadata_for_json(
            cache,
            tool="analyze_near_miss_oracles",
            stage=args.stage,
            score_threshold=args.score_thresh,
            quality_power=args.quality_power,
            top_k=args.top_k,
            iou_threshold=args.iou_thresh,
            near_miss_interval=[args.near_min_iou, args.near_max_iou],
            selection_iou_space="official_raster",
            oracle_iou_space="analytic_row_space",
            exact_raster_oracles=bool(args.exact_raster_oracles),
            oracle_names=args.oracle_names,
            line_width=args.line_width,
            shifts_px=[float(shifts.min()), float(shifts.max()), float(args.shift_step_px)],
            row_bounds_px=args.row_bounds_px,
            nms_distance_thresh_px=nms_distance,
            nms_min_overlap_points=nms_overlap,
        ),
        "counts": dict(counts),
        "baseline": {
            "official_near_miss_iou": _summary(baseline_official_ious),
            "row_space_near_miss_iou": _summary(baseline_row_ious),
        },
        "oracles": oracle_rows,
    }
    print("counts:", dict(counts))
    print("baseline row-space near-miss IoU:", payload["baseline"]["row_space_near_miss_iou"])
    for name, row in oracle_rows.items():
        print(
            f"{name:>30}: meanIoU={row['mean_row_iou']:.4f} "
            f"candidate_rescue={row['candidate_rescues']}/{counts['near_miss_fp']} "
            f"joint_TP_gain={row['joint_system_tp_gain']} killed={row['joint_system_tp_killed']}"
        )
        if args.exact_raster_oracles:
            exact = row["exact_official_iou"]
            print(
                f"{'':>30}  exactMeanIoU={exact['mean']:.4f} "
                f"exactCandidateRescue={row['exact_candidate_rescues']}/{exact['count']} "
                f"exactJointTPGain={row['exact_joint_system_tp_gain']} "
                f"killed={row['exact_joint_system_tp_killed']}"
            )
    write_json(args.output_json or None, payload)
    if args.output_json:
        print(f"output_json: {args.output_json}")


if __name__ == "__main__":
    main()
