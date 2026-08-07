from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    metadata_for_json,
    stage_scores,
    trace_postprocess,
    unique_candidate_labels,
    write_json,
)
from dynlaneseq_eg.evaluation.four_slot_decode import decode_four_slot_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate V4 candidate geometry capacity from duplicate-safe Top-K "
            "coverage using the same frozen predictions and official CULane IoU."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs/diagnostic_cache/v4_selection_coverage",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=8)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument("--stage", default="main")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75]
    )
    parser.add_argument("--near-min-iou", type=float, default=0.30)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument(
        "--hard-diversity-distances",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0, 40.0, 60.0],
    )
    parser.add_argument(
        "--mmr-penalties",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75],
    )
    parser.add_argument(
        "--mmr-sigmas",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 30.0],
    )
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


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _valid_ids(mask: torch.Tensor) -> list[int]:
    return torch.nonzero(mask.bool(), as_tuple=False).flatten().tolist()


def _curve_distance_matrix(
    stage: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    min_valid_rows: int,
    row_visibility_thresh: float,
    min_overlap_points: int,
) -> torch.Tensor:
    """Mean row distance in the same input-pixel frame used by lane NMS."""

    pred_x, masks, _candidate_valid = candidate_row_masks(
        stage,
        input_h=input_h,
        input_w=input_w,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=row_visibility_thresh,
    )
    overlap = masks.unsqueeze(1) & masks.unsqueeze(0)
    overlap_count = overlap.sum(dim=-1)
    difference = (pred_x.unsqueeze(1) - pred_x.unsqueeze(0)).abs()
    distance = (difference * overlap.to(difference.dtype)).sum(dim=-1)
    distance = distance / overlap_count.clamp_min(1).to(distance.dtype)
    distance = torch.where(
        overlap_count >= int(min_overlap_points),
        distance,
        torch.full_like(distance, float("inf")),
    )
    return distance.cpu()


def _mmr_ids(
    scores: torch.Tensor,
    distance: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    penalty: float,
    sigma: float,
    top_k: int,
) -> list[int]:
    """Greedy scalar ranking with an explicit curve-redundancy penalty."""

    if float(sigma) <= 0.0:
        raise ValueError("MMR sigma must be positive")
    available = candidate_valid.bool().clone()
    selected: list[int] = []
    similarity = torch.exp(-distance.float() / float(sigma))
    for _ in range(min(int(top_k), int(available.sum()))):
        utility = scores.float().clone()
        if selected:
            redundancy = similarity[:, selected].amax(dim=-1)
            utility = utility - float(penalty) * redundancy
        utility = utility.masked_fill(~available, float("-inf"))
        index = int(utility.argmax())
        selected.append(index)
        available[index] = False
    return selected


def _new_counter() -> dict[str, Any]:
    return {
        "images": 0,
        "gt": 0,
        "selected": 0,
        "hits": 0,
        "false_positive_labels": Counter(),
        "selected_pairs": 0,
        "close_pairs_20px": 0,
        "finite_pair_distance_sum": 0.0,
        "finite_pair_distances": 0,
    }


def _update_counter(
    counter: dict[str, Any],
    iou: torch.Tensor,
    selected_ids: Iterable[int],
    distance: torch.Tensor,
    *,
    threshold: float,
    near_min_iou: float,
) -> None:
    selected = [int(index) for index in selected_ids]
    assignment = evaluator_hungarian_assignment(iou, selected, threshold)
    assigned = set(int(index) for index in assignment.proposal_ids)
    gt_count = int(iou.shape[0])
    best_iou = (
        iou.max(dim=0).values
        if gt_count > 0
        else torch.zeros(int(iou.shape[1]), dtype=torch.float32)
    )
    counter["images"] += 1
    counter["gt"] += gt_count
    counter["selected"] += len(selected)
    counter["hits"] += int(assignment.hit_count)
    for proposal_index in selected:
        if proposal_index in assigned:
            continue
        if gt_count == 0:
            label = "empty_scene_fp"
        else:
            value = float(best_iou[proposal_index])
            if value > float(threshold):
                label = "duplicate_fp"
            elif value >= float(near_min_iou):
                label = "near_miss_fp"
            else:
                label = "background_fp"
        counter["false_positive_labels"][label] += 1
    for left_position, left in enumerate(selected):
        for right in selected[left_position + 1 :]:
            value = float(distance[left, right])
            counter["selected_pairs"] += 1
            if math.isfinite(value):
                counter["finite_pair_distance_sum"] += value
                counter["finite_pair_distances"] += 1
                if value < 20.0:
                    counter["close_pairs_20px"] += 1


def _finish_counter(counter: dict[str, Any]) -> dict[str, Any]:
    tp = int(counter["hits"])
    selected = int(counter["selected"])
    gt = int(counter["gt"])
    fp = max(selected - tp, 0)
    fn = max(gt - tp, 0)
    precision = float(tp) / max(tp + fp, 1)
    recall = float(tp) / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    labels = counter["false_positive_labels"]
    return {
        "images": int(counter["images"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_selected_per_image": float(selected)
        / max(int(counter["images"]), 1),
        "false_positive_breakdown": {
            name: {
                "count": int(labels.get(name, 0)),
                "fraction_of_fp": float(labels.get(name, 0)) / max(fp, 1),
            }
            for name in (
                "duplicate_fp",
                "near_miss_fp",
                "background_fp",
                "empty_scene_fp",
            )
        },
        "selected_curve_diversity": {
            "pair_count": int(counter["selected_pairs"]),
            "close_pair_count_below_20px": int(counter["close_pairs_20px"]),
            "close_pair_fraction_below_20px": float(
                counter["close_pairs_20px"]
            )
            / max(int(counter["selected_pairs"]), 1),
            "mean_finite_pair_distance_px": float(
                counter["finite_pair_distance_sum"]
            )
            / max(int(counter["finite_pair_distances"]), 1),
        },
    }


def _new_oracle_counter() -> dict[str, int]:
    return {"gt": 0, "hits": 0}


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(scores) != len(labels) or sum(labels) == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 0:
            continue
        positives += 1
        precision_sum += float(positives) / float(rank)
    return precision_sum / float(sum(labels))


def _update_oracle(
    counter: dict[str, int],
    iou: torch.Tensor,
    candidate_ids: Iterable[int],
    *,
    threshold: float,
    top_k: int,
) -> None:
    allowed = torch.zeros(int(iou.shape[1]), dtype=torch.bool)
    for index in candidate_ids:
        allowed[int(index)] = True
    result = cardinality_oracle_assignment(
        iou,
        threshold,
        int(top_k),
        candidate_valid=allowed,
    )
    counter["gt"] += int(iou.shape[0])
    counter["hits"] += int(result.hit_count)


def _finish_oracle(counter: dict[str, int]) -> dict[str, float | int]:
    return {
        "gt": int(counter["gt"]),
        "hits": int(counter["hits"]),
        "recall": float(counter["hits"]) / max(int(counter["gt"]), 1),
    }


def _build_verdict(
    methods: dict[str, dict[str, Any]],
    capacity: dict[str, dict[str, Any]],
    *,
    primary_threshold: float,
) -> dict[str, Any]:
    threshold_key = f"{float(primary_threshold):.2f}"
    raw_recall = float(methods["score_top4"][threshold_key]["recall"])
    oracle_recall = float(capacity[threshold_key]["all_candidate_oracle"]["recall"])
    diversity_names = [name for name in methods if name != "score_top4"]
    best_name = max(
        diversity_names,
        key=lambda name: (
            float(methods[name][threshold_key]["recall"]),
            float(methods[name][threshold_key]["precision"]),
        ),
    )
    best_recall = float(methods[best_name][threshold_key]["recall"])
    gap = max(oracle_recall - raw_recall, 0.0)
    gain = best_recall - raw_recall
    recovery = gain / gap if gap > 1e-12 else 0.0
    raw_fp = methods["score_top4"][threshold_key]["false_positive_breakdown"]
    duplicate_fraction = float(raw_fp["duplicate_fp"]["fraction_of_fp"])
    if gap < 0.05:
        interpretation = "no_material_top4_selection_gap"
    elif gain >= 0.10 and recovery >= 0.40:
        interpretation = "duplicate_coverage_is_a_primary_selection_bottleneck"
    elif gain >= 0.03:
        interpretation = "mixed_duplicate_coverage_and_scalar_ranking_bottleneck"
    else:
        interpretation = "scalar_score_quality_dominates_over_simple_diversity"
    return {
        "primary_iou_threshold": float(primary_threshold),
        "score_top4_recall": raw_recall,
        "all_candidate_oracle_top4_recall": oracle_recall,
        "selection_gap_points": 100.0 * gap,
        "best_diversity_method": best_name,
        "best_diversity_recall": best_recall,
        "best_diversity_gain_points": 100.0 * gain,
        "fraction_of_oracle_gap_recovered": recovery,
        "score_top4_duplicate_fraction_of_fp": duplicate_fraction,
        "interpretation": interpretation,
        "warning": (
            "The best diversity setting is selected on this diagnostic sample "
            "and is not a validation-selected deployment result. Inspect the "
            "fixed 20px hard-diversity row and the complete grid before drawing "
            "an architectural conclusion."
        ),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")
    thresholds = tuple(float(value) for value in args.iou_thresholds)
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    if not 0.0 <= float(args.near_min_iou) < min(thresholds):
        raise ValueError("near-min-iou must be below every IoU threshold")
    if any(float(value) <= 0.0 for value in args.hard_diversity_distances):
        raise ValueError("hard diversity distances must be positive")
    if any(float(value) < 0.0 for value in args.mmr_penalties):
        raise ValueError("MMR penalties must be non-negative")
    if any(float(value) <= 0.0 for value in args.mmr_sigmas):
        raise ValueError("MMR sigmas must be positive")

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
        desc="V4 selection-coverage cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        min_valid_rows=args.min_valid_rows,
        row_visibility_thresh=args.row_visibility_thresh,
        workers=args.metric_workers,
    )
    metadata = cache["metadata"]
    input_h = int(metadata["input_h"])
    input_w = int(metadata["input_w"])
    score_mode = str(
        metadata.get("postprocess", {}).get("score_mode", "exist")
    )

    method_names = ["score_top4"]
    pointer_mode = score_mode in {
        "pointer",
        "sequential_pointer",
        "pointer_stop",
    }
    slot_mode = score_mode in {"four_slot", "four_slots", "slot", "slots"}
    if pointer_mode:
        method_names.append("pointer_greedy")
    if slot_mode:
        method_names.append("four_slot_global_unique")
    method_names.extend(
        f"hard_diverse_{_float_tag(distance)}px"
        for distance in args.hard_diversity_distances
    )
    method_names.extend(
        f"mmr_sigma{_float_tag(sigma)}_penalty{_float_tag(penalty)}"
        for sigma in args.mmr_sigmas
        for penalty in args.mmr_penalties
    )
    counters = {
        name: {threshold: _new_counter() for threshold in thresholds}
        for name in method_names
    }
    all_candidate_oracle = {
        threshold: _new_oracle_counter() for threshold in thresholds
    }
    hard_pool_oracle = {
        float(distance): {
            threshold: _new_oracle_counter() for threshold in thresholds
        }
        for distance in args.hard_diversity_distances
    }
    candidate_pool = {
        threshold: {
            label: {"count": 0, "score_sum": 0.0}
            for label in ("unique_tp", "duplicate", "background", "invalid_range")
        }
        for threshold in thresholds
    }
    unique_scores: dict[float, list[float]] = {
        threshold: [] for threshold in thresholds
    }
    unique_labels_by_threshold: dict[float, list[int]] = {
        threshold: [] for threshold in thresholds
    }
    probability_mass: list[float] = []
    target_lane_counts: list[float] = []
    slot_raw_collisions = 0
    slot_global_repairs = 0
    slot_dustbins = 0
    slot_total = 0
    slot_semantic_duplicates = 0
    slot_active = 0
    slot_route_entropies: list[float] = []

    iterator = tqdm(
        cache["records"],
        ncols=100,
        desc="V4 score/diversity/oracle comparison",
    )
    for record in iterator:
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
        if pointer_mode:
            unary_logits = stage.get("selection_logits")
            if not isinstance(unary_logits, torch.Tensor):
                raise ValueError("pointer diagnostics require selection_logits")
            scores = torch.sigmoid(unary_logits.float()).cpu()
        elif slot_mode:
            # Retain the frozen V5 direct ownership score as a proposal-level
            # control.  Slot selection itself is added separately below.
            scores = stage_scores(
                stage,
                quality_power=0.0,
                score_mode="exist",
            ).cpu()
        else:
            scores = stage_scores(
                stage,
                quality_power=0.0,
                score_mode=score_mode,
            ).cpu()
        valid_ids = _valid_ids(candidate_valid)
        probability_mass.append(float(scores[candidate_valid].sum()))
        target_lane_counts.append(float(iou.shape[0]))
        raw_ids = sorted(
            valid_ids,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[: int(args.top_k)]
        distance = _curve_distance_matrix(
            stage,
            input_h=input_h,
            input_w=input_w,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=args.row_visibility_thresh,
            min_overlap_points=args.nms_min_overlap_points,
        )
        selections: dict[str, list[int]] = {"score_top4": raw_ids}
        if pointer_mode:
            pointer_indices = stage.get("selection_pointer_indices")
            if not isinstance(pointer_indices, torch.Tensor):
                raise ValueError(
                    "pointer diagnostics require selection_pointer_indices"
                )
            pointer_ids: list[int] = []
            valid_id_set = set(valid_ids)
            for value in pointer_indices.tolist():
                candidate = int(value)
                if candidate < 0:
                    break
                if candidate in valid_id_set and candidate not in pointer_ids:
                    pointer_ids.append(candidate)
                if len(pointer_ids) >= int(args.top_k):
                    break
            selections["pointer_greedy"] = pointer_ids
        if slot_mode:
            slot_logits = stage.get("selection_slot_logits")
            slot_valid = stage.get("selection_slot_candidate_valid")
            if not isinstance(slot_logits, torch.Tensor) or not isinstance(
                slot_valid, torch.Tensor
            ):
                raise ValueError(
                    "four-slot diagnostics require slot logits and valid mask"
                )
            decoded = decode_four_slot_logits(
                slot_logits,
                slot_valid.bool() & candidate_valid.bool(),
            )
            slot_ids = [
                int(value)
                for value in decoded["indices"].tolist()
                if int(value) >= 0
            ]
            selections["four_slot_global_unique"] = slot_ids
            slot_total += int(slot_logits.shape[0])
            slot_active += len(slot_ids)
            slot_dustbins += int(slot_logits.shape[0]) - len(slot_ids)
            slot_raw_collisions += int(decoded["raw_collision_count"])
            slot_global_repairs += int(decoded["repair_count"])
            entropy = stage.get("selection_slot_route_entropy")
            if isinstance(entropy, torch.Tensor):
                slot_route_entropies.extend(float(value) for value in entropy)
            if int(iou.shape[0]) > 0 and slot_ids:
                selected_quality = iou[:, slot_ids]
                best_quality, best_gt = selected_quality.max(dim=0)
                meaningful = best_quality >= float(args.near_min_iou)
                cluster_ids = best_gt[meaningful]
                slot_semantic_duplicates += int(cluster_ids.numel()) - int(
                    cluster_ids.unique().numel()
                )
        hard_pools: dict[float, list[int]] = {}
        for hard_distance in args.hard_diversity_distances:
            hard_distance = float(hard_distance)
            trace = trace_postprocess(
                stage,
                input_h=input_h,
                input_w=input_w,
                score_thresh=0.0,
                quality_power=0.0,
                min_valid_rows=args.min_valid_rows,
                nms_distance_thresh_px=hard_distance,
                nms_min_overlap_points=args.nms_min_overlap_points,
                top_k=args.top_k,
                row_visibility_thresh=args.row_visibility_thresh,
                allowed_ids=valid_ids,
                score_mode=(
                    "selection"
                    if pointer_mode
                    else ("exist" if slot_mode else score_mode)
                ),
            )
            name = f"hard_diverse_{_float_tag(hard_distance)}px"
            selections[name] = [int(index) for index in trace["selected_ids"]]
            hard_pools[hard_distance] = [
                int(index) for index in trace["nms_kept_ids"]
            ]
        for sigma in args.mmr_sigmas:
            for penalty in args.mmr_penalties:
                name = (
                    f"mmr_sigma{_float_tag(sigma)}_"
                    f"penalty{_float_tag(penalty)}"
                )
                selections[name] = _mmr_ids(
                    scores,
                    distance,
                    candidate_valid,
                    penalty=float(penalty),
                    sigma=float(sigma),
                    top_k=args.top_k,
                )

        for threshold in thresholds:
            for name, selected_ids in selections.items():
                _update_counter(
                    counters[name][threshold],
                    iou,
                    selected_ids,
                    distance,
                    threshold=threshold,
                    near_min_iou=args.near_min_iou,
                )
            _update_oracle(
                all_candidate_oracle[threshold],
                iou,
                valid_ids,
                threshold=threshold,
                top_k=args.top_k,
            )
            for hard_distance, pool_ids in hard_pools.items():
                _update_oracle(
                    hard_pool_oracle[hard_distance][threshold],
                    iou,
                    pool_ids,
                    threshold=threshold,
                    top_k=args.top_k,
                )
            labels, _assignment = unique_candidate_labels(
                iou,
                threshold,
                candidate_valid,
            )
            unique_scores[threshold].extend(
                float(scores[index]) for index in valid_ids
            )
            unique_labels_by_threshold[threshold].extend(
                int(labels[index] == "unique_tp") for index in valid_ids
            )
            for proposal_index, label in enumerate(labels):
                row = candidate_pool[threshold][label]
                row["count"] += 1
                row["score_sum"] += float(scores[proposal_index])

    method_summary = {
        name: {
            f"{threshold:.2f}": _finish_counter(counter)
            for threshold, counter in by_threshold.items()
        }
        for name, by_threshold in counters.items()
    }
    capacity_summary = {
        f"{threshold:.2f}": {
            "all_candidate_oracle": _finish_oracle(
                all_candidate_oracle[threshold]
            ),
            "hard_diversity_pool_oracle": {
                f"{distance:g}px": _finish_oracle(
                    hard_pool_oracle[distance][threshold]
                )
                for distance in sorted(hard_pool_oracle)
            },
        }
        for threshold in thresholds
    }
    candidate_pool_summary = {
        f"{threshold:.2f}": {
            label: {
                "count": int(row["count"]),
                "mean_score": float(row["score_sum"]) / max(int(row["count"]), 1),
            }
            for label, row in by_label.items()
        }
        for threshold, by_label in candidate_pool.items()
    }
    score_diagnostics = {
        "score_mode": score_mode,
        "mean_foreground_probability_mass": (
            sum(probability_mass) / max(len(probability_mass), 1)
        ),
        "mean_target_lane_count": (
            sum(target_lane_counts) / max(len(target_lane_counts), 1)
        ),
        "unique_candidate_ap": {
            f"{threshold:.2f}": _average_precision(
                unique_scores[threshold],
                unique_labels_by_threshold[threshold],
            )
            for threshold in thresholds
        },
    }
    slot_diagnostics = None
    if slot_mode:
        slot_diagnostics = {
            "total_slots": int(slot_total),
            "active_slots": int(slot_active),
            "dustbin_slots": int(slot_dustbins),
            "dustbin_fraction": float(slot_dustbins) / max(slot_total, 1),
            "raw_argmax_collision_count": int(slot_raw_collisions),
            "raw_argmax_collision_per_image": float(slot_raw_collisions)
            / max(len(cache["records"]), 1),
            "global_assignment_repair_count": int(slot_global_repairs),
            "global_assignment_repair_fraction": float(slot_global_repairs)
            / max(slot_total, 1),
            "semantic_duplicate_cluster_count": int(
                slot_semantic_duplicates
            ),
            "semantic_duplicate_cluster_fraction": float(
                slot_semantic_duplicates
            )
            / max(slot_active, 1),
            "mean_route_entropy": (
                sum(slot_route_entropies) / max(len(slot_route_entropies), 1)
            ),
        }
    verdict = _build_verdict(
        method_summary,
        capacity_summary,
        primary_threshold=thresholds[0],
    )
    payload = {
        "diagnostic_only": True,
        "warning": (
            "Frozen-prediction selection diagnosis only. Oracle selections and "
            "the best diversity grid row are not deployable benchmark results."
        ),
        "selection_rule": (
            "Identical frozen candidates: unary scalar Top-4, optional learned "
            "sequential pointer with explicit STOP, score-ordered hard curve "
            "diversity, MMR curve diversity, and maximum-cardinality official-"
            "IoU Oracle Top-4."
        ),
        "metadata": metadata_for_json(
            cache,
            tool="analyze_v4_selection_coverage",
            iou_space="official_raster",
            top_k=args.top_k,
            iou_thresholds=list(thresholds),
            hard_diversity_distances=list(args.hard_diversity_distances),
            mmr_sigmas=list(args.mmr_sigmas),
            mmr_penalties=list(args.mmr_penalties),
            nms_min_overlap_points=args.nms_min_overlap_points,
        ),
        "methods": method_summary,
        "capacity": capacity_summary,
        "candidate_pool_score_by_official_status": candidate_pool_summary,
        "score_diagnostics": score_diagnostics,
        "four_slot_diagnostics": slot_diagnostics,
        "verdict": verdict,
    }
    write_json(args.output_json, payload)
    compact = {
        "score_top4": method_summary["score_top4"],
        "four_slot_global_unique": method_summary.get(
            "four_slot_global_unique"
        ),
        "hard_diverse_20px": method_summary.get("hard_diverse_20px"),
        "capacity": capacity_summary,
        "candidate_pool": candidate_pool_summary,
        "score_diagnostics": score_diagnostics,
        "four_slot_diagnostics": slot_diagnostics,
        "verdict": verdict,
    }
    print(json.dumps(compact, indent=2))
    print(f"output_json: {Path(args.output_json)}")


if __name__ == "__main__":
    main()
