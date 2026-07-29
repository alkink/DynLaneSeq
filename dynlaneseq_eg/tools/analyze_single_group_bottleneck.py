from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    official_proposal_gt_iou_matrix,
    stage_scores,
)
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate score/ranking, visible-range, and horizontal-geometry "
            "bottlenecks inside one structured-query group."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=("sequential", "uniform"), default="uniform")
    parser.add_argument("--stage-name", default="main")
    parser.add_argument("--num-query-groups", type=int, default=4)
    parser.add_argument("--query-group-index", type=int, default=0)
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--quality-power", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _slice_group(
    stage: dict[str, torch.Tensor],
    *,
    num_query_groups: int,
    query_group_index: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    pred_x = stage.get("pred_x_rows")
    if not isinstance(pred_x, torch.Tensor) or pred_x.ndim != 2:
        raise ValueError("Expected per-image pred_x_rows with shape [queries, rows]")
    num_candidates = int(pred_x.shape[0])
    groups = int(num_query_groups)
    group_index = int(query_group_index)
    if groups < 1 or num_candidates % groups != 0:
        raise ValueError(
            f"num_candidates={num_candidates} is not divisible by num_query_groups={groups}"
        )
    if not 0 <= group_index < groups:
        raise ValueError(f"query_group_index={group_index} is outside [0, {groups - 1}]")
    group_size = num_candidates // groups
    start = group_index * group_size
    ids = list(range(start, start + group_size))
    sliced = {
        key: value[ids].clone()
        for key, value in stage.items()
        if isinstance(value, torch.Tensor)
        and value.ndim >= 1
        and int(value.shape[0]) == num_candidates
    }
    return sliced, ids


def _pairwise_row_iou(
    pred_x: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_x: torch.Tensor,
    gt_masks: torch.Tensor,
    *,
    line_width: float,
) -> torch.Tensor:
    if gt_x.numel() == 0 or pred_x.numel() == 0:
        return pred_x.new_zeros((gt_x.shape[0], pred_x.shape[0]))
    gt = gt_x[:, None, :]
    pred = pred_x[None, :, :]
    gt_valid = gt_masks[:, None, :]
    pred_valid = pred_masks[None, :, :]
    both = gt_valid & pred_valid
    either = gt_valid | pred_valid
    overlap = (float(line_width) - (gt - pred).abs()).clamp(min=0.0)
    overlap = torch.where(both, overlap, torch.zeros_like(overlap))
    union = torch.where(
        both,
        2.0 * float(line_width) - overlap,
        torch.where(
            either,
            torch.full_like(overlap, float(line_width)),
            torch.zeros_like(overlap),
        ),
    )
    return overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)


def _counterfactual_matrices(
    stage: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    line_width: float,
    min_valid_rows: int,
) -> dict[str, torch.Tensor]:
    pred_x = stage["pred_x_rows"].float().clamp(0.0, float(input_w - 1))
    num_candidates, num_rows = pred_x.shape
    y_rows = fixed_y_rows(num_rows, input_h, device=pred_x.device, dtype=pred_x.dtype)
    ranges = sort_range_norm(stage["range_norm"].float())
    pred_masks = (
        (y_rows.view(1, -1) >= ranges[:, 0:1] * float(input_h))
        & (y_rows.view(1, -1) <= ranges[:, 1:2] * float(input_h))
        & torch.isfinite(pred_x)
    )
    candidate_valid = pred_masks.sum(dim=-1) >= int(min_valid_rows)

    gt_x_all = target["x_rows"].float()
    gt_masks_all = target["valid_mask"].bool() & torch.isfinite(gt_x_all)
    valid_gt = gt_masks_all.sum(dim=-1) >= int(min_valid_rows)
    gt_x = gt_x_all[valid_gt]
    gt_masks = gt_masks_all[valid_gt]

    predicted = _pairwise_row_iou(
        pred_x,
        pred_masks,
        gt_x,
        gt_masks,
        line_width=line_width,
    )
    if predicted.numel() > 0:
        predicted[:, ~candidate_valid] = 0.0

    # Replace only the candidate's visible range with the paired GT range.
    oracle_gt_range_masks = gt_masks[:, None, :].expand(
        gt_x.shape[0], num_candidates, num_rows
    )
    if gt_x.numel() == 0:
        oracle_gt_range = predicted.clone()
    else:
        diff = (gt_x[:, None, :] - pred_x[None, :, :]).abs()
        overlap = (float(line_width) - diff).clamp(min=0.0)
        overlap = torch.where(
            oracle_gt_range_masks,
            overlap,
            torch.zeros_like(overlap),
        )
        union = torch.where(
            oracle_gt_range_masks,
            2.0 * float(line_width) - overlap,
            torch.zeros_like(overlap),
        )
        oracle_gt_range = overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(1e-6)

    # Replace only horizontal coordinates with the paired GT x values while
    # retaining each candidate's predicted visible range.
    if gt_x.numel() == 0:
        oracle_x_pred_range = predicted.clone()
    else:
        gt_valid = gt_masks[:, None, :]
        pred_valid = pred_masks[None, :, :]
        both = gt_valid & pred_valid
        either = gt_valid | pred_valid
        overlap = torch.where(
            both,
            torch.full(
                (gt_x.shape[0], num_candidates, num_rows),
                float(line_width),
                dtype=pred_x.dtype,
                device=pred_x.device,
            ),
            torch.zeros(
                (gt_x.shape[0], num_candidates, num_rows),
                dtype=pred_x.dtype,
                device=pred_x.device,
            ),
        )
        union = torch.where(
            either,
            torch.full_like(overlap, float(line_width)),
            torch.zeros_like(overlap),
        )
        oracle_x_pred_range = overlap.sum(dim=-1) / union.sum(dim=-1).clamp_min(
            1e-6
        )
        oracle_x_pred_range[:, ~candidate_valid] = 0.0

    oracle_x_gt_range = pred_x.new_ones((gt_x.shape[0], num_candidates))
    if gt_x.shape[0] == 0:
        oracle_x_gt_range = pred_x.new_zeros((0, num_candidates))

    if gt_x.shape[0] == 0:
        best_x_mae = pred_x.new_zeros((0,))
    else:
        errors = (gt_x[:, None, :] - pred_x[None, :, :]).abs()
        errors = torch.where(
            gt_masks[:, None, :],
            errors,
            torch.zeros_like(errors),
        )
        per_candidate_mae = errors.sum(dim=-1) / gt_masks.sum(dim=-1).clamp_min(
            1
        )[:, None]
        best_x_mae = per_candidate_mae.min(dim=1).values

    return {
        "predicted": predicted,
        "oracle_gt_range": oracle_gt_range,
        "oracle_x_pred_range": oracle_x_pred_range,
        "oracle_x_gt_range": oracle_x_gt_range,
        "candidate_valid": candidate_valid,
        "valid_gt": valid_gt,
        "best_x_mae_px": best_x_mae,
    }


def _rank_model_ids(
    stage: dict[str, torch.Tensor],
    candidate_valid: torch.Tensor,
    *,
    score_thresh: float,
    quality_power: float,
    top_k: int,
) -> tuple[list[int], list[int]]:
    scores = stage_scores(stage, quality_power=quality_power)
    eligible = [
        index
        for index in range(int(scores.shape[0]))
        if bool(candidate_valid[index]) and float(scores[index]) >= float(score_thresh)
    ]
    eligible.sort(key=lambda index: float(scores[index]), reverse=True)
    selected = eligible[: int(top_k)] if int(top_k) > 0 else eligible
    return selected, eligible


def _empty_counter() -> dict[str, float]:
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "gt": 0.0,
        "predictions": 0.0,
        "best_iou_sum": 0.0,
    }


def _update_counter(
    counter: dict[str, float],
    matrix: torch.Tensor,
    proposal_ids: Iterable[int],
    *,
    threshold: float,
) -> set[int]:
    ids = [int(index) for index in proposal_ids]
    assignment = evaluator_hungarian_assignment(matrix, ids, threshold=threshold)
    gt_count = int(matrix.shape[0])
    hits = int(assignment.hit_count)
    if gt_count == 0:
        best = matrix.new_zeros((0,))
    elif ids:
        best = matrix[:, ids].max(dim=1).values
    else:
        best = matrix.new_zeros((gt_count,))
    counter["tp"] += hits
    counter["fp"] += max(0, len(ids) - hits)
    counter["fn"] += max(0, gt_count - hits)
    counter["gt"] += gt_count
    counter["predictions"] += len(ids)
    counter["best_iou_sum"] += float(best.sum())
    return {int(gt_index) for gt_index, _proposal_index in assignment.pairs}


def _finish_counter(counter: dict[str, float]) -> dict[str, float | int]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    gt = int(counter["gt"])
    predictions = int(counter["predictions"])
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "gt_lanes": gt,
        "predictions": predictions,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_best_iou": float(counter["best_iou_sum"]) / float(max(gt, 1)),
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "fraction_le_2px": 0.0,
            "fraction_le_4px": 0.0,
            "fraction_le_8px": 0.0,
            "fraction_le_16px": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": float(sum(ordered)) / float(len(ordered)),
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "fraction_le_2px": sum(value <= 2.0 for value in ordered) / len(ordered),
        "fraction_le_4px": sum(value <= 4.0 for value in ordered) / len(ordered),
        "fraction_le_8px": sum(value <= 8.0 for value in ordered) / len(ordered),
        "fraction_le_16px": sum(value <= 16.0 for value in ordered) / len(ordered),
    }


def _strict_hit_set(
    matrix: torch.Tensor,
    *,
    threshold: float,
) -> set[int]:
    if matrix.shape[1] == 0:
        return set()
    best = matrix.max(dim=1).values
    return {
        index
        for index, value in enumerate(best.tolist())
        if float(value) > float(threshold)
    }


def _derive_recall_gains(
    official: dict[str, Any],
    row_space: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for threshold, official_modes in official.items():
        row_modes = row_space[threshold]
        official_model = float(official_modes["model_top4"]["recall"])
        row_oracle_pred = float(
            row_modes["oracle_top4__predicted"]["recall"]
        )
        output[threshold] = {
            "official_score_threshold_all_vs_model_top4_points": 100.0
            * (
                float(official_modes["score_threshold_all"]["recall"])
                - official_model
            ),
            "official_all8_geometry_vs_model_top4_points": 100.0
            * (
                float(official_modes["all_group_candidates"]["recall"])
                - official_model
            ),
            "official_oracle_top4_vs_model_top4_points": 100.0
            * (
                float(official_modes["oracle_top4_predicted"]["recall"])
                - official_model
            ),
            "row_oracle_gt_range_vs_predicted_points": 100.0
            * (
                float(row_modes["oracle_top4__oracle_gt_range"]["recall"])
                - row_oracle_pred
            ),
            "row_oracle_x_vs_predicted_points": 100.0
            * (
                float(
                    row_modes["oracle_top4__oracle_x_pred_range"]["recall"]
                )
                - row_oracle_pred
            ),
            "row_joint_x_range_vs_predicted_points": 100.0
            * (
                float(row_modes["oracle_top4__oracle_x_gt_range"]["recall"])
                - row_oracle_pred
            ),
        }
    return output


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    cache = load_or_collect_cache(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root or None,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=bool(args.reuse_cache),
        max_batches=int(args.max_batches),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy=str(args.sample_strategy),
        desc="single-group bottleneck cache",
    )
    records = cache.get("records", [])
    if not records:
        raise ValueError("Diagnostic cache contains no records")
    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])
    thresholds = [float(value) for value in args.iou_thresholds]

    official_counters = {
        threshold: defaultdict(_empty_counter) for threshold in thresholds
    }
    row_counters = {
        threshold: defaultdict(_empty_counter) for threshold in thresholds
    }
    causal_buckets = {
        threshold: defaultdict(int) for threshold in thresholds
    }
    x_mae_values: list[float] = []
    group_global_ids: list[int] | None = None
    row_gt_total = 0
    official_gt_total = 0

    for record in tqdm(records, ncols=80, desc="single-group causal audit"):
        stage = record.get("stages", {}).get(args.stage_name)
        if stage is None:
            raise KeyError(f"Stage {args.stage_name!r} is missing from a cache record")
        group_stage, current_global_ids = _slice_group(
            stage,
            num_query_groups=args.num_query_groups,
            query_group_index=args.query_group_index,
        )
        if group_global_ids is None:
            group_global_ids = current_global_ids
        elif group_global_ids != current_global_ids:
            raise ValueError("Query-group indices changed across records")

        matrices = _counterfactual_matrices(
            group_stage,
            record["target"],
            input_h=input_h,
            input_w=input_w,
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
        )
        model_ids, eligible_ids = _rank_model_ids(
            group_stage,
            matrices["candidate_valid"],
            score_thresh=args.score_thresh,
            quality_power=args.quality_power,
            top_k=args.top_k,
        )
        all_valid_ids = torch.nonzero(
            matrices["candidate_valid"], as_tuple=False
        ).flatten().tolist()
        x_mae_values.extend(float(value) for value in matrices["best_x_mae_px"])
        row_gt_total += int(matrices["predicted"].shape[0])

        temporary_record = {
            **record,
            "stages": {"single_group": group_stage},
        }
        official_matrix, official_valid = official_proposal_gt_iou_matrix(
            temporary_record,
            "single_group",
            line_width=args.line_width,
            min_valid_rows=args.min_valid_rows,
            row_visibility_thresh=0.0,
        )
        official_gt_total += int(official_matrix.shape[0])
        official_all_ids = torch.nonzero(
            official_valid, as_tuple=False
        ).flatten().tolist()

        for threshold in thresholds:
            _update_counter(
                official_counters[threshold]["model_top4"],
                official_matrix,
                model_ids,
                threshold=threshold,
            )
            _update_counter(
                official_counters[threshold]["score_threshold_all"],
                official_matrix,
                eligible_ids,
                threshold=threshold,
            )
            _update_counter(
                official_counters[threshold]["all_group_candidates"],
                official_matrix,
                official_all_ids,
                threshold=threshold,
            )
            official_oracle = cardinality_oracle_assignment(
                official_matrix,
                threshold,
                args.top_k,
                official_valid,
            )
            _update_counter(
                official_counters[threshold]["oracle_top4_predicted"],
                official_matrix,
                official_oracle.proposal_ids,
                threshold=threshold,
            )

            predicted = matrices["predicted"]
            gt_range = matrices["oracle_gt_range"]
            x_pred_range = matrices["oracle_x_pred_range"]
            both = matrices["oracle_x_gt_range"]
            matrix_by_name = {
                "predicted": predicted,
                "oracle_gt_range": gt_range,
                "oracle_x_pred_range": x_pred_range,
                "oracle_x_gt_range": both,
            }
            for name, matrix in matrix_by_name.items():
                _update_counter(
                    row_counters[threshold][f"fixed_model_selection__{name}"],
                    matrix,
                    model_ids,
                    threshold=threshold,
                )
                validity = (
                    matrices["candidate_valid"]
                    if name in {"predicted", "oracle_x_pred_range"}
                    else torch.ones(
                        matrix.shape[1],
                        dtype=torch.bool,
                        device=matrix.device,
                    )
                )
                oracle = cardinality_oracle_assignment(
                    matrix,
                    threshold,
                    args.top_k,
                    validity,
                )
                _update_counter(
                    row_counters[threshold][f"oracle_top4__{name}"],
                    matrix,
                    oracle.proposal_ids,
                    threshold=threshold,
                )

            predicted_hits = _strict_hit_set(predicted, threshold=threshold)
            range_hits = _strict_hit_set(gt_range, threshold=threshold)
            x_hits = _strict_hit_set(x_pred_range, threshold=threshold)
            both_hits = _strict_hit_set(both, threshold=threshold)
            for gt_index in range(int(predicted.shape[0])):
                if gt_index in predicted_hits:
                    bucket = "predicted_geometry_available"
                elif gt_index in range_hits and gt_index not in x_hits:
                    bucket = "range_only_rescue"
                elif gt_index in x_hits and gt_index not in range_hits:
                    bucket = "x_only_rescue"
                elif gt_index in range_hits and gt_index in x_hits:
                    bucket = "either_single_intervention_rescues"
                elif gt_index in both_hits:
                    bucket = "requires_joint_x_and_range_oracle"
                else:
                    bucket = "unreachable_even_with_joint_oracle"
                causal_buckets[threshold][bucket] += 1

    official_output: dict[str, Any] = {}
    row_output: dict[str, Any] = {}
    bucket_output: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        official_output[key] = {
            name: _finish_counter(counter)
            for name, counter in official_counters[threshold].items()
        }
        row_output[key] = {
            name: _finish_counter(counter)
            for name, counter in row_counters[threshold].items()
        }
        total = sum(causal_buckets[threshold].values())
        bucket_output[key] = {
            name: {
                "lanes": int(count),
                "fraction": 0.0 if total == 0 else float(count) / float(total),
            }
            for name, count in sorted(causal_buckets[threshold].items())
        }

    gains = _derive_recall_gains(official_output, row_output)
    return {
        "diagnostic_only": True,
        "warning": (
            "Official-raster results use unchanged predicted candidates and isolate "
            "score/ranking. Range/x interventions are pairwise GT oracles in row "
            "space; they localize the output bottleneck but do not by themselves "
            "prove an internal positional-encoding bug."
        ),
        "metadata": {
            **cache["metadata"],
            "tool": "analyze_single_group_bottleneck",
            "stage_name": args.stage_name,
            "num_query_groups": int(args.num_query_groups),
            "query_group_index": int(args.query_group_index),
            "group_global_candidate_ids": group_global_ids or [],
            "score_thresh": float(args.score_thresh),
            "quality_power": float(args.quality_power),
            "top_k": int(args.top_k),
            "min_valid_rows": int(args.min_valid_rows),
            "line_width": float(args.line_width),
            "iou_thresholds": thresholds,
            "row_space_gt_lanes": int(row_gt_total),
            "official_gt_lanes": int(official_gt_total),
        },
        "official_raster_predicted_candidates": official_output,
        "row_space_causal_ladder": row_output,
        "recall_gain_summary_points": gains,
        "best_candidate_causal_buckets": bucket_output,
        "best_candidate_x_mae_px_on_gt_rows": _summary(x_mae_values),
    }


def _print_report(payload: dict[str, Any]) -> None:
    print("Single-query-group bottleneck audit")
    print(
        "group candidate ids:",
        payload["metadata"]["group_global_candidate_ids"],
    )
    for threshold, modes in payload["official_raster_predicted_candidates"].items():
        print(f"\nOfficial raster IoU {threshold}")
        for name, values in modes.items():
            print(
                f"  {name:>26}: R={values['recall']:.4f} "
                f"P={values['precision']:.4f} F1={values['F1']:.4f} "
                f"TP={values['TP']} FP={values['FP']} FN={values['FN']}"
            )
    for threshold, modes in payload["row_space_causal_ladder"].items():
        print(f"\nRow-space causal ladder IoU {threshold}")
        for name, values in modes.items():
            print(
                f"  {name:>42}: R={values['recall']:.4f} "
                f"meanIoU={values['mean_best_iou']:.4f}"
            )
    print("\nRecall gain summary (points)")
    for threshold, values in payload["recall_gain_summary_points"].items():
        print(f"  IoU {threshold}: {values}")
    print("\nBest-candidate x MAE:", payload["best_candidate_x_mae_px_on_gt_rows"])


def main() -> None:
    args = parse_args()
    payload = analyze(args)
    _print_report(payload)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
