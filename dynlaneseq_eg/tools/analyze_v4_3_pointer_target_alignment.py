from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    diagnostic_iou_matrix,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    proposal_gt_iou_matrix,
)
from dynlaneseq_eg.losses.loss_s0 import build_pointer_sequence_targets
from dynlaneseq_eg.tools.audit_v4_3_pointer_full_validation import (
    _selected_pointer_ids,
    _value_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the existing V4.3 uniform cache to distinguish row-strip "
            "training-target mismatch from pointer learning/exposure error."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-strategy", default="uniform")
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75]
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-12:
        return 0.0
    return float((x * y).sum() / denominator)


def compare_target_alignment_image(
    row_iou: torch.Tensor,
    official_iou: torch.Tensor,
    candidate_valid: torch.Tensor,
    pointer_indices: torch.Tensor,
    training_target_indices: torch.Tensor,
    *,
    thresholds: tuple[float, ...] = (0.50, 0.75),
    top_k: int = 4,
) -> dict[str, Any]:
    row = row_iou.float().cpu()
    official = official_iou.float().cpu()
    valid = candidate_valid.bool().cpu()
    if row.shape != official.shape:
        raise ValueError(
            "row/official IoU shape mismatch: "
            f"{tuple(row.shape)} vs {tuple(official.shape)}"
        )
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    pointer_ids, _invalid = _selected_pointer_ids(
        pointer_indices.cpu(),
        valid,
        top_k=top_k,
    )

    target_ids: list[int] = []
    candidate_count = int(row.shape[1])
    for value in training_target_indices.detach().cpu().flatten().tolist()[:top_k]:
        index = int(value)
        if index == candidate_count:  # Learned STOP class.
            break
        if index == -100:  # Positions after STOP are ignored by the loss.
            continue
        if not 0 <= index < candidate_count:
            raise ValueError(f"invalid pointer teacher index: {index}")
        if index in target_ids:
            raise ValueError(f"duplicate pointer teacher index: {index}")
        target_ids.append(index)

    threshold_rows: dict[str, Any] = {}
    for threshold in thresholds:
        pointer_tp = evaluator_hungarian_assignment(
            official,
            pointer_ids,
            threshold=float(threshold),
        ).hit_count
        target_tp = evaluator_hungarian_assignment(
            official,
            target_ids,
            threshold=float(threshold),
        ).hit_count
        oracle_tp = cardinality_oracle_assignment(
            official,
            threshold=float(threshold),
            top_k=int(top_k),
            candidate_valid=valid,
        ).hit_count
        threshold_rows[f"{float(threshold):.2f}"] = {
            "pointer_tp": int(pointer_tp),
            "training_target_set_tp": int(target_tp),
            "official_oracle_tp": int(oracle_tp),
        }

    top1_agreement = 0
    official_regrets: list[float] = []
    row_official_pairs: list[tuple[float, float]] = []
    official_best_values: list[float] = []
    row_best_official_values: list[float] = []
    for gt_index in range(int(row.shape[0])):
        if not valid_ids:
            continue
        ids = torch.tensor(valid_ids, dtype=torch.long)
        row_values = row[gt_index, ids]
        official_values = official[gt_index, ids]
        row_best_local = int(row_values.argmax())
        official_best_local = int(official_values.argmax())
        top1_agreement += int(row_best_local == official_best_local)
        row_best_official = float(official_values[row_best_local])
        official_best = float(official_values[official_best_local])
        official_regrets.append(max(official_best - row_best_official, 0.0))
        official_best_values.append(official_best)
        row_best_official_values.append(row_best_official)
        row_official_pairs.extend(
            (float(row_value), float(official_value))
            for row_value, official_value in zip(row_values, official_values)
        )

    pointer_set = set(pointer_ids)
    target_set = set(target_ids)
    union = pointer_set | target_set
    return {
        "gt_count": int(row.shape[0]),
        "pointer_count": len(pointer_ids),
        "training_target_count": len(target_ids),
        "target_pointer_overlap": len(pointer_set & target_set),
        "target_pointer_union": len(union),
        "exact_target_set": pointer_set == target_set,
        "thresholds": threshold_rows,
        "top1_agreement": top1_agreement,
        "official_regrets": official_regrets,
        "official_best_values": official_best_values,
        "row_best_official_values": row_best_official_values,
        "row_official_pairs": row_official_pairs,
    }


def summarize_alignment(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    gt = sum(int(row["gt_count"]) for row in rows)
    pointer_count = sum(int(row["pointer_count"]) for row in rows)
    target_count = sum(int(row["training_target_count"]) for row in rows)
    overlap = sum(int(row["target_pointer_overlap"]) for row in rows)
    union = sum(int(row["target_pointer_union"]) for row in rows)
    exact_sets = sum(bool(row["exact_target_set"]) for row in rows)
    top1_agreement = sum(int(row["top1_agreement"]) for row in rows)
    regrets = [
        value for row in rows for value in row["official_regrets"]
    ]
    official_best = [
        value for row in rows for value in row["official_best_values"]
    ]
    row_best_official = [
        value for row in rows for value in row["row_best_official_values"]
    ]
    pair_rows = [pair for row in rows for pair in row["row_official_pairs"]]
    row_values = [left for left, _right in pair_rows]
    official_values = [right for _left, right in pair_rows]

    threshold_summary: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        pointer_tp = sum(int(row["thresholds"][key]["pointer_tp"]) for row in rows)
        target_tp = sum(
            int(row["thresholds"][key]["training_target_set_tp"])
            for row in rows
        )
        oracle_tp = sum(
            int(row["thresholds"][key]["official_oracle_tp"])
            for row in rows
        )
        total_gap = max(oracle_tp - pointer_tp, 0)
        target_definition_gap = max(oracle_tp - target_tp, 0)
        # This term is deliberately signed.  A negative value means the
        # deployed pointer accidentally outperforms its own teacher set, which
        # is itself evidence that the representative target is misaligned.
        learning_exposure_gap = target_tp - pointer_tp
        threshold_summary[key] = {
            "pointer_tp": pointer_tp,
            "training_target_set_tp": target_tp,
            "official_oracle_tp": oracle_tp,
            "total_pointer_to_oracle_gap": total_gap,
            "training_target_definition_gap": target_definition_gap,
            "pointer_learning_or_exposure_gap": learning_exposure_gap,
            "target_definition_fraction": (
                float(target_definition_gap) / float(max(total_gap, 1))
            ),
            "learning_exposure_fraction_signed": (
                float(learning_exposure_gap) / float(max(total_gap, 1))
            ),
        }

    strict_key = f"{max(thresholds):.2f}"
    strict = threshold_summary[strict_key]
    if strict["training_target_definition_gap"] > strict[
        "pointer_learning_or_exposure_gap"
    ]:
        verdict = "row_strip_target_misalignment_is_primary"
        next_action = "replace_or_calibrate_the_representative_quality_target"
    else:
        verdict = "pointer_learning_or_exposure_is_primary"
        next_action = "strengthen_cluster_winner_learning_and_pointer_rollout"
    return {
        "images": len(rows),
        "gt_lanes": gt,
        "pointer_predictions": pointer_count,
        "training_target_representatives": target_count,
        "pointer_target_set_jaccard": float(overlap) / float(max(union, 1)),
        "exact_pointer_target_set_rate": float(exact_sets) / float(max(len(rows), 1)),
        "row_strip_vs_official": {
            "pairwise_pearson": _pearson(row_values, official_values),
            "best_candidate_top1_agreement": float(top1_agreement) / float(max(gt, 1)),
            "official_regret_when_using_row_strip_top1": _value_summary(regrets),
            "mean_official_best_iou": (
                sum(official_best) / float(max(len(official_best), 1))
            ),
            "mean_official_iou_of_row_strip_best": (
                sum(row_best_official) / float(max(len(row_best_official), 1))
            ),
        },
        "official_iou_gap_decomposition": threshold_summary,
        "verdict": verdict,
        "next_action": next_action,
    }


def main() -> None:
    args = parse_args()
    thresholds = tuple(sorted(set(float(value) for value in args.iou_thresholds)))
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root or None,
        device="cpu",
        cache_dir=args.cache_dir,
        reuse_cache=True,
        require_cache=True,
        max_batches=int(args.max_batches),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy=str(args.sample_strategy),
        desc="unused-cache-only",
    )
    rows: list[dict[str, Any]] = []
    for record in cache["records"]:
        stage_name = (
            "final"
            if "final" in record["stages"]
            else "stage2"
            if "stage2" in record["stages"]
            else "main"
        )
        stage = record["stages"][stage_name]
        official, _valid_gt, official_valid = diagnostic_iou_matrix(
            record,
            stage_name,
            use_official=True,
        )
        row_iou, valid_gt, row_valid = proposal_gt_iou_matrix(
            stage,
            record["target"],
            input_h=int(cache["metadata"].get("input_h", 288)),
            input_w=int(cache["metadata"].get("input_w", 800)),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            row_visibility_thresh=0.0,
        )
        if int(valid_gt.sum()) != int(official.shape[0]):
            raise ValueError("row/official valid GT count mismatch")
        teacher = build_pointer_sequence_targets(
            {
                "pred_x_rows": stage["pred_x_rows"].unsqueeze(0),
                "range_norm": stage["range_norm"].unsqueeze(0),
            },
            [record["target"]],
            max_selections=int(args.top_k),
            input_h=int(cache["metadata"].get("input_h", 288)),
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
        )[0]
        candidate_valid = official_valid.bool() & row_valid.bool()
        rows.append(
            compare_target_alignment_image(
                row_iou,
                official,
                candidate_valid,
                stage["selection_pointer_indices"],
                teacher,
                thresholds=thresholds,
                top_k=int(args.top_k),
            )
        )

    report = {
        "experiment": "V4.3 pointer training-target vs official-IoU alignment",
        "diagnostic_only": True,
        "cache": cache["metadata"].get("cache_path", ""),
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "sample": {
            "split": str(args.split),
            "max_batches": int(args.max_batches),
            "eval_batch_size": int(args.eval_batch_size),
            "strategy": str(args.sample_strategy),
        },
        "analysis": summarize_alignment(rows, thresholds),
        "warning": (
            "This audit selects the next loss/decoder intervention. It is not "
            "a benchmark result and does not authorize test-set tuning."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
