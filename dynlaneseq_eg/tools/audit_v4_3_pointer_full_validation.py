from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    STAGE_TENSOR_FIELDS,
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
    official_proposal_gt_iou_matrix,
)
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream full validation without a candidate cache and separate "
            "V4.3 pointer representative regret from STOP/cardinality error."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=[0.50, 0.75]
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _stage_batch(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    stages = collect_prediction_stages(outputs)
    for name in ("final", "stage2", "main", "coarse"):
        stage = stages.get(name)
        if isinstance(stage, dict):
            return stage
    if len(stages) == 1:
        return next(iter(stages.values()))
    raise ValueError("V4.3 audit could not resolve a final prediction stage")


def _slice_stage(
    stage: dict[str, Any],
    batch_index: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in STAGE_TENSOR_FIELDS:
        value = stage.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = value[batch_index].detach().cpu()
    required = {
        "pred_x_rows",
        "range_norm",
        "selection_pointer_indices",
        "selection_pointer_logits",
        "selection_logits",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(
            "V4.3 pointer output is missing: " + ", ".join(missing)
        )
    return result


def _selected_pointer_ids(
    indices: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    top_k: int,
) -> tuple[list[int], int]:
    valid = candidate_valid.bool()
    selected: list[int] = []
    invalid = 0
    for value in indices.flatten().tolist():
        candidate = int(value)
        if candidate < 0:
            break
        if candidate >= int(valid.numel()) or not bool(valid[candidate]):
            invalid += 1
            continue
        if candidate in selected:
            invalid += 1
            continue
        selected.append(candidate)
        if len(selected) >= int(top_k):
            break
    return selected, invalid


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(max(float(denominator), 1.0))


def _value_summary(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    def percentile(fraction: float) -> float:
        index = int(round(float(fraction) * float(len(ordered) - 1)))
        return ordered[min(max(index, 0), len(ordered) - 1)]

    return {
        "count": len(ordered),
        "mean": sum(ordered) / float(len(ordered)),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def analyze_pointer_image(
    official_iou: torch.Tensor,
    candidate_valid: torch.Tensor,
    pointer_indices: torch.Tensor,
    pointer_logits: torch.Tensor,
    unary_logits: torch.Tensor,
    *,
    thresholds: tuple[float, ...] = (0.50, 0.75),
    top_k: int = 4,
) -> dict[str, Any]:
    """Return sufficient per-image statistics for the streaming audit."""

    iou = official_iou.float().cpu()
    valid = candidate_valid.bool().cpu()
    selected, invalid_selections = _selected_pointer_ids(
        pointer_indices.cpu(),
        valid,
        top_k=top_k,
    )
    gt_count = int(iou.shape[0])
    selected_count = len(selected)
    threshold_rows: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        pointer = evaluator_hungarian_assignment(
            iou,
            selected,
            threshold=float(threshold),
        )
        same_count = cardinality_oracle_assignment(
            iou,
            threshold=float(threshold),
            top_k=selected_count,
            candidate_valid=valid,
        )
        top4 = cardinality_oracle_assignment(
            iou,
            threshold=float(threshold),
            top_k=int(top_k),
            candidate_valid=valid,
        )
        prefix_hits = []
        for step in range(int(top_k)):
            prefix = selected[: min(step + 1, selected_count)]
            prefix_hits.append(
                evaluator_hungarian_assignment(
                    iou,
                    prefix,
                    threshold=float(threshold),
                ).hit_count
            )
        threshold_rows[f"{float(threshold):.2f}"] = {
            "pointer_tp": int(pointer.hit_count),
            "same_count_oracle_tp": int(same_count.hit_count),
            "top4_oracle_tp": int(top4.hit_count),
            "representative_gap": int(same_count.hit_count - pointer.hit_count),
            "stop_cardinality_gap": int(top4.hit_count - same_count.hit_count),
            "prefix_hits": prefix_hits,
        }

    full_assignment = evaluator_hungarian_assignment(
        iou,
        selected,
        threshold=-1.0,
    )
    selected_by_gt = {
        int(gt_index): int(candidate_index)
        for gt_index, candidate_index in full_assignment.pairs
    }
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    representative_rows: list[dict[str, Any]] = []
    strict_threshold = max(float(value) for value in thresholds)
    for gt_index in range(gt_count):
        if valid_ids.numel() > 0:
            values = iou[gt_index, valid_ids]
            best_local = int(values.argmax())
            best_candidate = int(valid_ids[best_local])
            best_iou = float(values[best_local])
        else:
            best_candidate = -1
            best_iou = 0.0
        chosen_candidate = selected_by_gt.get(gt_index, -1)
        chosen_iou = (
            float(iou[gt_index, chosen_candidate])
            if chosen_candidate >= 0
            else 0.0
        )
        if chosen_candidate >= 0 and valid_ids.numel() > 0:
            rank = 1 + int(
                (iou[gt_index, valid_ids] > chosen_iou + 1e-12).sum()
            )
        else:
            rank = 0
        representative_rows.append(
            {
                "assigned": chosen_candidate >= 0,
                "best_candidate": best_candidate,
                "chosen_candidate": chosen_candidate,
                "best_iou": best_iou,
                "chosen_iou": chosen_iou,
                "regret": max(best_iou - chosen_iou, 0.0),
                "chosen_rank": rank,
                "strict_recoverable": best_iou > strict_threshold,
                "strict_hit": chosen_iou > strict_threshold,
            }
        )

    probabilities = torch.softmax(pointer_logits.float().cpu(), dim=-1)
    candidate_count = int(iou.shape[1])
    first_stop_probability = (
        float(probabilities[0, candidate_count])
        if probabilities.ndim == 2 and probabilities.shape[0] > 0
        else 0.0
    )
    unary_probability = torch.sigmoid(unary_logits.float().cpu())
    valid_unary = unary_probability[valid]
    unary_max_probability = (
        float(valid_unary.max()) if valid_unary.numel() > 0 else 0.0
    )
    return {
        "gt_count": gt_count,
        "selected_count": selected_count,
        "selected_ids": selected,
        "invalid_or_repeat_selections": int(invalid_selections),
        "initial_stop": selected_count == 0,
        "first_stop_probability": first_stop_probability,
        "unary_max_probability": unary_max_probability,
        "thresholds": threshold_rows,
        "representatives": representative_rows,
    }


class PointerAuditAccumulator:
    def __init__(self, thresholds: tuple[float, ...], top_k: int) -> None:
        self.thresholds = thresholds
        self.top_k = int(top_k)
        self.images = 0
        self.gt_lanes = 0
        self.selected = 0
        self.invalid_or_repeat = 0
        self.by_gt_count: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "images": 0,
                "gt_lanes": 0,
                "selected": 0,
                "exact_count": 0,
                "under_count": 0,
                "over_count": 0,
                "initial_stop": 0,
                "absolute_count_error": 0,
                "first_stop_probability_sum": 0.0,
                "unary_max_probability_sum": 0.0,
                "emitted_histogram": Counter(),
            }
        )
        self.threshold_stats = {
            f"{threshold:.2f}": {
                "pointer_tp": 0,
                "same_count_oracle_tp": 0,
                "top4_oracle_tp": 0,
                "prefix_hits": [0 for _ in range(self.top_k)],
            }
            for threshold in thresholds
        }
        self.assigned_regrets: list[float] = []
        self.all_gt_regrets: list[float] = []
        self.chosen_ranks: list[float] = []
        self.strict_recoverable = 0
        self.strict_hit = 0
        self.strict_wrong_representative = 0
        self.strict_unassigned = 0

    def update(self, row: dict[str, Any]) -> None:
        gt_count = int(row["gt_count"])
        selected_count = int(row["selected_count"])
        self.images += 1
        self.gt_lanes += gt_count
        self.selected += selected_count
        self.invalid_or_repeat += int(row["invalid_or_repeat_selections"])
        group = self.by_gt_count[gt_count]
        group["images"] += 1
        group["gt_lanes"] += gt_count
        group["selected"] += selected_count
        group["exact_count"] += int(selected_count == gt_count)
        group["under_count"] += int(selected_count < gt_count)
        group["over_count"] += int(selected_count > gt_count)
        group["initial_stop"] += int(row["initial_stop"])
        group["absolute_count_error"] += abs(selected_count - gt_count)
        group["first_stop_probability_sum"] += float(
            row["first_stop_probability"]
        )
        group["unary_max_probability_sum"] += float(
            row["unary_max_probability"]
        )
        group["emitted_histogram"][selected_count] += 1

        for key, values in row["thresholds"].items():
            target = self.threshold_stats[key]
            for name in (
                "pointer_tp",
                "same_count_oracle_tp",
                "top4_oracle_tp",
            ):
                target[name] += int(values[name])
            for step, hits in enumerate(values["prefix_hits"]):
                target["prefix_hits"][step] += int(hits)

        for representative in row["representatives"]:
            regret = float(representative["regret"])
            self.all_gt_regrets.append(regret)
            if bool(representative["assigned"]):
                self.assigned_regrets.append(regret)
                self.chosen_ranks.append(float(representative["chosen_rank"]))
            if bool(representative["strict_recoverable"]):
                self.strict_recoverable += 1
                if bool(representative["strict_hit"]):
                    self.strict_hit += 1
                elif bool(representative["assigned"]):
                    self.strict_wrong_representative += 1
                else:
                    self.strict_unassigned += 1

    def finish(self) -> dict[str, Any]:
        threshold_summary: dict[str, Any] = {}
        for key, row in self.threshold_stats.items():
            pointer_tp = int(row["pointer_tp"])
            same_count_tp = int(row["same_count_oracle_tp"])
            top4_tp = int(row["top4_oracle_tp"])
            selected = self.selected
            gt = self.gt_lanes
            precision = _safe_ratio(pointer_tp, selected)
            recall = _safe_ratio(pointer_tp, gt)
            f1 = (
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
            total_gap = max(top4_tp - pointer_tp, 0)
            representative_gap = max(same_count_tp - pointer_tp, 0)
            cardinality_gap = max(top4_tp - same_count_tp, 0)
            threshold_summary[key] = {
                "pointer": {
                    "tp": pointer_tp,
                    "fp": selected - pointer_tp,
                    "fn": gt - pointer_tp,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                "same_emitted_count_oracle": {
                    "tp": same_count_tp,
                    "recall": _safe_ratio(same_count_tp, gt),
                },
                "top4_candidate_oracle": {
                    "tp": top4_tp,
                    "recall": _safe_ratio(top4_tp, gt),
                },
                "oracle_gap_decomposition": {
                    "total_recoverable_tp": total_gap,
                    "representative_or_ordering_tp": representative_gap,
                    "stop_or_cardinality_tp": cardinality_gap,
                    "representative_fraction": _safe_ratio(
                        representative_gap, total_gap
                    ),
                    "stop_cardinality_fraction": _safe_ratio(
                        cardinality_gap, total_gap
                    ),
                },
                "pointer_prefix_tp": {
                    str(step + 1): int(value)
                    for step, value in enumerate(row["prefix_hits"])
                },
            }

        count_groups: dict[str, Any] = {}
        for gt_count in sorted(self.by_gt_count):
            row = self.by_gt_count[gt_count]
            images = int(row["images"])
            count_groups[str(gt_count)] = {
                "images": images,
                "mean_gt_lanes": _safe_ratio(row["gt_lanes"], images),
                "mean_emitted_lanes": _safe_ratio(row["selected"], images),
                "exact_count_rate": _safe_ratio(row["exact_count"], images),
                "under_count_rate": _safe_ratio(row["under_count"], images),
                "over_count_rate": _safe_ratio(row["over_count"], images),
                "initial_stop_rate": _safe_ratio(row["initial_stop"], images),
                "mean_absolute_count_error": _safe_ratio(
                    row["absolute_count_error"], images
                ),
                "mean_first_stop_probability": _safe_ratio(
                    row["first_stop_probability_sum"], images
                ),
                "mean_unary_max_probability": _safe_ratio(
                    row["unary_max_probability_sum"], images
                ),
                "emitted_histogram": {
                    str(key): int(value)
                    for key, value in sorted(row["emitted_histogram"].items())
                },
            }

        strict_unrecovered = max(self.strict_recoverable - self.strict_hit, 0)
        zero_gt = count_groups.get("0", {"images": 0})
        strict_key = f"{max(self.thresholds):.2f}"
        strict_gap = threshold_summary[strict_key]["oracle_gap_decomposition"]
        if int(zero_gt.get("images", 0)) > 0 and float(
            zero_gt.get("mean_emitted_lanes", 0.0)
        ) > 0.10:
            stop_verdict = "empty_scene_stop_is_a_confirmed_failure"
        elif int(zero_gt.get("images", 0)) == 0:
            stop_verdict = "validation_has_no_zero_gt_images"
        else:
            stop_verdict = "empty_scene_stop_is_well_controlled"
        if int(strict_gap["representative_or_ordering_tp"]) > int(
            strict_gap["stop_or_cardinality_tp"]
        ):
            strict_verdict = "representative_ordering_is_the_larger_strict_gap"
        else:
            strict_verdict = "stop_cardinality_is_the_larger_strict_gap"

        return {
            "images": self.images,
            "gt_lanes": self.gt_lanes,
            "selected_predictions": self.selected,
            "mean_gt_lanes_per_image": _safe_ratio(
                self.gt_lanes, self.images
            ),
            "mean_selected_per_image": _safe_ratio(
                self.selected, self.images
            ),
            "invalid_or_repeat_pointer_selections": self.invalid_or_repeat,
            "official_iou": threshold_summary,
            "cardinality_by_gt_count": count_groups,
            "representative_quality": {
                "assigned_gt_regret": _value_summary(self.assigned_regrets),
                "all_gt_regret_including_unassigned": _value_summary(
                    self.all_gt_regrets
                ),
                "chosen_candidate_rank": _value_summary(self.chosen_ranks),
                "top1_chosen_rate": _safe_ratio(
                    sum(rank == 1.0 for rank in self.chosen_ranks),
                    len(self.chosen_ranks),
                ),
                "top2_chosen_rate": _safe_ratio(
                    sum(rank <= 2.0 for rank in self.chosen_ranks),
                    len(self.chosen_ranks),
                ),
                "strict_recoverable_gt": self.strict_recoverable,
                "strict_hit_with_pointer_assignment": self.strict_hit,
                "strict_recoverable_but_wrong_representative": (
                    self.strict_wrong_representative
                ),
                "strict_recoverable_but_unassigned_after_stop": (
                    self.strict_unassigned
                ),
                "strict_unrecovered_total": strict_unrecovered,
            },
            "verdict": {
                "empty_scene_stop": stop_verdict,
                "strict_iou": strict_verdict,
                "next_action": (
                    "repair_pointer_representative_quality_before_long_geometry_training"
                    if strict_verdict
                    == "representative_ordering_is_the_larger_strict_gap"
                    else "audit_geometry_and_stop_cardinality_trajectory_before_278k"
                ),
            },
        }


def _official_for_record(
    record: dict[str, Any],
    *,
    line_width: float,
    min_valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return official_proposal_gt_iou_matrix(
        record,
        "main",
        line_width=float(line_width),
        min_valid_rows=int(min_valid_rows),
        row_visibility_thresh=0.0,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    thresholds = tuple(sorted(set(float(value) for value in args.iou_thresholds)))
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    if int(args.top_k) < 1:
        raise ValueError("top-k must be positive")
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = str(args.dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(
        args.eval_batch_size
    )
    cfg["dataloader"]["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    checkpoint_iteration = int(
        load_checkpoint(args.checkpoint, model, strict=False)
    )
    model.requires_grad_(False).eval()
    supports_inference_only = bool(
        getattr(model, "supports_inference_only", False)
    )
    if supports_inference_only and hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    loader = build_dataloader(cfg, split=args.split, training=False)
    accumulator = PointerAuditAccumulator(thresholds, args.top_k)

    metric_workers = max(int(args.metric_workers), 1)
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    executor = ThreadPoolExecutor(max_workers=metric_workers)
    try:
        iterator = tqdm(
            loader,
            ncols=100,
            desc="V4.3 full-validation pointer forensics",
        )
        for batch_index, (images, _targets, metas) in enumerate(iterator):
            if int(args.max_batches) > 0 and batch_index >= int(args.max_batches):
                break
            if channels_last:
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                images = images.to(device, non_blocking=True)
            with _amp_context(device, args.amp_dtype):
                outputs = (
                    model(images, inference_only=True)
                    if supports_inference_only
                    else model(images)
                )
            stage_batch = _stage_batch(outputs)
            records: list[dict[str, Any]] = []
            stages: list[dict[str, torch.Tensor]] = []
            for image_index, meta in enumerate(metas):
                stage = _slice_stage(stage_batch, image_index)
                stages.append(stage)
                records.append(
                    {
                        "meta": dict(meta),
                        "stages": {"main": stage},
                    }
                )
            futures = [
                executor.submit(
                    _official_for_record,
                    record,
                    line_width=args.line_width,
                    min_valid_rows=args.min_valid_rows,
                )
                for record in records
            ]
            for stage, future in zip(stages, futures):
                official_iou, candidate_valid = future.result()
                accumulator.update(
                    analyze_pointer_image(
                        official_iou,
                        candidate_valid,
                        stage["selection_pointer_indices"],
                        stage["selection_pointer_logits"],
                        stage["selection_logits"],
                        thresholds=thresholds,
                        top_k=args.top_k,
                    )
                )
    finally:
        executor.shutdown(wait=True)
        cv2.setNumThreads(previous_cv_threads)

    report = {
        "experiment": "V4.3 full-validation streaming pointer/STOP forensics",
        "diagnostic_only": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_iteration": checkpoint_iteration,
        "split": str(args.split),
        "protocol": {
            "official_culane_raster_iou": True,
            "line_width": float(args.line_width),
            "thresholds": list(thresholds),
            "top_k_maximum": int(args.top_k),
            "pointer_stop": True,
            "score_threshold": 0.0,
            "nms": False,
            "candidate_cache_written": False,
            "max_batches": int(args.max_batches),
        },
        "audit": accumulator.finish(),
        "warning": (
            "Use this full-validation report for the next architectural "
            "decision. Do not tune the already-observed CULane test split."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
