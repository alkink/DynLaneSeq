from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    ensure_official_iou_cache,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    official_proposal_gt_iou_matrix,
    proposal_gt_iou_matrix,
)
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm
from dynlaneseq_eg.tools.analyze_selection_assignment_gradient_conflict import (
    _target_to_official_mapping,
)


def _install_numpy_pickle_compatibility() -> None:
    """Read NumPy-2 checkpoints from the supported local NumPy-1 runtime.

    NumPy 2 moved the private pickle module from ``numpy.core`` to
    ``numpy._core``.  Torch checkpoints containing a captured NumPy RNG state
    therefore need module aliases when audited on the project's older local
    environment.  This does not alter arrays or checkpoint tensors.
    """

    if hasattr(np, "_core"):
        return
    core = np.core
    sys.modules.setdefault("numpy._core", core)
    for suffix in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        module = getattr(core, suffix, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{suffix}", module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the zero-training geometry ceiling of the V6-B routed "
            "slot refiner. The audit separates learned x correction, a "
            "second bounded x correction, inherited-range error, and route "
            "coverage in official CULane raster-IoU space."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument(
        "--sample-strategy",
        choices=("uniform", "sequential"),
        default="uniform",
    )
    parser.add_argument(
        "--iou-thresholds",
        nargs="+",
        type=float,
        default=(0.50, 0.75),
    )
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--min-valid-rows", type=int, default=5)
    parser.add_argument("--match-min-quality", type=float, default=0.20)
    parser.add_argument("--delta-bound-px", type=float, default=24.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _stage_name(record: dict[str, Any]) -> str:
    for name in ("final", "stage2", "main"):
        if name in record.get("stages", {}):
            return name
    if not record.get("stages"):
        raise ValueError("diagnostic record has no prediction stage")
    return next(iter(record["stages"]))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _value_summary(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(finite),
        "mean": sum(finite) / float(len(finite)),
        "p50": _percentile(finite, 0.50),
        "p90": _percentile(finite, 0.90),
        "p95": _percentile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def _gather_routed_geometry(
    stage: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    routes = stage["selection_slot_indices"].long()
    proposal_x = stage["pred_x_rows"].float()
    proposal_range = sort_range_norm(stage["range_norm"].float())
    candidates = int(proposal_x.shape[0])
    rows = int(proposal_x.shape[-1])
    safe = routes.clamp(min=0, max=max(candidates - 1, 0))
    active = routes >= 0
    reference_x = proposal_x.gather(
        0,
        safe.unsqueeze(-1).expand(-1, rows),
    )
    reference_range = proposal_range.gather(
        0,
        safe.unsqueeze(-1).expand(-1, 2),
    )
    reference_x = torch.where(
        active.unsqueeze(-1), reference_x, torch.zeros_like(reference_x)
    )
    reference_range = torch.where(
        active.unsqueeze(-1),
        reference_range,
        torch.zeros_like(reference_range),
    )
    return reference_x, reference_range, active


def _gt_ranges(
    valid_mask: torch.Tensor,
    *,
    input_h: int,
) -> torch.Tensor:
    valid_mask = valid_mask.bool()
    rows = int(valid_mask.shape[-1])
    y_rows = fixed_y_rows(rows, input_h, dtype=torch.float32)
    output = torch.zeros((int(valid_mask.shape[0]), 2), dtype=torch.float32)
    for gt_index, valid in enumerate(valid_mask):
        ids = torch.nonzero(valid, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        values = y_rows[ids] / float(input_h)
        output[gt_index, 0] = values.min()
        output[gt_index, 1] = values.max()
    return output


def _reference_matches(
    reference_x: torch.Tensor,
    reference_range: torch.Tensor,
    active: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    line_width: float,
    min_valid_rows: int,
    match_min_quality: float,
) -> tuple[list[tuple[int, int, float]], torch.Tensor]:
    stage = {
        "pred_x_rows": reference_x,
        "range_norm": reference_range,
    }
    quality, valid_gt, valid_slot = proposal_gt_iou_matrix(
        stage,
        target,
        input_h=input_h,
        input_w=input_w,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=0.0,
    )
    gt_ids = torch.nonzero(valid_gt, as_tuple=False).flatten()
    slot_ids = torch.nonzero(valid_slot & active, as_tuple=False).flatten()
    if gt_ids.numel() == 0 or slot_ids.numel() == 0:
        return [], quality
    local = quality[:, slot_ids]
    gt_rows, slot_columns = linear_sum_assignment(
        1.0 - local.detach().cpu().numpy()
    )
    pairs = []
    for gt_row, slot_column in zip(gt_rows.tolist(), slot_columns.tolist()):
        value = float(local[int(gt_row), int(slot_column)])
        if value < float(match_min_quality):
            continue
        pairs.append(
            (
                int(slot_ids[int(slot_column)]),
                int(gt_ids[int(gt_row)]),
                value,
            )
        )
    return pairs, quality


def _replace_ranges(
    ranges: torch.Tensor,
    pairs: list[tuple[int, int, float]],
    gt_ranges: torch.Tensor,
) -> torch.Tensor:
    output = ranges.clone()
    for slot_index, gt_index, _quality in pairs:
        output[int(slot_index)] = gt_ranges[int(gt_index)]
    return sort_range_norm(output)


def _replace_x(
    base_x: torch.Tensor,
    reference_x: torch.Tensor,
    pairs: list[tuple[int, int, float]],
    target: dict[str, torch.Tensor],
    *,
    bound_px: float | None,
) -> torch.Tensor:
    output = base_x.clone()
    gt_x = target["x_rows"].float()
    gt_valid = target["valid_mask"].bool() & torch.isfinite(gt_x)
    for slot_index, gt_index, _quality in pairs:
        slot_index = int(slot_index)
        gt_index = int(gt_index)
        valid = gt_valid[gt_index]
        desired = gt_x[gt_index]
        if bound_px is None:
            replacement = desired
        else:
            residual = (desired - reference_x[slot_index]).clamp(
                min=-float(bound_px), max=float(bound_px)
            )
            replacement = reference_x[slot_index] + residual
        output[slot_index] = torch.where(
            valid,
            replacement,
            output[slot_index],
        )
    return output


def build_counterfactual_geometries(
    stage: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    input_h: int,
    input_w: int,
    line_width: float,
    min_valid_rows: int,
    match_min_quality: float,
    delta_bound_px: float,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    list[tuple[int, int, float]],
    dict[str, Any],
]:
    reference_x, reference_range, active = _gather_routed_geometry(stage)
    refined_x = stage["selection_slot_pred_x_rows"].float()
    refined_range = sort_range_norm(
        stage.get("selection_slot_range_norm", reference_range).float()
    )
    pairs, _quality = _reference_matches(
        reference_x,
        reference_range,
        active,
        target,
        input_h=input_h,
        input_w=input_w,
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        match_min_quality=match_min_quality,
    )
    gt_ranges = _gt_ranges(target["valid_mask"], input_h=input_h)
    reference_gt_range = _replace_ranges(reference_range, pairs, gt_ranges)
    refined_gt_range = _replace_ranges(refined_range, pairs, gt_ranges)
    bounded_x = _replace_x(
        reference_x,
        reference_x,
        pairs,
        target,
        bound_px=delta_bound_px,
    )
    second_bounded_x = _replace_x(
        refined_x,
        refined_x,
        pairs,
        target,
        bound_px=delta_bound_px,
    )
    exact_x = _replace_x(
        reference_x,
        reference_x,
        pairs,
        target,
        bound_px=None,
    )
    methods = {
        "reference": (reference_x, reference_range),
        "learned_refined": (refined_x, refined_range),
        "reference_gt_range_oracle": (
            reference_x,
            reference_gt_range,
        ),
        "learned_refined_gt_range_oracle": (
            refined_x,
            refined_gt_range,
        ),
        "bounded_x_oracle_current_range": (
            bounded_x,
            reference_range,
        ),
        "bounded_x_plus_gt_range_oracle": (
            bounded_x,
            reference_gt_range,
        ),
        "second_bounded_x_oracle_current_range": (
            second_bounded_x,
            refined_range,
        ),
        "second_bounded_x_plus_gt_range_oracle": (
            second_bounded_x,
            refined_gt_range,
        ),
        "exact_x_oracle_current_range": (exact_x, reference_range),
        "exact_x_plus_gt_range_oracle": (exact_x, reference_gt_range),
    }

    residuals: list[float] = []
    range_ious: list[float] = []
    gt_x = target["x_rows"].float()
    gt_valid = target["valid_mask"].bool() & torch.isfinite(gt_x)
    y_rows = fixed_y_rows(
        int(reference_x.shape[-1]), input_h, dtype=torch.float32
    )
    for slot_index, gt_index, _quality_value in pairs:
        valid = gt_valid[int(gt_index)]
        residuals.extend(
            (gt_x[int(gt_index), valid] - reference_x[int(slot_index), valid])
            .abs()
            .tolist()
        )
        low, high = reference_range[int(slot_index)] * float(input_h)
        pred_valid = (y_rows >= low) & (y_rows <= high)
        intersection = int((pred_valid & valid).sum())
        union = int((pred_valid | valid).sum())
        range_ious.append(float(intersection) / float(max(union, 1)))
    residual_array = np.asarray(residuals, dtype=np.float64)
    diagnostics = {
        "active_slots": int(active.sum()),
        "matched_slots": len(pairs),
        "reference_match_quality": _value_summary(
            [quality for _slot, _gt, quality in pairs]
        ),
        "absolute_target_residual_px": _value_summary(residuals),
        "target_residual_fraction_over": {
            "6px": float(np.mean(residual_array > 6.0))
            if residual_array.size
            else 0.0,
            "12px": float(np.mean(residual_array > 12.0))
            if residual_array.size
            else 0.0,
            f"{float(delta_bound_px):g}px": float(
                np.mean(residual_array > float(delta_bound_px))
            )
            if residual_array.size
            else 0.0,
        },
        "reference_vs_gt_range_iou": _value_summary(range_ious),
    }
    return methods, pairs, diagnostics


def _synthetic_official_iou(
    record: dict[str, Any],
    x_rows: torch.Tensor,
    range_norm: torch.Tensor,
    *,
    line_width: float,
    min_valid_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    synthetic = {
        "meta": record["meta"],
        "stages": {
            "synthetic": {
                "pred_x_rows": x_rows,
                "range_norm": range_norm,
            }
        },
    }
    return official_proposal_gt_iou_matrix(
        synthetic,
        "synthetic",
        line_width=line_width,
        min_valid_rows=min_valid_rows,
        row_visibility_thresh=0.0,
    )


@dataclass
class MetricAccumulator:
    gt: int = 0
    predictions: int = 0
    tp: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def update(
        self,
        matrix: torch.Tensor,
        candidate_valid: torch.Tensor,
        thresholds: tuple[float, ...],
    ) -> None:
        valid_ids = torch.nonzero(
            candidate_valid.bool(), as_tuple=False
        ).flatten().tolist()
        self.gt += int(matrix.shape[0])
        self.predictions += len(valid_ids)
        for threshold in thresholds:
            assignment = evaluator_hungarian_assignment(
                matrix,
                valid_ids,
                float(threshold),
            )
            self.tp[f"{float(threshold):.2f}"] += int(
                assignment.hit_count
            )

    def summary(self, thresholds: tuple[float, ...]) -> dict[str, Any]:
        output: dict[str, Any] = {
            "gt": self.gt,
            "predictions": self.predictions,
        }
        for threshold in thresholds:
            key = f"{float(threshold):.2f}"
            tp = int(self.tp[key])
            fp = int(self.predictions - tp)
            fn = int(self.gt - tp)
            precision = float(tp) / float(max(tp + fp, 1))
            recall = float(tp) / float(max(tp + fn, 1))
            f1 = 2.0 * float(tp) / float(max(self.predictions + self.gt, 1))
            output[key] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return output


@dataclass
class FixedPairStats:
    deltas: list[float] = field(default_factory=list)
    source_values: list[float] = field(default_factory=list)
    refined_values: list[float] = field(default_factory=list)
    transitions: dict[str, dict[str, int]] = field(default_factory=dict)

    def update(
        self,
        pairs: list[tuple[int, int, float]],
        target_to_official: dict[int, int],
        reference_iou: torch.Tensor,
        refined_iou: torch.Tensor,
        thresholds: tuple[float, ...],
    ) -> None:
        for slot_index, target_index, _quality in pairs:
            official_index = target_to_official.get(int(target_index))
            if official_index is None:
                continue
            source = float(reference_iou[int(official_index), int(slot_index)])
            refined = float(refined_iou[int(official_index), int(slot_index)])
            self.source_values.append(source)
            self.refined_values.append(refined)
            self.deltas.append(refined - source)
            for threshold in thresholds:
                key = f"{float(threshold):.2f}"
                row = self.transitions.setdefault(
                    key,
                    {
                        "miss_to_hit": 0,
                        "hit_to_miss": 0,
                        "hit_to_hit": 0,
                        "miss_to_miss": 0,
                    },
                )
                source_hit = source > float(threshold)
                refined_hit = refined > float(threshold)
                if not source_hit and refined_hit:
                    row["miss_to_hit"] += 1
                elif source_hit and not refined_hit:
                    row["hit_to_miss"] += 1
                elif source_hit and refined_hit:
                    row["hit_to_hit"] += 1
                else:
                    row["miss_to_miss"] += 1

    def summary(self) -> dict[str, Any]:
        positive = sum(value > 1.0e-9 for value in self.deltas)
        negative = sum(value < -1.0e-9 for value in self.deltas)
        return {
            "source_official_iou": _value_summary(self.source_values),
            "refined_official_iou": _value_summary(self.refined_values),
            "delta_refined_minus_source": _value_summary(self.deltas),
            "positive_pairs": positive,
            "negative_pairs": negative,
            "unchanged_pairs": len(self.deltas) - positive - negative,
            "transitions": self.transitions,
        }


def _oracle_update(
    accumulator: dict[str, defaultdict[str, int]],
    matrix: torch.Tensor,
    candidate_valid: torch.Tensor,
    active_count: int,
    thresholds: tuple[float, ...],
) -> None:
    accumulator["meta"]["gt"] += int(matrix.shape[0])
    accumulator["meta"]["emitted"] += int(active_count)
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        all_four = cardinality_oracle_assignment(
            matrix,
            float(threshold),
            top_k=4,
            candidate_valid=candidate_valid,
        )
        same_count = cardinality_oracle_assignment(
            matrix,
            float(threshold),
            top_k=int(active_count),
            candidate_valid=candidate_valid,
        )
        accumulator["top4"][key] += int(all_four.hit_count)
        accumulator["same_count"][key] += int(same_count.hit_count)


def _oracle_summary(
    accumulator: dict[str, defaultdict[str, int]],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    gt = int(accumulator["meta"]["gt"])
    output: dict[str, Any] = {
        "gt": gt,
        "emitted": int(accumulator["meta"]["emitted"]),
    }
    for name in ("same_count", "top4"):
        output[name] = {}
        for threshold in thresholds:
            key = f"{float(threshold):.2f}"
            tp = int(accumulator[name][key])
            output[name][key] = {
                "tp": tp,
                "recall": float(tp) / float(max(gt, 1)),
            }
    return output


def _recommendation(methods: dict[str, Any]) -> dict[str, Any]:
    learned = methods["learned_refined"]["0.75"]
    learned_range = methods["learned_refined_gt_range_oracle"]["0.75"]
    second = methods["second_bounded_x_oracle_current_range"]["0.75"]
    exact_current = methods["exact_x_oracle_current_range"]["0.75"]
    exact_range = methods["exact_x_plus_gt_range_oracle"]["0.75"]
    gains = {
        "gt_range_on_learned_x_tp": int(learned_range["tp"] - learned["tp"]),
        "second_bounded_x_tp": int(second["tp"] - learned["tp"]),
        "exact_x_current_range_tp": int(
            exact_current["tp"] - learned["tp"]
        ),
        "exact_x_plus_gt_range_tp": int(
            exact_range["tp"] - learned["tp"]
        ),
    }
    largest = max(gains, key=gains.get)
    if gains["gt_range_on_learned_x_tp"] >= max(
        8, gains["second_bounded_x_tp"]
    ):
        next_step = "add_bounded_slot_range_refinement_before_router_unfreeze"
    elif gains["second_bounded_x_tp"] >= 8:
        next_step = "add_a_second_iterative_bounded_x_refinement_stage"
    elif gains["exact_x_plus_gt_range_tp"] >= 8:
        next_step = "coadapt_router_and_slot_geometry_with_protected_gradients"
    else:
        next_step = "stop_local_refinement_and_reaudit_route_targets"
    return {
        "recoverable_strict_tp": gains,
        "largest_counterfactual": largest,
        "next_step": next_step,
        "warning": (
            "All oracle geometries are diagnostic counterfactuals and are "
            "not deployable predictions."
        ),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    _install_numpy_pickle_compatibility()
    thresholds = tuple(sorted(set(float(v) for v in args.iou_thresholds)))
    cache = load_or_collect_cache(
        args.config,
        args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root,
        device=args.device,
        cache_dir=args.cache_dir,
        reuse_cache=True,
        require_cache=False,
        max_batches=int(args.max_batches),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        sample_strategy=str(args.sample_strategy),
        desc="V6-B refinement ceiling cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=float(args.line_width),
        min_valid_rows=int(args.min_valid_rows),
        row_visibility_thresh=0.0,
        workers=int(args.metric_workers),
    )
    input_h = int(cache["metadata"]["input_h"])
    input_w = int(cache["metadata"]["input_w"])
    metric_accumulators: dict[str, MetricAccumulator] = {}
    fixed_pairs = FixedPairStats()
    residuals: list[float] = []
    range_ious: list[float] = []
    match_quality: list[float] = []
    active_slots = 0
    matched_slots = 0
    mapping_quality: list[float] = []
    oracle_accumulator: dict[str, defaultdict[str, int]] = {
        "meta": defaultdict(int),
        "same_count": defaultdict(int),
        "top4": defaultdict(int),
    }

    for record in tqdm(
        cache["records"], ncols=80, desc="V6-B geometry ceilings"
    ):
        name = _stage_name(record)
        stage = record["stages"][name]
        if not isinstance(
            stage.get("selection_slot_pred_x_rows"), torch.Tensor
        ):
            raise ValueError(
                "checkpoint/cache has no V6-B refined slot geometry"
            )
        methods, pairs, diagnostics = build_counterfactual_geometries(
            stage,
            record["target"],
            input_h=input_h,
            input_w=input_w,
            line_width=float(args.line_width),
            min_valid_rows=int(args.min_valid_rows),
            match_min_quality=float(args.match_min_quality),
            delta_bound_px=float(args.delta_bound_px),
        )
        matrices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for method_name, (x_rows, ranges) in methods.items():
            matrix, candidate_valid = _synthetic_official_iou(
                record,
                x_rows,
                ranges,
                line_width=float(args.line_width),
                min_valid_rows=int(args.min_valid_rows),
            )
            matrices[method_name] = (matrix, candidate_valid)
            metric_accumulators.setdefault(
                method_name, MetricAccumulator()
            ).update(matrix, candidate_valid, thresholds)

        target_to_official, qualities = _target_to_official_mapping(
            record["target"],
            record["meta"],
            line_width=float(args.line_width),
        )
        mapping_quality.extend(qualities)
        fixed_pairs.update(
            pairs,
            target_to_official,
            matrices["reference"][0],
            matrices["learned_refined"][0],
            thresholds,
        )
        # Retain exact per-record summaries only through weighted scalar
        # expansion. The full raw row list would make the JSON unnecessarily
        # large, while the global threshold fractions are recomputed below.
        reference_x, _ranges, _active = _gather_routed_geometry(stage)
        gt_x = record["target"]["x_rows"].float()
        gt_valid = record["target"]["valid_mask"].bool()
        for slot_index, gt_index, quality in pairs:
            valid = gt_valid[int(gt_index)] & torch.isfinite(gt_x[int(gt_index)])
            residuals.extend(
                (
                    gt_x[int(gt_index), valid]
                    - reference_x[int(slot_index), valid]
                )
                .abs()
                .tolist()
            )
            match_quality.append(float(quality))
        range_row = diagnostics["reference_vs_gt_range_iou"]
        if int(range_row["count"]) > 0:
            # Exact range IoUs are cheap to reconstruct and avoid averaging
            # per-image percentiles.
            reference_range = _gather_routed_geometry(stage)[1]
            y_rows = fixed_y_rows(
                int(reference_x.shape[-1]), input_h, dtype=torch.float32
            )
            for slot_index, gt_index, _quality in pairs:
                low, high = reference_range[int(slot_index)] * float(input_h)
                pred_valid = (y_rows >= low) & (y_rows <= high)
                valid = gt_valid[int(gt_index)]
                intersection = int((pred_valid & valid).sum())
                union = int((pred_valid | valid).sum())
                range_ious.append(float(intersection) / float(max(union, 1)))
        active_slots += int(diagnostics["active_slots"])
        matched_slots += int(diagnostics["matched_slots"])

        proposal_matrix = stage["official_iou"].float()
        proposal_valid = stage["official_candidate_valid"].bool()
        _oracle_update(
            oracle_accumulator,
            proposal_matrix,
            proposal_valid,
            int(diagnostics["active_slots"]),
            thresholds,
        )

    method_summary = {
        name: accumulator.summary(thresholds)
        for name, accumulator in metric_accumulators.items()
    }
    residual_array = np.asarray(residuals, dtype=np.float64)
    payload = {
        "experiment": "V6-B zero-training slot refinement ceiling audit",
        "diagnostic_only": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "cache": cache["metadata"].get("cache_path", ""),
        "protocol": {
            "split": str(args.split),
            "images": len(cache["records"]),
            "sample_strategy": str(args.sample_strategy),
            "sampled_dataset_indices": cache["metadata"].get(
                "sampled_dataset_indices", []
            ),
            "official_raster_iou": True,
            "line_width": float(args.line_width),
            "thresholds": list(thresholds),
            "match_min_quality": float(args.match_min_quality),
            "delta_bound_px": float(args.delta_bound_px),
        },
        "route_and_target_contract": {
            "active_slots": active_slots,
            "matched_slots": matched_slots,
            "matched_fraction_of_active": float(matched_slots)
            / float(max(active_slots, 1)),
            "target_to_official_mapping_iou": _value_summary(
                mapping_quality
            ),
            "reference_match_quality": _value_summary(match_quality),
            "reference_vs_gt_range_iou": _value_summary(range_ious),
            "absolute_target_residual_px": _value_summary(residuals),
            "target_residual_fraction_over": {
                "6px": float(np.mean(residual_array > 6.0))
                if residual_array.size
                else 0.0,
                "12px": float(np.mean(residual_array > 12.0))
                if residual_array.size
                else 0.0,
                f"{float(args.delta_bound_px):g}px": float(
                    np.mean(residual_array > float(args.delta_bound_px))
                )
                if residual_array.size
                else 0.0,
            },
        },
        "fixed_route_learned_transition": fixed_pairs.summary(),
        "methods": method_summary,
        "proposal_capacity": _oracle_summary(
            oracle_accumulator, thresholds
        ),
        "decision": _recommendation(method_summary),
        "warning": (
            "GT-range, bounded-x, second-bounded-x, and exact-x methods are "
            "oracle counterfactuals used only to localize the remaining "
            "computation-graph bottleneck."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
