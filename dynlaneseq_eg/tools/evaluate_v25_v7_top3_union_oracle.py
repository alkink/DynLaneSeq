from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import (
    eval_predictions,
    interp,
    load_culane_img_data,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.modeling.v25_dual_energy_multi_path import diverse_viterbi_paths
from dynlaneseq_eg.tools.evaluate_v25_g1b_multi_path_capacity import (
    _model_lane_to_original,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_THRESHOLDS = (0.50, 0.75)
FIXED_LINE_WIDTH = 30
FIXED_IMAGE_SHAPE = (590, 1640)
FIXED_MINIMUM_SELECTOR_HEADROOM_POINTS = 0.80
FIXED_MEANINGFUL_HEADROOM_POINTS = 1.50
FIXED_STRONG_ABSOLUTE_F1_50_POINTS = 83.0
PATH_LABELS = ("v7", "g0_path0", "g0_path1", "g0_path2", "dustbin")


Lane = list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the training-free immutable-path union capacity of exact "
            "V7 plus the three diverse G0 coherent paths on all official CULane "
            "validation images."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--v7-prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--oracle-workers", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(
    path: str,
    root: Path,
    *,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(path))
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(root / "list/val.txt")
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _bottom_x(lane: Lane) -> float:
    return float(max(lane, key=lambda point: point[1])[0]) if lane else float("inf")


def _quantize_writer_lane(lane: Lane) -> Lane:
    """Apply the writer's exact three-decimal serialization before scoring."""

    return [(float(f"{x:.3f}"), float(f"{y:.3f}")) for x, y in lane]


def _prediction_path(root: Path, meta: dict[str, Any]) -> Path:
    image_path = Path(str(meta["image_path"]))
    return root / Path(*image_path.parts[-3:]).with_suffix(".lines.txt")


def _raster_lane_crop_official(
    lane: Lane,
    *,
    image_h: int = FIXED_IMAGE_SHAPE[0],
    image_w: int = FIXED_IMAGE_SHAPE[1],
    width: int = FIXED_LINE_WIDTH,
) -> tuple[np.ndarray, int, int]:
    """Tight-crop equivalent of the official evaluator's integer rasterizer."""

    points = interp(lane, n=5)
    if len(points) < 2:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0
    # ``draw_lane`` in culane_metric truncates rather than rounds.
    points = points.astype(np.int32)
    margin = int(width) + 2
    left = max(int(points[:, 0].min()) - margin, 0)
    right = min(int(points[:, 0].max()) + margin + 1, int(image_w))
    top = max(int(points[:, 1].min()) - margin, 0)
    bottom = min(int(points[:, 1].max()) + margin + 1, int(image_h))
    if right <= left or bottom <= top:
        return np.zeros((0, 0), dtype=np.uint8), top, left
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    shifted = points - np.asarray([left, top], dtype=np.int32)
    for first, second in zip(shifted[:-1], shifted[1:]):
        cv2.line(
            mask,
            tuple(first),
            tuple(second),
            color=1,
            thickness=int(width),
        )
    return mask, top, left


def _crop_intersection(
    first: tuple[np.ndarray, int, int],
    second: tuple[np.ndarray, int, int],
) -> int:
    first_mask, first_top, first_left = first
    second_mask, second_top, second_left = second
    if first_mask.size == 0 or second_mask.size == 0:
        return 0
    top = max(first_top, second_top)
    left = max(first_left, second_left)
    bottom = min(first_top + first_mask.shape[0], second_top + second_mask.shape[0])
    right = min(first_left + first_mask.shape[1], second_left + second_mask.shape[1])
    if bottom <= top or right <= left:
        return 0
    first_view = first_mask[
        top - first_top : bottom - first_top,
        left - first_left : right - first_left,
    ]
    second_view = second_mask[
        top - second_top : bottom - second_top,
        left - second_left : right - second_left,
    ]
    return int(cv2.countNonZero(cv2.bitwise_and(first_view, second_view)))


def official_iou_matrix(predictions: Sequence[Lane], targets: Sequence[Lane]) -> np.ndarray:
    if not predictions or not targets:
        return np.zeros((len(predictions), len(targets)), dtype=np.float32)
    pred_crops = [_raster_lane_crop_official(lane) for lane in predictions]
    target_crops = [_raster_lane_crop_official(lane) for lane in targets]
    pred_areas = [int(cv2.countNonZero(crop[0])) for crop in pred_crops]
    target_areas = [int(cv2.countNonZero(crop[0])) for crop in target_crops]
    matrix = np.zeros((len(predictions), len(targets)), dtype=np.float32)
    for pred_index, pred_crop in enumerate(pred_crops):
        for target_index, target_crop in enumerate(target_crops):
            intersection = _crop_intersection(pred_crop, target_crop)
            union = pred_areas[pred_index] + target_areas[target_index] - intersection
            matrix[pred_index, target_index] = (
                0.0 if union <= 0 else float(intersection) / float(union)
            )
    return matrix


def _official_assignment_summary(iou: np.ndarray) -> tuple[int, int, float]:
    if iou.shape[0] == 0 or iou.shape[1] == 0:
        return 0, 0, 0.0
    pred_ids, target_ids = linear_sum_assignment(1.0 - iou)
    matched = iou[pred_ids, target_ids]
    return (
        int((matched > FIXED_THRESHOLDS[0]).sum()),
        int((matched > FIXED_THRESHOLDS[1]).sum()),
        float(matched.sum()),
    )


def _oracle_objective(
    summary: tuple[int, int, float],
    *,
    edit_count: int,
    rank_cost: int,
) -> tuple[int, int, int, float, int]:
    # Official threshold crossings dominate.  Among equal TP outcomes, retain
    # exact V7 rather than making threshold-neutral edits.  IoU only resolves
    # choices with the same minimum number of edits.
    tp50, tp75, matched_iou = summary
    return tp50, tp75, -int(edit_count), float(matched_iou), -int(rank_cost)


def select_slot_union_oracle(
    banks: Sequence[Sequence[tuple[str, Lane | None]]],
    targets: Sequence[Lane],
    *,
    fixed_v7_count: bool,
) -> dict[str, Any]:
    """Exact source-on-tie oracle over immutable per-slot path members."""

    if len(banks) != 4:
        raise ValueError("union oracle requires four canonical slot banks")
    source_active = [bank[0][1] is not None and bank[0][0] == "v7" for bank in banks]
    flattened: list[Lane] = []
    flat_index: dict[tuple[int, int], int] = {}
    choice_sets: list[list[int]] = []
    for slot, bank in enumerate(banks):
        allowed: list[int] = []
        for option, (_label, lane) in enumerate(bank):
            if fixed_v7_count:
                if source_active[slot] and lane is None:
                    continue
                if not source_active[slot] and lane is not None:
                    continue
            if lane is not None:
                flat_index[(slot, option)] = len(flattened)
                flattened.append(lane)
            allowed.append(option)
        if not allowed:
            raise RuntimeError(f"slot {slot} has no legal union-oracle choice")
        choice_sets.append(allowed)
    all_iou = official_iou_matrix(flattened, targets)
    best: tuple[int, int, int, float, int] | None = None
    best_choices: tuple[int, ...] | None = None
    best_summary = (0, 0, 0.0)
    best_lanes: list[Lane] = []
    combinations = 0
    for choices in product(*choice_sets):
        combinations += 1
        selected_rows: list[int] = []
        selected_lanes: list[Lane] = []
        edit_count = 0
        rank_cost = 0
        for slot, option in enumerate(choices):
            label, lane = banks[slot][option]
            baseline = "v7" if source_active[slot] else "dustbin"
            edit_count += int(label != baseline)
            if label.startswith("g0_path"):
                rank_cost += 1 + int(label[-1])
            if lane is not None:
                selected_rows.append(flat_index[(slot, option)])
                selected_lanes.append(lane)
        subset = (
            all_iou[np.asarray(selected_rows, dtype=np.int64)]
            if selected_rows
            else np.zeros((0, len(targets)), dtype=np.float32)
        )
        summary = _official_assignment_summary(subset)
        objective = _oracle_objective(
            summary,
            edit_count=edit_count,
            rank_cost=rank_cost,
        )
        if best is None or objective > best:
            best = objective
            best_choices = tuple(int(value) for value in choices)
            best_summary = summary
            best_lanes = selected_lanes
    if best_choices is None:
        raise RuntimeError("union oracle produced no complete set")
    return {
        "choices": best_choices,
        "labels": tuple(banks[slot][choice][0] for slot, choice in enumerate(best_choices)),
        "lanes": best_lanes,
        "tp50": best_summary[0],
        "tp75": best_summary[1],
        "matched_iou": best_summary[2],
        "edit_count": -int(best[2]),
        "combinations": combinations,
    }


def _write_lane_file(path: Path, lanes: Iterable[Lane]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for lane in lanes:
            values: list[float] = []
            for x_value, y_value in lane:
                values.extend((x_value, y_value))
            handle.write(" ".join(f"{value:.3f}" for value in values) + "\n")


def _tree_sha256(root: Path, rel_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for relative in rel_paths:
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _delta_histogram(source: tuple[int, int], result: dict[str, Any]) -> dict[str, int]:
    return {
        "tp50": int(result["tp50"]) - int(source[0]),
        "tp75": int(result["tp75"]) - int(source[1]),
    }


def _score_union_task(
    task: tuple[
        list[list[tuple[str, Lane | None]]],
        list[Lane],
        list[Lane],
    ],
) -> tuple[tuple[int, int, float], dict[str, Any], dict[str, Any]]:
    """CPU-only exact set scoring, safe to execute in spawned workers."""

    banks, target_lanes, v7_lanes = task
    source_summary = _official_assignment_summary(
        official_iou_matrix(v7_lanes, target_lanes)
    )
    fixed = select_slot_union_oracle(
        banks,
        target_lanes,
        fixed_v7_count=True,
    )
    flexible = select_slot_union_oracle(
        banks,
        target_lanes,
        fixed_v7_count=False,
    )
    return source_summary, fixed, flexible


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    cfg = _configured(
        args.config,
        root,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    loader = build_dataloader(cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(loader.dataset) != expected:
        raise ValueError("union oracle altered official validation population")
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    device = torch.device(args.device)
    model.requires_grad_(False).eval().to(device)
    channels_last = bool(cfg.get("training", {}).get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)

    v7_root = Path(args.v7_prediction_dir).expanduser().resolve()
    if not v7_root.is_dir():
        raise FileNotFoundError(v7_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    fixed_dir = output_dir / "predictions" / "fixed_v7_count_union_oracle"
    flexible_dir = output_dir / "predictions" / "slot_dustbin_union_oracle"

    source_selection = {label: 0 for label in PATH_LABELS}
    fixed_selection = {label: 0 for label in PATH_LABELS}
    flexible_selection = {label: 0 for label in PATH_LABELS}
    image_outcomes = {
        "fixed": {"tp50_improved": 0, "tp50_tied": 0, "tp50_worsened": 0,
                  "tp75_improved": 0, "tp75_tied": 0, "tp75_worsened": 0,
                  "images_edited": 0},
        "flexible": {"tp50_improved": 0, "tp50_tied": 0, "tp50_worsened": 0,
                     "tp75_improved": 0, "tp75_tied": 0, "tp75_worsened": 0,
                     "images_edited": 0},
    }
    changed_records: list[dict[str, Any]] = []
    rel_paths: list[Path] = []
    images_seen = 0
    combinations_fixed = 0
    combinations_flexible = 0
    started = time.perf_counter()
    worker_count = max(1, int(args.oracle_workers))
    oracle_pool = (
        mp.get_context("spawn").Pool(processes=worker_count)
        if worker_count > 1
        else None
    )
    try:
        for batch_index, (images, _targets, metas) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=False):
                source = model(images.float())
            diverse = diverse_viterbi_paths(
                source["unary_logits"],
                num_hypotheses=3,
                transition_radius_bins=model.detector.transition_radius_bins,
                transition_penalty=model.detector.transition_penalty,
                suppression_radius_bins=5,
                suppression_penalty=8.0,
            )
            bin_width = float(model.detector.input_w) / float(model.detector.x_bins)
            hypotheses = (diverse.indices.float() + 0.5) * bin_width

            prepared: list[dict[str, Any]] = []
            tasks = []
            for image_index, meta in enumerate(metas):
                v7_path = _prediction_path(v7_root, meta)
                if not v7_path.is_file():
                    raise FileNotFoundError(v7_path)
                relative = v7_path.relative_to(v7_root)
                rel_paths.append(relative)
                v7_lanes = sorted(load_culane_img_data(v7_path), key=_bottom_x)[:4]
                target_lanes = load_culane_img_data(meta["anno_path"])
                banks: list[list[tuple[str, Lane | None]]] = []
                for slot in range(4):
                    v7_lane = v7_lanes[slot] if slot < len(v7_lanes) else None
                    bank: list[tuple[str, Lane | None]] = [("v7", v7_lane)]
                    for hypothesis in range(3):
                        lane = _model_lane_to_original(
                            hypotheses[image_index, slot, hypothesis],
                            source["range_norm"][image_index, slot],
                            meta,
                        )
                        bank.append(
                            (
                                f"g0_path{hypothesis}",
                                _quantize_writer_lane(lane) if len(lane) >= 2 else None,
                            )
                        )
                    bank.append(("dustbin", None))
                    banks.append(bank)
                prepared.append(
                    {
                        "relative": relative,
                        "v7_lanes": v7_lanes,
                    }
                )
                tasks.append((banks, target_lanes, v7_lanes))
            scored = (
                oracle_pool.map(_score_union_task, tasks, chunksize=1)
                if oracle_pool is not None
                else [_score_union_task(task) for task in tasks]
            )

            for record, (source_summary, fixed, flexible) in zip(prepared, scored):
                relative = record["relative"]
                v7_lanes = record["v7_lanes"]
                combinations_fixed += int(fixed["combinations"])
                combinations_flexible += int(flexible["combinations"])
                _write_lane_file(fixed_dir / relative, fixed["lanes"])
                _write_lane_file(flexible_dir / relative, flexible["lanes"])

                for slot in range(4):
                    source_selection[
                        "v7" if slot < len(v7_lanes) else "dustbin"
                    ] += 1
                for result, histogram, outcome_name in (
                    (fixed, fixed_selection, "fixed"),
                    (flexible, flexible_selection, "flexible"),
                ):
                    for label in result["labels"]:
                        histogram[label] += 1
                    image_outcomes[outcome_name]["images_edited"] += int(
                        result["edit_count"] > 0
                    )
                    for key, source_value, result_value in (
                        ("tp50", source_summary[0], result["tp50"]),
                        ("tp75", source_summary[1], result["tp75"]),
                    ):
                        suffix = (
                            "improved" if result_value > source_value
                            else "worsened" if result_value < source_value
                            else "tied"
                        )
                        image_outcomes[outcome_name][f"{key}_{suffix}"] += 1
                if fixed["edit_count"] > 0 or flexible["edit_count"] > 0:
                    changed_records.append(
                        {
                            "image": str(relative).replace(".lines.txt", ".jpg"),
                            "source_tp50": source_summary[0],
                            "source_tp75": source_summary[1],
                            "fixed_labels": fixed["labels"],
                            "fixed_delta": _delta_histogram(source_summary[:2], fixed),
                            "flexible_labels": flexible["labels"],
                            "flexible_delta": _delta_histogram(
                                source_summary[:2], flexible
                            ),
                        }
                    )
            images_seen += int(images.shape[0])
            if batch_index == 1 or (
                args.log_interval > 0 and batch_index % args.log_interval == 0
            ):
                elapsed = max(time.perf_counter() - started, 1.0e-9)
                print(
                    json.dumps(
                        {
                            "phase": "v25_v7_top3_union_oracle",
                            "images": images_seen,
                            "images_per_second": images_seen / elapsed,
                            "changed_records": len(changed_records),
                            "oracle_workers": worker_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        if oracle_pool is not None:
            oracle_pool.close()
            oracle_pool.join()
    if images_seen != expected or len(rel_paths) != expected:
        raise RuntimeError("union oracle did not consume full official validation")

    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        "source_v7_exact": eval_predictions(
            pred_dir=v7_root,
            anno_dir=root,
            list_path=root / "list/val.txt",
            iou_thresholds=FIXED_THRESHOLDS,
            width=FIXED_LINE_WIDTH,
            official=True,
            sequential=False,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        ),
        "fixed_v7_count_union_oracle": eval_predictions(
            pred_dir=fixed_dir,
            anno_dir=root,
            list_path=root / "list/val.txt",
            iou_thresholds=FIXED_THRESHOLDS,
            width=FIXED_LINE_WIDTH,
            official=True,
            sequential=False,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        ),
        "slot_dustbin_union_oracle": eval_predictions(
            pred_dir=flexible_dir,
            anno_dir=root,
            list_path=root / "list/val.txt",
            iou_thresholds=FIXED_THRESHOLDS,
            width=FIXED_LINE_WIDTH,
            official=True,
            sequential=False,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        ),
    }
    metrics = {
        name: {str(key): value for key, value in result.items()}
        for name, result in metrics.items()
    }
    source50 = float(metrics["source_v7_exact"]["0.5"]["F1"])
    fixed50 = float(metrics["fixed_v7_count_union_oracle"]["0.5"]["F1"])
    flexible50 = float(metrics["slot_dustbin_union_oracle"]["0.5"]["F1"])
    fixed_gain = 100.0 * (fixed50 - source50)
    flexible_gain = 100.0 * (flexible50 - source50)
    checks = {
        "source_exact_expected_population": len(rel_paths) == expected,
        "fixed_oracle_never_worsened_image_tp50": image_outcomes["fixed"]["tp50_worsened"] == 0,
        "fixed_official_tp50_not_lower_than_source": int(
            metrics["fixed_v7_count_union_oracle"]["0.5"]["TP"]
        ) >= int(metrics["source_v7_exact"]["0.5"]["TP"]),
        "fixed_activity_count_exact_v7": (
            sum(value for label, value in fixed_selection.items() if label != "dustbin")
            == source_selection["v7"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("immutable union-oracle guard failed: " + json.dumps(checks))
    report = {
        "experiment": "V25 exact-V7 plus G0-top3 immutable path union oracle",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": iteration,
        "v7_prediction_dir": str(v7_root),
        "v7_prediction_tree_sha256": _tree_sha256(v7_root, rel_paths),
        "official_validation_population_contract": population,
        "metrics": metrics,
        "selection_histograms": {
            "source_v7": source_selection,
            "fixed_v7_count_union_oracle": fixed_selection,
            "slot_dustbin_union_oracle": flexible_selection,
        },
        "per_image_outcomes": image_outcomes,
        "changed_images": changed_records,
        "search_contract": {
            "fixed_combinations_evaluated": combinations_fixed,
            "flexible_combinations_evaluated": combinations_flexible,
            "members_per_slot": list(PATH_LABELS),
            "primary_activity_count": "exact V7",
            "geometry_mutation": False,
            "coordinate_blending": False,
            "objective_order": [
                "TP@.50", "TP@.75", "minimum edit/source-on-tie", "matched IoU", "lower path rank"
            ],
            "official_assignment": "SciPy Hungarian on exact discrete raster IoU",
        },
        "gate": {
            "checks": checks,
            "fixed_v7_count_gain_f1_50_points": fixed_gain,
            "flexible_gain_f1_50_points": flexible_gain,
            "selector_training_authorized": fixed_gain
            >= FIXED_MINIMUM_SELECTOR_HEADROOM_POINTS,
            "meaningful_bank_headroom": fixed_gain
            >= FIXED_MEANINGFUL_HEADROOM_POINTS,
            "strong_bank_capacity": 100.0 * fixed50
            >= FIXED_STRONG_ABSOLUTE_F1_50_POINTS,
            "predeclared_thresholds_points": {
                "minimum_selector_authorization": FIXED_MINIMUM_SELECTOR_HEADROOM_POINTS,
                "meaningful": FIXED_MEANINGFUL_HEADROOM_POINTS,
                "strong_absolute_f1_50": FIXED_STRONG_ABSOLUTE_F1_50_POINTS,
            },
        },
        "contract": {
            "training_performed": False,
            "oracle_is_deployable": False,
            "gt_used_for_selection": True,
            "official_val_rows": expected,
            "validation_subset_used": False,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "test_set_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v7_top3_union_oracle_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    headline = {
        "report": str(report_path),
        "source_f1_50_points": 100.0 * source50,
        "fixed_union_f1_50_points": 100.0 * fixed50,
        "fixed_gain_points": fixed_gain,
        "flexible_union_f1_50_points": 100.0 * flexible50,
        "flexible_gain_points": flexible_gain,
        "selector_training_authorized": report["gate"]["selector_training_authorized"],
        "meaningful_bank_headroom": report["gate"]["meaningful_bank_headroom"],
        "strong_bank_capacity": report["gate"]["strong_bank_capacity"],
    }
    print(json.dumps(headline, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
