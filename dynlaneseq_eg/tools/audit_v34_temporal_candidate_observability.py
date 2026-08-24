from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    candidate_row_masks,
    ensure_official_iou_cache,
    load_or_collect_cache,
)
from dynlaneseq_eg.modeling.common import fixed_y_rows


AUDIT_VERSION = 1


@dataclass(frozen=True)
class TemporalSample:
    fold: str
    clip: str
    target: str
    previous: str
    following: str
    wrong_context: str


@dataclass(frozen=True)
class WarpResult:
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    forward_backward_error: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether adjacent CULane frames distinguish an oracle-good "
            "V7 proposal from the currently selected wrong proposal."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--val-list", default="list/val.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-fold", type=int, default=512)
    parser.add_argument("--frame-step", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--metric-workers", type=int, default=16)
    parser.add_argument("--flow-workers", type=int, default=8)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--fb-threshold", type=float, default=3.0)
    parser.add_argument("--min-warp-rows", type=int, default=20)
    parser.add_argument("--line-width", type=float, default=30.0)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def _normalize_image_path(value: str) -> str:
    return Path(value.strip().split()[0].lstrip("/")).as_posix()


def _clip_id(image: str) -> str:
    return Path(_normalize_image_path(image)).parent.as_posix()


def _frame_index(image: str) -> int:
    stem = Path(_normalize_image_path(image)).stem
    if not stem.isdigit():
        raise ValueError(f"frame filename is not numeric: {image}")
    return int(stem)


def _frame_path(image: str, frame: int) -> str:
    path = Path(_normalize_image_path(image))
    return (path.parent / f"{int(frame):05d}{path.suffix}").as_posix()


def _balanced_clip_split(clips: Iterable[str], seed: int) -> dict[str, str]:
    def key(clip: str) -> str:
        return hashlib.sha256(f"{int(seed)}:{clip}".encode("utf-8")).hexdigest()

    ordered = sorted(set(clips), key=lambda item: (key(item), item))
    return {clip: ("a" if index % 2 == 0 else "b") for index, clip in enumerate(ordered)}


def _uniform_take(rows: list[str], count: int) -> list[str]:
    if count <= 0 or len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    return [rows[int(index)] for index in indices.tolist()]


def build_temporal_manifest(
    val_list: str | Path,
    *,
    sample_per_fold: int,
    frame_step: int,
    seed: int,
) -> dict[str, Any]:
    rows = [
        _normalize_image_path(line)
        for line in Path(val_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(set(rows)):
        raise ValueError("validation list contains duplicate image paths")
    row_set = set(rows)
    split = _balanced_clip_split((_clip_id(row) for row in rows), seed)
    eligible: dict[str, list[str]] = {"a": [], "b": []}
    for row in rows:
        frame = _frame_index(row)
        previous = _frame_path(row, frame - int(frame_step))
        following = _frame_path(row, frame + int(frame_step))
        if previous in row_set and following in row_set:
            eligible[split[_clip_id(row)]].append(row)

    selected = {
        fold: _uniform_take(fold_rows, int(sample_per_fold))
        for fold, fold_rows in eligible.items()
    }
    samples: list[TemporalSample] = []
    for fold in ("a", "b"):
        fold_rows = selected[fold]
        if not fold_rows:
            raise ValueError(f"fold {fold} has no eligible temporal samples")
        for index, target in enumerate(fold_rows):
            wrong = None
            for offset in range(1, len(fold_rows) + 1):
                candidate = fold_rows[(index + offset) % len(fold_rows)]
                if _clip_id(candidate) != _clip_id(target):
                    wrong = candidate
                    break
            if wrong is None:
                raise ValueError(f"fold {fold} does not contain two distinct clips")
            frame = _frame_index(target)
            samples.append(
                TemporalSample(
                    fold=fold,
                    clip=_clip_id(target),
                    target=target,
                    previous=_frame_path(target, frame - int(frame_step)),
                    following=_frame_path(target, frame + int(frame_step)),
                    wrong_context=wrong,
                )
            )

    union = sorted(
        {
            path
            for sample in samples
            for path in (
                sample.target,
                sample.previous,
                sample.following,
                sample.wrong_context,
            )
        }
    )
    sampled_clip_counts = {
        fold: len({sample.clip for sample in samples if sample.fold == fold})
        for fold in ("a", "b")
    }
    assigned_clip_counts = {
        fold: sum(value == fold for value in split.values()) for fold in ("a", "b")
    }
    return {
        "audit_version": AUDIT_VERSION,
        "seed": int(seed),
        "frame_step": int(frame_step),
        "validation_rows": len(rows),
        "validation_clips": len(set(split)),
        "assigned_clip_counts": assigned_clip_counts,
        "sampled_clip_counts": sampled_clip_counts,
        "eligible_by_fold": {key: len(value) for key, value in eligible.items()},
        "selected_by_fold": {key: len(value) for key, value in selected.items()},
        "union_images": union,
        "samples": [sample.__dict__ for sample in samples],
    }


def bilinear_sample_flow(
    flow: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (y >= 0.0)
        & (x <= float(width - 1))
        & (y <= float(height - 1))
    )
    clipped_x = np.clip(x, 0.0, float(width - 1))
    clipped_y = np.clip(y, 0.0, float(height - 1))
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (clipped_x - x0).astype(np.float32)
    wy = (clipped_y - y0).astype(np.float32)
    sampled = (
        flow[y0, x0] * ((1.0 - wx) * (1.0 - wy))[..., None]
        + flow[y0, x1] * (wx * (1.0 - wy))[..., None]
        + flow[y1, x0] * ((1.0 - wx) * wy)[..., None]
        + flow[y1, x1] * (wx * wy)[..., None]
    )
    sampled[~valid] = 0.0
    return sampled.astype(np.float32), valid


def warp_curve(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    row_mask: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    input_w: int,
    input_h: int,
    fb_threshold: float,
) -> WarpResult:
    flow_h, flow_w = forward_flow.shape[:2]
    scale_x = float(flow_w) / float(input_w)
    scale_y = float(flow_h) / float(input_h)
    x0 = np.asarray(x_rows, dtype=np.float32) * scale_x
    y0 = np.asarray(y_rows, dtype=np.float32) * scale_y
    forward, valid0 = bilinear_sample_flow(forward_flow, x0, y0)
    x1 = x0 + forward[:, 0]
    y1 = y0 + forward[:, 1]
    backward, valid1 = bilinear_sample_flow(backward_flow, x1, y1)
    fb_error = np.linalg.norm(forward + backward, axis=-1)
    valid = (
        np.asarray(row_mask, dtype=bool)
        & valid0
        & valid1
        & np.isfinite(fb_error)
        & (fb_error <= float(fb_threshold))
    )
    return WarpResult(
        x=(x1 / scale_x).astype(np.float32),
        y=(y1 / scale_y).astype(np.float32),
        valid=valid,
        forward_backward_error=fb_error.astype(np.float32),
    )


def identity_warp(
    x_rows: np.ndarray, y_rows: np.ndarray, row_mask: np.ndarray
) -> WarpResult:
    valid = np.asarray(row_mask, dtype=bool) & np.isfinite(x_rows) & np.isfinite(y_rows)
    return WarpResult(
        x=np.asarray(x_rows, dtype=np.float32).copy(),
        y=np.asarray(y_rows, dtype=np.float32).copy(),
        valid=valid,
        forward_backward_error=np.zeros_like(np.asarray(y_rows, dtype=np.float32)),
    )


def curve_soft_iou(
    warped: WarpResult,
    neighbor_x: np.ndarray,
    neighbor_y: np.ndarray,
    neighbor_mask: np.ndarray,
    *,
    line_width: float,
    min_rows: int,
) -> float:
    source_ids = np.flatnonzero(warped.valid)
    neighbor_ids = np.flatnonzero(np.asarray(neighbor_mask, dtype=bool))
    if source_ids.size < int(min_rows) or neighbor_ids.size < 2:
        return float("nan")
    ny = np.asarray(neighbor_y, dtype=np.float32)[neighbor_ids]
    nx = np.asarray(neighbor_x, dtype=np.float32)[neighbor_ids]
    order = np.argsort(ny)
    ny = ny[order]
    nx = nx[order]
    query_y = warped.y[source_ids]
    inside = (query_y >= float(ny[0])) & (query_y <= float(ny[-1]))
    neighbor_at_y = np.interp(query_y, ny, nx).astype(np.float32)
    dx = np.abs(warped.x[source_ids] - neighbor_at_y)
    overlap = np.maximum(0.0, float(line_width) - dx)
    overlap[~inside] = 0.0
    union = np.where(
        inside,
        2.0 * float(line_width) - overlap,
        float(line_width),
    )
    denominator = float(union.sum())
    return 0.0 if denominator <= 0.0 else float(overlap.sum() / denominator)


def candidate_support_score(
    warped: WarpResult,
    neighbor_x: np.ndarray,
    neighbor_masks: np.ndarray,
    neighbor_y: np.ndarray,
    *,
    line_width: float,
    min_rows: int,
) -> float:
    scores = [
        curve_soft_iou(
            warped,
            neighbor_x[index],
            neighbor_y,
            neighbor_masks[index],
            line_width=line_width,
            min_rows=min_rows,
        )
        for index in range(int(neighbor_x.shape[0]))
    ]
    finite = [score for score in scores if math.isfinite(score)]
    return max(finite) if finite else float("nan")


def _slot_gt_pairs(stage: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    official = stage["selection_slot_official_iou"].float()
    active = stage["selection_slot_active"].bool()
    valid = stage.get("selection_slot_official_candidate_valid", active).bool() & active
    slot_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    if official.shape[0] == 0 or not slot_ids:
        return []
    selected = official[:, slot_ids].detach().cpu().numpy()
    gt_ids, local_slot_ids = linear_sum_assignment(1.0 - selected)
    return [
        (int(gt_id), int(slot_ids[local_slot_id]))
        for gt_id, local_slot_id in zip(gt_ids.tolist(), local_slot_ids.tolist())
    ]


def _record_key(record: dict[str, Any], dataset_root: Path) -> str:
    image_path = Path(str(record.get("meta", {}).get("image_path", record.get("image_id", ""))))
    try:
        return image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return Path(*image_path.parts[-3:]).as_posix()


def _stage_banks(
    stage: dict[str, torch.Tensor], input_h: int, input_w: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw_x, raw_masks, raw_valid = candidate_row_masks(
        stage, input_h=input_h, input_w=input_w, min_valid_rows=5
    )
    raw_masks &= raw_valid[:, None]
    slot_x = stage["selection_slot_pred_x_rows"].float()
    rows = int(slot_x.shape[-1])
    y_rows = fixed_y_rows(rows, input_h, dtype=torch.float32)
    ranges = stage["selection_slot_range_norm"].float()
    lower = torch.minimum(ranges[:, 0], ranges[:, 1])[:, None] * float(input_h)
    upper = torch.maximum(ranges[:, 0], ranges[:, 1])[:, None] * float(input_h)
    slot_masks = (y_rows[None] >= lower) & (y_rows[None] <= upper)
    slot_masks &= stage["selection_slot_active"].bool()[:, None]
    slot_valid = stage.get(
        "selection_slot_official_candidate_valid",
        torch.ones(slot_x.shape[0], dtype=torch.bool),
    ).bool()
    slot_masks &= slot_valid[:, None]
    return {
        "selected": (slot_x.numpy(), slot_masks.numpy()),
        "bank": (raw_x.numpy(), raw_masks.numpy()),
    }


def _read_flow_image(record: dict[str, Any], flow_scale: float) -> np.ndarray:
    meta = record["meta"]
    image = cv2.imread(str(meta["image_path"]), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {meta['image_path']}")
    input_h = int(meta["input_h"])
    input_w = int(meta["input_w"])
    crop_x = int(round(float(meta.get("crop_x", 0.0))))
    crop_y = int(round(float(meta.get("crop_y", 0.0))))
    crop_w = int(round(float(input_w) / float(meta.get("scale_x", 1.0))))
    crop_h = int(round(float(input_h) / float(meta.get("scale_y", 1.0))))
    crop = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    width = max(2, int(round(float(input_w) * float(flow_scale))))
    height = max(2, int(round(float(input_h) * float(flow_scale))))
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)


def _dense_flow(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return estimator.calc(source, target, None)


def _candidate_cases(
    sample: TemporalSample,
    record: dict[str, Any],
    *,
    input_h: int,
    input_w: int,
) -> list[dict[str, Any]]:
    stage = record["stages"]["main"]
    official = stage["official_iou"].float()
    valid = stage["official_candidate_valid"].bool()
    valid &= stage.get("selection_slot_candidate_valid", valid).bool()
    raw_x, raw_masks, _ = candidate_row_masks(
        stage, input_h=input_h, input_w=input_w, min_valid_rows=5
    )
    source_routes = stage["selection_slot_indices"].long()
    source_refined = stage["selection_slot_official_iou"].float()
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    cases: list[dict[str, Any]] = []
    for gt_id, slot_id in _slot_gt_pairs(stage):
        if not valid_ids:
            continue
        source = int(source_routes[slot_id])
        if source not in valid_ids:
            continue
        oracle = max(valid_ids, key=lambda idx: (float(official[gt_id, idx]), -idx))
        source_iou = float(official[gt_id, source])
        oracle_iou = float(official[gt_id, oracle])
        thresholds = [
            threshold
            for threshold in (0.5, 0.75)
            if source != oracle and source_iou <= threshold < oracle_iou
        ]
        if not thresholds:
            continue
        cases.append(
            {
                "fold": sample.fold,
                "clip": sample.clip,
                "image_id": sample.target,
                "gt_id": int(gt_id),
                "slot_id": int(slot_id),
                "good_candidate": int(oracle),
                "wrong_candidate": int(source),
                "good_iou": oracle_iou,
                "wrong_iou": source_iou,
                "source_refined_iou": float(source_refined[gt_id, slot_id]),
                "thresholds": thresholds,
                "candidate_x": raw_x[[oracle, source]].numpy(),
                "candidate_masks": raw_masks[[oracle, source]].numpy(),
            }
        )
    return cases


def _score_context(
    cases: list[dict[str, Any]],
    target_image: np.ndarray,
    context_image: np.ndarray,
    context_banks: dict[str, tuple[np.ndarray, np.ndarray]],
    y_rows: np.ndarray,
    *,
    input_w: int,
    input_h: int,
    fb_threshold: float,
    line_width: float,
    min_rows: int,
) -> list[dict[str, Any]]:
    forward = _dense_flow(target_image, context_image)
    backward = _dense_flow(context_image, target_image)
    outputs: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {}
        for local_id, role in enumerate(("good", "wrong")):
            warped = warp_curve(
                case["candidate_x"][local_id],
                y_rows,
                case["candidate_masks"][local_id],
                forward,
                backward,
                input_w=input_w,
                input_h=input_h,
                fb_threshold=fb_threshold,
            )
            identity = identity_warp(
                case["candidate_x"][local_id],
                y_rows,
                case["candidate_masks"][local_id],
            )
            row[f"{role}_reliable_rows"] = int(warped.valid.sum())
            for bank_name, (neighbor_x, neighbor_masks) in context_banks.items():
                row[f"{bank_name}_{role}"] = candidate_support_score(
                    warped,
                    neighbor_x,
                    neighbor_masks,
                    y_rows,
                    line_width=line_width,
                    min_rows=min_rows,
                )
                row[f"identity_{bank_name}_{role}"] = candidate_support_score(
                    identity,
                    neighbor_x,
                    neighbor_masks,
                    y_rows,
                    line_width=line_width,
                    min_rows=min_rows,
                )
        outputs.append(row)
    return outputs


def _mean_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _process_sample(
    sample: TemporalSample,
    records: dict[str, dict[str, Any]],
    *,
    flow_scale: float,
    fb_threshold: float,
    line_width: float,
    min_rows: int,
) -> list[dict[str, Any]]:
    target_record = records[sample.target]
    input_h = int(target_record["meta"]["input_h"])
    input_w = int(target_record["meta"]["input_w"])
    cases = _candidate_cases(
        sample, target_record, input_h=input_h, input_w=input_w
    )
    if not cases:
        return []
    rows = int(cases[0]["candidate_x"].shape[-1])
    y_rows = fixed_y_rows(rows, input_h, dtype=torch.float32).numpy()
    target_image = _read_flow_image(target_record, flow_scale)
    contexts = {
        "previous": sample.previous,
        "following": sample.following,
        "wrong_context": sample.wrong_context,
    }
    context_scores: dict[str, list[dict[str, Any]]] = {}
    for name, image_id in contexts.items():
        context_record = records[image_id]
        context_scores[name] = _score_context(
            cases,
            target_image,
            _read_flow_image(context_record, flow_scale),
            _stage_banks(context_record["stages"]["main"], input_h, input_w),
            y_rows,
            input_w=input_w,
            input_h=input_h,
            fb_threshold=fb_threshold,
            line_width=line_width,
            min_rows=min_rows,
        )
    output: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        row = {key: value for key, value in case.items() if not key.startswith("candidate_")}
        for context_name, scored in context_scores.items():
            for key, value in scored[index].items():
                row[f"{context_name}_{key}"] = value
        for bank in ("selected", "bank"):
            for role in ("good", "wrong"):
                row[f"bidirectional_{bank}_{role}"] = _mean_finite(
                    (
                        row[f"previous_{bank}_{role}"],
                        row[f"following_{bank}_{role}"],
                    )
                )
                row[f"identity_bidirectional_{bank}_{role}"] = _mean_finite(
                    (
                        row[f"previous_identity_{bank}_{role}"],
                        row[f"following_identity_{bank}_{role}"],
                    )
                )
        output.append(row)
    return output


def _roc_auc(rows: list[dict[str, Any]], good_key: str, wrong_key: str) -> float | None:
    pairs = [
        (float(row[good_key]), float(row[wrong_key]))
        for row in rows
        if math.isfinite(float(row.get(good_key, float("nan"))))
        and math.isfinite(float(row.get(wrong_key, float("nan"))))
    ]
    if not pairs:
        return None
    scores = np.asarray([value for pair in pairs for value in pair], dtype=np.float64)
    labels = np.asarray([label for _pair in pairs for label in (1, 0)], dtype=np.int64)
    ranks = rankdata(scores, method="average")
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2.0)
        / float(positive * negative)
    )


def _metric(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    good_key = f"{prefix}_good"
    wrong_key = f"{prefix}_wrong"
    pairs = [
        (float(row[good_key]), float(row[wrong_key]))
        for row in rows
        if math.isfinite(float(row.get(good_key, float("nan"))))
        and math.isfinite(float(row.get(wrong_key, float("nan"))))
    ]
    wins = [float(good > wrong) + 0.5 * float(good == wrong) for good, wrong in pairs]
    return {
        "pairs": len(pairs),
        "pair_accuracy": float(np.mean(wins)) if wins else None,
        "auc": _roc_auc(rows, good_key, wrong_key),
        "mean_good_score": float(np.mean([pair[0] for pair in pairs])) if pairs else None,
        "mean_wrong_score": float(np.mean([pair[1] for pair in pairs])) if pairs else None,
        "mean_margin": float(np.mean([a - b for a, b in pairs])) if pairs else None,
    }


def _bootstrap_metric(
    rows: list[dict[str, Any]],
    prefix: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_clip.setdefault(str(row["clip"]), []).append(row)
    clips = sorted(by_clip)
    if len(clips) < 2 or repetitions <= 0:
        return {"repetitions": 0, "pair_accuracy_ci95": None, "auc_ci95": None}
    rng = np.random.default_rng(int(seed))
    accuracies: list[float] = []
    aucs: list[float] = []
    for _ in range(int(repetitions)):
        chosen = rng.choice(clips, size=len(clips), replace=True)
        sampled = [row for clip in chosen.tolist() for row in by_clip[str(clip)]]
        metric = _metric(sampled, prefix)
        if metric["pair_accuracy"] is not None:
            accuracies.append(float(metric["pair_accuracy"]))
        if metric["auc"] is not None:
            aucs.append(float(metric["auc"]))

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "repetitions": int(repetitions),
        "pair_accuracy_ci95": interval(accuracies),
        "auc_ci95": interval(aucs),
    }


def summarize_pairs(
    rows: list[dict[str, Any]], *, bootstrap_reps: int, seed: int
) -> dict[str, Any]:
    contexts = {
        "previous_selected": "previous_selected",
        "following_selected": "following_selected",
        "bidirectional_selected_primary": "bidirectional_selected",
        "bidirectional_bank": "bidirectional_bank",
        "identity_bidirectional_selected_control": "identity_bidirectional_selected",
        "wrong_clip_selected_control": "wrong_context_selected",
        "wrong_clip_bank_control": "wrong_context_bank",
    }
    results: dict[str, Any] = {}
    for fold in ("a", "b"):
        for threshold in (0.5, 0.75):
            subset = [
                row
                for row in rows
                if row["fold"] == fold and float(threshold) in row["thresholds"]
            ]
            key = f"fold_{fold}_iou_{threshold:.2f}"
            metrics: dict[str, Any] = {}
            for name, prefix in contexts.items():
                metrics[name] = {
                    **_metric(subset, prefix),
                    "clip_bootstrap": _bootstrap_metric(
                        subset,
                        prefix,
                        repetitions=bootstrap_reps,
                        seed=seed + (0 if fold == "a" else 10_000) + int(threshold * 100),
                    ),
                }
            primary = metrics["bidirectional_selected_primary"]
            wrong = metrics["wrong_clip_selected_control"]
            primary_auc = primary["auc"]
            wrong_auc = wrong["auc"]
            metrics["primary_minus_wrong_clip_auc"] = (
                None
                if primary_auc is None or wrong_auc is None
                else float(primary_auc - wrong_auc)
            )
            results[key] = {
                "fold": fold,
                "threshold": threshold,
                "eligible_pairs": len(subset),
                "clips": len({row["clip"] for row in subset}),
                "metrics": metrics,
            }

    checks: dict[str, bool] = {}
    for key, item in results.items():
        primary = item["metrics"]["bidirectional_selected_primary"]
        delta = item["metrics"]["primary_minus_wrong_clip_auc"]
        checks[f"{key}_at_least_30_pairs"] = int(primary["pairs"]) >= 30
        checks[f"{key}_auc_at_least_0p70"] = (
            primary["auc"] is not None and float(primary["auc"]) >= 0.70
        )
        checks[f"{key}_pair_accuracy_at_least_0p65"] = (
            primary["pair_accuracy"] is not None
            and float(primary["pair_accuracy"]) >= 0.65
        )
        checks[f"{key}_correct_context_auc_advantage_at_least_0p10"] = (
            delta is not None and float(delta) >= 0.10
        )
    return {
        "predeclared_primary": "bidirectional_selected",
        "pass_contract": {
            "minimum_pairs_per_fold_threshold": 30,
            "minimum_auc": 0.70,
            "minimum_pair_accuracy": 0.65,
            "minimum_auc_advantage_over_wrong_clip": 0.10,
            "all_checks_required": True,
        },
        "by_fold_threshold": results,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V34 temporal candidate observability audit",
        "",
        "This is a GT-labelled diagnostic audit on the validation split. GT is used only to form good-versus-current-wrong candidate pairs; no GT enters the temporal score.",
        "",
        "## Primary result",
        "",
        f"- Gate passed: `{payload['summary']['passed']}`",
        f"- V7 checkpoint: `{payload['checkpoint']}`",
        f"- Evaluated route-recoverable pairs: `{len(payload['pairs'])}`",
        "- Primary score: bidirectional optical-flow alignment to adjacent-frame deployed V7 lanes.",
        "",
        "| Fold | IoU | Pairs | Pair accuracy | AUC | Wrong-clip AUC | AUC advantage | Bank AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["summary"]["by_fold_threshold"].values():
        metrics = item["metrics"]
        primary = metrics["bidirectional_selected_primary"]
        wrong = metrics["wrong_clip_selected_control"]
        bank = metrics["bidirectional_bank"]

        def show(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            "| "
            f"{item['fold']} | {item['threshold']:.2f} | {primary['pairs']} | "
            f"{show(primary['pair_accuracy'])} | {show(primary['auc'])} | "
            f"{show(wrong['auc'])} | "
            f"{show(metrics['primary_minus_wrong_clip_auc'])} | {show(bank['auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision contract",
            "",
            "A temporal model is justified only if every fold/threshold independently has at least 30 pairs, AUC >= 0.70, pair accuracy >= 0.65, and at least +0.10 AUC over the wrong-clip control.",
            "",
            "## Interpretation",
            "",
            (
                "The gate passed: adjacent frames contain reproducible evidence for choosing the better proposal. A learned temporal selector is justified."
                if payload["summary"]["passed"]
                else "The gate failed: this fixed optical-flow temporal evidence does not reliably distinguish the lucky-tail proposal. Do not build a temporal selector from this signal without a new causal observation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    val_list = Path(args.val_list).expanduser()
    if not val_list.is_absolute():
        val_list = dataset_root / val_list

    manifest = build_temporal_manifest(
        val_list,
        sample_per_fold=args.sample_per_fold,
        frame_step=args.frame_step,
        seed=args.seed,
    )
    manifest_path = output_dir / "temporal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    union_list = output_dir / "temporal_union_val.txt"
    union_list.write_text(
        "\n".join(f"/{path}" for path in manifest["union_images"]) + "\n",
        encoding="utf-8",
    )

    cache = load_or_collect_cache(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split="val",
        list_path=union_list,
        dataset_root=dataset_root,
        device=args.device,
        cache_dir=output_dir / "cache",
        reuse_cache=bool(args.reuse_cache),
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        amp_dtype=args.amp_dtype,
        desc="V34 temporal union cache",
    )
    cache = ensure_official_iou_cache(
        cache,
        line_width=args.line_width,
        workers=args.metric_workers,
    )
    records = {
        _record_key(record, dataset_root): record for record in cache["records"]
    }
    missing = sorted(set(manifest["union_images"]) - set(records))
    if missing:
        raise ValueError(f"cache is missing {len(missing)} manifest images: {missing[:5]}")

    samples = [TemporalSample(**row) for row in manifest["samples"]]
    cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        worker_count = max(1, int(args.flow_workers))

        def process(sample: TemporalSample) -> list[dict[str, Any]]:
            return _process_sample(
                sample,
                records,
                flow_scale=args.flow_scale,
                fb_threshold=args.fb_threshold,
                line_width=args.line_width,
                min_rows=args.min_warp_rows,
            )

        if worker_count == 1:
            nested = [
                process(sample)
                for sample in tqdm(samples, ncols=90, desc="temporal flow audit")
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                nested = list(
                    tqdm(
                        executor.map(process, samples),
                        total=len(samples),
                        ncols=90,
                        desc="temporal flow audit",
                    )
                )
    finally:
        cv2.setNumThreads(cv_threads)

    pairs = [row for group in nested for row in group]
    summary = summarize_pairs(
        pairs, bootstrap_reps=args.bootstrap_reps, seed=args.seed
    )
    serializable_pairs = []
    for row in pairs:
        serializable_pairs.append(
            {
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    payload = {
        "audit_version": AUDIT_VERSION,
        "experiment": "V34 training-free temporal candidate observability",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "cache": cache["metadata"].get("cache_path"),
        "settings": {
            "sample_per_fold": int(args.sample_per_fold),
            "frame_step": int(args.frame_step),
            "flow_scale": float(args.flow_scale),
            "forward_backward_threshold_flow_px": float(args.fb_threshold),
            "min_warp_rows": int(args.min_warp_rows),
            "line_width": float(args.line_width),
            "optical_flow": "OpenCV DIS FAST, forward/backward consistency",
            "gt_enters_temporal_score": False,
            "test_set_used": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
        },
        "manifest_summary": {
            key: value for key, value in manifest.items() if key not in {"samples", "union_images"}
        },
        "summary": summary,
        "pairs": serializable_pairs,
    }
    report_json = output_dir / "v34_temporal_observability.json"
    report_md = output_dir / "v34_temporal_observability.md"
    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "passed": summary["passed"]}, indent=2))


if __name__ == "__main__":
    main()
