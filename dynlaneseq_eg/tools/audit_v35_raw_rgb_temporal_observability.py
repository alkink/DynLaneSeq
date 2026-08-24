from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.stats import rankdata
import torch
from tqdm import tqdm

from dynlaneseq_eg.evaluation.candidate_diagnostics import candidate_row_masks
from dynlaneseq_eg.modeling.common import fixed_y_rows
from dynlaneseq_eg.modeling.v35_raw_rgb_scorer import (
    ARM_NAMES,
    CandidateRibbonScorer,
    V35LossWeights,
    parameter_count,
    v35_pair_loss,
)
from dynlaneseq_eg.tools.audit_v34_temporal_candidate_observability import (
    TemporalSample,
    _dense_flow,
    _read_flow_image,
    _record_key,
    _slot_gt_pairs,
    identity_warp,
    warp_curve,
)


AUDIT_VERSION = 1
DEFAULT_SEEDS = (3407, 5741, 9011)


@dataclass(frozen=True)
class PairSpec:
    fold: str
    clip: str
    image_id: str
    slot_id: int
    gt_id: int
    good_candidate: int
    wrong_candidate: int
    good_quality: float
    wrong_quality: float
    thresholds: tuple[float, ...]
    evaluation_pair: bool


@dataclass(frozen=True)
class EntrySpec:
    image_id: str
    candidate_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a compact raw-RGB candidate scorer on V34's fixed "
            "validation clips and compare geometry-only, target-only, and "
            "three-frame temporal observability."
        )
    )
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--union-cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--v34-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-samples", type=int, default=64)
    parser.add_argument("--ribbon-samples", type=int, default=25)
    parser.add_argument("--fine-radius", type=float, default=32.0)
    parser.add_argument("--coarse-radius", type=float, default=160.0)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--fb-threshold", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--reuse-ribbon-cache", action="store_true")
    parser.add_argument("--reuse-training", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_image_path(value: str) -> str:
    return Path(str(value).strip().split()[0].lstrip("/")).as_posix()


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, TemporalSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = {
        _normalize_image_path(row["target"]): TemporalSample(**row)
        for row in payload["samples"]
    }
    if len(samples) != len(payload["samples"]):
        raise ValueError("V35 manifest contains duplicate target images")
    return payload, samples


def _fold_clip_contract(samples: dict[str, TemporalSample]) -> dict[str, Any]:
    clips = {
        fold: {sample.clip for sample in samples.values() if sample.fold == fold}
        for fold in ("a", "b")
    }
    checks = {
        "fold_a_nonempty": bool(clips["a"]),
        "fold_b_nonempty": bool(clips["b"]),
        "clip_intersection_empty": not bool(clips["a"] & clips["b"]),
    }
    return {
        "clips": {fold: sorted(value) for fold, value in clips.items()},
        "clip_counts": {fold: len(value) for fold, value in clips.items()},
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_learning_pairs(
    target_records: dict[str, dict[str, Any]],
    samples: dict[str, TemporalSample],
    evaluation_rows: list[dict[str, Any]],
    *,
    minimum_quality_gap: float = 0.03,
) -> tuple[list[PairSpec], list[PairSpec]]:
    """Build train pairs without proposal-ID features and exact V34 eval pairs."""

    learning: list[PairSpec] = []
    for image_id, sample in samples.items():
        record = target_records[image_id]
        stage = record["stages"]["main"]
        official = stage["official_iou"].float()
        valid = stage["official_candidate_valid"].bool()
        valid &= stage.get("selection_slot_candidate_valid", valid).bool()
        valid_ids = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        routes = stage["selection_slot_indices"].long()
        for gt_id, slot_id in _slot_gt_pairs(stage):
            source = int(routes[slot_id])
            if source not in valid_ids or not valid_ids:
                continue
            oracle = max(
                valid_ids,
                key=lambda index: (float(official[gt_id, index]), -index),
            )
            good_quality = float(official[gt_id, oracle])
            wrong_quality = float(official[gt_id, source])
            gap = good_quality - wrong_quality
            if source == oracle or good_quality < 0.50 or gap < minimum_quality_gap:
                continue
            thresholds = tuple(
                threshold
                for threshold in (0.50, 0.75)
                if wrong_quality <= threshold < good_quality
            )
            learning.append(
                PairSpec(
                    fold=sample.fold,
                    clip=sample.clip,
                    image_id=image_id,
                    slot_id=int(slot_id),
                    gt_id=int(gt_id),
                    good_candidate=int(oracle),
                    wrong_candidate=int(source),
                    good_quality=good_quality,
                    wrong_quality=wrong_quality,
                    thresholds=thresholds,
                    evaluation_pair=False,
                )
            )

    evaluation: list[PairSpec] = []
    for row in evaluation_rows:
        image_id = _normalize_image_path(row["image_id"])
        sample = samples.get(image_id)
        if sample is None:
            raise ValueError(f"V34 evaluation row not in manifest: {image_id}")
        evaluation.append(
            PairSpec(
                fold=str(row["fold"]),
                clip=str(row["clip"]),
                image_id=image_id,
                slot_id=int(row["slot_id"]),
                gt_id=int(row["gt_id"]),
                good_candidate=int(row["good_candidate"]),
                wrong_candidate=int(row["wrong_candidate"]),
                good_quality=float(row["good_iou"]),
                wrong_quality=float(row["wrong_iou"]),
                thresholds=tuple(float(value) for value in row["thresholds"]),
                evaluation_pair=True,
            )
        )

    if not learning or not evaluation:
        raise ValueError("V35 requires nonempty learning and evaluation pairs")
    return learning, evaluation


def _entry_specs(
    learning: Iterable[PairSpec], evaluation: Iterable[PairSpec]
) -> tuple[list[EntrySpec], dict[tuple[str, int], int]]:
    keys = {
        (pair.image_id, candidate)
        for pair in (*tuple(learning), *tuple(evaluation))
        for candidate in (pair.good_candidate, pair.wrong_candidate)
    }
    entries = [EntrySpec(*key) for key in sorted(keys)]
    return entries, {
        (entry.image_id, entry.candidate_id): index
        for index, entry in enumerate(entries)
    }


def _read_rgb_image(record: dict[str, Any]) -> np.ndarray:
    meta = record["meta"]
    image = cv2.imread(str(meta["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {meta['image_path']}")
    input_h = int(meta["input_h"])
    input_w = int(meta["input_w"])
    crop_x = int(round(float(meta.get("crop_x", 0.0))))
    crop_y = int(round(float(meta.get("crop_y", 0.0))))
    crop_w = int(round(float(input_w) / float(meta.get("scale_x", 1.0))))
    crop_h = int(round(float(input_h) / float(meta.get("scale_y", 1.0))))
    crop = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _sample_ribbon(
    image: np.ndarray,
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    valid_rows: np.ndarray,
    *,
    radius: float,
    samples: int,
) -> np.ndarray:
    x_rows = np.asarray(x_rows, dtype=np.float32)
    y_rows = np.asarray(y_rows, dtype=np.float32)
    valid_rows = np.asarray(valid_rows, dtype=bool)
    offsets = np.linspace(-float(radius), float(radius), int(samples), dtype=np.float32)
    map_x = x_rows[:, :, None] + offsets[None, None, :]
    map_y = np.broadcast_to(y_rows[:, :, None], map_x.shape).astype(np.float32)
    count, rows, width = map_x.shape
    sampled = cv2.remap(
        image,
        map_x.reshape(count * rows, width),
        map_y.reshape(count * rows, width),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(count, rows, width, 3)
    sampled[~valid_rows] = 0
    return sampled.transpose(0, 3, 1, 2).copy()


def _frame_ribbons(
    image: np.ndarray,
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    valid_rows: np.ndarray,
    *,
    fine_radius: float,
    coarse_radius: float,
    ribbon_samples: int,
) -> np.ndarray:
    return np.concatenate(
        (
            _sample_ribbon(
                image,
                x_rows,
                y_rows,
                valid_rows,
                radius=fine_radius,
                samples=ribbon_samples,
            ),
            _sample_ribbon(
                image,
                x_rows,
                y_rows,
                valid_rows,
                radius=coarse_radius,
                samples=ribbon_samples,
            ),
        ),
        axis=1,
    )


def _downsample_warp(
    warp,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(warp.x, dtype=np.float32)[..., row_indices],
        np.asarray(warp.y, dtype=np.float32)[..., row_indices],
        np.asarray(warp.valid, dtype=bool)[..., row_indices],
    )


def _warp_candidates(
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    masks: np.ndarray,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    *,
    input_w: int,
    input_h: int,
    fb_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _dense_flow(source_gray, target_gray)
    backward = _dense_flow(target_gray, source_gray)
    warped = [
        warp_curve(
            x_rows[index],
            y_rows,
            masks[index],
            forward,
            backward,
            input_w=input_w,
            input_h=input_h,
            fb_threshold=fb_threshold,
        )
        for index in range(int(x_rows.shape[0]))
    ]
    return (
        np.stack([value.x for value in warped]),
        np.stack([value.y for value in warped]),
        np.stack([value.valid for value in warped]),
    )


def _geometry_features(
    x_rows: np.ndarray,
    masks: np.ndarray,
    row_indices: np.ndarray,
    *,
    input_w: int,
) -> np.ndarray:
    x = np.asarray(x_rows, dtype=np.float32)[:, row_indices]
    valid = np.asarray(masks, dtype=bool)[:, row_indices]
    x_norm = x / max(float(input_w - 1), 1.0) * 2.0 - 1.0
    slope = np.gradient(x_norm, axis=-1)
    curvature = np.gradient(slope, axis=-1)
    geometry = np.stack((x_norm, slope * 16.0, curvature * 64.0, valid), axis=1)
    geometry[:, :3] *= valid[:, None]
    return geometry.astype(np.float16)


def _sequence_ribbons(
    *,
    base_record: dict[str, Any],
    previous_record: dict[str, Any],
    following_record: dict[str, Any],
    x_rows: np.ndarray,
    masks: np.ndarray,
    y_rows: np.ndarray,
    row_indices: np.ndarray,
    flow_scale: float,
    fb_threshold: float,
    fine_radius: float,
    coarse_radius: float,
    ribbon_samples: int,
) -> np.ndarray:
    input_h = int(base_record["meta"]["input_h"])
    input_w = int(base_record["meta"]["input_w"])
    base_gray = _read_flow_image(base_record, flow_scale)
    previous_gray = _read_flow_image(previous_record, flow_scale)
    following_gray = _read_flow_image(following_record, flow_scale)
    previous_x, previous_y, previous_valid = _warp_candidates(
        x_rows,
        y_rows,
        masks,
        base_gray,
        previous_gray,
        input_w=input_w,
        input_h=input_h,
        fb_threshold=fb_threshold,
    )
    following_x, following_y, following_valid = _warp_candidates(
        x_rows,
        y_rows,
        masks,
        base_gray,
        following_gray,
        input_w=input_w,
        input_h=input_h,
        fb_threshold=fb_threshold,
    )
    target_x = x_rows[:, row_indices]
    target_y = np.broadcast_to(
        y_rows[row_indices][None], target_x.shape
    ).astype(np.float32)
    target_valid = masks[:, row_indices]
    frame_arguments = (
        (
            previous_record,
            previous_x[:, row_indices],
            previous_y[:, row_indices],
            previous_valid[:, row_indices],
        ),
        (base_record, target_x, target_y, target_valid),
        (
            following_record,
            following_x[:, row_indices],
            following_y[:, row_indices],
            following_valid[:, row_indices],
        ),
    )
    ribbons = [
        _frame_ribbons(
            _read_rgb_image(record),
            frame_x,
            frame_y,
            frame_valid,
            fine_radius=fine_radius,
            coarse_radius=coarse_radius,
            ribbon_samples=ribbon_samples,
        )
        for record, frame_x, frame_y, frame_valid in frame_arguments
    ]
    return np.concatenate(ribbons, axis=1)


def _cache_paths(output_dir: Path) -> dict[str, Path]:
    cache_dir = output_dir / "ribbon_cache"
    return {
        "dir": cache_dir,
        "contract": cache_dir / "contract.json",
        "correct": cache_dir / "correct_ribbons.npy",
        "wrong": cache_dir / "wrong_ribbons.npy",
        "geometry": cache_dir / "geometry.npy",
    }


def _expected_cache_contract(
    *,
    entries: list[EntrySpec],
    learning: list[PairSpec],
    evaluation: list[PairSpec],
    target_cache: Path,
    union_cache: Path,
    manifest: Path,
    v34_report: Path,
    row_samples: int,
    ribbon_samples: int,
    fine_radius: float,
    coarse_radius: float,
    flow_scale: float,
    fb_threshold: float,
) -> dict[str, Any]:
    payload = {
        "audit_version": AUDIT_VERSION,
        "target_cache": str(target_cache.resolve()),
        "target_cache_sha256": _sha256(target_cache),
        "union_cache": str(union_cache.resolve()),
        "union_cache_sha256": _sha256(union_cache),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "v34_report": str(v34_report.resolve()),
        "v34_report_sha256": _sha256(v34_report),
        "entry_count": len(entries),
        "learning_pair_count": len(learning),
        "evaluation_pair_count": len(evaluation),
        "row_samples": int(row_samples),
        "ribbon_samples": int(ribbon_samples),
        "fine_radius": float(fine_radius),
        "coarse_radius": float(coarse_radius),
        "flow_scale": float(flow_scale),
        "fb_threshold": float(fb_threshold),
        "ribbon_shape": [len(entries), 18, int(row_samples), int(ribbon_samples)],
        "geometry_shape": [len(entries), 4, int(row_samples)],
        "entries": [asdict(entry) for entry in entries],
        "learning_pairs": [asdict(pair) for pair in learning],
        "evaluation_pairs": [asdict(pair) for pair in evaluation],
    }
    return json.loads(json.dumps(payload))


def _contract_matches(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    ignored = {"elapsed_seconds", "completed", "array_sha256"}
    return {
        key: value for key, value in observed.items() if key not in ignored
    } == expected


def build_or_load_ribbon_cache(
    *,
    output_dir: Path,
    entries: list[EntrySpec],
    learning: list[PairSpec],
    evaluation: list[PairSpec],
    target_cache_path: Path,
    union_cache_path: Path,
    manifest_path: Path,
    v34_report_path: Path,
    target_records: dict[str, dict[str, Any]],
    union_records: dict[str, dict[str, Any]],
    samples: dict[str, TemporalSample],
    row_samples: int,
    ribbon_samples: int,
    fine_radius: float,
    coarse_radius: float,
    flow_scale: float,
    fb_threshold: float,
    reuse: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    paths = _cache_paths(output_dir)
    expected = _expected_cache_contract(
        entries=entries,
        learning=learning,
        evaluation=evaluation,
        target_cache=target_cache_path,
        union_cache=union_cache_path,
        manifest=manifest_path,
        v34_report=v34_report_path,
        row_samples=row_samples,
        ribbon_samples=ribbon_samples,
        fine_radius=fine_radius,
        coarse_radius=coarse_radius,
        flow_scale=flow_scale,
        fb_threshold=fb_threshold,
    )
    if reuse and paths["contract"].is_file():
        observed = json.loads(paths["contract"].read_text(encoding="utf-8"))
        if not observed.get("completed") or not _contract_matches(observed, expected):
            raise ValueError("existing V35 ribbon cache does not match the contract")
        correct = np.load(paths["correct"], mmap_mode="r")
        wrong = np.load(paths["wrong"], mmap_mode="r")
        geometry = np.load(paths["geometry"], mmap_mode="r")
        if list(correct.shape) != expected["ribbon_shape"]:
            raise ValueError("existing V35 correct-ribbon shape mismatch")
        if list(wrong.shape) != expected["ribbon_shape"]:
            raise ValueError("existing V35 wrong-ribbon shape mismatch")
        if list(geometry.shape) != expected["geometry_shape"]:
            raise ValueError("existing V35 geometry shape mismatch")
        return observed, correct, wrong, geometry

    paths["dir"].mkdir(parents=True, exist_ok=True)
    correct = np.lib.format.open_memmap(
        paths["correct"], mode="w+", dtype=np.uint8, shape=tuple(expected["ribbon_shape"])
    )
    wrong = np.lib.format.open_memmap(
        paths["wrong"], mode="w+", dtype=np.uint8, shape=tuple(expected["ribbon_shape"])
    )
    geometry = np.lib.format.open_memmap(
        paths["geometry"], mode="w+", dtype=np.float16, shape=tuple(expected["geometry_shape"])
    )
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entry_index, entry in enumerate(entries):
        grouped[entry.image_id].append((entry_index, entry.candidate_id))

    started = time.perf_counter()
    for image_id, indexed_candidates in tqdm(
        sorted(grouped.items()), ncols=100, desc="V35 raw RGB ribbon cache"
    ):
        sample = samples[image_id]
        target_record = target_records[image_id]
        stage = target_record["stages"]["main"]
        input_h = int(target_record["meta"]["input_h"])
        input_w = int(target_record["meta"]["input_w"])
        all_x, all_masks, all_valid = candidate_row_masks(
            stage, input_h=input_h, input_w=input_w, min_valid_rows=5
        )
        all_masks &= all_valid[:, None]
        candidate_ids = [candidate for _entry, candidate in indexed_candidates]
        x_rows = all_x[candidate_ids].numpy().astype(np.float32)
        masks = all_masks[candidate_ids].numpy().astype(bool)
        rows = int(x_rows.shape[-1])
        y_rows = fixed_y_rows(rows, input_h, dtype=torch.float32).numpy()
        row_indices = np.linspace(0, rows - 1, num=row_samples).round().astype(np.int64)

        previous_record = union_records[_normalize_image_path(sample.previous)]
        following_record = union_records[_normalize_image_path(sample.following)]
        correct_values = _sequence_ribbons(
            base_record=target_record,
            previous_record=previous_record,
            following_record=following_record,
            x_rows=x_rows,
            masks=masks,
            y_rows=y_rows,
            row_indices=row_indices,
            flow_scale=flow_scale,
            fb_threshold=fb_threshold,
            fine_radius=fine_radius,
            coarse_radius=coarse_radius,
            ribbon_samples=ribbon_samples,
        )

        wrong_sample = samples[_normalize_image_path(sample.wrong_context)]
        wrong_base = union_records[_normalize_image_path(wrong_sample.target)]
        wrong_previous = union_records[_normalize_image_path(wrong_sample.previous)]
        wrong_following = union_records[_normalize_image_path(wrong_sample.following)]
        wrong_values = _sequence_ribbons(
            base_record=wrong_base,
            previous_record=wrong_previous,
            following_record=wrong_following,
            x_rows=x_rows,
            masks=masks,
            y_rows=y_rows,
            row_indices=row_indices,
            flow_scale=flow_scale,
            fb_threshold=fb_threshold,
            fine_radius=fine_radius,
            coarse_radius=coarse_radius,
            ribbon_samples=ribbon_samples,
        )
        geometry_values = _geometry_features(
            x_rows, masks, row_indices, input_w=input_w
        )
        for local_index, (entry_index, _candidate) in enumerate(indexed_candidates):
            correct[entry_index] = correct_values[local_index]
            wrong[entry_index] = wrong_values[local_index]
            geometry[entry_index] = geometry_values[local_index]

    correct.flush()
    wrong.flush()
    geometry.flush()
    completed = {
        **expected,
        "completed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "array_sha256": {
            "correct": _sha256(paths["correct"]),
            "wrong": _sha256(paths["wrong"]),
            "geometry": _sha256(paths["geometry"]),
        },
    }
    paths["contract"].write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (
        completed,
        np.load(paths["correct"], mmap_mode="r"),
        np.load(paths["wrong"], mmap_mode="r"),
        np.load(paths["geometry"], mmap_mode="r"),
    )


def _pair_entry_indices(
    pairs: list[PairSpec], entry_map: dict[tuple[str, int], int]
) -> tuple[np.ndarray, np.ndarray]:
    good = np.asarray(
        [entry_map[(pair.image_id, pair.good_candidate)] for pair in pairs],
        dtype=np.int64,
    )
    wrong = np.asarray(
        [entry_map[(pair.image_id, pair.wrong_candidate)] for pair in pairs],
        dtype=np.int64,
    )
    return good, wrong


def _tensor_batch(
    array: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    image: bool,
) -> torch.Tensor:
    values = np.asarray(array[indices])
    tensor = torch.from_numpy(values).to(device=device, non_blocking=True)
    if image:
        return tensor.float().div_(127.5).sub_(1.0)
    return tensor.float()


def _augment_pair_batch(
    ribbons: torch.Tensor,
    geometry: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same photometric/temporal augmentation to both candidates."""

    batch = int(ribbons.shape[0])
    contrast = torch.empty(
        (batch, 1, 1, 1, 1), device=ribbons.device
    ).uniform_(0.82, 1.18, generator=generator)
    brightness = torch.empty(
        (batch, 1, 1, 1, 1), device=ribbons.device
    ).uniform_(-0.12, 0.12, generator=generator)
    ribbons = (ribbons * contrast + brightness).clamp_(-1.0, 1.0)
    noise = torch.randn(
        ribbons.shape,
        device=ribbons.device,
        dtype=ribbons.dtype,
        generator=generator,
    ) * 0.015
    ribbons = (ribbons + noise).clamp_(-1.0, 1.0)

    reverse = torch.rand(batch, device=ribbons.device, generator=generator) < 0.5
    if bool(reverse.any()):
        selected = ribbons[reverse]
        ribbons[reverse] = torch.cat(
            (selected[:, :, 12:18], selected[:, :, 6:12], selected[:, :, 0:6]),
            dim=2,
        )
    return ribbons, geometry


def _learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup: int = 100,
    minimum_ratio: float = 0.05,
) -> float:
    if step <= warmup:
        ratio = float(step) / float(max(warmup, 1))
    else:
        progress = float(step - warmup) / float(max(total_steps - warmup, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        ratio = minimum_ratio + (1.0 - minimum_ratio) * cosine
    return float(base_lr) * ratio


def train_scorer(
    *,
    pairs: list[PairSpec],
    entry_map: dict[tuple[str, int], int],
    correct: np.ndarray,
    geometry: np.ndarray,
    arm: str,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    quality_weight: float,
    device: torch.device,
    log_interval: int,
) -> tuple[CandidateRibbonScorer, dict[str, Any]]:
    if not pairs:
        raise ValueError("cannot train V35 scorer with zero pairs")
    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = CandidateRibbonScorer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp = device.type == "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 17)
    good_entries, wrong_entries = _pair_entry_indices(pairs, entry_map)
    good_quality = np.asarray([pair.good_quality for pair in pairs], dtype=np.float32)
    wrong_quality = np.asarray([pair.wrong_quality for pair in pairs], dtype=np.float32)
    weights = np.asarray(
        [2.0 if pair.thresholds else 1.0 for pair in pairs], dtype=np.float64
    )
    probabilities = weights / weights.sum()
    loss_weights = V35LossWeights(ranking=1.0, quality=quality_weight)
    logs: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, int(steps) + 1):
        chosen = np_rng.choice(
            len(pairs), size=int(batch_size), replace=True, p=probabilities
        )
        selected_good = good_entries[chosen]
        selected_wrong = wrong_entries[chosen]
        ribbons_good = _tensor_batch(
            correct, selected_good, device=device, image=True
        )
        ribbons_wrong = _tensor_batch(
            correct, selected_wrong, device=device, image=True
        )
        geometry_good = _tensor_batch(
            geometry, selected_good, device=device, image=False
        )
        geometry_wrong = _tensor_batch(
            geometry, selected_wrong, device=device, image=False
        )
        ribbons_pair = torch.stack((ribbons_good, ribbons_wrong), dim=1)
        geometry_pair = torch.stack((geometry_good, geometry_wrong), dim=1)
        ribbons_pair, geometry_pair = _augment_pair_batch(
            ribbons_pair, geometry_pair, generator=generator
        )
        ribbons_flat = ribbons_pair.flatten(0, 1)
        geometry_flat = geometry_pair.flatten(0, 1)
        target_good = torch.from_numpy(good_quality[chosen]).to(device)
        target_wrong = torch.from_numpy(wrong_quality[chosen]).to(device)

        lr = _learning_rate(step, steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            scores = model(ribbons_flat, geometry_flat, arm=arm).view(-1, 2)
            total, diagnostics = v35_pair_loss(
                scores[:, 0],
                scores[:, 1],
                target_good,
                target_wrong,
                weights=loss_weights,
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % int(log_interval) == 0 or step == int(steps):
            row = {
                "step": step,
                "lr": lr,
                **{key: float(value) for key, value in diagnostics.items()},
            }
            logs.append(row)
            print(
                f"V35 arm={arm} seed={seed} step={step}/{steps} "
                f"loss={row['loss_total']:.5f} margin={row['mean_margin']:.4f}",
                flush=True,
            )
    report = {
        "arm": arm,
        "seed": int(seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "training_pairs": len(pairs),
        "parameter_count": parameter_count(model),
        "elapsed_seconds": time.perf_counter() - started,
        "logs": logs,
    }
    return model, report


@torch.inference_mode()
def score_pairs(
    model: CandidateRibbonScorer,
    *,
    pairs: list[PairSpec],
    entry_map: dict[tuple[str, int], int],
    ribbons: np.ndarray,
    geometry: np.ndarray,
    arm: str,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    good_entries, wrong_entries = _pair_entry_indices(pairs, entry_map)
    good_scores: list[np.ndarray] = []
    wrong_scores: list[np.ndarray] = []
    model.eval()
    amp = device.type == "cuda"
    for start in range(0, len(pairs), int(batch_size)):
        stop = min(start + int(batch_size), len(pairs))
        good_images = _tensor_batch(
            ribbons, good_entries[start:stop], device=device, image=True
        )
        wrong_images = _tensor_batch(
            ribbons, wrong_entries[start:stop], device=device, image=True
        )
        good_geometry = _tensor_batch(
            geometry, good_entries[start:stop], device=device, image=False
        )
        wrong_geometry = _tensor_batch(
            geometry, wrong_entries[start:stop], device=device, image=False
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            scores = model(
                torch.cat((good_images, wrong_images), dim=0),
                torch.cat((good_geometry, wrong_geometry), dim=0),
                arm=arm,
            ).float()
        size = stop - start
        good_scores.append(scores[:size].cpu().numpy())
        wrong_scores.append(scores[size:].cpu().numpy())
    return np.concatenate(good_scores), np.concatenate(wrong_scores)


def _roc_auc(good: np.ndarray, wrong: np.ndarray) -> float | None:
    if good.size == 0 or wrong.size == 0:
        return None
    scores = np.concatenate((good, wrong)).astype(np.float64)
    labels = np.concatenate(
        (np.ones(good.size, dtype=np.int64), np.zeros(wrong.size, dtype=np.int64))
    )
    ranks = rankdata(scores, method="average")
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2.0)
        / float(positive * negative)
    )


def _metric(good: np.ndarray, wrong: np.ndarray) -> dict[str, Any]:
    if good.size == 0:
        return {
            "pairs": 0,
            "pair_accuracy": None,
            "auc": None,
            "mean_good_score": None,
            "mean_wrong_score": None,
            "mean_margin": None,
        }
    wins = (good > wrong).astype(np.float64) + 0.5 * (good == wrong)
    return {
        "pairs": int(good.size),
        "pair_accuracy": float(wins.mean()),
        "auc": _roc_auc(good, wrong),
        "mean_good_score": float(good.mean()),
        "mean_wrong_score": float(wrong.mean()),
        "mean_margin": float((good - wrong).mean()),
    }


def _bootstrap_metrics(
    pairs: list[PairSpec],
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    clips = sorted({pair.clip for pair in pairs})
    indices_by_clip = {
        clip: np.asarray(
            [index for index, pair in enumerate(pairs) if pair.clip == clip],
            dtype=np.int64,
        )
        for clip in clips
    }
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(repetitions)):
        sampled = rng.choice(clips, size=len(clips), replace=True)
        indices = np.concatenate([indices_by_clip[str(clip)] for clip in sampled])
        metrics = {
            name: _metric(good[indices], wrong[indices])
            for name, (good, wrong) in scores.items()
        }
        for name, metric in metrics.items():
            values[f"{name}.auc"].append(float(metric["auc"]))
            values[f"{name}.pair_accuracy"].append(float(metric["pair_accuracy"]))
        if "S" in metrics and "G" in metrics:
            values["S_minus_G.auc"].append(
                float(metrics["S"]["auc"]) - float(metrics["G"]["auc"])
            )
        if "T" in metrics and "S" in metrics:
            values["T_minus_S.auc"].append(
                float(metrics["T"]["auc"]) - float(metrics["S"]["auc"])
            )
            values["T_minus_S.pair_accuracy"].append(
                float(metrics["T"]["pair_accuracy"])
                - float(metrics["S"]["pair_accuracy"])
            )
        for arm in ("S", "T"):
            wrong_name = f"{arm}_wrong"
            if arm in metrics and wrong_name in metrics:
                values[f"{arm}_minus_wrong.auc"].append(
                    float(metrics[arm]["auc"])
                    - float(metrics[wrong_name]["auc"])
                )
    return {
        key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for key, value in values.items()
        if value
    }


def _fold_take(pairs: list[PairSpec], count: int) -> list[PairSpec]:
    output: list[PairSpec] = []
    for fold in ("a", "b"):
        output.extend([pair for pair in pairs if pair.fold == fold][: int(count)])
    return output


def _json_state_dict(model: CandidateRibbonScorer) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    def show(value: Any) -> str:
        return "—" if value is None else f"{float(value):.4f}"

    lines = [
        "# V35 Raw-RGB Temporal Observability Results",
        "",
        f"- Overall single-frame PASS: `{payload['verdict']['single_frame_pass']}`",
        f"- Overall temporal-increment PASS: `{payload['verdict']['temporal_increment_pass']}`",
        f"- Final decision: **{payload['verdict']['decision']}**",
        "- Test split used: `False`",
        "",
        "| Direction | IoU | Arm | Pairs | Pair acc | AUC | Wrong-image AUC | Δ vs G | Δ T−S |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in payload["summary"].items():
        for arm in ARM_NAMES:
            metric = item["arms"][arm]["correct"]
            wrong = item["arms"][arm]["wrong_image"]
            lines.append(
                f"| {item['direction']} | {item['threshold']:.2f} | {arm} | "
                f"{metric['pairs']} | {show(metric['pair_accuracy'])} | "
                f"{show(metric['auc'])} | {show(wrong['auc'])} | "
                f"{show(item['deltas'].get(f'{arm}_minus_G_auc'))} | "
                f"{show(item['deltas'].get('T_minus_S_auc') if arm == 'T' else None)} |"
            )
    lines.extend(
        (
            "",
            "## Yorum",
            "",
            payload["verdict"]["interpretation"],
            "",
            "Bu cross-fit bir observability diagnostic'idir; official validation F1 veya deploy sonucu değildir.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_cache_path = Path(args.target_cache).expanduser().resolve()
    union_cache_path = Path(args.union_cache).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    v34_report_path = Path(args.v34_report).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    manifest_payload, samples = _load_manifest(manifest_path)
    fold_contract = _fold_clip_contract(samples)
    if not fold_contract["passed"]:
        raise ValueError("V35 clip-disjoint fold contract failed")
    target_payload = _load_torch(target_cache_path)
    union_payload = _load_torch(union_cache_path)
    target_records = {
        _record_key(record, dataset_root): record
        for record in target_payload["records"]
    }
    union_records = {
        _record_key(record, dataset_root): record
        for record in union_payload["records"]
    }
    missing_targets = sorted(set(samples) - set(target_records))
    if missing_targets:
        raise ValueError(f"target cache is missing {len(missing_targets)} samples")
    required_union = {
        _normalize_image_path(value)
        for sample in samples.values()
        for value in (sample.target, sample.previous, sample.following)
    }
    missing_union = sorted(required_union - set(union_records))
    if missing_union:
        raise ValueError(f"union cache is missing {len(missing_union)} images")
    v34_payload = json.loads(v34_report_path.read_text(encoding="utf-8"))
    learning, evaluation = build_learning_pairs(
        target_records, samples, list(v34_payload["pairs"])
    )
    if args.smoke:
        learning = _fold_take(learning, 8)
        evaluation = _fold_take(evaluation, 8)
        args.steps = min(int(args.steps), 2)
        args.seeds = [int(args.seeds[0])]
        args.row_samples = min(int(args.row_samples), 16)
        args.ribbon_samples = min(int(args.ribbon_samples), 9)
        args.bootstrap_reps = min(int(args.bootstrap_reps), 10)
    entries, entry_map = _entry_specs(learning, evaluation)
    print(
        json.dumps(
            {
                "entries": len(entries),
                "learning_pairs": len(learning),
                "evaluation_pairs": len(evaluation),
                "learning_by_fold": {
                    fold: sum(pair.fold == fold for pair in learning)
                    for fold in ("a", "b")
                },
                "evaluation_by_fold": {
                    fold: sum(pair.fold == fold for pair in evaluation)
                    for fold in ("a", "b")
                },
            },
            indent=2,
        ),
        flush=True,
    )
    cache_contract, correct, wrong, geometry = build_or_load_ribbon_cache(
        output_dir=output_dir,
        entries=entries,
        learning=learning,
        evaluation=evaluation,
        target_cache_path=target_cache_path,
        union_cache_path=union_cache_path,
        manifest_path=manifest_path,
        v34_report_path=v34_report_path,
        target_records=target_records,
        union_records=union_records,
        samples=samples,
        row_samples=int(args.row_samples),
        ribbon_samples=int(args.ribbon_samples),
        fine_radius=float(args.fine_radius),
        coarse_radius=float(args.coarse_radius),
        flow_scale=float(args.flow_scale),
        fb_threshold=float(args.fb_threshold),
        reuse=bool(args.reuse_ribbon_cache),
    )
    if args.cache_only:
        print(json.dumps({"cache": cache_contract, "cache_only": True}, indent=2))
        return

    directions = (
        ("a_to_b", "a", "b"),
        ("b_to_a", "b", "a"),
    )
    training_root = output_dir / "training"
    score_payload: dict[str, Any] = {}
    training_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for direction, train_fold, evaluation_fold in directions:
        train_pairs = [pair for pair in learning if pair.fold == train_fold]
        eval_pairs = [pair for pair in evaluation if pair.fold == evaluation_fold]
        direction_scores: dict[str, Any] = {}
        for arm in ARM_NAMES:
            correct_good: list[np.ndarray] = []
            correct_wrong: list[np.ndarray] = []
            wrong_good: list[np.ndarray] = []
            wrong_wrong: list[np.ndarray] = []
            seed_metrics: list[dict[str, Any]] = []
            for seed in [int(value) for value in args.seeds]:
                checkpoint = training_root / direction / arm / f"seed_{seed}.pt"
                if bool(args.reuse_training) and checkpoint.is_file():
                    saved = _load_torch(checkpoint)
                    if (
                        saved.get("direction") != direction
                        or saved.get("arm") != arm
                        or int(saved.get("seed", -1)) != seed
                        or int(saved.get("steps", -1)) != int(args.steps)
                    ):
                        raise ValueError(f"V35 checkpoint contract mismatch: {checkpoint}")
                    model = CandidateRibbonScorer().to(device)
                    model.load_state_dict(saved["model"], strict=True)
                    train_report = dict(saved["training_report"])
                else:
                    model, train_report = train_scorer(
                        pairs=train_pairs,
                        entry_map=entry_map,
                        correct=correct,
                        geometry=geometry,
                        arm=arm,
                        seed=seed,
                        steps=int(args.steps),
                        batch_size=int(args.batch_size),
                        learning_rate=float(args.learning_rate),
                        weight_decay=float(args.weight_decay),
                        quality_weight=float(args.quality_weight),
                        device=device,
                        log_interval=int(args.log_interval),
                    )
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "audit_version": AUDIT_VERSION,
                            "direction": direction,
                            "train_fold": train_fold,
                            "evaluation_fold": evaluation_fold,
                            "arm": arm,
                            "seed": seed,
                            "steps": int(args.steps),
                            "model": _json_state_dict(model),
                            "training_report": train_report,
                        },
                        checkpoint,
                    )
                cg, cw = score_pairs(
                    model,
                    pairs=eval_pairs,
                    entry_map=entry_map,
                    ribbons=correct,
                    geometry=geometry,
                    arm=arm,
                    device=device,
                )
                wg, ww = score_pairs(
                    model,
                    pairs=eval_pairs,
                    entry_map=entry_map,
                    ribbons=wrong,
                    geometry=geometry,
                    arm=arm,
                    device=device,
                )
                correct_good.append(cg)
                correct_wrong.append(cw)
                wrong_good.append(wg)
                wrong_wrong.append(ww)
                seed_metrics.append(
                    {
                        "seed": seed,
                        "correct": _metric(cg, cw),
                        "wrong_image": _metric(wg, ww),
                    }
                )
                training_reports.append(
                    {"direction": direction, **train_report}
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            direction_scores[arm] = {
                "correct_good": np.mean(np.stack(correct_good), axis=0),
                "correct_wrong": np.mean(np.stack(correct_wrong), axis=0),
                "wrong_good": np.mean(np.stack(wrong_good), axis=0),
                "wrong_wrong": np.mean(np.stack(wrong_wrong), axis=0),
                "per_seed": seed_metrics,
            }
        score_payload[direction] = {
            "train_fold": train_fold,
            "evaluation_fold": evaluation_fold,
            "pairs": eval_pairs,
            "scores": direction_scores,
        }

    summary: dict[str, Any] = {}
    gate_checks: dict[str, bool] = {}
    serialized_pair_scores: dict[str, list[dict[str, Any]]] = {}
    for direction, _train_fold, evaluation_fold in directions:
        data = score_payload[direction]
        pairs = data["pairs"]
        scores = data["scores"]
        serialized_pair_scores[direction] = []
        for pair_index, pair in enumerate(pairs):
            serialized_pair_scores[direction].append(
                {
                    **asdict(pair),
                    "scores": {
                        arm: {
                            "correct_good": float(scores[arm]["correct_good"][pair_index]),
                            "correct_wrong": float(scores[arm]["correct_wrong"][pair_index]),
                            "wrong_image_good": float(scores[arm]["wrong_good"][pair_index]),
                            "wrong_image_wrong": float(scores[arm]["wrong_wrong"][pair_index]),
                        }
                        for arm in ARM_NAMES
                    },
                }
            )
        for threshold in (0.50, 0.75):
            selected = np.asarray(
                [threshold in pair.thresholds for pair in pairs], dtype=bool
            )
            selected_pairs = [pair for pair in pairs if threshold in pair.thresholds]
            arm_metrics: dict[str, Any] = {}
            bootstrap_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for arm in ARM_NAMES:
                correct_pair = (
                    scores[arm]["correct_good"][selected],
                    scores[arm]["correct_wrong"][selected],
                )
                wrong_pair = (
                    scores[arm]["wrong_good"][selected],
                    scores[arm]["wrong_wrong"][selected],
                )
                arm_metrics[arm] = {
                    "correct": _metric(*correct_pair),
                    "wrong_image": _metric(*wrong_pair),
                    "per_seed": scores[arm]["per_seed"],
                }
                bootstrap_inputs[arm] = correct_pair
                bootstrap_inputs[f"{arm}_wrong"] = wrong_pair
            g_auc = float(arm_metrics["G"]["correct"]["auc"])
            s_auc = float(arm_metrics["S"]["correct"]["auc"])
            t_auc = float(arm_metrics["T"]["correct"]["auc"])
            s_wrong_auc = float(arm_metrics["S"]["wrong_image"]["auc"])
            t_wrong_auc = float(arm_metrics["T"]["wrong_image"]["auc"])
            s_accuracy = float(arm_metrics["S"]["correct"]["pair_accuracy"])
            t_accuracy = float(arm_metrics["T"]["correct"]["pair_accuracy"])
            deltas = {
                "G_minus_G_auc": 0.0,
                "S_minus_G_auc": s_auc - g_auc,
                "T_minus_G_auc": t_auc - g_auc,
                "S_correct_minus_wrong_auc": s_auc - s_wrong_auc,
                "T_correct_minus_wrong_auc": t_auc - t_wrong_auc,
                "T_minus_S_auc": t_auc - s_auc,
                "T_minus_S_pair_accuracy": t_accuracy - s_accuracy,
            }
            cell = f"{direction}_iou_{threshold:.2f}"
            single_checks = {
                "pairs_at_least_150": len(selected_pairs) >= 150,
                "pair_accuracy_at_least_0p65": s_accuracy >= 0.65,
                "auc_at_least_0p70": s_auc >= 0.70,
                "correct_minus_wrong_auc_at_least_0p10": s_auc - s_wrong_auc >= 0.10,
                "auc_minus_geometry_at_least_0p05": s_auc - g_auc >= 0.05,
            }
            temporal_checks = {
                "pairs_at_least_150": len(selected_pairs) >= 150,
                "pair_accuracy_at_least_0p65": t_accuracy >= 0.65,
                "auc_at_least_0p70": t_auc >= 0.70,
                "correct_minus_wrong_auc_at_least_0p10": t_auc - t_wrong_auc >= 0.10,
                "auc_minus_geometry_at_least_0p05": t_auc - g_auc >= 0.05,
                "auc_minus_single_at_least_0p05": t_auc - s_auc >= 0.05,
                "accuracy_minus_single_at_least_0p03": t_accuracy - s_accuracy >= 0.03,
            }
            for name, value in single_checks.items():
                gate_checks[f"{cell}_single_{name}"] = bool(value)
            for name, value in temporal_checks.items():
                gate_checks[f"{cell}_temporal_{name}"] = bool(value)
            summary[cell] = {
                "direction": direction,
                "train_fold": data["train_fold"],
                "evaluation_fold": evaluation_fold,
                "threshold": threshold,
                "arms": arm_metrics,
                "deltas": deltas,
                "bootstrap_ci95": _bootstrap_metrics(
                    selected_pairs,
                    bootstrap_inputs,
                    repetitions=int(args.bootstrap_reps),
                    seed=int(args.seeds[0]) + int(threshold * 1000),
                ),
                "single_frame_checks": single_checks,
                "single_frame_pass": all(single_checks.values()),
                "temporal_checks": temporal_checks,
                "temporal_pass": all(temporal_checks.values()),
            }

    single_pass = all(item["single_frame_pass"] for item in summary.values())
    temporal_pass = all(item["temporal_pass"] for item in summary.values())
    if temporal_pass and not single_pass:
        decision = "TEMPORAL_RAW_RGB_DEPLOYMENT_GATE_JUSTIFIED"
        interpretation = (
            "Tek kare gate'i geçmezken üç ham kare bütün önceden kilitli koşulları "
            "geçti. Yalnız temporal raw-image belief modeli için structured-route "
            "deployment gate'i açılabilir."
        )
    elif single_pass:
        decision = (
            "SINGLE_FRAME_GATE_JUSTIFIED_TEMPORAL_ADDS"
            if temporal_pass
            else "SINGLE_FRAME_GATE_JUSTIFIED_TEMPORAL_NOT_JUSTIFIED"
        )
        interpretation = (
            "Ham hedef görüntü proposal çiftlerini clip-disjoint biçimde ayırdı. "
            "Yeni bilgi gerektirmeyen en küçük raw-image structured-route replay "
            "gate'i açılabilir; temporal kol yalnız temporal PASS de sağlandıysa gerekçelidir."
        )
    else:
        decision = "CLOSE_PROPOSAL_SELECTION_RESCUE"
        interpretation = (
            "Ne tek-kare ne de üç-kare ham RGB scorer bütün fold/threshold "
            "koşullarını geçti. Frozen-head, ribbon, corridor, AGF ve temporal "
            "proposal-selection rescue ailesi kapatılmalıdır."
        )
    payload = {
        "experiment": "V35 raw-RGB temporal observability cross-fit gate",
        "audit_version": AUDIT_VERSION,
        "contract": str(
            Path("docs/V35_RAW_RGB_TEMPORAL_OBSERVABILITY_CONTRACT_20260824.md")
        ),
        "inputs": {
            "target_cache": str(target_cache_path),
            "target_cache_sha256": _sha256(target_cache_path),
            "union_cache": str(union_cache_path),
            "union_cache_sha256": _sha256(union_cache_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "v34_report": str(v34_report_path),
            "v34_report_sha256": _sha256(v34_report_path),
        },
        "fold_contract": fold_contract,
        "manifest_summary": {
            key: manifest_payload[key]
            for key in (
                "validation_rows",
                "validation_clips",
                "selected_by_fold",
                "sampled_clip_counts",
                "frame_step",
                "seed",
            )
        },
        "settings": {
            "row_samples": int(args.row_samples),
            "ribbon_samples": int(args.ribbon_samples),
            "fine_radius": float(args.fine_radius),
            "coarse_radius": float(args.coarse_radius),
            "flow_scale": float(args.flow_scale),
            "fb_threshold": float(args.fb_threshold),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seeds": [int(value) for value in args.seeds],
            "arms": list(ARM_NAMES),
        },
        "counts": {
            "entries": len(entries),
            "learning_pairs": len(learning),
            "evaluation_pairs": len(evaluation),
        },
        "ribbon_cache_contract": cache_contract,
        "training_reports": training_reports,
        "summary": summary,
        "pair_scores": serialized_pair_scores,
        "gate_checks": gate_checks,
        "verdict": {
            "single_frame_pass": single_pass,
            "temporal_increment_pass": temporal_pass,
            "decision": decision,
            "interpretation": interpretation,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "test_set_used": False,
        "validation_used_for_checkpoint_selection": False,
        "posthoc_threshold_selection_performed": False,
    }
    json_path = output_dir / "v35_raw_rgb_temporal_observability.json"
    markdown_path = output_dir / "v35_raw_rgb_temporal_observability.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(payload, markdown_path)
    print(json.dumps(payload["verdict"], indent=2), flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
