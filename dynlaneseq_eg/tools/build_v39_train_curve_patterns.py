from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.data.culane_dataset import CULaneDataset
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.factory import build_dataset
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V39 full-curve query patterns from official train labels."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--pattern-count", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=20_000)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _complete_curve(x: np.ndarray, valid: np.ndarray, input_w: int) -> np.ndarray:
    ids = np.flatnonzero(valid)
    if ids.size < 2:
        raise ValueError("a V39 pattern curve needs at least two valid rows")
    rows = np.arange(x.shape[0], dtype=np.float32)
    completed = np.interp(rows, ids.astype(np.float32), x[ids].astype(np.float32))
    return np.clip(completed / float(input_w), 0.0, 1.0).astype(np.float32)


def _ordered_curves(target: dict[str, np.ndarray], input_w: int) -> list[np.ndarray]:
    x = target["x_rows"]
    valid = target["valid_mask"]
    if int(x.shape[0]) == 0:
        return []
    last_x = []
    for curve, mask in zip(x, valid):
        ids = np.flatnonzero(mask)
        last_x.append(float(curve[ids[-1]]) if ids.size else float("inf"))
    order = np.argsort(np.asarray(last_x), kind="stable")
    return [_complete_curve(x[index], valid[index], input_w) for index in order]


def _kmeans_plus_plus(
    values: np.ndarray, count: int, rng: np.random.RandomState
) -> np.ndarray:
    centres = [values[int(rng.randint(0, values.shape[0]))].copy()]
    closest = np.full((values.shape[0],), np.inf, dtype=np.float64)
    for _ in range(1, count):
        distance = np.square(values - centres[-1]).sum(axis=1, dtype=np.float64)
        closest = np.minimum(closest, distance)
        total = float(closest.sum())
        if not np.isfinite(total) or total <= 0.0:
            index = int(rng.randint(0, values.shape[0]))
        else:
            index = int(rng.choice(values.shape[0], p=closest / total))
        centres.append(values[index].copy())
    return np.stack(centres).astype(np.float32)


def _fit_patterns(
    values: np.ndarray,
    *,
    count: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if values.ndim != 2 or values.shape[0] < count:
        raise ValueError(
            f"not enough curves for {count} patterns: shape={tuple(values.shape)}"
        )
    rng = np.random.RandomState(seed)
    centres = _kmeans_plus_plus(values, count, rng)
    assignment = np.zeros((values.shape[0],), dtype=np.int64)
    for _ in range(iterations):
        value_norm = np.square(values).sum(axis=1, keepdims=True)
        centre_norm = np.square(centres).sum(axis=1, keepdims=True).T
        distance = value_norm + centre_norm - 2.0 * values @ centres.T
        assignment = distance.argmin(axis=1)
        updated = centres.copy()
        for cluster in range(count):
            members = values[assignment == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
            else:
                farthest = int(distance.min(axis=1).argmax())
                updated[cluster] = values[farthest]
        if float(np.abs(updated - centres).max()) < 1.0e-6:
            centres = updated
            break
        centres = updated
    final_distance = np.square(values - centres[assignment]).sum(axis=1)
    order = np.argsort(centres[:, -1], kind="stable")
    centres = centres[order]
    counts = np.bincount(assignment, minlength=count)
    return centres.clip(0.0, 1.0).astype(np.float32), {
        "curves": int(values.shape[0]),
        "mean_squared_curve_distance": float(final_distance.mean()),
        "minimum_cluster_members": int(counts.min()),
        "maximum_cluster_members": int(counts.max()),
    }


def _configured(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="train")
    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["train"] = str(population["list_path"])
    return cfg, population


def main() -> None:
    args = parse_args()
    if args.pattern_count < 2 or args.max_images < args.pattern_count:
        raise ValueError("invalid V39 pattern sampling contract")
    cfg, population = _configured(args)
    dataset = build_dataset(cfg, split="train", training=False)
    if not isinstance(dataset, CULaneDataset):
        raise TypeError("V39 pattern builder requires CULane")
    if len(dataset) != int(population["expected_nonempty_rows"]):
        raise ValueError("V39 pattern builder altered official train.txt")
    rng = np.random.RandomState(int(args.seed))
    sample_count = min(int(args.max_images), len(dataset))
    sampled = np.sort(rng.choice(len(dataset), sample_count, replace=False))
    slots = 4
    input_w = int(cfg["model"]["input_w"])
    rows = int(cfg["model"]["num_rows"])
    cut_height = int(cfg.get("augmentation", {}).get("cut_height", 0))
    curves: list[list[np.ndarray]] = [[] for _ in range(slots)]
    skipped_overflow = 0
    for progress, index in enumerate(sampled, start=1):
        record = dataset.records[int(index)]
        with Image.open(record.image_path) as image:
            width, height = image.size
        crop_y = max(0, min(cut_height, height - 1))
        lanes = CULaneDataset._read_lines_txt(record.anno_path)
        cropped = [[(x, y - crop_y) for x, y in lane] for lane in lanes]
        target = dataset.target_builder.build(
            cropped, orig_w=width, orig_h=height - crop_y
        )
        ordered = _ordered_curves(target, input_w)
        if len(ordered) > slots:
            skipped_overflow += 1
            continue
        for slot, curve in enumerate(ordered):
            if curve.shape != (rows,):
                raise ValueError("V39 target row count changed during pattern build")
            curves[slot].append(curve)
        if progress == 1 or progress % 2000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "collect_v39_train_curves",
                        "images": progress,
                        "slot_curves": [len(value) for value in curves],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    patterns = []
    slot_reports = []
    for slot, values in enumerate(curves):
        centres, report = _fit_patterns(
            np.stack(values).astype(np.float32),
            count=int(args.pattern_count),
            iterations=int(args.kmeans_iters),
            seed=int(args.seed) + slot * 1009,
        )
        patterns.append(centres)
        slot_reports.append(report)
    bank = np.stack(patterns)
    payload = {
        "experiment": "V39 train-only full-curve pattern bank",
        "patterns": bank.tolist(),
        "shape": list(bank.shape),
        "normalization": "x / input_w",
        "input_w": input_w,
        "num_rows": rows,
        "num_slots": slots,
        "pattern_count": int(args.pattern_count),
        "seed": int(args.seed),
        "sampled_images": sample_count,
        "sampled_dataset_indices_sha256": hashlib.sha256(
            sampled.astype(np.int64).tobytes()
        ).hexdigest(),
        "skipped_images_with_more_than_four_valid_lanes": skipped_overflow,
        "slot_reports": slot_reports,
        "official_train_population_contract": population,
        "train_list_sha256": sha256_file(population["list_path"]),
        "source_config": str(Path(args.config).expanduser().resolve()),
        "source_config_sha256": sha256_file(args.config),
        "test_set_used": False,
        "validation_set_used": False,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "patterns"}, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
