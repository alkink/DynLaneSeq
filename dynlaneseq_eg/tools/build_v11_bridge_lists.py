from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic clip-aware V11 bridge train/evaluation lists."
        )
    )
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-clips", type=int, default=512)
    parser.add_argument("--train-images", type=int, default=4096)
    parser.add_argument(
        "--balanced-train-remainder",
        action="store_true",
        help=(
            "Allow train-images not divisible by train-clips and distribute "
            "the remainder deterministically, at most one extra image/clip."
        ),
    )
    parser.add_argument("--seen-images", type=int, default=256)
    parser.add_argument("--same-clip-unseen-images", type=int, default=256)
    parser.add_argument("--heldout-clips", type=int, default=64)
    parser.add_argument("--heldout-images", type=int, default=256)
    parser.add_argument("--val-images", type=int, default=256)
    parser.add_argument("--optimizer-steps", type=int, default=3000)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument(
        "--experiment-name",
        default="V11 clip-disjoint bridge-list contract",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _image_path(line: str) -> str:
    fields = line.split()
    if not fields:
        raise ValueError("empty dataset-list row")
    return fields[0]


def _clip(path: str) -> str:
    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot derive clip from image path: {path}")
    return "/".join(parts[:-1])


def _driver(clip: str) -> str:
    parts = clip.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def _rank(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(
        f"{int(seed)}:{namespace}:{value}".encode("utf-8")
    ).digest()


def _ordered(values: Iterable[str], *, seed: int, namespace: str) -> list[str]:
    return sorted(values, key=lambda value: (_rank(seed, namespace, value), value))


def _read(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({_image_path(line) for line in lines}) != len(lines):
        raise ValueError(f"dataset list contains duplicate image paths: {path}")
    return lines


def _group(lines: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        result[_clip(_image_path(line))].append(line)
    return dict(result)


def _choose_lines(
    lines: list[str],
    count: int,
    *,
    seed: int,
    namespace: str,
) -> list[str]:
    if int(count) > len(lines):
        raise ValueError(
            f"requested {count} rows from a population of {len(lines)}"
        )
    return _ordered(
        lines,
        seed=seed,
        namespace=namespace,
    )[: int(count)]


def _write(path: Path, lines: list[str], order: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(lines, key=lambda line: order[_image_path(line)])
    path.write_text("\n".join(ordered) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statistics(lines: list[str]) -> dict[str, object]:
    clips = [_clip(_image_path(line)) for line in lines]
    return {
        "images": len(lines),
        "clips": len(set(clips)),
        "drivers": dict(sorted(Counter(_driver(clip) for clip in clips).items())),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    train_source = Path(args.train_list).expanduser().resolve()
    val_source = Path(args.val_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_lines = _read(train_source)
    val_lines = _read(val_source)
    train_by_clip = _group(train_lines)
    val_by_clip = _group(val_lines)

    per_train_clip, train_remainder = divmod(
        int(args.train_images), int(args.train_clips)
    )
    balanced_remainder = bool(
        getattr(args, "balanced_train_remainder", False)
    )
    if train_remainder and not balanced_remainder:
        raise ValueError("train-images must divide evenly across train-clips")
    max_train_per_clip = per_train_clip + int(train_remainder > 0)
    eligible_train_clips = [
        clip
        for clip, lines in train_by_clip.items()
        if len(lines) >= max_train_per_clip
    ]
    ordered_train_clips = _ordered(
        eligible_train_clips,
        seed=args.seed,
        namespace="train-clips",
    )
    if len(ordered_train_clips) < int(args.train_clips) + int(args.heldout_clips):
        raise ValueError("not enough clips for disjoint bridge partitions")
    selected_train_clips = ordered_train_clips[: int(args.train_clips)]
    selected_train_clip_set = set(selected_train_clips)
    remaining_clips = [
        clip for clip in ordered_train_clips if clip not in selected_train_clip_set
    ]
    selected_heldout_clips = _ordered(
        remaining_clips,
        seed=args.seed,
        namespace="heldout-clips",
    )[: int(args.heldout_clips)]

    train_subset: list[str] = []
    selected_by_clip: dict[str, list[str]] = {}
    for clip_index, clip in enumerate(selected_train_clips):
        clip_image_count = per_train_clip + int(clip_index < train_remainder)
        chosen = _choose_lines(
            train_by_clip[clip],
            clip_image_count,
            seed=args.seed,
            namespace=f"train-frames:{clip}",
        )
        selected_by_clip[clip] = chosen
        train_subset.extend(chosen)

    seen_clips = _ordered(
        selected_train_clips,
        seed=args.seed,
        namespace="seen-eval-clips",
    )[: int(args.seen_images)]
    if len(seen_clips) != int(args.seen_images):
        raise ValueError("seen-images cannot exceed selected train clips")
    seen_subset = [
        _choose_lines(
            selected_by_clip[clip],
            1,
            seed=args.seed,
            namespace=f"seen-frame:{clip}",
        )[0]
        for clip in seen_clips
    ]

    selected_paths_by_clip = {
        clip: {_image_path(value) for value in selected_by_clip[clip]}
        for clip in selected_train_clips
    }
    same_clip_candidates = {
        clip: [
            line
            for line in train_by_clip[clip]
            if _image_path(line) not in selected_paths_by_clip[clip]
        ]
        for clip in selected_train_clips
    }
    same_clip_pool = [
        clip for clip, lines in same_clip_candidates.items() if lines
    ]
    same_clip_eval_clips = _ordered(
        same_clip_pool,
        seed=args.seed,
        namespace="same-clip-unseen-clips",
    )[: int(args.same_clip_unseen_images)]
    if len(same_clip_eval_clips) != int(args.same_clip_unseen_images):
        raise ValueError("not enough selected clips with unseen frames")
    same_clip_unseen = [
        _choose_lines(
            same_clip_candidates[clip],
            1,
            seed=args.seed,
            namespace=f"same-clip-unseen-frame:{clip}",
        )[0]
        for clip in same_clip_eval_clips
    ]

    if int(args.heldout_images) % int(args.heldout_clips):
        raise ValueError("heldout-images must divide evenly across heldout-clips")
    per_heldout_clip = int(args.heldout_images) // int(args.heldout_clips)
    heldout_subset: list[str] = []
    for clip in selected_heldout_clips:
        heldout_subset.extend(
            _choose_lines(
                train_by_clip[clip],
                per_heldout_clip,
                seed=args.seed,
                namespace=f"heldout-frames:{clip}",
            )
        )

    ordered_val_clips = _ordered(
        val_by_clip,
        seed=args.seed,
        namespace="val-clips",
    )
    val_base, val_extra = divmod(int(args.val_images), len(ordered_val_clips))
    if val_base < 1:
        raise ValueError("val-images must cover every validation clip")
    val_extra_clips = set(
        _ordered(
            ordered_val_clips,
            seed=args.seed,
            namespace="val-extra-clips",
        )[:val_extra]
    )
    val_subset: list[str] = []
    for clip in ordered_val_clips:
        count = val_base + int(clip in val_extra_clips)
        val_subset.extend(
            _choose_lines(
                val_by_clip[clip],
                count,
                seed=args.seed,
                namespace=f"val-frames:{clip}",
            )
        )

    train_paths = {_image_path(line) for line in train_subset}
    seen_paths = {_image_path(line) for line in seen_subset}
    same_paths = {_image_path(line) for line in same_clip_unseen}
    heldout_paths = {_image_path(line) for line in heldout_subset}
    val_paths = {_image_path(line) for line in val_subset}
    train_clips = {_clip(path) for path in train_paths}
    heldout_clips = {_clip(path) for path in heldout_paths}
    val_clips = {_clip(path) for path in val_paths}
    checks = {
        "train_count_exact": len(train_paths) == int(args.train_images),
        "seen_count_exact": len(seen_paths) == int(args.seen_images),
        "same_clip_unseen_count_exact": len(same_paths)
        == int(args.same_clip_unseen_images),
        "heldout_count_exact": len(heldout_paths) == int(args.heldout_images),
        "val_count_exact": len(val_paths) == int(args.val_images),
        "seen_is_train_subset": seen_paths <= train_paths,
        "same_clip_images_disjoint_from_train": not (same_paths & train_paths),
        "same_clip_uses_train_clips": {
            _clip(path) for path in same_paths
        }
        <= train_clips,
        "heldout_images_disjoint_from_train": not (heldout_paths & train_paths),
        "heldout_clips_disjoint_from_train": not (heldout_clips & train_clips),
        "val_images_disjoint_from_train": not (val_paths & train_paths),
        "val_clips_disjoint_from_train": not (val_clips & train_clips),
        "same_clip_and_heldout_images_disjoint": not (
            same_paths & heldout_paths
        ),
        "same_clip_and_heldout_clips_disjoint": not (
            {_clip(path) for path in same_paths} & heldout_clips
        ),
        "heldout_and_val_images_disjoint": not (heldout_paths & val_paths),
        "heldout_and_val_clips_disjoint": not (heldout_clips & val_clips),
    }
    if not all(checks.values()):
        raise RuntimeError(f"bridge-list contract failed: {checks}")

    train_order = {
        _image_path(line): index for index, line in enumerate(train_lines)
    }
    val_order = {_image_path(line): index for index, line in enumerate(val_lines)}
    destinations = {
        "train": output_dir
        / f"train_clip{int(args.train_clips)}_image{int(args.train_images)}.txt",
        "seen_train": output_dir
        / f"seen_train_image{int(args.seen_images)}.txt",
        "same_clip_unseen": output_dir
        / f"same_clip_unseen_image{int(args.same_clip_unseen_images)}.txt",
        "heldout_clip": output_dir
        / f"heldout_clip_image{int(args.heldout_images)}.txt",
        "val": output_dir
        / f"val_clip_balanced_image{int(args.val_images)}.txt",
    }
    _write(destinations["train"], train_subset, train_order)
    _write(destinations["seen_train"], seen_subset, train_order)
    _write(destinations["same_clip_unseen"], same_clip_unseen, train_order)
    _write(destinations["heldout_clip"], heldout_subset, train_order)
    _write(destinations["val"], val_subset, val_order)

    optimizer_steps = int(getattr(args, "optimizer_steps", 3000))
    effective_batch_size = int(
        getattr(args, "effective_batch_size", 16)
    )
    if optimizer_steps < 1 or effective_batch_size < 1:
        raise ValueError("bridge exposure contract must be positive")
    sample_exposures = optimizer_steps * effective_batch_size
    report: dict[str, object] = {
        "experiment": str(
            getattr(
                args,
                "experiment_name",
                "V11 clip-disjoint bridge-list contract",
            )
        ),
        "seed": int(args.seed),
        "sources": {
            "train": str(train_source),
            "val": str(val_source),
        },
        "parameters": {
            "train_clips": int(args.train_clips),
            "train_images": int(args.train_images),
            "balanced_train_remainder": balanced_remainder,
            "train_images_per_clip_base": per_train_clip,
            "train_clips_with_one_extra_image": train_remainder,
            "seen_images": int(args.seen_images),
            "same_clip_unseen_images": int(args.same_clip_unseen_images),
            "heldout_clips": int(args.heldout_clips),
            "heldout_images": int(args.heldout_images),
            "val_images": int(args.val_images),
            "effective_batch_size": effective_batch_size,
            "optimizer_steps": optimizer_steps,
            "training_sample_exposures": sample_exposures,
            "approximate_subset_epochs": float(sample_exposures)
            / int(args.train_images),
        },
        "lists": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                **_statistics(
                    {
                        "train": train_subset,
                        "seen_train": seen_subset,
                        "same_clip_unseen": same_clip_unseen,
                        "heldout_clip": heldout_subset,
                        "val": val_subset,
                    }[name]
                ),
            }
            for name, path in destinations.items()
        },
        "checks": checks,
        "passed": all(checks.values()),
        "test_set_used": False,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
