from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
FIXED_FOLDS = 2


def _image_path(row: str) -> str:
    fields = row.split()
    if not fields:
        raise ValueError("empty CULane list row")
    return fields[0]


def _clip(row: str) -> str:
    path = Path(_image_path(row))
    if len(path.parts) < 3:
        raise ValueError(f"cannot derive CULane clip from {path}")
    return path.parent.as_posix()


def _rank(clip: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:v25-s1-oof:{clip}".encode()).digest()


def build_oof_folds(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    folds: int = FIXED_FOLDS,
    seed: int = FIXED_SEED,
) -> dict[str, Any]:
    if int(folds) != FIXED_FOLDS:
        raise ValueError("V25-S1 mechanism gate is predeclared as exactly two folds")
    if int(seed) != FIXED_SEED:
        raise ValueError("V25-S1 uses the fixed seed 3407")
    root = Path(dataset_root).expanduser().resolve()
    official = official_v23_culane_list_contract(root, split="train")
    source = Path(official["list_path"])
    rows = [row for row in source.read_text(encoding="utf-8").splitlines() if row]
    clips: dict[str, list[str]] = {}
    for row in rows:
        clips.setdefault(_clip(row), []).append(row)

    # Deterministic greedy bin packing keeps every clip intact while balancing
    # image counts. Hash ranking prevents dependence on filesystem/list order.
    ordered_clips = sorted(
        clips,
        key=lambda name: (-len(clips[name]), _rank(name, seed=seed), name),
    )
    fold_clips: list[set[str]] = [set() for _ in range(folds)]
    fold_counts = [0 for _ in range(folds)]
    for clip in ordered_clips:
        target = min(range(folds), key=lambda index: (fold_counts[index], index))
        fold_clips[target].add(clip)
        fold_counts[target] += len(clips[clip])

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    heldout_union: list[str] = []
    for fold in range(folds):
        holdout = [row for row in rows if _clip(row) in fold_clips[fold]]
        train = [row for row in rows if _clip(row) not in fold_clips[fold]]
        train_path = destination / f"fold_{fold}_train.txt"
        holdout_path = destination / f"fold_{fold}_holdout.txt"
        train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
        holdout_path.write_text("\n".join(holdout) + "\n", encoding="utf-8")
        train_clips = {_clip(row) for row in train}
        holdout_clips = {_clip(row) for row in holdout}
        if train_clips & holdout_clips:
            raise RuntimeError("V25-S1 fold leaked a clip across train/holdout")
        heldout_union.extend(holdout)
        records.append(
            {
                "fold": fold,
                "train_list": str(train_path),
                "train_sha256": sha256_file(train_path),
                "train_rows": len(train),
                "train_clips": len(train_clips),
                "holdout_list": str(holdout_path),
                "holdout_sha256": sha256_file(holdout_path),
                "holdout_rows": len(holdout),
                "holdout_clips": len(holdout_clips),
                "clip_disjoint": True,
            }
        )

    checks = {
        "official_population_exact": len(rows)
        == int(official["expected_nonempty_rows"]),
        "heldout_union_count_exact": len(heldout_union) == len(rows),
        "heldout_union_rows_exact": sorted(heldout_union) == sorted(rows),
        "heldout_folds_disjoint": len(heldout_union) == len(set(heldout_union)),
        "all_fold_train_holdout_unions_exact": all(
            record["train_rows"] + record["holdout_rows"] == len(rows)
            for record in records
        ),
        "no_filtering_or_deduplication": True,
    }
    if not all(checks.values()):
        raise RuntimeError("V25-S1 OOF fold contract failed: " + json.dumps(checks))
    manifest = {
        "experiment": "V25-S1 two-fold out-of-fold immutable-bank selector",
        "seed": int(seed),
        "fold_count": int(folds),
        "official_train_population_contract": official,
        "folds": records,
        "checks": checks,
        "passed": True,
        "validation_used": False,
        "test_set_used": False,
    }
    manifest_path = destination / "oof_fold_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_fold_training_population(
    manifest_path: str | Path,
    *,
    fold: int,
    supplied_train_list: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("passed") is not True or int(manifest.get("fold_count", -1)) != 2:
        raise ValueError("invalid V25-S1 OOF fold manifest")
    rows = manifest.get("folds", [])
    if not 0 <= int(fold) < len(rows):
        raise ValueError(f"invalid V25-S1 fold index {fold}")
    record = rows[int(fold)]
    train_path = Path(supplied_train_list).expanduser().resolve()
    expected_path = Path(record["train_list"]).expanduser().resolve()
    checks = {
        "path_exact": train_path == expected_path,
        "sha256_exact": sha256_file(train_path) == str(record["train_sha256"]),
        "row_count_exact": len(
            [row for row in train_path.read_text(encoding="utf-8").splitlines() if row]
        )
        == int(record["train_rows"]),
        "clip_disjoint": record.get("clip_disjoint") is True,
    }
    if not all(checks.values()):
        raise ValueError("V25-S1 fold training contract failed: " + json.dumps(checks))
    return {
        "passed": True,
        "split": f"oof_fold_{fold}_train",
        "list_path": str(train_path),
        "list_sha256": str(record["train_sha256"]),
        "expected_nonempty_rows": int(record["train_rows"]),
        "observed_nonempty_rows": int(record["train_rows"]),
        "holdout_list": str(record["holdout_list"]),
        "holdout_sha256": str(record["holdout_sha256"]),
        "holdout_rows": int(record["holdout_rows"]),
        "fold": int(fold),
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "checks": checks,
        "new_list_constructed": True,
        "rows_removed": 0,
        "deduplication_performed": False,
        "clip_filtering_performed": False,
        "image_filtering_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed clip-disjoint V25-S1 two-fold OOF lists."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=FIXED_FOLDS)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_oof_folds(
        args.dataset_root,
        args.output_dir,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
