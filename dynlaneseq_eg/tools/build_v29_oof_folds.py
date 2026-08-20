from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FOLD_NAMES = ("a", "b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build two deterministic, image-balanced, clip-disjoint CULane "
            "training folds for V29 cross-fitted support models."
        )
    )
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--train-gt-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_path(row: str) -> str:
    fields = row.split()
    if not fields:
        raise ValueError("empty CULane list row")
    return fields[0]


def _clip(image_path: str) -> str:
    parts = image_path.split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot derive clip from image path: {image_path}")
    return "/".join(parts[:-1])


def _rank(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(
        f"{int(seed)}:{namespace}:{value}".encode("utf-8")
    ).digest()


def _read(path: Path) -> list[str]:
    rows = [
        row
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if not rows:
        raise ValueError(f"empty CULane list: {path}")
    images = [_image_path(row) for row in rows]
    if len(images) != len(set(images)):
        raise ValueError(f"duplicate image paths in CULane list: {path}")
    return rows


def _assign_clips(rows: list[str], seed: int) -> dict[str, str]:
    counts = Counter(_clip(_image_path(row)) for row in rows)
    ordered = sorted(
        counts,
        key=lambda clip: (
            -counts[clip],
            _rank(seed, "v29-fold-clip", clip),
            clip,
        ),
    )
    fold_sizes = {name: 0 for name in FOLD_NAMES}
    assignment: dict[str, str] = {}
    for clip in ordered:
        destination = min(
            FOLD_NAMES,
            key=lambda name: (
                fold_sizes[name],
                _rank(seed, f"v29-fold-tie-{clip}", name),
                name,
            ),
        )
        assignment[clip] = destination
        fold_sizes[destination] += int(counts[clip])
    return assignment


def _write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build(
    *,
    train_list: Path,
    train_gt_list: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    train_list = train_list.expanduser().resolve()
    train_gt_list = train_gt_list.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    image_rows = _read(train_list)
    gt_rows = _read(train_gt_list)
    image_paths = [_image_path(row) for row in image_rows]
    gt_paths = [_image_path(row) for row in gt_rows]
    if image_paths != gt_paths:
        raise ValueError(
            "train.txt and train_gt.txt must contain identical ordered images"
        )

    assignment = _assign_clips(image_rows, int(seed))
    fold_image_rows: dict[str, list[str]] = defaultdict(list)
    fold_gt_rows: dict[str, list[str]] = defaultdict(list)
    for image_row, gt_row in zip(image_rows, gt_rows):
        fold = assignment[_clip(_image_path(image_row))]
        fold_image_rows[fold].append(image_row)
        fold_gt_rows[fold].append(gt_row)

    fold_reports: dict[str, Any] = {}
    for fold in FOLD_NAMES:
        image_path = output_dir / f"fold_{fold}_train.txt"
        gt_path = output_dir / f"fold_{fold}_train_gt.txt"
        _write_rows(image_path, fold_image_rows[fold])
        _write_rows(gt_path, fold_gt_rows[fold])
        paths = [_image_path(row) for row in fold_image_rows[fold]]
        clips = {_clip(path) for path in paths}
        fold_reports[fold] = {
            "fold": fold,
            "image_list": str(image_path),
            "image_list_sha256": _sha256(image_path),
            "gt_list": str(gt_path),
            "gt_list_sha256": _sha256(gt_path),
            "row_count": len(paths),
            "clip_count": len(clips),
            "clips": sorted(clips),
        }

    path_sets = {
        fold: {_image_path(row) for row in fold_image_rows[fold]}
        for fold in FOLD_NAMES
    }
    clip_sets = {
        fold: {_clip(path) for path in path_sets[fold]}
        for fold in FOLD_NAMES
    }
    union = path_sets["a"] | path_sets["b"]
    checks = {
        "source_lists_ordered_images_exact": image_paths == gt_paths,
        "image_union_exact": union == set(image_paths),
        "image_intersection_empty": not (path_sets["a"] & path_sets["b"]),
        "clip_intersection_empty": not (clip_sets["a"] & clip_sets["b"]),
        "row_count_exact": sum(len(value) for value in path_sets.values())
        == len(image_paths),
        "fold_size_difference_at_most_largest_clip": abs(
            len(path_sets["a"]) - len(path_sets["b"])
        )
        <= max(Counter(_clip(path) for path in image_paths).values()),
    }
    report: dict[str, Any] = {
        "experiment": "V29 deterministic clip-disjoint OOF support folds",
        "seed": int(seed),
        "source": {
            "train_list": str(train_list),
            "train_list_sha256": _sha256(train_list),
            "train_gt_list": str(train_gt_list),
            "train_gt_list_sha256": _sha256(train_gt_list),
            "row_count": len(image_paths),
            "clip_count": len(set(_clip(path) for path in image_paths)),
        },
        "folds": fold_reports,
        "checks": checks,
        "passed": all(checks.values()),
    }
    report_path = output_dir / "fold_contract.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_fold_contract(
    report_path: str | Path,
    *,
    fold: str,
    list_path: str | Path,
) -> dict[str, Any]:
    fold = str(fold).lower()
    if fold not in FOLD_NAMES:
        raise ValueError(f"invalid V29 fold: {fold!r}")
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("V29 fold contract did not pass")
    entry = report.get("folds", {}).get(fold)
    if not isinstance(entry, dict):
        raise ValueError(f"V29 fold contract is missing fold {fold}")
    list_path = Path(list_path).expanduser().resolve()
    expected_path = Path(str(entry["image_list"])).expanduser().resolve()
    if list_path != expected_path:
        raise ValueError(
            f"V29 fold {fold} list mismatch: {list_path} != {expected_path}"
        )
    if _sha256(list_path) != str(entry["image_list_sha256"]):
        raise ValueError(f"V29 fold {fold} image-list SHA mismatch")
    rows = _read(list_path)
    if len(rows) != int(entry["row_count"]):
        raise ValueError(f"V29 fold {fold} row-count mismatch")
    return {
        "passed": True,
        "split": f"oof_fold_{fold}",
        "fold": fold,
        "list_path": str(list_path),
        "list_sha256": _sha256(list_path),
        "fold_contract": str(report_path),
        "fold_contract_sha256": _sha256(report_path),
        "expected_nonempty_rows": len(rows),
        "observed_nonempty_rows": len(rows),
        "physical_line_count": len(rows),
        "clip_count": int(entry["clip_count"]),
        "new_list_constructed": True,
        "rows_removed": int(report["source"]["row_count"]) - len(rows),
        "deduplication_performed": False,
        "clip_filtering_performed": True,
        "image_filtering_performed": False,
    }


def validate_support_fold_contract(
    report_path: str | Path,
    *,
    fold: str,
    gt_list_path: str | Path,
) -> dict[str, Any]:
    fold = str(fold).lower()
    if fold not in FOLD_NAMES:
        raise ValueError(f"invalid V29 fold: {fold!r}")
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("V29 fold contract did not pass")
    entry = report.get("folds", {}).get(fold)
    if not isinstance(entry, dict):
        raise ValueError(f"V29 fold contract is missing fold {fold}")
    gt_list_path = Path(gt_list_path).expanduser().resolve()
    expected_path = Path(str(entry["gt_list"])).expanduser().resolve()
    if gt_list_path != expected_path:
        raise ValueError(
            f"V29 fold {fold} GT-list mismatch: {gt_list_path} != {expected_path}"
        )
    if _sha256(gt_list_path) != str(entry["gt_list_sha256"]):
        raise ValueError(f"V29 fold {fold} GT-list SHA mismatch")
    rows = _read(gt_list_path)
    if len(rows) != int(entry["row_count"]):
        raise ValueError(f"V29 fold {fold} GT row-count mismatch")
    return {
        "passed": True,
        "fold": fold,
        "gt_list_path": str(gt_list_path),
        "gt_list_sha256": _sha256(gt_list_path),
        "fold_contract": str(report_path),
        "fold_contract_sha256": _sha256(report_path),
        "row_count": len(rows),
        "clip_count": int(entry["clip_count"]),
    }


def main() -> None:
    args = parse_args()
    report = build(
        train_list=Path(args.train_list),
        train_gt_list=Path(args.train_gt_list),
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
