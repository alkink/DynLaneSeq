from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

OFFICIAL_CULANE_LISTS = {
    "train": ("list/train_gt.txt", 88_880),
    "val": ("list/val.txt", 9_675),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_culane_list_contract(
    dataset_root: str | Path,
    *,
    split: str,
    supplied_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require the untouched official CULane train/validation population.

    This contract deliberately does not construct a replacement list and does
    not remove repeated entries.  Repeated rows, if present in the official
    file, remain part of the training/evaluation population.
    """

    if split not in OFFICIAL_CULANE_LISTS:
        raise ValueError(f"unsupported official CULane split: {split!r}")
    relative_path, expected_rows = OFFICIAL_CULANE_LISTS[split]
    root = Path(dataset_root).expanduser().resolve()
    expected_path = (root / relative_path).resolve()
    actual_path = (
        expected_path
        if supplied_path is None
        else Path(supplied_path).expanduser().resolve()
    )
    if actual_path != expected_path:
        raise ValueError(
            f"V22 requires the official CULane {split} list at "
            f"{expected_path}, got {actual_path}"
        )
    if not actual_path.is_file():
        raise FileNotFoundError(actual_path)
    physical_lines = actual_path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in physical_lines if line.strip()]
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"official CULane {split} population mismatch: "
            f"expected {expected_rows}, found {len(rows)}"
        )
    repeated_rows = len(rows) - len(set(rows))
    return {
        "passed": True,
        "split": split,
        "dataset_root": str(root),
        "list_path": str(actual_path),
        "list_relative_path": relative_path,
        "list_sha256": _sha256_file(actual_path),
        "expected_nonempty_rows": int(expected_rows),
        "observed_nonempty_rows": len(rows),
        "physical_line_count": len(physical_lines),
        "repeated_row_count_preserved": repeated_rows,
        "new_list_constructed": False,
        "rows_removed": 0,
        "deduplication_performed": False,
        "clip_filtering_performed": False,
        "image_filtering_performed": False,
    }
