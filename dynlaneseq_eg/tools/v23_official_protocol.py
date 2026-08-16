from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


OFFICIAL_V23_CULANE_LISTS = {
    "train": ("list/train.txt", 88_880),
    "val": ("list/val.txt", 9_675),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_v23_culane_list_contract(
    dataset_root: str | Path,
    *,
    split: str,
    supplied_path: str | Path | None = None,
) -> dict[str, Any]:
    """Lock V23 to the untouched official CULane image populations.

    V23 deliberately uses ``list/train.txt`` rather than the convenience
    ``train_gt.txt`` list.  Both contain 88,880 image rows in the reference
    release, but ``train_gt.txt`` appends segmentation paths and flags.  No
    line, repeated entry, clip, or image is removed by this contract.
    """

    if split not in OFFICIAL_V23_CULANE_LISTS:
        raise ValueError(f"unsupported V23 CULane split: {split!r}")
    relative_path, expected_rows = OFFICIAL_V23_CULANE_LISTS[split]
    root = Path(dataset_root).expanduser().resolve()
    expected_path = (root / relative_path).resolve()
    actual_path = (
        expected_path
        if supplied_path is None
        else Path(supplied_path).expanduser().resolve()
    )
    if actual_path != expected_path:
        raise ValueError(
            f"V23 requires the official CULane {split} list at "
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
        "repeated_row_count_preserved": len(rows) - len(set(rows)),
        "new_list_constructed": False,
        "rows_removed": 0,
        "deduplication_performed": False,
        "clip_filtering_performed": False,
        "image_filtering_performed": False,
    }
