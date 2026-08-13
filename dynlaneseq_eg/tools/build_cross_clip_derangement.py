from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic one-to-one wrong-image list in which no "
            "image is paired with itself or with an image from the same clip."
        )
    )
    parser.add_argument("--input-list", required=True)
    parser.add_argument("--output-list", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=3407)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[str]:
    rows = [row for row in path.read_text(encoding="utf-8").splitlines() if row]
    paths = [_image_path(row) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError(f"input list contains duplicate image paths: {path}")
    if len(rows) < 2:
        raise ValueError("cross-clip derangement requires at least two images")
    return rows


def _cyclic_derangement(rows: list[str], seed: int) -> list[int]:
    """Return partner indices as a permutation with no same-clip edges.

    We search all cyclic shifts of a deterministic hash ordering.  Among
    feasible shifts we maximize different-driver pairs, which gives a stronger
    negative control on train-derived lists while remaining feasible on the
    single-driver validation split.
    """

    paths = [_image_path(row) for row in rows]
    clips = [_clip(path) for path in paths]
    drivers = [_driver(clip) for clip in clips]
    # Keep every clip in one contiguous block.  If the largest clip contains
    # at most half the list, a cyclic shift by that block size is guaranteed
    # to leave every block; scanning all shifts then lets us prefer a stronger
    # different-driver pairing without sacrificing feasibility.
    clip_order = sorted(
        set(clips),
        key=lambda clip: (_rank(seed, "cross-clip-order", clip), clip),
    )
    ordered = [
        index
        for clip in clip_order
        for index in sorted(
            (index for index, value in enumerate(clips) if value == clip),
            key=lambda index: (
                _rank(seed, "cross-clip-frame", paths[index]),
                paths[index],
            ),
        )
    ]
    best: tuple[int, bytes, list[int]] | None = None
    for shift in range(1, len(rows)):
        partners = [-1] * len(rows)
        valid = True
        different_driver = 0
        for position, source_index in enumerate(ordered):
            partner_index = ordered[(position + shift) % len(rows)]
            if clips[source_index] == clips[partner_index]:
                valid = False
                break
            partners[source_index] = partner_index
            different_driver += int(
                drivers[source_index] != drivers[partner_index]
            )
        if not valid:
            continue
        tie_break = _rank(seed, "cross-clip-shift", str(shift))
        candidate = (different_driver, tie_break, partners)
        if best is None or candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[1] < best[1]
        ):
            best = candidate
    if best is None:
        clip_counts: dict[str, int] = {}
        for clip in clips:
            clip_counts[clip] = clip_counts.get(clip, 0) + 1
        raise ValueError(
            "no deterministic cyclic cross-clip derangement exists; "
            f"clip counts={dict(sorted(clip_counts.items()))}"
        )
    return best[2]


def build(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.input_list).expanduser().resolve()
    destination = Path(args.output_list).expanduser().resolve()
    report_path = Path(args.output_json).expanduser().resolve()
    rows = _read(source)
    partner_indices = _cyclic_derangement(rows, int(args.seed))
    partner_rows = [rows[index] for index in partner_indices]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(partner_rows) + "\n", encoding="utf-8")

    source_paths = [_image_path(row) for row in rows]
    partner_paths = [_image_path(row) for row in partner_rows]
    source_clips = [_clip(path) for path in source_paths]
    partner_clips = [_clip(path) for path in partner_paths]
    source_drivers = [_driver(clip) for clip in source_clips]
    partner_drivers = [_driver(clip) for clip in partner_clips]
    same_image = sum(
        left == right for left, right in zip(source_paths, partner_paths)
    )
    same_clip = sum(
        left == right for left, right in zip(source_clips, partner_clips)
    )
    same_driver = sum(
        left == right for left, right in zip(source_drivers, partner_drivers)
    )
    permutation_exact = sorted(source_paths) == sorted(partner_paths)
    checks = {
        "same_image_partner_count_zero": same_image == 0,
        "same_clip_partner_count_zero": same_clip == 0,
        "partner_is_exact_permutation": permutation_exact,
        "count_exact": len(source_paths) == len(partner_paths),
    }
    report: dict[str, object] = {
        "experiment": "deterministic cross-clip wrong-image derangement",
        "seed": int(args.seed),
        "input_list": str(source),
        "output_list": str(destination),
        "input_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
        "image_count": len(rows),
        "clip_count": len(set(source_clips)),
        "driver_count": len(set(source_drivers)),
        "same_image_partner_count": same_image,
        "same_clip_partner_count": same_clip,
        "same_driver_partner_count": same_driver,
        "different_driver_partner_fraction": 1.0
        - float(same_driver) / float(len(rows)),
        "checks": checks,
        "passed": all(checks.values()),
        "pairs": [
            {
                "source_image": source_path,
                "source_clip": source_clip,
                "partner_image": partner_path,
                "partner_clip": partner_clip,
            }
            for source_path, source_clip, partner_path, partner_clip in zip(
                source_paths, source_clips, partner_paths, partner_clips
            )
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build(parse_args())
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "pairs"},
            indent=2,
            sort_keys=True,
        )
    )
    if not bool(report["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
