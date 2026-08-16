from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple

import cv2
from tqdm import tqdm


EXPECTED_TRAIN_IMAGES = 88880
EXPECTED_VAL_IMAGES = 9675


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CondLSTR CULane PKLs from the untouched official train.txt and val.txt populations."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-list", default="")
    parser.add_argument("--val-list", default="")
    parser.add_argument("--version", default="official_train_val_v1")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_population(path: Path) -> List[str]:
    population: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        image_path = raw_line.split()[0]
        population.append(image_path[1:] if image_path.startswith("/") else image_path)
    return population


def annotation_path(dataset_root: Path, relative_image: str) -> Path:
    return dataset_root / str(Path(relative_image).with_suffix(".lines.txt"))


def audit_population(dataset_root: Path, population: Sequence[str]) -> Dict[str, Any]:
    missing_images: List[str] = []
    missing_annotations: List[str] = []
    empty_annotations = 0
    for relative_image in population:
        if not (dataset_root / relative_image).is_file():
            missing_images.append(relative_image)
        anno_path = annotation_path(dataset_root, relative_image)
        if not anno_path.is_file():
            missing_annotations.append(relative_image)
        elif anno_path.stat().st_size == 0:
            empty_annotations += 1
    return {
        "count": len(population),
        "unique_count": len(set(population)),
        "duplicate_entries": len(population) - len(set(population)),
        "empty_annotation_files_kept": empty_annotations,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
    }


def _dedupe_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # CondLSTR removes duplicate points inside a polyline.  Keep that behavior
    # deterministic without removing or deduplicating any image records.
    seen = set()
    output: List[Tuple[float, float]] = []
    for point in points:
        if point in seen:
            continue
        seen.add(point)
        output.append(point)
    return output


def build_info(task: Tuple[str, str]) -> Dict[str, Any]:
    dataset_root_raw, relative_image = task
    dataset_root = Path(dataset_root_raw)
    image_path = dataset_root / relative_image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    lanes: List[List[Tuple[float, float]]] = []
    for line in annotation_path(dataset_root, relative_image).read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        points = [
            (values[index], values[index + 1])
            for index in range(0, len(values) - 1, 2)
            if values[index] >= 0.0 and values[index + 1] >= 0.0
        ]
        points = sorted(_dedupe_points(points), key=lambda point: point[1])
        if len(points) >= 2:
            lanes.append(points)

    comma_name = relative_image.replace("/", ",")
    return {
        "img_name": comma_name,
        "img_path": str(image_path),
        "img_mask": str(Path("img_mask") / comma_name.replace(".jpg", ".npz")),
        "img_shape": image.shape,
        "lane_points": lanes,
        "lane_attris": [0 for _ in lanes],
    }


def build_split(dataset_root: Path, population: Sequence[str], workers: int, description: str) -> List[Dict[str, Any]]:
    tasks = ((str(dataset_root), relative_image) for relative_image in population)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        return list(tqdm(executor.map(build_info, tasks, chunksize=32), total=len(population), desc=description, ncols=90))


def write_pickle(path: Path, infos: List[Dict[str, Any]], version: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"infos": infos, "metadata": {"version": version}}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_list = Path(args.train_list).expanduser().resolve() if args.train_list else dataset_root / "list" / "train.txt"
    val_list = Path(args.val_list).expanduser().resolve() if args.val_list else dataset_root / "list" / "val.txt"
    train_population = read_population(train_list)
    val_population = read_population(val_list)

    manifest: Dict[str, Any] = {
        "protocol": "official CULane train.txt -> val.txt; no image exclusion or deduplication",
        "dataset_root": str(dataset_root),
        "version": args.version,
        "train_list": str(train_list),
        "train_list_sha256": sha256_file(train_list),
        "val_list": str(val_list),
        "val_list_sha256": sha256_file(val_list),
        "train": audit_population(dataset_root, train_population),
        "val": audit_population(dataset_root, val_population),
        "train_val_overlap": len(set(train_population) & set(val_population)),
    }
    if len(train_population) != EXPECTED_TRAIN_IMAGES or len(val_population) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Official population mismatch: train={len(train_population)} (expected {EXPECTED_TRAIN_IMAGES}), "
            f"val={len(val_population)} (expected {EXPECTED_VAL_IMAGES})"
        )
    for split in ("train", "val"):
        if manifest[split]["missing_images"] or manifest[split]["missing_annotations"]:
            raise RuntimeError(f"Missing CULane files in {split}; see manifest")
    if manifest["train_val_overlap"]:
        raise RuntimeError(f"Official train/val overlap is non-zero: {manifest['train_val_overlap']}")

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else dataset_root / f"culane_official_protocol_{args.version}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.audit_only:
        train_infos = build_split(dataset_root, train_population, args.workers, "CondLSTR official train PKL")
        val_infos = build_split(dataset_root, val_population, args.workers, "CondLSTR official val PKL")
        train_output = dataset_root / f"culane_infos_train_{args.version}.pkl"
        val_output = dataset_root / f"culane_infos_val_{args.version}.pkl"
        write_pickle(train_output, train_infos, args.version)
        write_pickle(val_output, val_infos, args.version)
        manifest["outputs"] = {
            "train_pkl": str(train_output),
            "train_pkl_sha256": sha256_file(train_output),
            "val_pkl": str(val_output),
            "val_pkl_sha256": sha256_file(val_output),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
