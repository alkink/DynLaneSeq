from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.curvelanes_metric import evaluate_image
from dynlaneseq_eg.evaluation.curvelanes_writer import outputs_to_curvelanes_records
from dynlaneseq_eg.factory import build_dataloader, build_model


CACHE_VERSION = 2
PREDICTION_FIELDS = (
    "pred_x_rows",
    "exist_logits",
    "quality_logits",
    "range_norm",
    "row_visibility_logits",
)


def _source_fingerprint() -> str:
    """Fingerprint inference-relevant project code to reject stale caches.

    A checkpoint and config can stay unchanged while preprocessing or model
    forward code changes between branches.  Reusing raw outputs across that
    change would make a sweep silently evaluate the old implementation.
    Hashing the package sources is intentionally conservative and cheap
    compared with one full validation inference pass.
    """
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if "__pycache__" in relative.parts or "tests" in relative.parts:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _prediction_stage(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(outputs.get("stage2"), dict):
        outputs = outputs["stage2"]
    elif isinstance(outputs.get("final"), dict):
        outputs = outputs["final"]
    stage = {
        field: outputs[field].detach().to(device="cpu")
        for field in PREDICTION_FIELDS
        if isinstance(outputs.get(field), torch.Tensor)
    }
    required = {"pred_x_rows", "exist_logits", "range_norm"}
    missing = required - set(stage)
    if missing:
        raise KeyError(f"Model output is missing prediction fields: {sorted(missing)}")
    return stage


def _cache_path(
    cache_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    split: str,
) -> Path:
    checkpoint_stat = checkpoint_path.stat()
    config_stat = config_path.stat()
    identity = "|".join(
        (
            str(CACHE_VERSION),
            str(config_path.resolve()),
            str(config_stat.st_size),
            str(config_stat.st_mtime_ns),
            str(checkpoint_path.resolve()),
            str(checkpoint_stat.st_size),
            str(checkpoint_stat.st_mtime_ns),
            str(dataset_root.resolve()),
            str(split),
            _source_fingerprint(),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{checkpoint_path.stem}_{digest}.pt"


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def collect_checkpoint_cache(
    config_path: Path,
    checkpoint_path: Path,
    dataset_root: Path,
    split: str,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    cache_dir: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    source_fingerprint = _source_fingerprint()
    cache_path = _cache_path(cache_dir, config_path, checkpoint_path, dataset_root, split)
    if reuse_cache and cache_path.exists():
        cache = _load_cache(cache_path)
        if int(cache.get("cache_version", -1)) != CACHE_VERSION:
            raise ValueError(f"Unsupported CurveLanes cache version in {cache_path}")
        cached_fingerprint = str(cache.get("metadata", {}).get("source_fingerprint", ""))
        if cached_fingerprint != source_fingerprint:
            raise ValueError(f"CurveLanes cache was produced by different project code: {cache_path}")
        cache.setdefault("metadata", {})["cache_path"] = str(cache_path)
        return cache

    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["pin_memory"] = bool(pin_memory)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    if num_workers > 0:
        cfg["dataloader"]["prefetch_factor"] = 2
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False

    train_cfg = cfg.get("training", {})
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")

    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    # A sweep must never partially load a checkpoint into the wrong
    # architecture and continue with randomly initialized missing weights.
    load_checkpoint(checkpoint_path, model, strict=True)
    model.eval()
    loader = build_dataloader(cfg, split=split, training=False)

    batches: list[dict[str, Any]] = []
    for images, _, metas in tqdm(loader, ncols=88, desc=f"cache {checkpoint_path.stem}"):
        if channels_last:
            images = images.to(device, non_blocking=bool(pin_memory), memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=bool(pin_memory))
        batches.append(
            {
                "outputs": _prediction_stage(model(images)),
                "metas": [_plain_value(meta) for meta in metas],
            }
        )

    payload = {
        "cache_version": CACHE_VERSION,
        "metadata": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "split": str(split),
            "num_samples": sum(len(batch["metas"]) for batch in batches),
            "source_fingerprint": source_fingerprint,
            "cache_path": str(cache_path),
        },
        "batches": batches,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    torch.save(payload, temporary_path)
    temporary_path.replace(cache_path)
    return payload


def _split_dir(split: str) -> str:
    return {"val": "valid", "train": "train", "test": "test"}.get(str(split), str(split))


def _image_sizes_from_cache(cache: dict[str, Any]) -> dict[str, tuple[int, int]]:
    image_sizes: dict[str, tuple[int, int]] = {}
    for batch in cache.get("batches", []):
        for meta in batch.get("metas", []):
            raw_file = str(meta["raw_file"])
            if raw_file in image_sizes:
                raise ValueError(f"Duplicate CurveLanes metadata for {raw_file!r}")
            width = int(meta["orig_w"])
            height = int(meta["orig_h"])
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid CurveLanes image size for {raw_file!r}: {width}x{height}")
            image_sizes[raw_file] = (width, height)
    if not image_sizes:
        raise ValueError("CurveLanes cache contains no image metadata")
    return image_sizes


def load_ground_truth(
    dataset_root: Path,
    split: str,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    split_dir = _split_dir(split)
    if split_dir == "test":
        raise ValueError("CurveLanes test/images is unlabeled; use split=val for post-process selection")
    list_path = dataset_root / split_dir / f"{split_dir}.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"CurveLanes split list not found: {list_path}")
    raw_files = [line.strip().lstrip("/") for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_files:
        raise ValueError(f"CurveLanes split list is empty: {list_path}")
    if image_sizes is not None:
        expected = set(raw_files)
        missing = expected - set(image_sizes)
        unexpected = set(image_sizes) - expected
        if missing or unexpected:
            raise ValueError(
                "Cached CurveLanes metadata does not exactly match the selected split: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    ground_truth: list[dict[str, Any]] = []
    for raw_file in tqdm(raw_files, ncols=88, desc=f"load CurveLanes {split_dir} ground truth"):
        image_path = dataset_root / split_dir / raw_file
        annotation_path = dataset_root / split_dir / "labels" / f"{Path(raw_file).stem}.lines.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"CurveLanes annotation not found: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8") as handle:
            gt_data = json.load(handle)
        if image_sizes is None:
            with Image.open(image_path) as image:
                image_size = image.size
        else:
            image_size = image_sizes[raw_file]
        ground_truth.append({"raw_file": raw_file, "gt_data": gt_data, "image_size": image_size})
    return ground_truth


def evaluate_records(records: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, float | int]:
    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        raw_file = str(record["raw_file"])
        if raw_file in predictions:
            raise ValueError(f"Duplicate CurveLanes prediction for {raw_file!r}")
        predictions[raw_file] = record

    expected = {str(item["raw_file"]) for item in ground_truth}
    missing = expected - set(predictions)
    unexpected = set(predictions) - expected
    if missing or unexpected:
        raise ValueError(
            "Prediction records do not exactly match the selected CurveLanes split: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    true_positive = false_positive = false_negative = 0
    for item in ground_truth:
        hits, pred_count, gt_count = evaluate_image(
            item["gt_data"],
            predictions[str(item["raw_file"])],
            item["image_size"],
        )
        true_positive += hits
        false_positive += pred_count - hits
        false_negative += gt_count - hits
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP": int(true_positive),
        "FP": int(false_positive),
        "FN": int(false_negative),
        "samples": int(len(ground_truth)),
    }


def _checkpoint_iteration(path: Path) -> int:
    if path.stem.startswith("iter_"):
        try:
            return int(path.stem[len("iter_") :])
        except ValueError:
            pass
    return 0


def evaluate_grid(
    cache: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    score_thresholds: list[float],
    quality_powers: list[float],
    top_ks: list[int],
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    row_visibility_thresh: float,
) -> list[dict[str, Any]]:
    checkpoint = Path(cache["metadata"]["checkpoint"])
    iteration = _checkpoint_iteration(checkpoint)
    combinations = [
        (score_threshold, quality_power, top_k)
        for top_k in top_ks
        for quality_power in quality_powers
        for score_threshold in score_thresholds
    ]
    rows: list[dict[str, Any]] = []
    for score_threshold, quality_power, top_k in tqdm(
        combinations,
        ncols=88,
        desc=f"grid {checkpoint.stem}",
    ):
        records: list[dict[str, Any]] = []
        for batch in cache["batches"]:
            records.extend(
                outputs_to_curvelanes_records(
                    batch["outputs"],
                    batch["metas"],
                    score_thresh=float(score_threshold),
                    min_pred_points=int(min_pred_points),
                    nms_distance_thresh_px=float(nms_distance_thresh_px),
                    nms_min_overlap_points=int(nms_min_overlap_points),
                    top_k=int(top_k),
                    row_visibility_thresh=float(row_visibility_thresh),
                    quality_score_power=float(quality_power),
                )
            )
        metrics = evaluate_records(records, ground_truth)
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "iteration": int(iteration),
                "score_thresh": float(score_threshold),
                "quality_power": float(quality_power),
                "top_k": int(top_k),
                **metrics,
            }
        )
    return rows


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    return (
        float(row["F1"]),
        float(row["Precision"]),
        float(row["Recall"]),
        -float(row["FP"]),
        -float(row["FN"]),
        -int(row["iteration"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cached CurveLanes validation sweep for score, quality calibration, and top-k post-processing."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--score-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--quality-powers", nargs="+", type=float, required=True)
    parser.add_argument("--top-ks", nargs="+", type=int, required=True)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--cache-dir", default="outputs/curvelanes_val_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_paths = [Path(path).expanduser().resolve() for path in args.checkpoints]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    for path in (config_path, dataset_root, *checkpoint_paths):
        if not path.exists():
            raise FileNotFoundError(path)

    if _split_dir(str(args.split)) == "test":
        raise ValueError("CurveLanes post-process selection must use the labelled validation split, not test")
    ground_truth: list[dict[str, Any]] | None = None
    all_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        cache = collect_checkpoint_cache(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            split=str(args.split),
            device=torch.device(args.device),
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory),
            cache_dir=Path(args.cache_dir),
            reuse_cache=bool(args.reuse_cache),
        )
        if ground_truth is None:
            ground_truth = load_ground_truth(
                dataset_root,
                str(args.split),
                image_sizes=_image_sizes_from_cache(cache),
            )
        all_rows.extend(
            evaluate_grid(
                cache=cache,
                ground_truth=ground_truth,
                score_thresholds=list(args.score_thresholds),
                quality_powers=list(args.quality_powers),
                top_ks=list(args.top_ks),
                min_pred_points=int(args.min_pred_points),
                nms_distance_thresh_px=float(args.nms_distance_thresh_px),
                nms_min_overlap_points=int(args.nms_min_overlap_points),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
        )

    if not all_rows:
        raise RuntimeError("No CurveLanes sweep results were produced")
    ranked = sorted(all_rows, key=selection_key, reverse=True)
    best = ranked[0]
    payload = {
        "selection_metric": "CurveLanes validation F1",
        "tie_breakers": ["Precision desc", "Recall desc", "FP asc", "FN asc", "iteration asc"],
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "score_thresholds": list(args.score_thresholds),
        "quality_powers": list(args.quality_powers),
        "top_ks": list(args.top_ks),
        "best": best,
        "results": ranked,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)

    print("best validation setting:")
    print(json.dumps(best, indent=2))
    print(f"json: {output_json}")
    print(f"csv: {output_csv}")


if __name__ == "__main__":
    main()
