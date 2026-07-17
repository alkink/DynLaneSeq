from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.tusimple_metric import TuSimpleLaneEval
from dynlaneseq_eg.evaluation.tusimple_writer import (
    outputs_to_tusimple_records,
    write_tusimple_json_lines,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.tools.evaluate_tusimple import resolve_ground_truth_path


CACHE_VERSION = 1
PREDICTION_FIELDS = (
    "pred_x_rows",
    "exist_logits",
    "quality_logits",
    "range_norm",
    "row_visibility_logits",
)


def _resolve_annotation_paths(cfg: dict[str, Any], split: str) -> set[Path]:
    dataset_cfg = cfg.get("dataset", {})
    root = Path(dataset_cfg.get("root", "dataset/tusimple")).expanduser().resolve()
    split_cfg = dataset_cfg.get("splits", {}).get(split, {})
    annotations = split_cfg.get("annotations", [])
    if isinstance(annotations, (str, Path)):
        annotations = [annotations]
    resolved: set[Path] = set()
    for annotation in annotations:
        path = Path(annotation).expanduser()
        resolved.add((path if path.is_absolute() else root / path).resolve())
    return resolved


def validate_held_out_split(cfg: dict[str, Any], split: str) -> None:
    if split != "val":
        return
    overlap = _resolve_annotation_paths(cfg, "train") & _resolve_annotation_paths(cfg, split)
    if overlap:
        joined = ", ".join(str(path) for path in sorted(overlap))
        raise ValueError(
            "Refusing a TuSimple validation sweep because training and validation annotations overlap: "
            f"{joined}"
        )


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
            split,
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
    cache_dir: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    cache_path = _cache_path(cache_dir, config_path, checkpoint_path, dataset_root, split)
    if reuse_cache and cache_path.exists():
        cache = _load_cache(cache_path)
        cache.setdefault("metadata", {})["cache_path"] = str(cache_path)
        return cache

    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(eval_batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg["dataloader"]["prefetch_factor"] = 2
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    validate_held_out_split(cfg, split)

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
    load_checkpoint(checkpoint_path, model, strict=False)
    model.eval()
    loader = build_dataloader(cfg, split=split, training=False)

    batches: list[dict[str, Any]] = []
    for images, _, metas in tqdm(
        loader,
        ncols=88,
        desc=f"cache {checkpoint_path.stem}",
    ):
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        outputs = model(images)
        batches.append(
            {
                "outputs": _prediction_stage(outputs),
                "metas": [_plain_value(meta) for meta in metas],
            }
        )

    payload = {
        "cache_version": CACHE_VERSION,
        "metadata": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(dataset_root),
            "split": split,
            "num_samples": sum(len(batch["metas"]) for batch in batches),
            "cache_path": str(cache_path),
        },
        "batches": batches,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _checkpoint_iteration(path: Path) -> int:
    stem = path.stem
    if stem.startswith("iter_"):
        try:
            return int(stem[len("iter_") :])
        except ValueError:
            pass
    payload = _load_cache(path)
    return int(payload.get("iteration", 0))


def evaluate_grid(
    cache: dict[str, Any],
    ground_truth_path: Path,
    score_thresholds: list[float],
    quality_powers: list[float],
    min_pred_points: int,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    top_k: int,
    row_visibility_thresh: float,
) -> list[dict[str, Any]]:
    checkpoint = Path(cache["metadata"]["checkpoint"])
    iteration = _checkpoint_iteration(checkpoint)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="dynlaneseq_tusimple_sweep_") as temp_dir:
        prediction_path = Path(temp_dir) / "predictions.jsonl"
        combinations = [
            (score_threshold, quality_power)
            for quality_power in quality_powers
            for score_threshold in score_thresholds
        ]
        for score_threshold, quality_power in tqdm(
            combinations,
            ncols=88,
            desc=f"grid {checkpoint.stem}",
        ):
            records: list[dict[str, Any]] = []
            for batch in cache["batches"]:
                records.extend(
                    outputs_to_tusimple_records(
                        batch["outputs"],
                        batch["metas"],
                        score_thresh=float(score_threshold),
                        min_pred_points=int(min_pred_points),
                        nms_distance_thresh_px=float(nms_distance_thresh_px),
                        nms_min_overlap_points=int(nms_min_overlap_points),
                        top_k=int(top_k),
                        row_visibility_thresh=float(row_visibility_thresh),
                        quality_score_power=float(quality_power),
                        run_time_ms=0.0,
                    )
                )
            write_tusimple_json_lines(records, prediction_path)
            metrics = TuSimpleLaneEval.evaluate(prediction_path, ground_truth_path)
            rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "iteration": int(iteration),
                    "score_thresh": float(score_threshold),
                    "quality_power": float(quality_power),
                    "Accuracy": float(metrics["Accuracy"]),
                    "F1_score": float(metrics["F1_score"]),
                    "FP": float(metrics["FP"]),
                    "FN": float(metrics["FN"]),
                    "num_samples": int(metrics["num_samples"]),
                }
            )
    return rows


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    """Official Accuracy first; deterministic public-metric tie breakers."""
    return (
        float(row["Accuracy"]),
        float(row["F1_score"]),
        -float(row["FP"]),
        -float(row["FN"]),
        -int(row["iteration"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cached TuSimple checkpoint, score-threshold and quality-power sweep."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--quality-powers", nargs="+", type=float, required=True)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--cache-dir", default="outputs/tusimple_val_cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_paths = [Path(path).expanduser().resolve() for path in args.checkpoints]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    for path in (config_path, dataset_root, *checkpoint_paths):
        if not path.exists():
            raise FileNotFoundError(path)

    cfg = load_config(config_path)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    validate_held_out_split(cfg, args.split)
    ground_truth_path = resolve_ground_truth_path(cfg, args.split).resolve()
    if not ground_truth_path.exists():
        raise FileNotFoundError(ground_truth_path)

    all_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        cache = collect_checkpoint_cache(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            split=args.split,
            device=torch.device(args.device),
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            cache_dir=Path(args.cache_dir),
            reuse_cache=bool(args.reuse_cache),
        )
        all_rows.extend(
            evaluate_grid(
                cache=cache,
                ground_truth_path=ground_truth_path,
                score_thresholds=list(args.score_thresholds),
                quality_powers=list(args.quality_powers),
                min_pred_points=int(args.min_pred_points),
                nms_distance_thresh_px=float(args.nms_distance_thresh_px),
                nms_min_overlap_points=int(args.nms_min_overlap_points),
                top_k=int(args.top_k),
                row_visibility_thresh=float(args.row_visibility_thresh),
            )
        )

    ranked = sorted(all_rows, key=selection_key, reverse=True)
    best = ranked[0]
    payload = {
        "selection_metric": "official TuSimple Accuracy",
        "tie_breakers": ["F1_score desc", "FP asc", "FN asc", "iteration asc"],
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "ground_truth": str(ground_truth_path),
        "score_thresholds": list(args.score_thresholds),
        "quality_powers": list(args.quality_powers),
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
