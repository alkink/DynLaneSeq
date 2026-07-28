from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import repeat
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.culane_metric import (
    culane_metric,
    list_image_rel_paths,
    load_culane_img_data,
)
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.evaluation.proposal_recall import collect_prediction_stages
from dynlaneseq_eg.factory import build_dataloader, build_model


STAGE_TENSOR_FIELDS = (
    "pred_x_rows",
    "exist_logits",
    "quality_logits",
    "range_norm",
    "row_visibility_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one model forward and evaluate each contiguous query group with "
            "NMS enabled and disabled under the official CULane raster metric."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-txt", default="")
    parser.add_argument("--num-query-groups", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--quality-power", type=float, default=0.50)
    parser.add_argument("--score-mode", default="exist_quality")
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--row-visibility-thresh", type=float, default=0.0)
    parser.add_argument("--nms-distance-thresh-px", type=float, default=20.0)
    parser.add_argument("--nms-min-overlap-points", type=int, default=5)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=0)
    parser.add_argument("--metric-chunksize", type=int, default=16)
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Reuse already completed prediction directories and run only metrics.",
    )
    return parser.parse_args()


def _resolve_list_path(cfg: dict[str, Any], split: str) -> Path:
    dataset = cfg.get("dataset", {})
    root = Path(dataset.get("root", "dataset"))
    rel = dataset.get("lists", {}).get(split)
    if rel is None:
        raise KeyError(f"No dataset list configured for split={split!r}")
    path = Path(rel)
    return path if path.is_absolute() else root / path


def _preferred_stage(outputs: dict[str, Any]) -> tuple[str, dict[str, torch.Tensor]]:
    stages = collect_prediction_stages(outputs)
    for name in ("main", "final", "stage2", "stage1", "coarse"):
        stage = stages.get(name)
        if stage is not None:
            return name, stage
    if not stages:
        raise ValueError("Model output contains no prediction stage")
    name = sorted(stages)[-1]
    return name, stages[name]


def _cpu_stage(stage: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: stage[key].detach().to(device="cpu")
        for key in STAGE_TENSOR_FIELDS
        if isinstance(stage.get(key), torch.Tensor)
    }


def _slice_candidates(
    stage: dict[str, Any],
    candidate_ids: list[int],
    num_candidates: int,
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in stage.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and int(value.shape[1]) == int(num_candidates)
        ):
            sliced[key] = value[:, candidate_ids]
        else:
            sliced[key] = value
    return sliced


def _variant_specs(num_groups: int) -> list[dict[str, Any]]:
    specs = [
        {
            "name": "full_nms",
            "group_index": None,
            "nms_enabled": True,
        }
    ]
    for group_index in range(num_groups):
        specs.extend(
            [
                {
                    "name": f"group{group_index}_no_nms",
                    "group_index": group_index,
                    "nms_enabled": False,
                },
                {
                    "name": f"group{group_index}_nms",
                    "group_index": group_index,
                    "nms_enabled": True,
                },
            ]
        )
    return specs


@torch.inference_mode()
def _write_all_variants(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_dir: Path,
    variants: list[dict[str, Any]],
    num_groups: int,
    score_thresh: float,
    quality_power: float,
    score_mode: str,
    top_k: int,
    min_pred_points: int,
    row_visibility_thresh: float,
    nms_distance_thresh_px: float,
    nms_min_overlap_points: int,
    channels_last: bool,
    amp_dtype: str,
    pass_targets: bool,
    inference_only: bool,
) -> tuple[str, int]:
    model.eval()
    stage_name = ""
    num_candidates = 0
    for images, targets, metas in tqdm(loader, ncols=80, desc="one forward, all query groups"):
        if channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)
        if amp_dtype == "float16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif amp_dtype == "bfloat16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_context = nullcontext()
        with amp_context:
            if pass_targets:
                outputs = model(images, targets=targets)
            elif inference_only:
                outputs = model(images, inference_only=True)
            else:
                outputs = model(images)
        current_stage_name, stage = _preferred_stage(outputs)
        if stage_name and stage_name != current_stage_name:
            raise ValueError(f"Prediction stage changed: {stage_name} -> {current_stage_name}")
        stage_name = current_stage_name
        stage = _cpu_stage(stage)
        current_candidates = int(stage["pred_x_rows"].shape[1])
        if num_candidates and num_candidates != current_candidates:
            raise ValueError("Candidate count changed across batches")
        num_candidates = current_candidates
        if num_candidates % int(num_groups) != 0:
            raise ValueError(
                f"num_candidates={num_candidates} is not divisible by num_groups={num_groups}"
            )
        group_size = num_candidates // int(num_groups)
        for spec in variants:
            group_index = spec["group_index"]
            if group_index is None:
                selected_stage = stage
            else:
                start = int(group_index) * group_size
                selected_stage = _slice_candidates(
                    stage,
                    list(range(start, start + group_size)),
                    num_candidates,
                )
            write_culane_predictions(
                selected_stage,
                metas,
                output_dir / spec["name"],
                score_thresh=score_thresh,
                min_pred_points=min_pred_points,
                nms_distance_thresh_px=(
                    nms_distance_thresh_px if spec["nms_enabled"] else 0.0
                ),
                nms_min_overlap_points=nms_min_overlap_points,
                top_k=top_k,
                row_visibility_thresh=row_visibility_thresh,
                quality_score_power=quality_power,
                score_mode=score_mode,
            )
    return stage_name, num_candidates


def _metric_all_variants(task: tuple[Any, ...]) -> dict[str, dict[float, list[int]]]:
    (
        relative_image,
        variant_dirs,
        annotation_root,
        width,
        iou_thresholds,
    ) = task
    annotation_path = Path(annotation_root) / relative_image.replace(
        ".jpg", ".lines.txt"
    )
    annotation = load_culane_img_data(annotation_path)
    results: dict[str, dict[float, list[int]]] = {}
    for name, directory in variant_dirs.items():
        prediction_path = Path(directory) / relative_image.replace(
            ".jpg", ".lines.txt"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing prediction for variant={name!r}: {prediction_path}"
            )
        prediction = load_culane_img_data(prediction_path)
        results[name] = culane_metric(
            prediction,
            annotation,
            width=int(width),
            iou_thresholds=tuple(iou_thresholds),
            official=True,
        )
    return results


def _aggregate_variant_metrics(
    per_image_results: list[dict[str, dict[float, list[int]]]],
    variants: list[dict[str, Any]],
    iou_thresholds: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for spec in variants:
        name = str(spec["name"])
        threshold_results: dict[str, Any] = {}
        for threshold in iou_thresholds:
            tp = sum(int(image[name][float(threshold)][0]) for image in per_image_results)
            fp = sum(int(image[name][float(threshold)][1]) for image in per_image_results)
            fn = sum(int(image[name][float(threshold)][2]) for image in per_image_results)
            precision = float(tp) / float(max(tp + fp, 1))
            recall = float(tp) / float(max(tp + fn, 1))
            f1 = (
                0.0
                if precision + recall == 0.0
                else 2.0 * precision * recall / (precision + recall)
            )
            threshold_results[f"{threshold:.2f}"] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        output[name] = {
            "group_index": spec["group_index"],
            "nms_enabled": bool(spec["nms_enabled"]),
            "results": threshold_results,
        }
    return output


def _evaluate_variants(
    *,
    output_dir: Path,
    variants: list[dict[str, Any]],
    dataset_root: Path,
    list_path: Path,
    width: int,
    iou_thresholds: list[float],
    metric_workers: int,
    metric_chunksize: int,
) -> dict[str, Any]:
    relative_images = list_image_rel_paths(list_path)
    variant_dirs = {
        str(spec["name"]): str((output_dir / str(spec["name"])).resolve())
        for spec in variants
    }
    tasks = zip(
        relative_images,
        repeat(variant_dirs),
        repeat(str(dataset_root.resolve())),
        repeat(int(width)),
        repeat(tuple(float(value) for value in iou_thresholds)),
    )
    workers = int(metric_workers) if int(metric_workers) > 0 else cpu_count()
    if workers <= 1:
        per_image = [
            _metric_all_variants(task)
            for task in tqdm(
                tasks,
                total=len(relative_images),
                ncols=80,
                desc="official group metrics",
            )
        ]
    else:
        with Pool(workers) as pool:
            per_image = list(
                tqdm(
                    pool.imap(
                        _metric_all_variants,
                        tasks,
                        chunksize=max(1, int(metric_chunksize)),
                    ),
                    total=len(relative_images),
                    ncols=80,
                    desc="official group metrics",
                )
            )
    return _aggregate_variant_metrics(per_image, variants, iou_thresholds)


def _comparison_to_full(metrics: dict[str, Any]) -> dict[str, Any]:
    reference = metrics["full_nms"]["results"]
    comparisons: dict[str, Any] = {}
    for name, values in metrics.items():
        if name == "full_nms":
            continue
        comparisons[name] = {
            threshold: {
                "delta_f1_points": 100.0
                * (
                    float(result["F1"])
                    - float(reference[threshold]["F1"])
                ),
                "delta_precision_points": 100.0
                * (
                    float(result["Precision"])
                    - float(reference[threshold]["Precision"])
                ),
                "delta_recall_points": 100.0
                * (
                    float(result["Recall"])
                    - float(reference[threshold]["Recall"])
                ),
            }
            for threshold, result in values["results"].items()
        }
    return comparisons


def _format_report(payload: dict[str, Any]) -> str:
    lines = [
        "CULane query-group official validation audit",
        f"checkpoint: {payload['checkpoint']}",
        f"score_thresh: {payload['settings']['score_thresh']}",
        f"quality_power: {payload['settings']['quality_power']}",
        f"top_k: {payload['settings']['top_k']}",
        "",
    ]
    for name, variant in payload["variants"].items():
        for threshold, result in variant["results"].items():
            lines.append(
                f"{name:>16} IoU {threshold}: "
                f"TP={result['TP']} FP={result['FP']} FN={result['FN']} "
                f"P={result['Precision']:.4f} R={result['Recall']:.4f} "
                f"F1={result['F1']:.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_root:
        cfg.setdefault("dataset", {})["root"] = str(Path(args.dataset_root).expanduser())
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg.setdefault("model", {})["require_pretrained_backbone"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    if int(args.num_workers) == 0:
        cfg["dataloader"]["persistent_workers"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = _variant_specs(args.num_query_groups)
    completion_marker = output_dir / "_PREDICTIONS_COMPLETE.json"
    stage_name = ""
    num_candidates = 0

    if not args.skip_write:
        device = torch.device(args.device)
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
        model = build_model(cfg)
        load_checkpoint(args.checkpoint, model, strict=False)
        pass_targets = bool(getattr(model, "oracle_coarse_enabled", False))
        inference_only = bool(getattr(model, "supports_inference_only", False))
        if inference_only and not pass_targets:
            cfg.setdefault("dataset", {})["load_targets"] = False
            cfg.setdefault("dataset", {})["infer_seg_labels"] = False
        loader = build_dataloader(cfg, split=args.split, training=False)
        if inference_only and hasattr(model, "prepare_for_inference"):
            model.prepare_for_inference()
        model = model.to(device)
        channels_last = bool(train_cfg.get("channels_last", False) and device.type == "cuda")
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        stage_name, num_candidates = _write_all_variants(
            model=model,
            loader=loader,
            device=device,
            output_dir=output_dir,
            variants=variants,
            num_groups=args.num_query_groups,
            score_thresh=args.score_thresh,
            quality_power=args.quality_power,
            score_mode=args.score_mode,
            top_k=args.top_k,
            min_pred_points=args.min_pred_points,
            row_visibility_thresh=args.row_visibility_thresh,
            nms_distance_thresh_px=args.nms_distance_thresh_px,
            nms_min_overlap_points=args.nms_min_overlap_points,
            channels_last=channels_last,
            amp_dtype=args.amp_dtype,
            pass_targets=pass_targets,
            inference_only=inference_only,
        )
        completion_marker.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "stage_name": stage_name,
                    "num_candidates": num_candidates,
                    "variants": [spec["name"] for spec in variants],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        if not completion_marker.is_file():
            raise FileNotFoundError(
                f"--skip-write requires completion marker: {completion_marker}"
            )
        marker = json.loads(completion_marker.read_text(encoding="utf-8"))
        stage_name = str(marker.get("stage_name", ""))
        num_candidates = int(marker.get("num_candidates", 0))

    dataset_root = Path(cfg.get("dataset", {}).get("root", "dataset"))
    list_path = _resolve_list_path(cfg, args.split)
    metrics = _evaluate_variants(
        output_dir=output_dir,
        variants=variants,
        dataset_root=dataset_root,
        list_path=list_path,
        width=args.width,
        iou_thresholds=[float(value) for value in args.iou_thresholds],
        metric_workers=args.metric_workers,
        metric_chunksize=args.metric_chunksize,
    )
    payload = {
        "diagnostic_only": True,
        "protocol": "official CULane raster IoU on the configured validation split",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "list_path": str(list_path),
        "output_dir": str(output_dir),
        "stage_name": stage_name,
        "num_candidates": num_candidates,
        "settings": {
            "num_query_groups": int(args.num_query_groups),
            "score_thresh": float(args.score_thresh),
            "quality_power": float(args.quality_power),
            "score_mode": str(args.score_mode),
            "top_k": int(args.top_k),
            "min_pred_points": int(args.min_pred_points),
            "row_visibility_thresh": float(args.row_visibility_thresh),
            "nms_distance_thresh_px": float(args.nms_distance_thresh_px),
            "nms_min_overlap_points": int(args.nms_min_overlap_points),
            "iou_thresholds": [float(value) for value in args.iou_thresholds],
            "width": int(args.width),
            "amp_dtype": str(args.amp_dtype),
        },
        "variants": metrics,
        "comparisons_to_full_nms": _comparison_to_full(metrics),
    }
    report = _format_report(payload)
    print(report)
    output_json = Path(args.output_json) if args.output_json else output_dir / "metrics.json"
    output_txt = Path(args.output_txt) if args.output_txt else output_dir / "metrics.txt"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_txt.write_text(report + "\n", encoding="utf-8")
    print(f"output_json: {output_json}")
    print(f"output_txt: {output_txt}")


if __name__ == "__main__":
    main()
