from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    interp,
    load_culane_img_data,
)
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_y_rows, sort_range_norm


def _install_numpy_checkpoint_compatibility() -> None:
    """Permit NumPy-2 checkpoints to load in the project's NumPy-1 env."""

    try:
        import numpy._core  # type: ignore[attr-defined]  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)


def _resolve_annotation_path(dataset_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path.with_suffix(".lines.txt")
    relative = Path(str(path).lstrip("/\\"))
    candidate = dataset_root / relative
    if candidate.exists():
        return candidate.with_suffix(".lines.txt")
    return (dataset_root / Path(*path.parts[-3:])).with_suffix(".lines.txt")


def _cpu_tensor(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = value.detach().to(device="cpu")
    return value.to(dtype=dtype) if dtype is not None else value


def _extract_batch_records(
    outputs: dict[str, torch.Tensor],
    metas: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    if "stage2" in outputs:
        outputs = outputs["stage2"]
    elif "final" in outputs:
        outputs = outputs["final"]

    required = (
        "selection_slot_active_logits",
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_candidate_valid",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    )
    missing = [key for key in required if not isinstance(outputs.get(key), torch.Tensor)]
    if missing:
        raise KeyError(f"V7 score audit is missing output tensors: {missing}")

    active_logits = outputs["selection_slot_active_logits"].float()
    active_probability = torch.sigmoid(active_logits)
    route_logits = outputs["selection_slot_real_route_logits"].float()
    candidate_valid = outputs["selection_slot_candidate_valid"].bool()
    route_logits = route_logits.masked_fill(~candidate_valid[:, None, :], -1.0e4)
    real_probability = torch.softmax(route_logits, dim=-1)
    geometry_indices = outputs["selection_slot_geometry_route_indices"].long()
    safe_indices = geometry_indices.clamp(min=0, max=max(int(route_logits.shape[-1]) - 1, 0))
    selected_route_probability = real_probability.gather(
        -1,
        safe_indices.unsqueeze(-1),
    ).squeeze(-1)
    geometry_valid = geometry_indices >= 0
    selected_route_probability = torch.where(
        geometry_valid,
        selected_route_probability,
        torch.zeros_like(selected_route_probability),
    )
    product_probability = active_probability * selected_route_probability
    current_active = (active_logits >= 0.0) & geometry_valid

    parity_error = 0.0
    model_scores = outputs.get("selection_slot_scores")
    if isinstance(model_scores, torch.Tensor) and bool(current_active.any()):
        parity_error = float(
            (model_scores.float()[current_active] - product_probability[current_active])
            .abs()
            .max()
            .item()
        )

    pred_x = _cpu_tensor(outputs["selection_slot_pred_x_rows"], torch.float32)
    ranges = _cpu_tensor(sort_range_norm(outputs["selection_slot_range_norm"].float()), torch.float32)
    active_probability = _cpu_tensor(active_probability, torch.float32)
    selected_route_probability = _cpu_tensor(selected_route_probability, torch.float32)
    product_probability = _cpu_tensor(product_probability, torch.float32)
    geometry_valid = _cpu_tensor(geometry_valid, torch.bool)
    current_active = _cpu_tensor(current_active, torch.bool)

    records: list[dict[str, Any]] = []
    for index, meta in enumerate(metas):
        records.append(
            {
                "pred_x": pred_x[index],
                "range_norm": ranges[index],
                "active_probability": active_probability[index],
                "route_probability": selected_route_probability[index],
                "product_probability": product_probability[index],
                "geometry_valid": geometry_valid[index],
                "current_active": current_active[index],
                "image_path": str(meta["image_path"]),
                "input_w": int(meta.get("input_w", 800)),
                "input_h": int(meta.get("input_h", 288)),
                "scale_x": float(meta.get("scale_x", 1.0)),
                "scale_y": float(meta.get("scale_y", 1.0)),
                "crop_x": float(meta.get("crop_x", 0.0)),
                "crop_y": float(meta.get("crop_y", 0.0)),
            }
        )
    return records, parity_error


@torch.inference_mode()
def _build_cache(
    *,
    cfg: dict,
    checkpoint: str,
    split: str,
    device: torch.device,
    amp_dtype: str,
    max_images: int,
) -> tuple[list[dict[str, Any]], float]:
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg.setdefault("model", {})["require_pretrained_backbone"] = False
    model = build_model(cfg)
    _install_numpy_checkpoint_compatibility()
    load_checkpoint(checkpoint, model, strict=False)
    inference_only = bool(getattr(model, "supports_inference_only", False))
    if inference_only:
        cfg.setdefault("dataset", {})["load_targets"] = False
        cfg.setdefault("dataset", {})["infer_seg_labels"] = False
    loader = build_dataloader(cfg, split=split, training=False)
    if inference_only and hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()
    channels_last = bool(
        cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    records: list[dict[str, Any]] = []
    max_parity_error = 0.0
    progress = tqdm(loader, ncols=88, desc="caching V7 slot outputs")
    for images, _, metas in progress:
        if channels_last:
            images = images.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
        else:
            images = images.to(device, non_blocking=True)
        if amp_dtype == "float16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif amp_dtype == "bfloat16" and device.type == "cuda":
            amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            amp_context = nullcontext()
        with amp_context:
            outputs = model(images, inference_only=True) if inference_only else model(images)
        batch_records, parity_error = _extract_batch_records(outputs, metas)
        records.extend(batch_records)
        max_parity_error = max(max_parity_error, parity_error)
        if max_images > 0 and len(records) >= max_images:
            records = records[:max_images]
            break
    return records, max_parity_error


def _prepare_metric_record(task: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    record, dataset_root_value, min_pred_points = task
    dataset_root = Path(dataset_root_value)
    # Metric workers must receive ordinary NumPy arrays rather than CPU
    # tensors.  Passing thousands of small tensors through multiprocessing
    # invokes PyTorch's file-descriptor based storage sharing and can exhaust
    # the process open-file limit long before the full validation split ends.
    pred_x = torch.as_tensor(record["pred_x"], dtype=torch.float32)
    ranges = sort_range_norm(
        torch.as_tensor(record["range_norm"], dtype=torch.float32)
    )
    input_w = int(record["input_w"])
    input_h = int(record["input_h"])
    y_rows = fixed_y_rows(
        int(pred_x.shape[-1]),
        input_h,
        device=pred_x.device,
        dtype=pred_x.dtype,
    )
    scale_x = float(record["scale_x"])
    scale_y = float(record["scale_y"])
    crop_x = float(record["crop_x"])
    crop_y = float(record["crop_y"])
    geometry_valid = torch.as_tensor(record["geometry_valid"], dtype=torch.bool).clone()
    slot_lanes: list[list[tuple[float, float]] | None] = []
    for slot in range(int(pred_x.shape[0])):
        y_min = float(ranges[slot, 0] * input_h)
        y_max = float(ranges[slot, 1] * input_h)
        mask = (y_rows >= y_min) & (y_rows <= y_max)
        if not bool(geometry_valid[slot]) or int(mask.sum().item()) < min_pred_points:
            geometry_valid[slot] = False
            slot_lanes.append(None)
            continue
        lane = [
            (
                float(x_value) / scale_x + crop_x,
                float(y_value) / scale_y + crop_y,
            )
            for x_value, y_value in zip(
                pred_x[slot].clamp(0, input_w - 1)[mask],
                y_rows[mask],
            )
        ]
        slot_lanes.append(lane)

    annotation_path = _resolve_annotation_path(dataset_root, record["image_path"])
    annotations = load_culane_img_data(annotation_path)
    gt_count = len(annotations)
    iou = np.zeros((int(pred_x.shape[0]), gt_count), dtype=np.float32)
    valid_indices = [index for index, lane in enumerate(slot_lanes) if lane is not None]
    if valid_indices and annotations:
        pred_interp = [interp(slot_lanes[index], n=5) for index in valid_indices]
        anno_interp = [interp(lane, n=5) for lane in annotations]
        iou_valid = discrete_cross_iou(
            pred_interp,
            anno_interp,
            width=30,
            img_shape=(590, 1640),
        )
        iou[np.asarray(valid_indices), :] = iou_valid

    return {
        "iou": iou,
        "gt_count": gt_count,
        "geometry_valid": geometry_valid.numpy(),
        "current_active": np.asarray(record["current_active"], dtype=np.bool_),
        "active_probability": np.asarray(record["active_probability"], dtype=np.float32),
        "route_probability": np.asarray(record["route_probability"], dtype=np.float32),
        "product_probability": np.asarray(record["product_probability"], dtype=np.float32),
    }


def _numpy_metric_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove torch storages before a record crosses a process boundary."""

    converted = dict(record)
    tensor_keys = (
        "pred_x",
        "range_norm",
        "active_probability",
        "route_probability",
        "product_probability",
        "geometry_valid",
        "current_active",
    )
    for key in tensor_keys:
        value = converted[key]
        if isinstance(value, torch.Tensor):
            converted[key] = value.detach().cpu().numpy()
        else:
            converted[key] = np.asarray(value)
    return converted


def _metric_for_selection(
    iou: np.ndarray,
    selected: tuple[int, ...],
    gt_count: int,
    iou_thresholds: tuple[float, ...],
) -> dict[float, tuple[int, int, int]]:
    pred_count = len(selected)
    if pred_count == 0 or gt_count == 0:
        return {
            threshold: (0, pred_count, gt_count)
            for threshold in iou_thresholds
        }
    selected_iou = iou[np.asarray(selected), :]
    row_indices, column_indices = linear_sum_assignment(1.0 - selected_iou)
    matched = selected_iou[row_indices, column_indices]
    result: dict[float, tuple[int, int, int]] = {}
    for threshold in iou_thresholds:
        tp = int((matched > threshold).sum())
        result[threshold] = (tp, pred_count - tp, gt_count - tp)
    return result


def _score_modes(record: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    active = record["active_probability"]
    route = record["route_probability"]
    current_gate = record["current_active"].astype(bool)
    no_gate = np.ones_like(current_gate, dtype=bool)
    return {
        "current_product_gated": (active * route, current_gate),
        "product_ungated": (active * route, no_gate),
        "active_only": (active, no_gate),
        "route_only_gated": (route, current_gate),
        "active_route_alpha_0p25_gated": (
            active * np.power(route, 0.25),
            current_gate,
        ),
        "active_route_alpha_0p50_gated": (
            active * np.power(route, 0.50),
            current_gate,
        ),
        "active_route_alpha_0p75_gated": (
            active * np.power(route, 0.75),
            current_gate,
        ),
    }


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    value = (
        float(ranks[labels].sum())
        - float(positives * (positives + 1)) / 2.0
    ) / float(positives * negatives)
    return value


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, len(sorted_labels) + 1)
    return float(precision[sorted_labels].sum() / positives)


def _finalize_counts(counts: dict[float, list[int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for threshold, (tp, fp, fn) in counts.items():
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        result[f"{threshold:.2f}"] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    return result


def _run_sweep(
    records: list[dict[str, Any]],
    thresholds: tuple[float, ...],
    iou_thresholds: tuple[float, ...],
    operating_threshold: float,
) -> dict[str, Any]:
    mode_names = tuple(_score_modes(records[0]).keys())
    aggregate = {
        mode: {
            threshold: {iou_threshold: [0, 0, 0] for iou_threshold in iou_thresholds}
            for threshold in thresholds
        }
        for mode in mode_names
    }
    observability: dict[str, dict[str, list[float] | list[bool]]] = {
        mode: {"scores": [], "labels_050": [], "labels_075": [], "ious": []}
        for mode in mode_names
    }

    for record in tqdm(records, ncols=88, desc="sweeping score factorizations"):
        geometry_valid = record["geometry_valid"].astype(bool)
        best_iou = (
            record["iou"].max(axis=1)
            if int(record["gt_count"]) > 0
            else np.zeros_like(record["active_probability"])
        )
        for mode, (scores, gate) in _score_modes(record).items():
            usable = geometry_valid & gate
            mode_observability = observability[mode]
            mode_observability["scores"].extend(scores[geometry_valid].tolist())
            mode_observability["labels_050"].extend((best_iou[geometry_valid] > 0.50).tolist())
            mode_observability["labels_075"].extend((best_iou[geometry_valid] > 0.75).tolist())
            mode_observability["ious"].extend(best_iou[geometry_valid].tolist())
            selection_cache: dict[tuple[int, ...], dict[float, tuple[int, int, int]]] = {}
            for score_threshold in thresholds:
                selected = tuple(
                    np.flatnonzero(usable & (scores >= score_threshold)).tolist()
                )
                metric = selection_cache.get(selected)
                if metric is None:
                    metric = _metric_for_selection(
                        record["iou"],
                        selected,
                        int(record["gt_count"]),
                        iou_thresholds,
                    )
                    selection_cache[selected] = metric
                for iou_threshold, values in metric.items():
                    destination = aggregate[mode][score_threshold][iou_threshold]
                    for index, value in enumerate(values):
                        destination[index] += int(value)

    result: dict[str, Any] = {}
    for mode in mode_names:
        curve: dict[str, Any] = {}
        for score_threshold in thresholds:
            finalized = _finalize_counts(aggregate[mode][score_threshold])
            mean_f1 = sum(
                finalized[f"{iou_threshold:.2f}"]["F1"]
                for iou_threshold in iou_thresholds
            ) / len(iou_thresholds)
            curve[f"{score_threshold:.2f}"] = {
                "results": finalized,
                "mean_f1": mean_f1,
            }

        best_050_threshold = max(
            thresholds,
            key=lambda value: curve[f"{value:.2f}"]["results"]["0.50"]["F1"],
        )
        best_075_threshold = max(
            thresholds,
            key=lambda value: curve[f"{value:.2f}"]["results"]["0.75"]["F1"],
        )
        best_mean_threshold = max(
            thresholds,
            key=lambda value: curve[f"{value:.2f}"]["mean_f1"],
        )
        operating_key = f"{min(thresholds, key=lambda value: abs(value - operating_threshold)):.2f}"
        observed = observability[mode]
        observed_scores = np.asarray(observed["scores"], dtype=np.float64)
        observed_iou = np.asarray(observed["ious"], dtype=np.float64)
        labels_050 = np.asarray(observed["labels_050"], dtype=bool)
        labels_075 = np.asarray(observed["labels_075"], dtype=bool)
        pearson = (
            float(np.corrcoef(observed_scores, observed_iou)[0, 1])
            if len(observed_scores) > 1
            and float(observed_scores.std()) > 0.0
            and float(observed_iou.std()) > 0.0
            else None
        )
        result[mode] = {
            "best_f1_050": {
                "threshold": best_050_threshold,
                **curve[f"{best_050_threshold:.2f}"],
            },
            "best_f1_075": {
                "threshold": best_075_threshold,
                **curve[f"{best_075_threshold:.2f}"],
            },
            "best_mean_f1": {
                "threshold": best_mean_threshold,
                **curve[f"{best_mean_threshold:.2f}"],
            },
            "operating_point": {
                "threshold": float(operating_key),
                **curve[operating_key],
            },
            "observability": {
                "slot_count": int(len(observed_scores)),
                "positive_050": int(labels_050.sum()),
                "positive_075": int(labels_075.sum()),
                "auc_050": _binary_auc(observed_scores, labels_050),
                "auc_075": _binary_auc(observed_scores, labels_075),
                "average_precision_050": _average_precision(observed_scores, labels_050),
                "average_precision_075": _average_precision(observed_scores, labels_075),
                "pearson_score_vs_best_iou": pearson,
            },
            "curve": curve,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V7 four-slot active/route score factorization using one "
            "model inference pass and exact official raster IoU."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=8)
    parser.add_argument("--min-pred-points", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--operating-threshold", type=float, default=0.20)
    parser.add_argument("--amp-dtype", choices=("none", "float16", "bfloat16"), default="none")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("dataset", {})["root"] = args.dataset_root
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(args.eval_batch_size)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cache_path = Path(args.cache)
    if args.reuse_cache and cache_path.exists():
        cache_payload = torch.load(cache_path, map_location="cpu")
        records = cache_payload["records"]
        max_parity_error = float(cache_payload.get("max_product_score_parity_error", 0.0))
    else:
        device = torch.device(args.device)
        records, max_parity_error = _build_cache(
            cfg=cfg,
            checkpoint=args.checkpoint,
            split=args.split,
            device=device,
            amp_dtype=args.amp_dtype,
            max_images=int(args.max_images),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "dataset_root": args.dataset_root,
                "records": records,
                "max_product_score_parity_error": max_parity_error,
            },
            cache_path,
        )

    tasks = [
        (
            _numpy_metric_record(record),
            args.dataset_root,
            int(args.min_pred_points),
        )
        for record in records
    ]
    if int(args.metric_workers) > 1:
        with mp.Pool(int(args.metric_workers)) as pool:
            metric_records = list(
                tqdm(
                    pool.imap(_prepare_metric_record, tasks, chunksize=16),
                    total=len(tasks),
                    ncols=88,
                    desc="computing official slot IoU",
                )
            )
    else:
        metric_records = [
            _prepare_metric_record(task)
            for task in tqdm(tasks, ncols=88, desc="computing official slot IoU")
        ]

    step = float(args.threshold_step)
    if not 0.0 < step <= 1.0:
        raise ValueError("--threshold-step must be in (0,1]")
    threshold_count = int(math.floor(1.0 / step + 1.0e-9))
    thresholds = tuple(round(index * step, 10) for index in range(threshold_count + 1))
    if thresholds[-1] < 1.0:
        thresholds = thresholds + (1.0,)
    modes = _run_sweep(
        metric_records,
        thresholds,
        (0.50, 0.75),
        float(args.operating_threshold),
    )
    payload = {
        "experiment": "V7 four-slot deployment score factorization audit",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "dataset_root": args.dataset_root,
        "sample_count": len(metric_records),
        "amp_dtype": args.amp_dtype,
        "threshold_step": step,
        "iou_thresholds": [0.50, 0.75],
        "max_product_score_parity_error": max_parity_error,
        "cache": str(cache_path),
        "modes": modes,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"sample_count: {len(metric_records)}")
    print(f"max_product_score_parity_error: {max_parity_error:.8g}")
    for mode, result in modes.items():
        best = result["best_f1_050"]
        strict = result["best_f1_075"]
        mean = result["best_mean_f1"]
        best_result = best["results"]
        print(
            f"{mode}: best@.50 threshold={best['threshold']:.2f} "
            f"F1@.50={best_result['0.50']['F1']:.4f} "
            f"F1@.75={best_result['0.75']['F1']:.4f}; "
            f"best@.75 threshold={strict['threshold']:.2f} "
            f"F1@.75={strict['results']['0.75']['F1']:.4f}; "
            f"best_mean threshold={mean['threshold']:.2f} "
            f"mean={mean['mean_f1']:.4f}"
        )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
