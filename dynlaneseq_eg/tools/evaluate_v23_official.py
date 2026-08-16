from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import (
    discrete_cross_iou,
    interp,
    list_image_rel_paths,
    load_culane_img_data,
)
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v23 import DynLaneSeqV23
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


FIXED_SEED = 3407
FIXED_THRESHOLDS = (0.50, 0.75)
FIXED_LINE_WIDTH = 30
FIXED_SCORE_THRESHOLD = 0.50
FIXED_SOURCE_SCORE_THRESHOLD = 0.0
FIXED_MIN_POINTS = 5
FIXED_TOP_K = 4
FIXED_MINIMUM_F1_50_GAIN_POINTS = 0.50
FIXED_FULL_VAL_TARGET_F1_50_GAIN_POINTS = 0.80
FIXED_MAXIMUM_SOURCE_CORRECT_LOSS = 0.01
POLICIES = ("source_v7", "v23", "cross_clip_wrong_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fixed V23 endpoint against its frozen V7 source on "
            "all 9,675 untouched official CULane validation images."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--wrong-image-list", required=True)
    parser.add_argument("--wrong-image-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=0)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(
    base: dict[str, Any],
    *,
    dataset_root: Path,
    list_path: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(dataset_root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(list_path)
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(num_workers)
    cfg["dataloader"]["persistent_workers"] = bool(num_workers > 0)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _clip(path: str) -> str:
    return str(Path(path).parent)


def _validate_wrong_image_control(
    source_list: Path,
    wrong_list: Path,
    wrong_report_path: Path,
) -> dict[str, Any]:
    report = json.loads(wrong_report_path.read_text(encoding="utf-8"))
    checks = {
        "report_passed": report.get("passed") is True,
        "source_sha256_exact": str(report.get("input_sha256"))
        == sha256_file(source_list),
        "wrong_sha256_exact": str(report.get("output_sha256"))
        == sha256_file(wrong_list),
        "same_image_zero": int(report.get("same_image_partner_count", -1)) == 0,
        "same_clip_zero": int(report.get("same_clip_partner_count", -1)) == 0,
        "image_count_exact": int(report.get("image_count", -1)) == 9_675,
    }
    if not all(checks.values()):
        raise ValueError(
            "V23 wrong-image control contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {"checks": checks, "report": report}


def _checkpoint_contract(
    checkpoint: Path,
    training_report_path: Path,
) -> dict[str, Any]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    digest = sha256_file(checkpoint)
    checks = {
        "checkpoint_sha256_exact": digest == str(report.get("checkpoint_sha256")),
        "iteration_exact_8000": int(report.get("iteration", -1)) == 8_000,
        "gate_zero_passed": report.get("gate_zero", {}).get("passed") is True,
        "teacher_still_exact": report.get("teacher_state_still_exact") is True,
        "no_checkpoint_selection": report.get("checkpoint_selection_performed")
        is False,
        "no_threshold_selection": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "V23 endpoint/training contract failed: "
            + json.dumps(checks, sort_keys=True)
        )
    return {
        "checks": checks,
        "checkpoint_sha256": digest,
        "training_report_sha256": sha256_file(training_report_path),
    }


def _move_images(
    images: torch.Tensor,
    *,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


@torch.inference_mode()
def _write_all_predictions(
    model: DynLaneSeqV23,
    source_loader,
    wrong_loader,
    *,
    device: torch.device,
    output_dirs: dict[str, Path],
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    model.requires_grad_(False).eval()
    model.prepare_for_inference()
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    images_written = 0
    same_image = 0
    same_clip = 0
    started = time.perf_counter()
    for batch_index, (source_batch, wrong_batch) in enumerate(
        zip(source_loader, wrong_loader), start=1
    ):
        images, _targets, metas = source_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        if int(images.shape[0]) != int(wrong_images.shape[0]):
            raise ValueError("V23 source/wrong batch-size mismatch")
        for meta, wrong_meta in zip(metas, wrong_metas):
            source_path = str(meta.get("image_path", ""))
            wrong_path = str(wrong_meta.get("image_path", ""))
            same_image += int(source_path == wrong_path)
            same_clip += int(_clip(source_path) == _clip(wrong_path))

        images = _move_images(
            images, device=device, channels_last=channels_last
        )
        wrong_images = _move_images(
            wrong_images, device=device, channels_last=channels_last
        )
        # V7 remains exact FP32. The student endpoint is evaluated in FP32;
        # no post-hoc AMP, threshold, or checkpoint variant is selected.
        with torch.autocast(device_type=device.type, enabled=False):
            teacher = model.teacher(images.float(), inference_only=True)
            correct = model.student(images.float(), teacher)
        write_culane_predictions(
            teacher,
            metas,
            output_dirs["source_v7"],
            score_thresh=FIXED_SOURCE_SCORE_THRESHOLD,
            min_pred_points=FIXED_MIN_POINTS,
            nms_distance_thresh_px=0.0,
            top_k=FIXED_TOP_K,
            quality_score_power=0.0,
            score_mode="four_slot",
        )
        write_culane_predictions(
            correct,
            metas,
            output_dirs["v23"],
            score_thresh=FIXED_SCORE_THRESHOLD,
            min_pred_points=FIXED_MIN_POINTS,
            nms_distance_thresh_px=0.0,
            top_k=FIXED_TOP_K,
            quality_score_power=0.0,
            score_mode="exist",
        )
        del correct
        with torch.autocast(device_type=device.type, enabled=False):
            wrong = model.student(wrong_images.float(), teacher)
        write_culane_predictions(
            wrong,
            metas,
            output_dirs["cross_clip_wrong_image"],
            score_thresh=FIXED_SCORE_THRESHOLD,
            min_pred_points=FIXED_MIN_POINTS,
            nms_distance_thresh_px=0.0,
            top_k=FIXED_TOP_K,
            quality_score_power=0.0,
            score_mode="exist",
        )
        images_written += int(images.shape[0])
        if batch_index == 1 or (
            int(log_interval) > 0 and batch_index % int(log_interval) == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "write_predictions",
                        "batches": batch_index,
                        "images": images_written,
                        "images_per_second": images_written / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if images_written != len(source_loader.dataset):
        raise RuntimeError(
            f"V23 writer consumed {images_written} of {len(source_loader.dataset)} images"
        )
    return {
        "images_written": images_written,
        "runtime_same_image_wrong_partner": same_image,
        "runtime_same_clip_wrong_partner": same_clip,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _matched_gt(
    pred: list[list[tuple[float, float]]],
    anno: list[list[tuple[float, float]]],
    threshold: float,
) -> tuple[int, int, int, frozenset[int]]:
    pred_interp = [interp(lane, n=5) for lane in pred if len(lane) >= 2]
    anno_interp = [interp(lane, n=5) for lane in anno if len(lane) >= 2]
    if not pred_interp or not anno_interp:
        return 0, len(pred_interp), len(anno_interp), frozenset()
    ious = discrete_cross_iou(
        pred_interp,
        anno_interp,
        width=FIXED_LINE_WIDTH,
        img_shape=(590, 1640),
    )
    pred_ids, gt_ids = linear_sum_assignment(1.0 - ious)
    hit = ious[pred_ids, gt_ids] > float(threshold)
    matched = frozenset(int(value) for value in gt_ids[hit])
    tp = len(matched)
    return tp, len(pred_interp) - tp, len(anno_interp) - tp, matched


def _evaluate_image(
    rel: str,
    dataset_root: str,
    prediction_roots: dict[str, str],
) -> dict[str, Any]:
    annotation = load_culane_img_data(
        Path(dataset_root) / rel.replace(".jpg", ".lines.txt")
    )
    predictions = {
        policy: load_culane_img_data(
            Path(root) / rel.replace(".jpg", ".lines.txt")
        )
        for policy, root in prediction_roots.items()
    }
    result: dict[str, Any] = {
        "prediction_count": {
            policy: len(lanes) for policy, lanes in predictions.items()
        },
        "thresholds": {},
    }
    for threshold in FIXED_THRESHOLDS:
        result["thresholds"][f"{threshold:.2f}"] = {
            policy: _matched_gt(lanes, annotation, threshold)
            for policy, lanes in predictions.items()
        }
    return result


def _empty_counts() -> dict[str, int]:
    return {"TP": 0, "FP": 0, "FN": 0}


def _finish_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        **counts,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def summarize_image_results(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        policy: {
            f"{threshold:.2f}": _empty_counts()
            for threshold in FIXED_THRESHOLDS
        }
        for policy in POLICIES
    }
    paired = {
        f"{threshold:.2f}": {"improved": 0, "worsened": 0, "tied": 0}
        for threshold in FIXED_THRESHOLDS
    }
    source_correct = {
        f"{threshold:.2f}": 0 for threshold in FIXED_THRESHOLDS
    }
    source_correct_lost = {
        f"{threshold:.2f}": 0 for threshold in FIXED_THRESHOLDS
    }
    count_equal = 0
    for row in rows:
        counts = row["prediction_count"]
        count_equal += int(
            counts["source_v7"] == counts["v23"]
            == counts["cross_clip_wrong_image"]
        )
        for threshold in FIXED_THRESHOLDS:
            key = f"{threshold:.2f}"
            threshold_row = row["thresholds"][key]
            for policy in POLICIES:
                tp, fp, fn, _matched = threshold_row[policy]
                aggregate[policy][key]["TP"] += int(tp)
                aggregate[policy][key]["FP"] += int(fp)
                aggregate[policy][key]["FN"] += int(fn)
            source_tp = int(threshold_row["source_v7"][0])
            endpoint_tp = int(threshold_row["v23"][0])
            label = (
                "improved"
                if endpoint_tp > source_tp
                else "worsened"
                if endpoint_tp < source_tp
                else "tied"
            )
            paired[key][label] += 1
            source_gt = threshold_row["source_v7"][3]
            endpoint_gt = threshold_row["v23"][3]
            source_correct[key] += len(source_gt)
            source_correct_lost[key] += len(source_gt - endpoint_gt)
    metrics = {
        policy: {
            key: _finish_counts(value)
            for key, value in thresholds.items()
        }
        for policy, thresholds in aggregate.items()
    }
    degradation = {
        key: {
            "source_correct": source_correct[key],
            "lost_by_v23": source_correct_lost[key],
            "fraction": source_correct_lost[key]
            / max(source_correct[key], 1),
        }
        for key in source_correct
    }
    return {
        "metrics": metrics,
        "paired_image_effects": paired,
        "source_correct_degradation": degradation,
        "cardinality": {
            "exact_source_count_images": count_equal,
            "images": len(rows),
            "exact_fraction": count_equal / max(len(rows), 1),
        },
    }


def _evaluate_predictions(
    *,
    list_path: Path,
    dataset_root: Path,
    output_dirs: dict[str, Path],
    metric_workers: int,
    metric_chunksize: int,
) -> dict[str, Any]:
    rels = list_image_rel_paths(list_path)
    roots = {name: str(path) for name, path in output_dirs.items()}
    tasks = zip(rels, repeat(str(dataset_root)), repeat(roots))
    workers = int(metric_workers) if int(metric_workers) > 0 else cpu_count()
    if workers <= 1:
        rows = [
            _evaluate_image(rel, str(dataset_root), roots)
            for rel in tqdm(rels, desc="V23 official raster", ncols=80)
        ]
    else:
        with Pool(workers) as pool:
            rows = list(
                tqdm(
                    pool.starmap(
                        _evaluate_image,
                        tasks,
                        chunksize=max(int(metric_chunksize), 1),
                    ),
                    total=len(rels),
                    desc="V23 official raster",
                    ncols=80,
                )
            )
    return summarize_image_results(rows)


def _gate(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    source_50 = metrics["source_v7"]["0.50"]
    source_75 = metrics["source_v7"]["0.75"]
    endpoint_50 = metrics["v23"]["0.50"]
    endpoint_75 = metrics["v23"]["0.75"]
    wrong_50 = metrics["cross_clip_wrong_image"]["0.50"]
    wrong_75 = metrics["cross_clip_wrong_image"]["0.75"]
    f1_gain_50_points = 100.0 * (
        float(endpoint_50["F1"]) - float(source_50["F1"])
    )
    f1_gain_75_points = 100.0 * (
        float(endpoint_75["F1"]) - float(source_75["F1"])
    )
    checks = {
        "f1_50_gain_at_least_0p50_points": f1_gain_50_points
        >= FIXED_MINIMUM_F1_50_GAIN_POINTS,
        "tp_50_strictly_positive_vs_v7": int(endpoint_50["TP"])
        > int(source_50["TP"]),
        "f1_75_nonregression": float(endpoint_75["F1"])
        >= float(source_75["F1"]),
        "source_correct_loss_below_1pct_50": float(
            summary["source_correct_degradation"]["0.50"]["fraction"]
        )
        < FIXED_MAXIMUM_SOURCE_CORRECT_LOSS,
        "source_correct_loss_below_1pct_75": float(
            summary["source_correct_degradation"]["0.75"]["fraction"]
        )
        < FIXED_MAXIMUM_SOURCE_CORRECT_LOSS,
        "correct_image_beats_wrong_image_tp_50": int(endpoint_50["TP"])
        > int(wrong_50["TP"]),
        "correct_image_beats_wrong_image_tp_75": int(endpoint_75["TP"])
        > int(wrong_75["TP"]),
        "prediction_count_exact_source_all_images": float(
            summary["cardinality"]["exact_fraction"]
        )
        == 1.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "v23_minus_v7_f1_50_points": f1_gain_50_points,
            "v23_minus_v7_f1_75_points": f1_gain_75_points,
            "v23_minus_v7_tp_50": int(endpoint_50["TP"])
            - int(source_50["TP"]),
            "v23_minus_v7_tp_75": int(endpoint_75["TP"])
            - int(source_75["TP"]),
            "v23_minus_wrong_tp_50": int(endpoint_50["TP"])
            - int(wrong_50["TP"]),
            "v23_minus_wrong_tp_75": int(endpoint_75["TP"])
            - int(wrong_75["TP"]),
        },
        "full_validation_target": {
            "minimum_f1_50_gain_points": (
                FIXED_FULL_VAL_TARGET_F1_50_GAIN_POINTS
            ),
            "met": f1_gain_50_points
            >= FIXED_FULL_VAL_TARGET_F1_50_GAIN_POINTS,
        },
    }


def main() -> None:
    args = parse_args()
    if int(args.eval_batch_size) < 1:
        raise ValueError("--eval-batch-size must be positive")
    seed_everything(FIXED_SEED)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(
        dataset_root, split="val"
    )
    source_list = Path(population["list_path"])
    wrong_list = Path(args.wrong_image_list).expanduser().resolve()
    wrong_report = Path(args.wrong_image_report).expanduser().resolve()
    wrong_contract = _validate_wrong_image_control(
        source_list, wrong_list, wrong_report
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    training_report = Path(args.training_report).expanduser().resolve()
    endpoint_contract = _checkpoint_contract(checkpoint, training_report)

    base_cfg: dict[str, Any] = load_config(args.config)
    if str(base_cfg.get("model", {}).get("name")) != "DynLaneSeqV23":
        raise ValueError("V23 evaluator requires model.name=DynLaneSeqV23")
    source_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=source_list,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    wrong_cfg = _configured(
        base_cfg,
        dataset_root=dataset_root,
        list_path=wrong_list,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    source_loader = build_dataloader(source_cfg, split="val", training=False)
    wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(source_loader.dataset) != expected or len(wrong_loader.dataset) != expected:
        raise ValueError("V23 evaluator altered the official validation population")

    device = torch.device(args.device)
    model = build_model(source_cfg)
    if not isinstance(model, DynLaneSeqV23):
        raise TypeError("factory did not construct DynLaneSeqV23")
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    if iteration != 8_000:
        raise ValueError(f"V23 endpoint iteration must be 8000, got {iteration}")
    channels_last = bool(
        source_cfg.get("training", {}).get("channels_last", False)
        and device.type == "cuda"
    )
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)

    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_dirs = {
        policy: output_dir / "predictions" / policy for policy in POLICIES
    }
    writer_contract = _write_all_predictions(
        model,
        source_loader,
        wrong_loader,
        device=device,
        output_dirs=prediction_dirs,
        channels_last=channels_last,
        log_interval=int(args.log_interval),
    )
    if (
        int(writer_contract["runtime_same_image_wrong_partner"]) != 0
        or int(writer_contract["runtime_same_clip_wrong_partner"]) != 0
    ):
        raise ValueError("V23 runtime wrong-image control was contaminated")
    del model, source_loader, wrong_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = _evaluate_predictions(
        list_path=source_list,
        dataset_root=dataset_root,
        output_dirs=prediction_dirs,
        metric_workers=int(args.metric_workers),
        metric_chunksize=int(args.metric_chunksize),
    )
    gate = _gate(summary)
    report = {
        "experiment": "V23 ordered slot-conditioned cost-volume official validation",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_file(args.config),
        "endpoint_contract": endpoint_contract,
        "official_validation_population_contract": population,
        "wrong_image_contract": wrong_contract,
        "writer_contract": writer_contract,
        **summary,
        "gate": gate,
        "contract": {
            "official_val_rows": expected,
            "validation_subset_used": False,
            "validation_rows_removed": 0,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "score_threshold": FIXED_SCORE_THRESHOLD,
            "source_score_threshold": FIXED_SOURCE_SCORE_THRESHOLD,
            "lane_nms_enabled": False,
            "line_width": FIXED_LINE_WIDTH,
            "iou_thresholds": list(FIXED_THRESHOLDS),
            "student_inference_dtype": "float32",
            "test_set_used": False,
        },
        "recommendation": (
            "eligible_for_user_review_before_test"
            if gate["passed"] and gate["full_validation_target"]["met"]
            else "stop_v23_official_validation_failed"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "official_val_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
