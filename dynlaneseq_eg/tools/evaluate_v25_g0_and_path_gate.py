from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.common import fixed_row_fractions, sort_range_norm
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


POLICIES = (
    "expectation",
    "row_argmax",
    "hard_path",
    "wrong_image_hard_path",
)
THRESHOLDS = (0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full official validation for V25 G0 and its path gate."
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
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(
    base: dict[str, Any],
    *,
    root: Path,
    list_path: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dataset", {})["root"] = str(root)
    cfg["dataset"].setdefault("lists", {})["val"] = str(list_path)
    cfg["dataset"]["load_targets"] = False
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _clip(path: str) -> str:
    return str(Path(path).parent)


def _validate_wrong_control(
    source_list: Path, wrong_list: Path, report_path: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
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
        raise ValueError("V25 wrong-image contract failed: " + json.dumps(checks))
    return {"checks": checks, "report": report}


def _training_contract(
    checkpoint: Path, training_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    digest = sha256_file(checkpoint)
    checks = {
        "checkpoint_sha256_exact": digest == report.get("checkpoint_sha256"),
        "one_epoch_exact": abs(
            float(report.get("complete_official_train_epochs_seen", -1.0)) - 1.0
        )
        < 1.0e-12,
        "gate_zero_passed": report.get("gate_zero", {}).get("passed") is True,
        "no_checkpoint_selection": report.get("checkpoint_selection_performed")
        is False,
        "no_threshold_selection": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    if not all(checks.values()):
        raise ValueError("V25 endpoint contract failed: " + json.dumps(checks))
    return {
        "checks": checks,
        "checkpoint_sha256": digest,
        "training_report_sha256": sha256_file(training_report_path),
    }, report


def _training_trajectory(training_report_path: Path) -> dict[str, Any]:
    metrics_path = training_report_path.parent / "train_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 4:
        return {"rows": len(rows), "passed_20pct_loss_reduction": False}
    width = max(len(rows) // 10, 1)
    first = rows[:width]
    last = rows[-width:]

    def median(name: str, population: list[dict[str, Any]]) -> float:
        return float(statistics.median(float(row[name]) for row in population))

    first_total = median("loss_total", first)
    last_total = median("loss_total", last)
    first_row = median("loss_row_distribution", first)
    last_row = median("loss_row_distribution", last)
    total_reduction = (first_total - last_total) / max(abs(first_total), 1.0e-12)
    row_reduction = (first_row - last_row) / max(abs(first_row), 1.0e-12)
    return {
        "rows": len(rows),
        "first_decile_median_total": first_total,
        "last_decile_median_total": last_total,
        "total_loss_fractional_reduction": total_reduction,
        "first_decile_median_row": first_row,
        "last_decile_median_row": last_row,
        "row_loss_fractional_reduction": row_reduction,
        "passed_20pct_loss_reduction": max(total_reduction, row_reduction) >= 0.20,
    }


def _policy_output(
    output: dict[str, torch.Tensor], x_rows: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "exist_logits": output["exist_logits"],
        "pred_x_rows": x_rows,
        "range_norm": output["range_norm"],
        "quality_logits": output["quality_logits"],
    }


def _geometry_counts(
    x_rows: torch.Tensor,
    ranges: torch.Tensor,
    active: torch.Tensor,
    *,
    abrupt_threshold_px: float = 32.0,
    duplicate_distance_px: float = 30.0,
) -> dict[str, float]:
    batch, slots, rows = x_rows.shape
    row_fraction = fixed_row_fractions(
        rows, device=x_rows.device, dtype=torch.float32
    ).view(1, 1, rows)
    ranges = sort_range_norm(ranges.float())
    valid = (
        active.unsqueeze(-1)
        & (row_fraction >= ranges[..., :1])
        & (row_fraction <= ranges[..., 1:])
    )
    crossing_images = torch.zeros(batch, device=x_rows.device, dtype=torch.bool)
    duplicate_images = torch.zeros_like(crossing_images)
    for left in range(slots - 1):
        for right in range(left + 1, slots):
            common = valid[:, left] & valid[:, right]
            crossing_images |= (
                common & (x_rows[:, right] <= x_rows[:, left])
            ).any(dim=-1)
            distance = (x_rows[:, right] - x_rows[:, left]).abs()
            mean = (distance * common.float()).sum(dim=-1) / common.sum(dim=-1).clamp_min(1)
            duplicate_images |= (common.sum(dim=-1) >= 5) & (
                mean < float(duplicate_distance_px)
            )
    consecutive = valid[..., 1:] & valid[..., :-1]
    abrupt = (x_rows[..., 1:] - x_rows[..., :-1]).abs() > float(
        abrupt_threshold_px
    )
    abrupt_rows = int((abrupt & consecutive).sum().item())
    curvature = (
        x_rows[..., 2:] - 2.0 * x_rows[..., 1:-1] + x_rows[..., :-2]
        if rows >= 3
        else x_rows.new_zeros((*x_rows.shape[:-1], 0))
    )
    triple = (
        valid[..., 2:] & valid[..., 1:-1] & valid[..., :-2]
        if rows >= 3
        else valid[..., :0]
    )
    return {
        "images": float(batch),
        "active_lanes": float(active.sum().item()),
        "crossing_images": float(crossing_images.sum().item()),
        "duplicate_images": float(duplicate_images.sum().item()),
        "abrupt_rows": float(abrupt_rows),
        "valid_transitions": float(consecutive.sum().item()),
        "absolute_curvature_sum": float(
            (curvature.abs() * triple.float()).sum().item()
        ),
        "valid_curvature_rows": float(triple.sum().item()),
    }


def _merge_geometry(
    total: dict[str, float], row: dict[str, float]
) -> dict[str, float]:
    for key, value in row.items():
        total[key] = total.get(key, 0.0) + float(value)
    return total


def _finish_geometry(total: dict[str, float]) -> dict[str, float]:
    images = max(total.get("images", 0.0), 1.0)
    transitions = max(total.get("valid_transitions", 0.0), 1.0)
    curve_rows = max(total.get("valid_curvature_rows", 0.0), 1.0)
    return {
        **total,
        "mean_active_lanes": total.get("active_lanes", 0.0) / images,
        "crossing_image_fraction": total.get("crossing_images", 0.0) / images,
        "duplicate_image_fraction": total.get("duplicate_images", 0.0) / images,
        "abrupt_transition_fraction": total.get("abrupt_rows", 0.0)
        / transitions,
        "mean_absolute_curvature_px": total.get(
            "absolute_curvature_sum", 0.0
        )
        / curve_rows,
    }


@torch.inference_mode()
def _write_predictions(
    model: DynLaneSeqV25,
    source_loader,
    wrong_loader,
    *,
    device: torch.device,
    directories: dict[str, Path],
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    model.requires_grad_(False).eval()
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    geometry: dict[str, dict[str, float]] = {name: {} for name in POLICIES}
    images_written = 0
    same_image = 0
    same_clip = 0
    correct_wrong_abs_sum = 0.0
    correct_wrong_elements = 0
    started = time.perf_counter()
    for batch_index, (source_batch, wrong_batch) in enumerate(
        zip(source_loader, wrong_loader), start=1
    ):
        images, _targets, metas = source_batch
        wrong_images, _wrong_targets, wrong_metas = wrong_batch
        for meta, wrong_meta in zip(metas, wrong_metas):
            source_path = str(meta.get("image_path", ""))
            wrong_path = str(wrong_meta.get("image_path", ""))
            same_image += int(source_path == wrong_path)
            same_clip += int(_clip(source_path) == _clip(wrong_path))
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
            wrong_images = wrong_images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            correct = model(images.float())
            wrong = model(wrong_images.float())
        variants = {
            "expectation": correct["soft_x_rows"],
            "row_argmax": correct["row_argmax_x_rows"],
            "hard_path": correct["hard_path_x_rows"],
            "wrong_image_hard_path": wrong["hard_path_x_rows"],
        }
        correct_wrong_abs_sum += float(
            (variants["hard_path"] - variants["wrong_image_hard_path"])
            .abs()
            .sum()
            .item()
        )
        correct_wrong_elements += variants["hard_path"].numel()
        for policy, x_rows in variants.items():
            owner = wrong if policy == "wrong_image_hard_path" else correct
            policy_output = _policy_output(owner, x_rows)
            write_culane_predictions(
                policy_output,
                metas,
                directories[policy],
                score_thresh=0.5,
                min_pred_points=5,
                nms_distance_thresh_px=0.0,
                top_k=4,
                quality_score_power=0.0,
                score_mode="exist",
            )
            active = torch.softmax(owner["exist_logits"].float(), dim=-1)[..., 0] >= 0.5
            _merge_geometry(
                geometry[policy],
                _geometry_counts(x_rows.float(), owner["range_norm"], active),
            )
        images_written += int(images.shape[0])
        if batch_index == 1 or (
            log_interval > 0 and batch_index % log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "write_v25_g0_predictions",
                        "batches": batch_index,
                        "images": images_written,
                        "images_per_second": images_written / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "images_written": images_written,
        "same_image_wrong_partner": same_image,
        "same_clip_wrong_partner": same_clip,
        "correct_wrong_hard_mean_abs_px": correct_wrong_abs_sum
        / max(correct_wrong_elements, 1),
        "elapsed_seconds": time.perf_counter() - started,
        "geometry": {
            policy: _finish_geometry(values)
            for policy, values in geometry.items()
        },
    }


def _json_metrics(metrics: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metrics.items()}


def _path_gate(
    metrics: dict[str, dict[str, Any]], geometry: dict[str, dict[str, float]]
) -> dict[str, Any]:
    hard50 = float(metrics["hard_path"]["0.5"]["F1"])
    alternatives50 = max(
        float(metrics["expectation"]["0.5"]["F1"]),
        float(metrics["row_argmax"]["0.5"]["F1"]),
    )
    hard_cross = float(geometry["hard_path"]["crossing_image_fraction"])
    row_cross = float(geometry["row_argmax"]["crossing_image_fraction"])
    hard_abrupt = float(geometry["hard_path"]["abrupt_transition_fraction"])
    row_abrupt = float(geometry["row_argmax"]["abrupt_transition_fraction"])
    checks = {
        "hard_f1_50_within_0p10_points_of_best_row_decode": 100.0
        * (hard50 - alternatives50)
        >= -0.10,
        "hard_crossing_not_worse_than_row_argmax": hard_cross <= row_cross,
        "hard_abruptness_not_worse_than_row_argmax": hard_abrupt <= row_abrupt,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "hard_minus_best_alternative_f1_50_points": 100.0
        * (hard50 - alternatives50),
    }


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    source_list = Path(population["list_path"])
    wrong_list = Path(args.wrong_image_list).expanduser().resolve()
    wrong_report = Path(args.wrong_image_report).expanduser().resolve()
    wrong_contract = _validate_wrong_control(source_list, wrong_list, wrong_report)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    training_report_path = Path(args.training_report).expanduser().resolve()
    endpoint_contract, training_report = _training_contract(
        checkpoint, training_report_path
    )
    trajectory = _training_trajectory(training_report_path)
    base_cfg = load_config(args.config)
    source_cfg = _configured(
        base_cfg,
        root=root,
        list_path=source_list,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    wrong_cfg = _configured(
        base_cfg,
        root=root,
        list_path=wrong_list,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    source_loader = build_dataloader(source_cfg, split="val", training=False)
    wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(source_loader.dataset) != expected or len(wrong_loader.dataset) != expected:
        raise ValueError("V25 evaluator altered the official validation population")
    model = build_model(source_cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    if iteration != int(training_report["iteration"]):
        raise ValueError("V25 checkpoint/report iteration mismatch")
    device = torch.device(args.device)
    model.to(device)
    channels_last = bool(source_cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_dirs = {
        policy: output_dir / "predictions" / policy for policy in POLICIES
    }
    writer = _write_predictions(
        model,
        source_loader,
        wrong_loader,
        device=device,
        directories=prediction_dirs,
        channels_last=channels_last,
        log_interval=args.log_interval,
    )
    if writer["images_written"] != expected:
        raise RuntimeError("V25 writer did not consume full official validation")
    if writer["same_image_wrong_partner"] or writer["same_clip_wrong_partner"]:
        raise ValueError("V25 runtime wrong-image pairing is contaminated")
    del model, source_loader, wrong_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics: dict[str, dict[str, Any]] = {}
    for policy, directory in prediction_dirs.items():
        metrics[policy] = _json_metrics(
            eval_predictions(
                pred_dir=directory,
                anno_dir=root,
                list_path=source_list,
                iou_thresholds=THRESHOLDS,
                width=30,
                official=True,
                sequential=False,
                num_workers=args.metric_workers,
                chunksize=args.metric_chunksize,
            )
        )
    correct_image_checks = {
        key: int(metrics["hard_path"][key]["TP"])
        > int(metrics["wrong_image_hard_path"][key]["TP"])
        for key in ("0.5", "0.75")
    }
    mechanism = {
        "training_loss_reduced_20pct": trajectory.get(
            "passed_20pct_loss_reduction", False
        ),
        "correct_image_beats_wrong_at_50": correct_image_checks["0.5"],
        "correct_image_beats_wrong_at_75": correct_image_checks["0.75"],
        "mean_active_lanes_at_least_two": float(
            writer["geometry"]["hard_path"]["mean_active_lanes"]
        )
        >= 2.0,
    }
    path_gate = _path_gate(metrics, writer["geometry"])
    report = {
        "experiment": "V25 G0 full official validation and G1 path replay",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "endpoint_contract": endpoint_contract,
        "training_trajectory": trajectory,
        "official_validation_population_contract": population,
        "wrong_image_contract": wrong_contract,
        "writer_contract": writer,
        "metrics": metrics,
        "g0_mechanism_gate": {
            "passed": all(mechanism.values()),
            "checks": mechanism,
        },
        "g1_hard_path_gate": path_gate,
        "contract": {
            "official_val_rows": expected,
            "validation_subset_used": False,
            "validation_rows_removed": 0,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "score_threshold": 0.5,
            "lane_nms_enabled": False,
            "line_width": 30,
            "iou_thresholds": list(THRESHOLDS),
            "inference_dtype": "float32",
            "test_set_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "g0_g1_official_val_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
