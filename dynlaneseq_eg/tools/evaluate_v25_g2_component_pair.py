from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.tools.evaluate_v25_g0_and_path_gate import (
    _finish_geometry,
    _geometry_counts,
    _merge_geometry,
    _validate_wrong_control,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


POLICIES = ("control", "image_ownership", "image_ownership_wrong_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the paired V25 G2 image-ownership component gate."
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--treatment-report", required=True)
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
    path: str,
    *,
    root: Path,
    list_path: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(path))
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


def _endpoint_contract(
    checkpoint: Path, report_path: Path, *, expected_component: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {
        "component_exact": report.get("component") == expected_component,
        "endpoint_sha_exact": report.get("endpoint_sha256")
        == sha256_file(checkpoint),
        "scientific_gate": report.get("scientific_gate") is True,
        "quarter_epoch": 0.249
        <= float(report.get("component_official_epoch_fraction", -1.0))
        <= 0.251,
        "no_checkpoint_selection": report.get("checkpoint_selection_performed")
        is False,
        "no_threshold_selection": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    if not all(checks.values()):
        raise ValueError("V25 G2 endpoint contract failed: " + json.dumps(checks))
    return {"checks": checks, "report_sha256": sha256_file(report_path)}, report


def _load_model(
    cfg: dict[str, Any], checkpoint: Path, device: torch.device
) -> tuple[DynLaneSeqV25, int]:
    model = build_model(cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("factory did not construct DynLaneSeqV25")
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    model.requires_grad_(False).eval().to(device)
    if bool(cfg.get("training", {}).get("channels_last", False)):
        model.to(memory_format=torch.channels_last)
    return model, iteration


def _minimal(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: output[name]
        for name in ("exist_logits", "pred_x_rows", "range_norm", "quality_logits")
    }


@torch.inference_mode()
def _write(
    control: DynLaneSeqV25,
    treatment: DynLaneSeqV25,
    source_loader,
    wrong_loader,
    *,
    device: torch.device,
    directories: dict[str, Path],
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    geometry: dict[str, dict[str, float]] = {policy: {} for policy in POLICIES}
    images_written = 0
    same_image = 0
    same_clip = 0
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
            same_clip += int(Path(source_path).parent == Path(wrong_path).parent)
        images = images.to(device, non_blocking=True)
        wrong_images = wrong_images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
            wrong_images = wrong_images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = {
                "control": control(images.float()),
                "image_ownership": treatment(images.float()),
                "image_ownership_wrong_image": treatment(wrong_images.float()),
            }
        for policy, output in outputs.items():
            write_culane_predictions(
                _minimal(output),
                metas,
                directories[policy],
                score_thresh=0.5,
                min_pred_points=5,
                nms_distance_thresh_px=0.0,
                top_k=4,
                quality_score_power=0.0,
                score_mode="exist",
            )
            active = torch.softmax(output["exist_logits"].float(), dim=-1)[..., 0] >= 0.5
            _merge_geometry(
                geometry[policy],
                _geometry_counts(
                    output["pred_x_rows"].float(),
                    output["range_norm"],
                    active,
                ),
            )
        images_written += int(images.shape[0])
        if batch_index == 1 or (
            log_interval > 0 and batch_index % log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "write_v25_g2_predictions",
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
        "elapsed_seconds": time.perf_counter() - started,
        "geometry": {
            policy: _finish_geometry(value) for policy, value in geometry.items()
        },
    }


def _json_metrics(result: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in result.items()}


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    source_list = Path(population["list_path"])
    wrong_list = Path(args.wrong_image_list).expanduser().resolve()
    wrong_contract = _validate_wrong_control(
        source_list,
        wrong_list,
        Path(args.wrong_image_report).expanduser().resolve(),
    )
    control_checkpoint = Path(args.control_checkpoint).expanduser().resolve()
    treatment_checkpoint = Path(args.treatment_checkpoint).expanduser().resolve()
    control_contract, control_report = _endpoint_contract(
        control_checkpoint,
        Path(args.control_report).expanduser().resolve(),
        expected_component="g2_control",
    )
    treatment_contract, treatment_report = _endpoint_contract(
        treatment_checkpoint,
        Path(args.treatment_report).expanduser().resolve(),
        expected_component="g2_image_ownership",
    )
    if control_report["initial_checkpoint_sha256"] != treatment_report["initial_checkpoint_sha256"]:
        raise ValueError("G2 control/treatment did not start from identical G0 endpoint")
    control_cfg = _configured(
        args.control_config,
        root=root,
        list_path=source_list,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    treatment_cfg = _configured(
        args.treatment_config,
        root=root,
        list_path=source_list,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    wrong_cfg = _configured(
        args.treatment_config,
        root=root,
        list_path=wrong_list,
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    source_loader = build_dataloader(control_cfg, split="val", training=False)
    wrong_loader = build_dataloader(wrong_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(source_loader.dataset) != expected or len(wrong_loader.dataset) != expected:
        raise ValueError("V25 G2 evaluator altered official validation population")
    device = torch.device(args.device)
    control, control_iteration = _load_model(
        control_cfg, control_checkpoint, device
    )
    treatment, treatment_iteration = _load_model(
        treatment_cfg, treatment_checkpoint, device
    )
    if control_iteration != treatment_iteration:
        raise ValueError("G2 paired endpoints have different iterations")
    channels_last = bool(control_cfg["training"].get("channels_last", False))
    output_dir = Path(args.output_dir).expanduser().resolve()
    directories = {
        policy: output_dir / "predictions" / policy for policy in POLICIES
    }
    writer = _write(
        control,
        treatment,
        source_loader,
        wrong_loader,
        device=device,
        directories=directories,
        channels_last=channels_last,
        log_interval=args.log_interval,
    )
    if writer["images_written"] != expected:
        raise RuntimeError("V25 G2 writer did not consume full validation")
    if writer["same_image_wrong_partner"] or writer["same_clip_wrong_partner"]:
        raise ValueError("V25 G2 wrong-image runtime contamination")
    del control, treatment, source_loader, wrong_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics: dict[str, dict[str, Any]] = {}
    for policy, directory in directories.items():
        metrics[policy] = _json_metrics(
            eval_predictions(
                pred_dir=directory,
                anno_dir=root,
                list_path=source_list,
                iou_thresholds=(0.50, 0.75),
                width=30,
                official=True,
                sequential=False,
                num_workers=args.metric_workers,
                chunksize=args.metric_chunksize,
            )
        )
    geometry = writer["geometry"]
    f1_delta_50 = 100.0 * (
        float(metrics["image_ownership"]["0.5"]["F1"])
        - float(metrics["control"]["0.5"]["F1"])
    )
    f1_delta_75 = 100.0 * (
        float(metrics["image_ownership"]["0.75"]["F1"])
        - float(metrics["control"]["0.75"]["F1"])
    )
    crossing_control = float(geometry["control"]["crossing_image_fraction"])
    crossing_treatment = float(
        geometry["image_ownership"]["crossing_image_fraction"]
    )
    duplicate_control = float(geometry["control"]["duplicate_image_fraction"])
    duplicate_treatment = float(
        geometry["image_ownership"]["duplicate_image_fraction"]
    )
    checks = {
        "crossing_not_worse": crossing_treatment <= crossing_control,
        "duplicate_not_worse": duplicate_treatment <= duplicate_control,
        "crossing_or_duplicate_strictly_better": (
            crossing_treatment < crossing_control
            or duplicate_treatment < duplicate_control
        ),
        "unique_tp50_not_lower": int(metrics["image_ownership"]["0.5"]["TP"])
        >= int(metrics["control"]["0.5"]["TP"]),
        "f1_50_not_worse_by_0p10_points": f1_delta_50 >= -0.10,
        "f1_75_not_worse_by_0p10_points": f1_delta_75 >= -0.10,
        "correct_image_beats_wrong_tp50": int(
            metrics["image_ownership"]["0.5"]["TP"]
        )
        > int(metrics["image_ownership_wrong_image"]["0.5"]["TP"]),
        "correct_image_beats_wrong_tp75": int(
            metrics["image_ownership"]["0.75"]["TP"]
        )
        > int(metrics["image_ownership_wrong_image"]["0.75"]["TP"]),
    }
    report = {
        "experiment": "V25 G2 image-mediated ridge ownership paired gate",
        "control_endpoint_contract": control_contract,
        "treatment_endpoint_contract": treatment_contract,
        "official_validation_population_contract": population,
        "wrong_image_contract": wrong_contract,
        "writer_contract": writer,
        "metrics": metrics,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "treatment_minus_control_f1_50_points": f1_delta_50,
            "treatment_minus_control_f1_75_points": f1_delta_75,
        },
        "contract": {
            "validation_subset_used": False,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "test_set_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "g2_official_val_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
