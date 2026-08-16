from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions
from dynlaneseq_eg.evaluation.culane_writer import write_culane_predictions
from dynlaneseq_eg.factory import build_dataloader
from dynlaneseq_eg.modeling.v25_denoising import (
    build_denoising_query_anchors,
)
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    build_ordered_lane_targets,
)
from dynlaneseq_eg.tools.evaluate_v25_g2_component_pair import (
    _endpoint_contract,
    _json_metrics,
    _load_model,
    _minimal,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paired V25 G4 training-only denoising queries."
    )
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--treatment-config", required=True)
    parser.add_argument("--treatment-checkpoint", required=True)
    parser.add_argument("--treatment-report", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4)
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
    cfg["dataset"]["load_targets"] = True
    cfg["dataset"]["infer_seg_labels"] = False
    cfg.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    cfg["dataloader"]["num_workers"] = int(workers)
    cfg["dataloader"]["persistent_workers"] = workers > 0
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    return cfg


def _accumulate_error(
    total: dict[str, float],
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    error = (prediction.float() - target.float()).abs()
    total[name] = total.get(name, 0.0) + float(
        (error * valid.float()).sum().item()
    )
    total[name + "_rows"] = total.get(name + "_rows", 0.0) + float(
        valid.sum().item()
    )


@torch.inference_mode()
def _write_and_measure(
    control,
    treatment,
    loader,
    *,
    device: torch.device,
    directories: dict[str, Path],
    input_w: int,
    minimum_valid_rows: int,
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    totals: dict[str, float] = {}
    images_written = 0
    started = time.perf_counter()
    for batch_index, (images, targets, metas) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            clean_control = control(images.float())
            clean_treatment = treatment(images.float())
        outputs = {
            "control": clean_control,
            "denoising": clean_treatment,
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
        ordered = build_ordered_lane_targets(
            targets,
            device=device,
            slots=4,
            rows=int(clean_control["soft_x_rows"].shape[-1]),
            input_w=input_w,
            minimum_valid_rows=minimum_valid_rows,
        )
        anchors = build_denoising_query_anchors(
            ordered,
            input_w=input_w,
            seed=74_003 + batch_index,
            held_out=True,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            perturbed_control = control(
                images.float(),
                query_anchor_x_rows=anchors.anchors_normalized,
            )
            perturbed_treatment = treatment(
                images.float(),
                query_anchor_x_rows=anchors.anchors_normalized,
            )
        valid = ordered["valid_mask"] & ordered["active"].unsqueeze(-1)
        initial_px = anchors.anchors_normalized * float(input_w)
        _accumulate_error(
            totals,
            "initial",
            initial_px,
            ordered["x_rows"],
            valid,
        )
        _accumulate_error(
            totals,
            "control",
            perturbed_control["soft_x_rows"],
            ordered["x_rows"],
            valid,
        )
        _accumulate_error(
            totals,
            "treatment",
            perturbed_treatment["soft_x_rows"],
            ordered["x_rows"],
            valid,
        )
        images_written += int(images.shape[0])
        if batch_index == 1 or (
            log_interval > 0 and batch_index % log_interval == 0
        ):
            elapsed = max(time.perf_counter() - started, 1.0e-9)
            print(
                json.dumps(
                    {
                        "phase": "evaluate_v25_g4",
                        "batches": batch_index,
                        "images": images_written,
                        "images_per_second": images_written / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    diagnostics = {
        name + "_mean_abs_px": totals[name]
        / max(totals[name + "_rows"], 1.0)
        for name in ("initial", "control", "treatment")
    }
    diagnostics["rows_measured"] = int(totals.get("initial_rows", 0.0))
    return {
        "images_written": images_written,
        "elapsed_seconds": time.perf_counter() - started,
        "held_out_perturbation": diagnostics,
    }


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    source_list = Path(population["list_path"])
    control_checkpoint = Path(args.control_checkpoint).expanduser().resolve()
    treatment_checkpoint = Path(args.treatment_checkpoint).expanduser().resolve()
    control_contract, control_report = _endpoint_contract(
        control_checkpoint,
        Path(args.control_report).expanduser().resolve(),
        expected_component="g4_denoising_control",
    )
    treatment_contract, treatment_report = _endpoint_contract(
        treatment_checkpoint,
        Path(args.treatment_report).expanduser().resolve(),
        expected_component="g4_denoising_treatment",
    )
    if (
        control_report["initial_checkpoint_sha256"]
        != treatment_report["initial_checkpoint_sha256"]
    ):
        raise ValueError("G4 paired arms did not start from the same endpoint")
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
    loader = build_dataloader(control_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(loader.dataset) != expected:
        raise ValueError("G4 evaluator altered official validation population")
    device = torch.device(args.device)
    control, control_iteration = _load_model(
        control_cfg, control_checkpoint, device
    )
    treatment, treatment_iteration = _load_model(
        treatment_cfg, treatment_checkpoint, device
    )
    if control_iteration != treatment_iteration:
        raise ValueError("G4 paired endpoints have different iterations")
    channels_last = bool(control_cfg["training"].get("channels_last", False))
    output_dir = Path(args.output_dir).expanduser().resolve()
    directories = {
        policy: output_dir / "predictions" / policy
        for policy in ("control", "denoising")
    }
    writer = _write_and_measure(
        control,
        treatment,
        loader,
        device=device,
        directories=directories,
        input_w=int(control_cfg["model"]["input_w"]),
        minimum_valid_rows=int(
            control_cfg["v25"]["loss"]["minimum_valid_rows"]
        ),
        channels_last=channels_last,
        log_interval=args.log_interval,
    )
    if writer["images_written"] != expected:
        raise RuntimeError("G4 writer did not consume full official validation")
    del control, treatment, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        policy: _json_metrics(
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
        for policy, directory in directories.items()
    }
    delta50 = 100.0 * (
        float(metrics["denoising"]["0.5"]["F1"])
        - float(metrics["control"]["0.5"]["F1"])
    )
    delta75 = 100.0 * (
        float(metrics["denoising"]["0.75"]["F1"])
        - float(metrics["control"]["0.75"]["F1"])
    )
    perturbation = writer["held_out_perturbation"]
    checks = {
        "held_out_perturbation_error_strictly_lower": float(
            perturbation["treatment_mean_abs_px"]
        )
        < float(perturbation["control_mean_abs_px"]),
        "clean_f1_50_not_worse_by_0p10_points": delta50 >= -0.10,
        "clean_f1_75_not_worse_by_0p10_points": delta75 >= -0.10,
    }
    report = {
        "experiment": "V25 G4 training-only denoising paired gate",
        "control_endpoint_contract": control_contract,
        "treatment_endpoint_contract": treatment_contract,
        "official_validation_population_contract": population,
        "writer_contract": writer,
        "metrics": metrics,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "treatment_minus_control_f1_50_points": delta50,
            "treatment_minus_control_f1_75_points": delta75,
            "held_out_treatment_minus_control_mae_px": float(
                perturbation["treatment_mean_abs_px"]
            )
            - float(perturbation["control_mean_abs_px"]),
        },
        "contract": {
            "training_perturbations_present_at_inference": False,
            "validation_subset_used": False,
            "validation_deduplication_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "test_set_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "g4_official_val_report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
