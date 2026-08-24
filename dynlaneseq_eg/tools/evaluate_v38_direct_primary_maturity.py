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
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v38_direct_primary_maturity import (
    FIXED_EFFECTIVE_BATCH,
    FIXED_ENDPOINT,
    FIXED_SCHEDULE,
    validate_v38_contract,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


THRESHOLDS = (0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed V38 direct-primary 50K maturity endpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--v7-reference-metrics", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--metric-workers", type=int, default=20)
    parser.add_argument("--metric-chunksize", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _configured(
    cfg: dict[str, Any], *, root: Path, list_path: Path, batch_size: int, workers: int
) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value.setdefault("dataset", {})["root"] = str(root)
    value["dataset"].setdefault("lists", {})["val"] = str(list_path)
    value["dataset"]["load_targets"] = False
    value["dataset"]["infer_seg_labels"] = False
    value.setdefault("dataloader", {})["eval_batch_size"] = int(batch_size)
    value["dataloader"]["num_workers"] = int(workers)
    value["dataloader"]["persistent_workers"] = workers > 0
    value.setdefault("model", {})["pretrained_backbone"] = False
    value["model"]["require_pretrained_backbone"] = False
    return value


def validate_training_endpoint(
    checkpoint: Path, report_path: Path, cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config_contract = validate_v38_contract(cfg)
    checks = {
        "config_contract": config_contract["passed"],
        "checkpoint_sha_exact": report.get("checkpoint_sha256")
        == sha256_file(checkpoint),
        "scientific_gate": report.get("scientific_gate") is True,
        "iteration_exact": int(report.get("iteration", -1)) == FIXED_ENDPOINT,
        "effective_batch_exact": int(report.get("effective_batch_size", -1))
        == FIXED_EFFECTIVE_BATCH,
        "schedule_exact": int(report.get("scheduler_total_iters", -1))
        == FIXED_SCHEDULE,
        "no_checkpoint_selection": report.get("checkpoint_selection_performed")
        is False,
        "no_threshold_selection": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    return {"passed": all(checks.values()), "checks": checks}, report


def _minimal(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "exist_logits": output["exist_logits"],
        "pred_x_rows": output["pred_x_rows"],
        "range_norm": output["range_norm"],
        "quality_logits": output["quality_logits"],
    }


@torch.inference_mode()
def _write(
    model: DynLaneSeqV25,
    loader,
    *,
    device: torch.device,
    output_dir: Path,
    channels_last: bool,
    log_interval: int,
) -> dict[str, Any]:
    model.requires_grad_(False).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_written = 0
    geometry: dict[str, float] = {}
    started = time.perf_counter()
    for batch_index, (images, _targets, metas) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(images.float())
        write_culane_predictions(
            _minimal(output),
            metas,
            output_dir,
            score_thresh=0.5,
            min_pred_points=5,
            nms_distance_thresh_px=0.0,
            top_k=4,
            quality_score_power=0.0,
            score_mode="exist",
        )
        active = torch.softmax(output["exist_logits"].float(), dim=-1)[..., 0] >= 0.5
        _merge_geometry(
            geometry,
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
                        "phase": "write_v38_predictions",
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
        "elapsed_seconds": time.perf_counter() - started,
        "geometry": _finish_geometry(geometry),
    }


def _metric_payload(metrics: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metrics.items()}


def _extract_reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict) or "0.5" not in results or "0.75" not in results:
        raise ValueError("V38 V7 reference metrics have an unsupported schema")
    checks = {
        "reference_checkpoint_50k": "0050000" in str(payload.get("checkpoint", "")),
        "reference_validation_split": payload.get("split") == "val",
        "reference_nms_off": float(payload.get("lane_nms_distance_thresh_px", -1.0))
        == 0.0,
        "reference_top4": int(payload.get("top_k", -1)) == 4,
        "reference_test_unused": payload.get("split") != "test",
    }
    if not all(checks.values()):
        raise ValueError("V38 V7 reference contract failed: " + json.dumps(checks))
    return payload, checks


def decide_v38(
    metrics: dict[str, Any], reference_results: dict[str, Any]
) -> dict[str, Any]:
    deltas = {
        threshold: 100.0
        * (
            float(metrics[threshold]["F1"])
            - float(reference_results[threshold]["F1"])
        )
        for threshold in ("0.5", "0.75")
    }
    strong_win = deltas["0.5"] >= 0.30 and deltas["0.75"] >= 0.0
    maturity_pass = deltas["0.5"] >= -0.50 and deltas["0.75"] >= -0.50
    decision = (
        "DIRECT_PRIMARY_50K_WIN"
        if strong_win
        else (
            "AUTHORIZE_DIRECT_PRIMARY_MATURE_EXTENSION"
            if maturity_pass
            else "DIRECT_PRIMARY_50K_FAIL"
        )
    )
    return {
        "decision": decision,
        "deltas_f1_points": deltas,
        "checks": {
            "strong_win_f1_50_at_least_plus_0p30": deltas["0.5"] >= 0.30,
            "strong_win_f1_75_non_regression": deltas["0.75"] >= 0.0,
            "maturity_f1_50_within_0p50": deltas["0.5"] >= -0.50,
            "maturity_f1_75_within_0p50": deltas["0.75"] >= -0.50,
        },
        "interpretation": (
            "Direct-primary 50K'da V7'yi geçti; ikinci seed ve uzun endpoint hak etti."
            if strong_win
            else (
                "Direct-primary 50K'da V7'ye yeterince yaklaştı; tek predeclared uzun endpoint hak etti."
                if maturity_pass
                else "Direct-primary eşit-horizon 50K gate'ini geçemedi; mevcut direct-primary detector kapatılmalı."
            )
        ),
    }


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    cfg = load_config(args.config)
    endpoint_contract, training_report = validate_training_endpoint(
        Path(args.checkpoint).expanduser().resolve(),
        Path(args.training_report).expanduser().resolve(),
        cfg,
    )
    if not endpoint_contract["passed"]:
        raise ValueError("V38 endpoint contract failed: " + json.dumps(endpoint_contract))
    reference, reference_checks = _extract_reference(
        Path(args.v7_reference_metrics).expanduser().resolve()
    )
    eval_cfg = _configured(
        cfg,
        root=root,
        list_path=Path(population["list_path"]),
        batch_size=args.eval_batch_size,
        workers=args.num_workers,
    )
    loader = build_dataloader(eval_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(loader.dataset) != expected:
        raise ValueError("V38 evaluator altered official validation")
    model = build_model(eval_cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("V38 evaluator requires DynLaneSeqV25")
    iteration = int(
        load_checkpoint(
            Path(args.checkpoint).expanduser().resolve(), model, strict=True
        )
    )
    if iteration != FIXED_ENDPOINT or iteration != int(training_report["iteration"]):
        raise ValueError("V38 checkpoint iteration mismatch")
    device = torch.device(args.device)
    model.to(device)
    channels_last = bool(eval_cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_dir = output_dir / "predictions"
    writer = _write(
        model,
        loader,
        device=device,
        output_dir=prediction_dir,
        channels_last=channels_last,
        log_interval=args.log_interval,
    )
    if int(writer["images_written"]) != expected:
        raise RuntimeError("V38 writer did not consume full validation")
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = _metric_payload(
        eval_predictions(
            pred_dir=prediction_dir,
            anno_dir=root,
            list_path=Path(population["list_path"]),
            iou_thresholds=THRESHOLDS,
            width=30,
            official=True,
            sequential=False,
            num_workers=args.metric_workers,
            chunksize=args.metric_chunksize,
        )
    )
    verdict = decide_v38(metrics, reference["results"])
    report = {
        "experiment": "V38 direct-primary matched-50K maturity gate",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_iteration": iteration,
        "endpoint_contract": endpoint_contract,
        "official_validation_population_contract": population,
        "writer_contract": writer,
        "metrics": metrics,
        "v7_reference": {
            "path": str(Path(args.v7_reference_metrics).expanduser().resolve()),
            "sha256": sha256_file(args.v7_reference_metrics),
            "checks": reference_checks,
            "results": reference["results"],
        },
        "verdict": verdict,
        "contract": {
            "fixed_endpoint": FIXED_ENDPOINT,
            "effective_batch": FIXED_EFFECTIVE_BATCH,
            "scheduler_total_iters": FIXED_SCHEDULE,
            "official_val_rows": expected,
            "validation_subset_used": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "score_threshold": 0.5,
            "decode_mode": "expectation",
            "lane_nms_enabled": False,
            "top_k": 4,
            "line_width": 30,
            "inference_dtype": "float32",
            "test_set_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "v38_direct_primary_50k_official_val.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
