from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.evaluation.culane_metric import eval_predictions
from dynlaneseq_eg.factory import build_dataloader, build_model
from dynlaneseq_eg.modeling.dynlaneseq_v25 import DynLaneSeqV25
from dynlaneseq_eg.tools.evaluate_v38_direct_primary_maturity import (
    THRESHOLDS,
    _configured,
    _extract_reference,
    _metric_payload,
    _write,
)
from dynlaneseq_eg.tools.train import seed_everything
from dynlaneseq_eg.tools.train_v39_pattern_query_initialization import (
    FIXED_EFFECTIVE_BATCH,
    FIXED_ENDPOINT,
    FIXED_SCHEDULE,
    SOURCE_ITERATION,
    validate_v39_contract,
)
from dynlaneseq_eg.tools.v23_official_protocol import (
    official_v23_culane_list_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-paired V39 control/treatment on full validation."
    )
    for arm in ("control", "treatment"):
        parser.add_argument(f"--{arm}-config", required=True)
        parser.add_argument(f"--{arm}-checkpoint", required=True)
        parser.add_argument(f"--{arm}-training-report", required=True)
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


def _training_contract(
    *,
    arm: str,
    cfg: dict[str, Any],
    checkpoint: Path,
    report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config_contract = validate_v39_contract(cfg)
    checks = {
        "config_contract": config_contract["passed"],
        "arm_exact": report.get("arm") == arm == config_contract["arm"],
        "checkpoint_sha_exact": report.get("checkpoint_sha256")
        == sha256_file(checkpoint),
        "source_iteration_exact": int(report.get("source_iteration", -1))
        == SOURCE_ITERATION,
        "iteration_exact": int(report.get("iteration", -1)) == FIXED_ENDPOINT,
        "effective_batch_exact": int(report.get("effective_batch_size", -1))
        == FIXED_EFFECTIVE_BATCH,
        "schedule_exact": int(report.get("scheduler_total_iters", -1))
        == FIXED_SCHEDULE,
        "scientific_gate": report.get("scientific_gate") is True,
        "no_checkpoint_selection": report.get("checkpoint_selection_performed")
        is False,
        "no_threshold_selection": report.get("threshold_selection_performed")
        is False,
        "test_unused": report.get("test_set_used") is False,
    }
    return {"passed": all(checks.values()), "checks": checks}, report


def _evaluate_arm(
    *,
    arm: str,
    cfg_path: Path,
    checkpoint: Path,
    training_report_path: Path,
    root: Path,
    population: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    metric_workers: int,
    metric_chunksize: int,
    log_interval: int,
    reference: dict[str, Any],
    reference_path: Path,
    reference_checks: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(cfg_path)
    endpoint_contract, training_report = _training_contract(
        arm=arm,
        cfg=cfg,
        checkpoint=checkpoint,
        report_path=training_report_path,
    )
    if not endpoint_contract["passed"]:
        raise ValueError(f"V39 {arm} endpoint contract failed: {endpoint_contract}")
    eval_cfg = _configured(
        cfg,
        root=root,
        list_path=Path(population["list_path"]),
        batch_size=eval_batch_size,
        workers=num_workers,
    )
    loader = build_dataloader(eval_cfg, split="val", training=False)
    expected = int(population["expected_nonempty_rows"])
    if len(loader.dataset) != expected:
        raise ValueError(f"V39 {arm} evaluator altered official validation")
    model = build_model(eval_cfg)
    if not isinstance(model, DynLaneSeqV25):
        raise TypeError("V39 evaluator requires DynLaneSeqV25")
    iteration = int(load_checkpoint(checkpoint, model, strict=True))
    if iteration != FIXED_ENDPOINT:
        raise ValueError(f"V39 {arm} checkpoint iteration mismatch")
    model.to(device)
    channels_last = bool(eval_cfg["training"].get("channels_last", False))
    if channels_last:
        model.to(memory_format=torch.channels_last)
    prediction_dir = output_dir / arm / "predictions"
    writer = _write(
        model,
        loader,
        device=device,
        output_dir=prediction_dir,
        channels_last=channels_last,
        log_interval=log_interval,
    )
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
            num_workers=metric_workers,
            chunksize=metric_chunksize,
        )
    )
    report = {
        "experiment": "V39 image-conditioned pattern-query full-validation arm",
        "arm": arm,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "endpoint_contract": endpoint_contract,
        "official_validation_population_contract": population,
        "writer_contract": writer,
        "metrics": metrics,
        "v7_reference": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "checks": reference_checks,
            "results": reference["results"],
        },
        "training_report": str(training_report_path),
        "contract": {
            "fixed_source_iteration": SOURCE_ITERATION,
            "fixed_endpoint": FIXED_ENDPOINT,
            "effective_batch": FIXED_EFFECTIVE_BATCH,
            "scheduler_total_iters": FIXED_SCHEDULE,
            "official_val_rows": expected,
            "full_validation": True,
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
    report_path = output_dir / arm / "official_val.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report, {"report": str(report_path), "predictions": str(prediction_dir)}


def main() -> None:
    args = parse_args()
    seed_everything(3407)
    root = Path(args.dataset_root).expanduser().resolve()
    population = official_v23_culane_list_contract(root, split="val")
    reference_path = Path(args.v7_reference_metrics).expanduser().resolve()
    reference, reference_checks = _extract_reference(reference_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for arm in ("control", "treatment"):
        report, artifact = _evaluate_arm(
            arm=arm,
            cfg_path=Path(getattr(args, f"{arm}_config")).expanduser().resolve(),
            checkpoint=Path(getattr(args, f"{arm}_checkpoint")).expanduser().resolve(),
            training_report_path=Path(
                getattr(args, f"{arm}_training_report")
            ).expanduser().resolve(),
            root=root,
            population=population,
            output_dir=output_dir,
            device=device,
            eval_batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            metric_workers=int(args.metric_workers),
            metric_chunksize=int(args.metric_chunksize),
            log_interval=int(args.log_interval),
            reference=reference,
            reference_path=reference_path,
            reference_checks=reference_checks,
        )
        reports[arm] = report
        artifacts[arm] = artifact
    deltas = {
        threshold: 100.0
        * (
            float(reports["treatment"]["metrics"][threshold]["F1"])
            - float(reports["control"]["metrics"][threshold]["F1"])
        )
        for threshold in ("0.5", "0.75")
    }
    first_control = json.loads(
        Path(args.control_training_report).read_text(encoding="utf-8")
    )["first_batch_manifest"]
    first_treatment = json.loads(
        Path(args.treatment_training_report).read_text(encoding="utf-8")
    )["first_batch_manifest"]
    if first_control != first_treatment:
        raise RuntimeError("V39 pair did not use the same first continuation batch")
    pair = {
        "experiment": "V39 exact-paired image-conditioned pattern-query gate",
        "source_iteration": SOURCE_ITERATION,
        "endpoint_iteration": FIXED_ENDPOINT,
        "arms": artifacts,
        "metrics": {arm: report["metrics"] for arm, report in reports.items()},
        "treatment_minus_control_f1_points": deltas,
        "exact_first_batch_manifest_match": True,
        "preliminary_official_checks": {
            "f1_50_non_regression": deltas["0.5"] >= 0.0,
            "f1_75_at_least_plus_0p50": deltas["0.75"] >= 0.50,
        },
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "test_set_used": False,
    }
    path = output_dir / "v39_official_pair.json"
    path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(pair, indent=2, sort_keys=True))
    print(f"output_json: {path}")


if __name__ == "__main__":
    main()
