from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize V6-A probe parity and target-distribution audits."
    )
    parser.add_argument("--reference-probe-report", required=True)
    parser.add_argument("--probe-import-report", required=True)
    parser.add_argument("--production-parity-report", required=True)
    parser.add_argument("--target-distribution-report", required=True)
    parser.add_argument("--failed-gate-summary", default="")
    parser.add_argument("--failed-train-log", default="")
    parser.add_argument("--parity-f1-tolerance", type=float, default=0.005)
    parser.add_argument("--parity-count-tolerance", type=float, default=0.10)
    parser.add_argument("--min-production-f1-050", type=float, default=0.795)
    parser.add_argument("--min-production-f1-075", type=float, default=0.56)
    parser.add_argument("--min-selected", type=float, default=3.0)
    parser.add_argument("--max-selected", type=float, default=3.5)
    parser.add_argument(
        "--min-calibrated-representable-fraction",
        type=float,
        default=0.95,
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_log_metric(path: str, name: str) -> dict[str, Any] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    values = [
        float(value)
        for value in re.findall(
            rf"(?<![A-Za-z0-9_]){re.escape(name)} ([0-9.eE+-]+)", text
        )
    ]
    if not values:
        return None
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "first": values[0],
        "last": values[-1],
        "minimum": min(values),
        "maximum": max(values),
    }


def _log_integer(path: str, name: str) -> int | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(rf"'{re.escape(name)}': ([0-9]+)", text)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    reference = _load(args.reference_probe_report)
    imported = _load(args.probe_import_report)
    parity = _load(args.production_parity_report)
    targets = _load(args.target_distribution_report)
    failed_gate = _load(args.failed_gate_summary) if args.failed_gate_summary else None

    reference_slot = reference["evaluation"]["strategies"]["learned_4_slots"]
    parity_050 = parity["methods"]["four_slot_global_unique"]["0.50"]
    parity_075 = parity["methods"]["four_slot_global_unique"]["0.75"]
    reference_images = int(
        reference.get("caches", {})
        .get("val", {})
        .get("sample_count", 256)
    )
    reference_count = float(reference_slot["pred"]) / max(reference_images, 1)
    parity_count = float(parity_050["mean_selected_per_image"])
    differences = {
        "f1_050": float(parity_050["f1"])
        - float(reference_slot["f1_050"]),
        "f1_075": float(parity_075["f1"])
        - float(reference_slot["f1_075"]),
        "mean_selected": parity_count - reference_count,
    }
    parity_checks = {
        "probe_import_contract": bool(imported.get("passed")),
        "production_f1_050_matches_probe": abs(differences["f1_050"])
        <= float(args.parity_f1_tolerance),
        "production_f1_075_matches_probe": abs(differences["f1_075"])
        <= float(args.parity_f1_tolerance),
        "production_count_matches_probe": abs(differences["mean_selected"])
        <= float(args.parity_count_tolerance),
    }
    parity_passed = all(parity_checks.values())

    modes = targets["modes"]
    literal_clean = modes["train_clean"]["literal_official_threshold"]
    literal_augmented = modes["train_augmented"][
        "literal_official_threshold"
    ]
    calibrated_clean = modes["train_clean"]["calibrated_narrow"]
    calibrated_augmented = modes["train_augmented"]["calibrated_narrow"]
    calibrated_validation = modes["val_clean"]["calibrated_narrow"]
    literal_aug_gap = float(literal_clean["mean_representable_lanes"]) - float(
        literal_augmented["mean_representable_lanes"]
    )
    calibrated_aug_gap = float(
        calibrated_clean["mean_representable_lanes"]
    ) - float(calibrated_augmented["mean_representable_lanes"])
    target_findings = {
        "literal_train_clean_mean_representable": float(
            literal_clean["mean_representable_lanes"]
        ),
        "literal_train_augmented_mean_representable": float(
            literal_augmented["mean_representable_lanes"]
        ),
        "literal_augmentation_drop": literal_aug_gap,
        "calibrated_train_clean_mean_representable": float(
            calibrated_clean["mean_representable_lanes"]
        ),
        "calibrated_train_augmented_mean_representable": float(
            calibrated_augmented["mean_representable_lanes"]
        ),
        "calibrated_augmentation_drop": calibrated_aug_gap,
        "literal_augmented_expected_dustbin_fraction": float(
            literal_augmented["expected_dustbin_fraction"]
        ),
        "calibrated_augmented_expected_dustbin_fraction": float(
            calibrated_augmented["expected_dustbin_fraction"]
        ),
        "calibrated_augmented_representable_gt_fraction": float(
            calibrated_augmented["representable_gt_fraction"]
        ),
        "calibrated_validation_mean_representable": float(
            calibrated_validation["mean_representable_lanes"]
        ),
        "calibrated_validation_count_minus_probe_output": float(
            calibrated_validation["mean_representable_lanes"]
        )
        - reference_count,
        "probe_output_mean_selected": reference_count,
    }

    probe_steps = int(reference.get("training", {}).get("steps", 0))
    probe_batch = int(reference.get("training", {}).get("batch_size", 0))
    probe_cache_images = int(
        reference.get("caches", {}).get("train", {}).get("sample_count", 0)
    )
    production_steps = _log_integer(args.failed_train_log, "iters")
    production_batch = _log_integer(
        args.failed_train_log, "effective_batch_size"
    )
    production_images = _log_integer(args.failed_train_log, "train_images")
    training_exposure = {
        "probe_optimizer_steps": probe_steps,
        "probe_batch_size": probe_batch,
        "probe_sample_presentations": probe_steps * probe_batch,
        "probe_fixed_cache_images": probe_cache_images,
        "probe_cache_passes": (
            float(probe_steps * probe_batch) / float(probe_cache_images)
            if probe_cache_images > 0
            else None
        ),
        "production_optimizer_steps": production_steps,
        "production_effective_batch_size": production_batch,
        "production_sample_presentations": (
            production_steps * production_batch
            if production_steps is not None and production_batch is not None
            else None
        ),
        "production_train_images": production_images,
        "production_dataset_epochs": (
            float(production_steps * production_batch)
            / float(production_images)
            if production_steps is not None
            and production_batch is not None
            and production_images
            else None
        ),
    }

    imported_probe_uniform_gate = {
        "f1_050": float(parity_050["f1"])
        >= float(args.min_production_f1_050),
        "f1_075": float(parity_075["f1"])
        >= float(args.min_production_f1_075),
        "selected_count": float(args.min_selected)
        <= parity_count
        <= float(args.max_selected),
    }
    imported_probe_uniform_passed = all(imported_probe_uniform_gate.values())
    calibrated_target_passed = (
        float(calibrated_augmented["representable_gt_fraction"])
        >= float(args.min_calibrated_representable_fraction)
    )

    if not parity_passed:
        next_step = "fix_production_feature_or_decode_parity_before_training"
    elif imported_probe_uniform_passed:
        next_step = (
            "run_full_validation_of_imported_probe_then_use_calibrated_"
            "target_for_any_further_training"
        )
    elif calibrated_target_passed:
        next_step = (
            "authorize_single_v6_a1_calibrated_target_and_exposure_gate_"
            "with_augmentation"
        )
    elif (
        float(calibrated_clean["representable_gt_fraction"])
        >= float(args.min_calibrated_representable_fraction)
        and calibrated_aug_gap >= 0.25
    ):
        next_step = "authorize_single_v6_a1_calibrated_target_gate_without_strong_augmentation"
    else:
        next_step = "do_not_train_recalibrate_row_strip_target_on_v5_1_source"

    train_log_metrics = {
        name: _mean_log_metric(args.failed_train_log, name)
        for name in (
            "four_slot_target_mean_representable_count",
            "four_slot_target_mean_support_size",
            "four_slot_target_mean_entropy",
            "loss_four_slot_permutation",
            "four_slot_mean_route_entropy",
        )
    }
    report = {
        "experiment": "V6-A production/probe mismatch root-cause audit",
        "diagnostic_only": True,
        "training_started": False,
        "reference_probe": {
            "checkpoint_sha256": reference.get("checkpoint_sha256"),
            "f1_050": float(reference_slot["f1_050"]),
            "f1_075": float(reference_slot["f1_075"]),
            "mean_selected": reference_count,
        },
        "production_with_exact_probe_weights": {
            "f1_050": float(parity_050["f1"]),
            "f1_075": float(parity_075["f1"]),
            "mean_selected": parity_count,
            "slot_diagnostics": parity.get("four_slot_diagnostics"),
        },
        "parity_differences": differences,
        "parity_checks": parity_checks,
        "parity_passed": parity_passed,
        "imported_probe_uniform_gate": imported_probe_uniform_gate,
        "imported_probe_uniform_passed": imported_probe_uniform_passed,
        "calibrated_target_passed": calibrated_target_passed,
        "target_findings": target_findings,
        "training_exposure": training_exposure,
        "target_distribution": modes,
        "failed_training_log_metrics": train_log_metrics,
        "failed_gate_best": failed_gate.get("best")
        if isinstance(failed_gate, dict)
        else None,
        "next_step": next_step,
        "warning": (
            "Full validation is authorized only for an imported probe that "
            "passes exact production parity and the declared uniform gate. "
            "Any later training must replace the literal official-IoU-as-row-"
            "strip threshold with the calibrated target contract."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
