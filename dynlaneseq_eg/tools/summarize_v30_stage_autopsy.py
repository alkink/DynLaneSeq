from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = {
    "proposal_support_oracle": None,
    "selected_raw_proposal": "four_slot_global_unique",
    "selected_refined_lane": "four_slot_refined",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the exact-paired V30 proposal-support, route-selection, "
            "refinement, and activity/geometry decomposition."
        )
    )
    parser.add_argument("--control-coverage", required=True)
    parser.add_argument("--treatment-coverage", required=True)
    parser.add_argument("--factorial-audit", required=True)
    parser.add_argument("--exact-pair-summary", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _coverage_row(
    report: dict[str, Any], threshold: str
) -> dict[str, dict[str, float | int]]:
    oracle = report["capacity"][threshold]["all_candidate_oracle"]
    output: dict[str, dict[str, float | int]] = {
        "proposal_support_oracle": {
            "gt": int(oracle["gt"]),
            "hits": int(oracle["hits"]),
            "recall": float(oracle["recall"]),
        }
    }
    for stage_name, method_name in STAGES.items():
        if method_name is None:
            continue
        row = report["methods"][method_name][threshold]
        output[stage_name] = {
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "mean_selected_per_image": float(row["mean_selected_per_image"]),
        }
    support_hits = max(int(oracle["hits"]), 1)
    selected_raw_tp = int(output["selected_raw_proposal"]["tp"])
    output["conversion"] = {
        "support_to_selected_raw": selected_raw_tp / support_hits,
        "refinement_delta_tp": (
            int(output["selected_refined_lane"]["tp"]) - selected_raw_tp
        ),
        "refinement_delta_f1": (
            float(output["selected_refined_lane"]["f1"])
            - float(output["selected_raw_proposal"]["f1"])
        ),
    }
    return output


def _delta(
    control: dict[str, float | int],
    treatment: dict[str, float | int],
) -> dict[str, float | int]:
    keys = sorted(set(control) & set(treatment))
    return {
        key: float(treatment[key]) - float(control[key])
        for key in keys
        if isinstance(control[key], (int, float))
        and isinstance(treatment[key], (int, float))
    }


def _verdict(thresholds: dict[str, Any]) -> dict[str, Any]:
    strict = thresholds["0.75"]
    support_gain = int(strict["treatment_minus_control"][
        "proposal_support_oracle"
    ]["hits"])
    raw_gain = int(strict["treatment_minus_control"][
        "selected_raw_proposal"
    ]["tp"])
    refined_gain = int(strict["treatment_minus_control"][
        "selected_refined_lane"
    ]["tp"])
    treatment_refinement = int(
        strict["treatment"]["conversion"]["refinement_delta_tp"]
    )
    control_refinement = int(
        strict["control"]["conversion"]["refinement_delta_tp"]
    )

    if support_gain > 0 and raw_gain <= 0:
        primary = "support_gain_not_converted_by_route_selection"
    elif raw_gain > 0 and refined_gain < raw_gain:
        primary = "refinement_erases_part_of_selection_gain"
    elif refined_gain < 0:
        primary = "final_route_geometry_regresses_despite_support"
    else:
        primary = "no_single_stage_bottleneck_identified"

    return {
        "primary_bottleneck": primary,
        "strict_support_hit_gain": support_gain,
        "strict_selected_raw_tp_gain": raw_gain,
        "strict_selected_refined_tp_gain": refined_gain,
        "strict_refinement_delta_tp_control": control_refinement,
        "strict_refinement_delta_tp_treatment": treatment_refinement,
        "field_only_is_not_authorized_for_long_training": bool(
            refined_gain < 0 or thresholds["0.75"]["public_delta_f1"] < 0.0
        ),
        "interpretation": (
            "Proposal support, selected raw proposals, and final refined lanes "
            "are measured on the same full validation images with official-"
            "raster IoU. Oracle rows are diagnostic only and never deployable."
        ),
    }


def main() -> None:
    args = parse_args()
    control = _load(args.control_coverage)
    treatment = _load(args.treatment_coverage)
    factorial = _load(args.factorial_audit)
    exact = _load(args.exact_pair_summary)

    control_meta = control["metadata"]
    treatment_meta = treatment["metadata"]
    checks = {
        "full_validation_control": int(control_meta["num_records"]) == 9675,
        "full_validation_treatment": int(treatment_meta["num_records"]) == 9675,
        "same_validation_list": control_meta["list_sha256"]
        == treatment_meta["list_sha256"],
        "official_raster_control": control_meta["iou_space"]
        == "official_raster",
        "official_raster_treatment": treatment_meta["iou_space"]
        == "official_raster",
        "sequential_complete_control": int(control_meta["max_batches"]) == 0
        and control_meta["sample_strategy"] == "sequential",
        "sequential_complete_treatment": int(treatment_meta["max_batches"]) == 0
        and treatment_meta["sample_strategy"] == "sequential",
        "factorial_full_validation": int(factorial["images"]) == 9675,
        "test_split_closed": not bool(exact.get("test_split_used", True))
        and not bool(factorial.get("test_set_used", True)),
    }

    thresholds: dict[str, Any] = {}
    for threshold in ("0.50", "0.75"):
        control_row = _coverage_row(control, threshold)
        treatment_row = _coverage_row(treatment, threshold)
        exact_row = exact["thresholds"][str(float(threshold))]
        factorial_row = factorial["thresholds"][str(float(threshold))]
        thresholds[threshold] = {
            "control": control_row,
            "treatment": treatment_row,
            "treatment_minus_control": {
                stage: _delta(control_row[stage], treatment_row[stage])
                for stage in STAGES
            },
            "conversion_delta": _delta(
                control_row["conversion"], treatment_row["conversion"]
            ),
            "public_control_f1": float(exact_row["control"]["f1"]),
            "public_treatment_f1": float(exact_row["treatment"]["f1"]),
            "public_delta_f1": float(exact_row["delta"]["f1"]),
            "activity_geometry_factorial": factorial_row,
        }

    report = {
        "experiment": "V30 exact-paired full-validation stage autopsy",
        "diagnostic_only": True,
        "test_split_used": False,
        "checks": checks,
        "passed_contract": all(checks.values()),
        "thresholds": thresholds,
        "verdict": _verdict(thresholds),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed_contract"]:
        raise SystemExit("V30 stage autopsy contract failed")


if __name__ == "__main__":
    main()
