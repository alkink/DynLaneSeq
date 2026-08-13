from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the immutable two-domain V16 Stage-A endpoint gate."
    )
    parser.add_argument("--preflight-summary", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _threshold(
    report: dict[str, Any], policy: str, threshold: str
) -> dict[str, Any]:
    return report["metrics"][policy]["writer_valid"]["thresholds"][threshold]


def _delta(
    report: dict[str, Any], comparison: str, threshold: str
) -> dict[str, Any]:
    return report["deltas"][comparison]["writer_valid"][threshold]


def _domain(report: dict[str, Any]) -> dict[str, Any]:
    gain_050 = _delta(
        report, "v16_minus_v7_anchor_reference", "0.50"
    )
    gain_075 = _delta(
        report, "v16_minus_v7_anchor_reference", "0.75"
    )
    wrong_050 = _delta(
        report, "correct_minus_cross_clip_wrong_p2", "0.50"
    )
    wrong_075 = _delta(
        report, "correct_minus_cross_clip_wrong_p2", "0.75"
    )
    zero_050 = _delta(report, "correct_minus_zero_p2", "0.50")
    zero_075 = _delta(report, "correct_minus_zero_p2", "0.75")
    source = _threshold(report, "v7_anchor_reference", "0.50")
    treatment = _threshold(report, "v16_selected_reference", "0.50")
    values = {
        "images": int(report.get("images", 0)),
        "delta_tp_050": int(gain_050["delta_tp"]),
        "delta_tp_075": int(gain_075["delta_tp"]),
        "delta_f1_points_050": float(gain_050["delta_f1_points"]),
        "delta_f1_points_075": float(gain_075["delta_f1_points"]),
        "prediction_count_drift": int(treatment["predictions"])
        - int(source["predictions"]),
        "correct_minus_wrong_p2_tp_050": int(wrong_050["delta_tp"]),
        "correct_minus_wrong_p2_tp_075": int(wrong_075["delta_tp"]),
        "correct_minus_zero_p2_tp_050": int(zero_050["delta_tp"]),
        "correct_minus_zero_p2_tp_075": int(zero_075["delta_tp"]),
        "correct_vs_wrong_selection_change_fraction": float(
            report.get("correct_vs_wrong_p2_selection_change_fraction", 0.0)
        ),
        "wrong_p2_mean_abs_score_change": float(
            report.get("mean_abs_score_change", {}).get(
                "cross_clip_wrong_p2", 0.0
            )
        ),
        "selected_duplicate_count": int(
            report.get("selected_duplicate_count", -1)
        ),
    }
    checks = {
        "exact_256_images": values["images"] == 256,
        "tp_050_at_least_v7_anchor_plus_3": values["delta_tp_050"] >= 3,
        "tp_075_at_least_v7_anchor_plus_6": values["delta_tp_075"] >= 6,
        "f1_050_strictly_improves": values["delta_f1_points_050"] > 0.0,
        "f1_075_strictly_improves": values["delta_f1_points_075"] > 0.0,
        "prediction_count_exact": values["prediction_count_drift"] == 0,
        "correct_p2_noninferior_to_wrong_050": (
            values["correct_minus_wrong_p2_tp_050"] >= 0
        ),
        "correct_p2_noninferior_to_wrong_075": (
            values["correct_minus_wrong_p2_tp_075"] >= 0
        ),
        "correct_p2_noninferior_to_zero_050": (
            values["correct_minus_zero_p2_tp_050"] >= 0
        ),
        "correct_p2_noninferior_to_zero_075": (
            values["correct_minus_zero_p2_tp_075"] >= 0
        ),
        "p2_changes_at_least_1pct_of_hard_decisions": (
            values["correct_vs_wrong_selection_change_fraction"] >= 0.01
        ),
        "p2_changes_candidate_scores": (
            values["wrong_p2_mean_abs_score_change"] > 1.0e-4
        ),
        "selected_duplicates_zero": values["selected_duplicate_count"] == 0,
        "cross_clip_runtime_exact": (
            int(report.get("cross_clip_runtime_same_image", -1)) == 0
            and int(report.get("cross_clip_runtime_same_clip", -1)) == 0
        ),
        "no_coordinate_averaging": (
            report.get("coordinate_averaging_used") is False
        ),
        "no_fixed_k_or_padding": (
            report.get("fixed_k_or_padding_used") is False
        ),
        "fixed_v7_activity_count_scores": (
            report.get("activity_count_scores_source") == "exact_v7"
        ),
        "test_closed": report.get("test_set_used") is False,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    preflight = _load(args.preflight_summary)
    contract = _load(args.contract)
    heldout_report = _load(args.heldout)
    validation_report = _load(args.validation)
    heldout = _domain(heldout_report)
    validation = _domain(validation_report)
    checks = {
        "candidate_group_preflight_passed": preflight.get("passed") is True,
        "zero_step_contract_passed": contract.get("passed") is True,
        "heldout_clip_256_passed": heldout["passed"],
        "validation_256_passed": validation["passed"],
        "all_reports_test_closed": all(
            report.get("test_set_used") is False
            for report in (contract, heldout_report, validation_report)
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V16 candidate-aligned coherent hard-reranker fixed gate",
        "gate_thresholds": {
            "tp_050_gain_min": 3,
            "tp_075_gain_min": 6,
            "f1_both_thresholds_strictly_improve": True,
            "prediction_count_exact_v7": True,
            "p2_wrong_and_zero_noninferiority": True,
            "p2_hard_decision_change_fraction_min": 0.01,
            "selected_duplicates": 0,
        },
        "heldout_clip_256": heldout,
        "validation_256": validation,
        "checks": checks,
        "passed": passed,
        "v16_complete": True,
        "long_training_authorized": False,
        "full_validation_authorized": False,
        "next_version_authorized": False,
        "test_set_used": False,
        "checkpoint_selection_performed": False,
        "decision": (
            "v16_stage_a_pass_stop_for_review"
            if passed
            else "v16_stage_a_fail_stop_for_review"
        ),
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
