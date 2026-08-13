from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the immutable V17 two-domain endpoint gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--heldout-source-coverage", required=True)
    parser.add_argument("--heldout-treatment-coverage", required=True)
    parser.add_argument("--val-source-coverage", required=True)
    parser.add_argument("--val-treatment-coverage", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _coverage(payload: dict[str, Any]) -> dict[str, Any]:
    slots = payload.get("four_slot_diagnostics") or {}
    capacity = payload["capacity"]
    return {
        "semantic_duplicate_fraction": float(
            slots.get("semantic_duplicate_cluster_fraction", 1.0)
        ),
        "proposal_oracle_hits": {
            threshold: int(capacity[threshold]["all_candidate_oracle"]["hits"])
            for threshold in THRESHOLDS
        },
    }


def _threshold(report: dict[str, Any], policy: str, mode: str, threshold: str) -> dict[str, Any]:
    return report["metrics"][policy][mode]["thresholds"][threshold]


def _delta(report: dict[str, Any], comparison: str, mode: str, threshold: str) -> dict[str, Any]:
    return report["deltas"][comparison][mode][threshold]


def _domain(
    report: dict[str, Any],
    source_coverage_payload: dict[str, Any],
    treatment_coverage_payload: dict[str, Any],
) -> dict[str, Any]:
    source_coverage = _coverage(source_coverage_payload)
    treatment_coverage = _coverage(treatment_coverage_payload)
    delta_050 = _delta(report, "v17_minus_v7", "writer_valid", "0.50")
    delta_075 = _delta(report, "v17_minus_v7", "writer_valid", "0.75")
    wrong_050 = _delta(report, "correct_minus_cross_clip_wrong_image", "writer_valid", "0.50")
    wrong_075 = _delta(report, "correct_minus_cross_clip_wrong_image", "writer_valid", "0.75")
    zero_050 = _delta(report, "correct_minus_zero_image", "writer_valid", "0.50")
    zero_075 = _delta(report, "correct_minus_zero_image", "writer_valid", "0.75")
    context_050 = _delta(report, "correct_minus_no_proposal_context", "writer_valid", "0.50")
    context_075 = _delta(report, "correct_minus_no_proposal_context", "writer_valid", "0.75")
    source_neural = _threshold(report, "v7_anchor", "neural_active", "0.50")
    treatment_neural = _threshold(report, "v17_final", "neural_active", "0.50")
    source_writer = _threshold(report, "v7_anchor", "writer_valid", "0.50")
    treatment_writer = _threshold(report, "v17_final", "writer_valid", "0.50")
    source_invalid = int(source_neural["predictions"]) - int(source_writer["predictions"])
    treatment_invalid = int(treatment_neural["predictions"]) - int(treatment_writer["predictions"])
    oracle_equal = all(
        source_coverage["proposal_oracle_hits"][threshold]
        == treatment_coverage["proposal_oracle_hits"][threshold]
        for threshold in THRESHOLDS
    )
    paired_050 = report["paired_image_effects"]["0.50"]
    paired_075 = report["paired_image_effects"]["0.75"]
    values = {
        "delta_tp_050": int(delta_050["delta_tp"]),
        "delta_tp_075": int(delta_075["delta_tp"]),
        "delta_f1_points_050": float(delta_050["delta_f1_points"]),
        "delta_f1_points_075": float(delta_075["delta_f1_points"]),
        "neural_prediction_count_drift": int(treatment_neural["predictions"]) - int(source_neural["predictions"]),
        "writer_prediction_count_drift": int(treatment_writer["predictions"]) - int(source_writer["predictions"]),
        "writer_invalid_increase": treatment_invalid - source_invalid,
        "correct_minus_wrong_tp_050": int(wrong_050["delta_tp"]),
        "correct_minus_wrong_tp_075": int(wrong_075["delta_tp"]),
        "correct_minus_zero_tp_050": int(zero_050["delta_tp"]),
        "correct_minus_zero_tp_075": int(zero_075["delta_tp"]),
        "correct_minus_no_context_tp_050": int(context_050["delta_tp"]),
        "correct_minus_no_context_tp_075": int(context_075["delta_tp"]),
        "improved_images_050": int(paired_050["improved"]),
        "worsened_images_050": int(paired_050["worsened"]),
        "improved_images_075": int(paired_075["improved"]),
        "worsened_images_075": int(paired_075["worsened"]),
        "semantic_duplicate_fraction": treatment_coverage["semantic_duplicate_fraction"],
        "proposal_oracle_equal": oracle_equal,
    }
    checks = {
        "tp_050_at_least_v7_plus_3": values["delta_tp_050"] >= 3,
        "tp_075_at_least_v7_plus_6": values["delta_tp_075"] >= 6,
        "f1_050_strictly_improves": values["delta_f1_points_050"] > 0.0,
        "f1_075_nonregression": values["delta_f1_points_075"] >= 0.0,
        "neural_prediction_count_exact": values["neural_prediction_count_drift"] == 0,
        "writer_prediction_count_exact": values["writer_prediction_count_drift"] == 0,
        "writer_invalid_increase_zero": values["writer_invalid_increase"] == 0,
        "correct_image_at_least_wrong_plus_5_tp_050": values["correct_minus_wrong_tp_050"] >= 5,
        "correct_image_at_least_wrong_plus_8_tp_075": values["correct_minus_wrong_tp_075"] >= 8,
        "correct_image_at_least_zero_plus_3_tp_050": values["correct_minus_zero_tp_050"] >= 3,
        "correct_image_at_least_zero_plus_3_tp_075": values["correct_minus_zero_tp_075"] >= 3,
        "improved_images_exceed_worsened_050": values["improved_images_050"] > values["worsened_images_050"],
        "improved_images_exceed_worsened_075": values["improved_images_075"] > values["worsened_images_075"],
        "semantic_duplicate_at_most_2pct": values["semantic_duplicate_fraction"] <= 0.02,
        "proposal_oracle_unchanged": oracle_equal,
        "cross_clip_runtime_exact": int(report.get("cross_clip_runtime_same_image", -1)) == 0 and int(report.get("cross_clip_runtime_same_clip", -1)) == 0,
        "fixed_activity_score_source": report.get("activity_score_route_source") == "exact_v7",
        "hard_proposal_id_not_used": report.get("hard_proposal_id_used_for_final_geometry") is False,
        "proposal_coordinate_average_not_used": report.get("proposal_coordinate_average_used_for_final_geometry") is False,
        "test_closed": report.get("test_set_used") is False,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout_report = _load(args.heldout)
    validation_report = _load(args.validation)
    heldout = _domain(
        heldout_report,
        _load(args.heldout_source_coverage),
        _load(args.heldout_treatment_coverage),
    )
    validation = _domain(
        validation_report,
        _load(args.val_source_coverage),
        _load(args.val_treatment_coverage),
    )
    checks = {
        "gate0_contract_passed": contract.get("passed") is True,
        "heldout_clip_256_passed": heldout["passed"],
        "validation_256_passed": validation["passed"],
        "all_reports_test_closed": all(
            report.get("test_set_used") is False
            for report in (contract, heldout_report, validation_report)
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V17 iterative multi-scale continuous geometry fixed gate",
        "gate_thresholds": {
            "tp_050_gain_min": 3,
            "tp_075_gain_min": 6,
            "correct_wrong_tp_min": {"0.50": 5, "0.75": 8},
            "correct_zero_tp_min": {"0.50": 3, "0.75": 3},
            "prediction_count_exact": True,
            "semantic_duplicate_max": 0.02,
        },
        "heldout_clip_256": heldout,
        "validation_256": validation,
        "checks": checks,
        "passed": passed,
        "long_training_authorized": False,
        "full_validation_authorized": False,
        "test_set_used": False,
        "checkpoint_selection_performed": False,
        "decision": "v17_fixed_gate_pass_stop_for_review" if passed else "v17_fixed_gate_fail_stop_for_review",
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
