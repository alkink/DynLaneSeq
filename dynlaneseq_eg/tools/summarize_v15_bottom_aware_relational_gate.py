from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the immutable V15 two-domain endpoint gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--gradient-population", required=True)
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
            threshold: int(
                capacity[threshold]["all_candidate_oracle"]["hits"]
            )
            for threshold in THRESHOLDS
        },
    }


def _threshold(
    report: dict[str, Any], policy: str, mode: str, threshold: str
) -> dict[str, Any]:
    return report["metrics"][policy][mode]["thresholds"][threshold]


def _delta(
    report: dict[str, Any], comparison: str, mode: str, threshold: str
) -> dict[str, Any]:
    return report["deltas"][comparison][mode][threshold]


def _domain(
    report: dict[str, Any],
    source_coverage_payload: dict[str, Any],
    treatment_coverage_payload: dict[str, Any],
) -> dict[str, Any]:
    source_coverage = _coverage(source_coverage_payload)
    treatment_coverage = _coverage(treatment_coverage_payload)
    v15_050 = _delta(report, "v15_minus_v7", "writer_valid", "0.50")
    v15_075 = _delta(report, "v15_minus_v7", "writer_valid", "0.75")
    p2_050 = _delta(
        report,
        "correct_minus_cross_clip_wrong_p2",
        "writer_valid",
        "0.50",
    )
    identity_050 = _delta(
        report, "correct_minus_identity_graph", "writer_valid", "0.50"
    )
    shuffled_050 = _delta(
        report,
        "correct_minus_geometry_shuffled_graph",
        "writer_valid",
        "0.50",
    )
    context_050 = _delta(
        report,
        "correct_minus_no_proposal_context",
        "writer_valid",
        "0.50",
    )
    context_075 = _delta(
        report,
        "correct_minus_no_proposal_context",
        "writer_valid",
        "0.75",
    )
    source_neural = _threshold(
        report, "v7_anchor", "neural_active", "0.50"
    )
    treatment_neural = _threshold(
        report, "v15_final", "neural_active", "0.50"
    )
    source_writer = _threshold(
        report, "v7_anchor", "writer_valid", "0.50"
    )
    treatment_writer = _threshold(
        report, "v15_final", "writer_valid", "0.50"
    )
    source_invalid = int(source_neural["predictions"]) - int(
        source_writer["predictions"]
    )
    treatment_invalid = int(treatment_neural["predictions"]) - int(
        treatment_writer["predictions"]
    )
    oracle_equal = all(
        source_coverage["proposal_oracle_hits"][threshold]
        == treatment_coverage["proposal_oracle_hits"][threshold]
        for threshold in THRESHOLDS
    )
    values = {
        "delta_tp_050": int(v15_050["delta_tp"]),
        "delta_tp_075": int(v15_075["delta_tp"]),
        "delta_f1_points_050": float(v15_050["delta_f1_points"]),
        "delta_f1_points_075": float(v15_075["delta_f1_points"]),
        "neural_prediction_count_drift": int(
            treatment_neural["predictions"]
        )
        - int(source_neural["predictions"]),
        "writer_prediction_count_drift": int(
            treatment_writer["predictions"]
        )
        - int(source_writer["predictions"]),
        "writer_invalid_increase": treatment_invalid - source_invalid,
        "correct_minus_wrong_p2_tp_050": int(p2_050["delta_tp"]),
        "correct_minus_identity_graph_tp_050": int(
            identity_050["delta_tp"]
        ),
        "correct_minus_shuffled_graph_tp_050": int(
            shuffled_050["delta_tp"]
        ),
        "correct_minus_no_context_tp_050": int(context_050["delta_tp"]),
        "correct_minus_no_context_tp_075": int(context_075["delta_tp"]),
        "semantic_duplicate_fraction": treatment_coverage[
            "semantic_duplicate_fraction"
        ],
        "proposal_oracle_equal": oracle_equal,
    }
    checks = {
        "tp_050_at_least_v7_plus_3": values["delta_tp_050"] >= 3,
        "f1_050_strictly_improves": values["delta_f1_points_050"] > 0.0,
        "f1_075_nonregression": values["delta_f1_points_075"] >= 0.0,
        "neural_prediction_count_exact": (
            values["neural_prediction_count_drift"] == 0
        ),
        "writer_prediction_count_exact": (
            values["writer_prediction_count_drift"] == 0
        ),
        "writer_invalid_increase_zero": (
            values["writer_invalid_increase"] == 0
        ),
        "correct_p2_at_least_wrong_plus_2_tp_050": (
            values["correct_minus_wrong_p2_tp_050"] >= 2
        ),
        "correct_graph_at_least_identity_plus_2_tp_050": (
            values["correct_minus_identity_graph_tp_050"] >= 2
        ),
        "correct_graph_at_least_shuffled_plus_2_tp_050": (
            values["correct_minus_shuffled_graph_tp_050"] >= 2
        ),
        "proposal_context_nonregression_050": (
            values["correct_minus_no_context_tp_050"] >= 0
        ),
        "proposal_context_nonregression_075": (
            values["correct_minus_no_context_tp_075"] >= 0
        ),
        "semantic_duplicate_at_most_2pct": (
            values["semantic_duplicate_fraction"] <= 0.02
        ),
        "proposal_oracle_unchanged": oracle_equal,
        "cross_clip_runtime_exact": (
            int(report.get("cross_clip_runtime_same_image", -1)) == 0
            and int(report.get("cross_clip_runtime_same_clip", -1)) == 0
        ),
        "no_hard_cluster_or_prototype": (
            report.get("hard_cluster_or_prototype_used") is False
        ),
        "no_proposal_id_supervision": (
            report.get("proposal_id_supervision_used") is False
        ),
        "fixed_activity_score_source": (
            report.get("activity_score_route_source") == "exact_v7"
        ),
        "test_closed": report.get("test_set_used") is False,
    }
    return {
        "metrics": values,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    gradient = _load(args.gradient_population)
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
        "endpoint_gradient_population_complete": gradient.get("passed") is True,
        "heldout_clip_256_passed": heldout["passed"],
        "validation_256_passed": validation["passed"],
        "all_reports_test_closed": all(
            report.get("test_set_used") is False
            for report in (
                contract,
                gradient,
                heldout_report,
                validation_report,
            )
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V15 bottom-aware relational slot geometry fixed gate",
        "gate_thresholds": {
            "tp_050_gain_min": 3,
            "f1_050_gain_strict": True,
            "f1_075_nonregression": True,
            "correct_wrong_p2_tp_050_min": 2,
            "correct_each_graph_control_tp_050_min": 2,
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
        "decision": (
            "v15_fixed_gate_pass_stop_for_review"
            if passed
            else "v15_fixed_gate_fail_stop_for_review"
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
