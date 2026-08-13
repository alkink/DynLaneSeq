from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply V14 Stage-B bridge or full-validation gate."
    )
    parser.add_argument("--mode", choices=("bridge", "full_validation"), required=True)
    parser.add_argument("--contract", default="")
    parser.add_argument("--heldout", default="")
    parser.add_argument("--validation", default="")
    parser.add_argument("--heldout-source-coverage", default="")
    parser.add_argument("--heldout-treatment-coverage", default="")
    parser.add_argument("--val-source-coverage", default="")
    parser.add_argument("--val-treatment-coverage", default="")
    parser.add_argument("--source-eval", default="")
    parser.add_argument("--treatment-eval", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _coverage(report: dict[str, Any]) -> dict[str, Any]:
    slot = report["four_slot_diagnostics"]
    return {
        "semantic_duplicate": float(slot["semantic_duplicate_cluster_fraction"]),
        "proposal_oracle": {
            threshold: float(
                report["capacity"][threshold]["all_candidate_oracle"]["recall"]
            )
            for threshold in ("0.50", "0.75")
        },
    }


def _domain(
    report: dict[str, Any],
    source_coverage: dict[str, Any],
    treatment_coverage: dict[str, Any],
) -> dict[str, Any]:
    delta = report["deltas"]["v14_minus_v7"]
    causal = report["deltas"]["correct_minus_cross_clip_wrong_p2"]
    metrics = report["metrics"]
    source_neural = metrics["v7_anchor"]["neural_active"]["thresholds"]["0.50"]
    treatment_neural = metrics["v14_final"]["neural_active"]["thresholds"]["0.50"]
    source_writer = metrics["v7_anchor"]["writer_valid"]["thresholds"]["0.50"]
    treatment_writer = metrics["v14_final"]["writer_valid"]["thresholds"]["0.50"]
    source_invalid = int(source_neural["predictions"]) - int(source_writer["predictions"])
    treatment_invalid = int(treatment_neural["predictions"]) - int(treatment_writer["predictions"])
    source_cov = _coverage(source_coverage)
    treatment_cov = _coverage(treatment_coverage)
    oracle_equal = all(
        abs(source_cov["proposal_oracle"][threshold] - treatment_cov["proposal_oracle"][threshold]) <= 1.0e-12
        for threshold in ("0.50", "0.75")
    )
    row050 = delta["writer_valid"]["0.50"]
    row075 = delta["writer_valid"]["0.75"]
    causal050 = causal["writer_valid"]["0.50"]
    causal075 = causal["writer_valid"]["0.75"]
    values = {
        "delta_tp_050": int(row050["delta_tp"]),
        "delta_tp_075": int(row075["delta_tp"]),
        "delta_f1_points_050": float(row050["delta_f1_points"]),
        "delta_f1_points_075": float(row075["delta_f1_points"]),
        "neural_prediction_count_drift": int(treatment_neural["predictions"]) - int(source_neural["predictions"]),
        "writer_invalid_increase": treatment_invalid - source_invalid,
        "correct_wrong_p2_delta_tp_050": int(causal050["delta_tp"]),
        "correct_wrong_p2_delta_tp_075": int(causal075["delta_tp"]),
        "semantic_duplicate_fraction": treatment_cov["semantic_duplicate"],
        "proposal_oracle_equal": oracle_equal,
    }
    checks = {
        "tp_050_at_least_v7_plus_3": values["delta_tp_050"] >= 3,
        "f1_050_strictly_improves": values["delta_f1_points_050"] > 0.0,
        "f1_075_nonregression": values["delta_f1_points_075"] >= 0.0,
        "activity_prediction_count_exact": values["neural_prediction_count_drift"] == 0,
        "writer_invalid_increase_zero": values["writer_invalid_increase"] <= 0,
        "correct_p2_advantage_positive_050": values["correct_wrong_p2_delta_tp_050"] > 0,
        "correct_p2_advantage_positive_075": values["correct_wrong_p2_delta_tp_075"] > 0,
        "semantic_duplicate_at_most_2pct": values["semantic_duplicate_fraction"] <= 0.02,
        "proposal_oracle_unchanged": oracle_equal,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def _eval_threshold(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    # evaluate_culane stores threshold rows under either `results` or
    # `thresholds`, depending on the historical report revision.
    for key in ("results", "thresholds", "metrics"):
        value = report.get(key)
        if isinstance(value, dict):
            for key, row in value.items():
                try:
                    matches = abs(float(key) - float(threshold)) <= 1.0e-12
                except (TypeError, ValueError):
                    matches = False
                if matches:
                    return {
                        "tp": int(row.get("tp", row.get("TP", 0))),
                        "fp": int(row.get("fp", row.get("FP", 0))),
                        "fn": int(row.get("fn", row.get("FN", 0))),
                        "precision": float(
                            row.get("precision", row.get("Precision", 0.0))
                        ),
                        "recall": float(
                            row.get("recall", row.get("Recall", 0.0))
                        ),
                        "f1": float(row.get("f1", row.get("F1", 0.0))),
                    }
    raise KeyError(f"cannot locate threshold {threshold} in evaluation report")


def main() -> None:
    args = parse_args()
    if args.mode == "bridge":
        contract = _load(args.contract)
        heldout_report = _load(args.heldout)
        val_report = _load(args.validation)
        heldout = _domain(
            heldout_report,
            _load(args.heldout_source_coverage),
            _load(args.heldout_treatment_coverage),
        )
        validation = _domain(
            val_report,
            _load(args.val_source_coverage),
            _load(args.val_treatment_coverage),
        )
        checks = {
            "stage_b_zero_step_contract_passed": contract.get("passed") is True,
            "heldout_clip_256_passed": heldout["passed"],
            "validation_256_passed": validation["passed"],
            "test_closed": all(
                report.get("test_set_used") is False
                for report in (contract, heldout_report, val_report)
            ),
        }
        passed = all(checks.values())
        report = {
            "experiment": "V14 Stage-B bridge gate",
            "heldout_clip_256": heldout,
            "validation_256": validation,
            "checks": checks,
            "passed": passed,
            "full_validation_authorized": passed,
            "long_training_authorized": False,
            "test_set_used": False,
            "decision": "authorize_single_fixed_full_validation" if passed else "stop_v14_stage_b",
        }
    else:
        source = _load(args.source_eval)
        treatment = _load(args.treatment_eval)
        source050 = _eval_threshold(source, "0.50")
        source075 = _eval_threshold(source, "0.75")
        treatment050 = _eval_threshold(treatment, "0.50")
        treatment075 = _eval_threshold(treatment, "0.75")
        delta050 = 100.0 * (float(treatment050["f1"]) - float(source050["f1"]))
        delta075 = 100.0 * (float(treatment075["f1"]) - float(source075["f1"]))
        checks = {
            "f1_050_at_least_v7_plus_0p50": delta050 >= 0.50,
            "f1_075_nonregression": delta075 >= 0.0,
            "test_closed": source.get("split") == "val" and treatment.get("split") == "val",
        }
        passed = all(checks.values())
        report = {
            "experiment": "V14 Stage-B single fixed full-validation gate",
            "source": {"0.50": source050, "0.75": source075},
            "treatment": {"0.50": treatment050, "0.75": treatment075},
            "delta_f1_points": {"0.50": delta050, "0.75": delta075},
            "checks": checks,
            "passed": passed,
            "long_training_authorized": False,
            "test_set_used": False,
            "decision": "v14_complete_pass_stop_for_sol" if passed else "v14_complete_fail_stop_for_sol",
        }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
