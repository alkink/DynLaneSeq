from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V13 official generalization gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _domain(report: dict[str, Any]) -> dict[str, Any]:
    writer = report["deltas"]["v13_minus_v7"]["writer_valid"]
    neural = report["deltas"]["v13_minus_v7"]["neural_active"]
    p2 = report["deltas"]["correct_minus_wrong_p2"]["writer_valid"]
    zero = report["deltas"]["correct_minus_zero_p2"]["writer_valid"]
    v7_050 = report["metrics"]["v7_anchor"]["writer_valid"]["thresholds"][
        "0.50"
    ]
    v13_050 = report["metrics"]["v13_final"]["writer_valid"]["thresholds"][
        "0.50"
    ]
    v7_neural_050 = report["metrics"]["v7_anchor"]["neural_active"][
        "thresholds"
    ]["0.50"]
    v13_neural_050 = report["metrics"]["v13_final"]["neural_active"][
        "thresholds"
    ]["0.50"]
    images = int(report["images"])
    metrics = {
        "writer_delta_tp_050": int(writer["0.50"]["delta_tp"]),
        "writer_delta_tp_075": int(writer["0.75"]["delta_tp"]),
        "writer_delta_f1_points_050": float(
            writer["0.50"]["delta_f1_points"]
        ),
        "writer_delta_f1_points_075": float(
            writer["0.75"]["delta_f1_points"]
        ),
        "same_active_delta_tp_050": int(neural["0.50"]["delta_tp"]),
        "same_active_delta_tp_075": int(neural["0.75"]["delta_tp"]),
        "same_active_delta_f1_points_050": float(
            neural["0.50"]["delta_f1_points"]
        ),
        "same_active_delta_f1_points_075": float(
            neural["0.75"]["delta_f1_points"]
        ),
        "correct_wrong_p2_delta_tp_050": int(p2["0.50"]["delta_tp"]),
        "correct_wrong_p2_delta_tp_075": int(p2["0.75"]["delta_tp"]),
        "correct_zero_p2_delta_tp_050": int(zero["0.50"]["delta_tp"]),
        "neural_active_prediction_count_drift": int(
            v13_neural_050["predictions"]
        )
        - int(v7_neural_050["predictions"]),
        "writer_prediction_count_drift": int(v13_050["predictions"])
        - int(v7_050["predictions"]),
        "writer_count_drift_limit": int(math.ceil(0.05 * images)),
    }
    checks = {
        # Five TP on a 256-image bridge is a real ~0.6-point signal, not the
        # one-TP noise that repeatedly misled earlier selector sidecars.
        "writer_v13_gains_at_least_5_tp_050": (
            metrics["writer_delta_tp_050"] >= 5
        ),
        "writer_v13_gains_at_least_5_tp_075": (
            metrics["writer_delta_tp_075"] >= 5
        ),
        "writer_v13_f1_nonregression_050": (
            metrics["writer_delta_f1_points_050"] >= 0.0
        ),
        "writer_v13_f1_nonregression_075": (
            metrics["writer_delta_f1_points_075"] >= 0.0
        ),
        # This is the count-controlled geometry result.  It prevents a
        # learned range from manufacturing a PASS by merely changing which
        # active slots survive the writer.
        "same_active_v13_gains_at_least_5_tp_050": (
            metrics["same_active_delta_tp_050"] >= 5
        ),
        "same_active_v13_gains_at_least_5_tp_075": (
            metrics["same_active_delta_tp_075"] >= 5
        ),
        "same_active_v13_f1_nonregression_050": (
            metrics["same_active_delta_f1_points_050"] >= 0.0
        ),
        "same_active_v13_f1_nonregression_075": (
            metrics["same_active_delta_f1_points_075"] >= 0.0
        ),
        "correct_p2_beats_wrong_by_5_tp_050": (
            metrics["correct_wrong_p2_delta_tp_050"] >= 5
        ),
        "correct_p2_beats_wrong_by_5_tp_075": (
            metrics["correct_wrong_p2_delta_tp_075"] >= 5
        ),
        "correct_p2_beats_zero_by_5_tp_050": (
            metrics["correct_zero_p2_delta_tp_050"] >= 5
        ),
        "neural_active_count_exact": (
            metrics["neural_active_prediction_count_drift"] == 0
        ),
        # Writer-validity may legitimately change when range geometry is
        # repaired.  Keep a broad anti-pathology bound instead of repeating
        # V11's scientifically misleading raw-count FAIL.
        "writer_count_drift_bounded": (
            abs(metrics["writer_prediction_count_drift"])
            <= metrics["writer_count_drift_limit"]
        ),
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout_report = _load(args.heldout)
    validation_report = _load(args.validation)
    heldout = _domain(heldout_report)
    validation = _domain(validation_report)
    checks = {
        "zero_step_contract_passed": contract.get("passed") is True,
        "heldout_clip_gate_passed": heldout["passed"],
        "validation_gate_passed": validation["passed"],
        "hard_proposal_id_absent": all(
            report.get("hard_proposal_id_produces_final_geometry") is False
            for report in (contract, heldout_report, validation_report)
        ),
        "test_closed": all(
            report.get("test_set_used") is False
            for report in (contract, heldout_report, validation_report)
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V13 visual-precision bridge gate",
        "heldout_clip": heldout,
        "validation": validation,
        "checks": checks,
        "passed": passed,
        "full_validation_authorized": passed,
        "long_training_authorized": False,
        "test_set_used": False,
        "decision": (
            "v13_pass_authorize_single_full_validation_checkpoint"
            if passed
            else "v13_fail_stop_visual_precision_family"
        ),
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
