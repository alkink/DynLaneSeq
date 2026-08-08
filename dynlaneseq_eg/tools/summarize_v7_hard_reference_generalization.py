from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_spec(value: str) -> tuple[int, Path]:
    iteration, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("report must be ITERATION=PATH")
    return int(iteration), Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V7 hard-reference full-train gate."
    )
    parser.add_argument(
        "--report",
        action="append",
        type=_parse_spec,
        default=[],
    )
    parser.add_argument("--gradient-contract", required=True)
    parser.add_argument(
        "--continuation-complete",
        action="store_true",
        help=(
            "Mark the pre-authorized 2k frozen-head continuation complete. "
            "A remaining conditional signal then stops this arm instead of "
            "requesting another extension."
        ),
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(iteration: int, path: Path) -> dict[str, Any]:
    report = _load(path)
    refined = report["methods"]["four_slot_refined"]
    unrefined = report["methods"]["four_slot_global_unique"]
    slot = report["four_slot_diagnostics"]
    cardinality = slot["cardinality"]
    refined_050 = refined["0.50"]
    refined_075 = refined["0.75"]
    unrefined_050 = unrefined["0.50"]
    unrefined_075 = unrefined["0.75"]
    return {
        "iteration": int(iteration),
        "report": str(path),
        "f1_050": float(refined_050["f1"]),
        "f1_075": float(refined_075["f1"]),
        "precision_050": float(refined_050["precision"]),
        "recall_050": float(refined_050["recall"]),
        "tp_050": int(refined_050["tp"]),
        "fp_050": int(refined_050["fp"]),
        "fn_050": int(refined_050["fn"]),
        "mean_selected": float(refined_050["mean_selected_per_image"]),
        "unrefined_f1_050": float(unrefined_050["f1"]),
        "unrefined_f1_075": float(unrefined_075["f1"]),
        "refinement_gain_050": float(
            refined_050["f1"] - unrefined_050["f1"]
        ),
        "refinement_gain_075": float(
            refined_075["f1"] - unrefined_075["f1"]
        ),
        "cardinality_exact": float(cardinality["exact_fraction"]),
        "cardinality_under": float(cardinality["under_fraction"]),
        "cardinality_over": float(cardinality["over_fraction"]),
        "cardinality_mae": float(cardinality["mean_absolute_error"]),
        "semantic_duplicate": float(
            slot["semantic_duplicate_cluster_fraction"]
        ),
        "close_pair_fraction_20px": float(
            refined_050["selected_curve_diversity"][
                "close_pair_fraction_below_20px"
            ]
        ),
        "repair_fraction": float(
            slot["global_assignment_repair_fraction"]
        ),
        "route_entropy": float(slot["mean_route_entropy"]),
        "oracle_recall_050": float(
            report["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        ),
        "oracle_recall_075": float(
            report["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        ),
    }


def summarize(
    specs: list[tuple[int, Path]],
    gradient_contract: dict[str, Any],
    *,
    continuation_complete: bool = False,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one generalization report is required")
    trajectory = [_row(iteration, path) for iteration, path in sorted(specs)]
    final = trajectory[-1]
    best_050 = max(float(row["f1_050"]) for row in trajectory)
    oracle_050 = float(trajectory[0]["oracle_recall_050"])
    oracle_075 = float(trajectory[0]["oracle_recall_075"])
    safety_checks = {
        "oracle_050_unchanged": all(
            abs(float(row["oracle_recall_050"]) - oracle_050) <= 1.0e-9
            for row in trajectory
        ),
        "oracle_075_unchanged": all(
            abs(float(row["oracle_recall_075"]) - oracle_075) <= 1.0e-9
            for row in trajectory
        ),
        "final_near_best": float(final["f1_050"]) >= best_050 - 0.01,
        "mean_selected_valid": 2.8 <= float(final["mean_selected"]) <= 3.6,
        "cardinality_exact": float(final["cardinality_exact"]) >= 0.65,
        "semantic_duplicate": float(final["semantic_duplicate"]) <= 0.02,
        "close_pair_fraction": (
            float(final["close_pair_fraction_20px"]) <= 0.02
        ),
        "global_repair": float(final["repair_fraction"]) <= 0.02,
        "refinement_050_non_destructive": (
            float(final["refinement_gain_050"]) >= -0.005
        ),
        "refinement_075_non_destructive": (
            float(final["refinement_gain_075"]) >= 0.0
        ),
        "gradient_contract": bool(gradient_contract.get("passed", False)),
    }
    safety_pass = all(safety_checks.values())
    strong_signal = bool(
        safety_pass
        and float(final["f1_050"]) >= 0.80
        and float(final["f1_075"]) >= 0.58
    )
    conditional_signal = bool(
        safety_pass
        and float(final["f1_050"]) >= 0.75
        and float(final["f1_075"]) >= 0.55
    )
    if strong_signal:
        verdict = "pass"
        next_action = "run_one_full_validation"
    elif conditional_signal:
        if continuation_complete:
            verdict = "frozen_head_plateau"
            next_action = "stop_frozen_head_and_test_joint_training_hypothesis"
        else:
            verdict = "conditional"
            next_action = "extend_same_frozen_head_by_2k_without_test"
    else:
        verdict = "fail"
        next_action = "stop_and_audit_generalization_objective"
    return {
        "experiment": "V7 hard-reference hard-ownership full-train gate",
        "diagnostic_only": True,
        "trajectory": trajectory,
        "gradient_contract": gradient_contract,
        "safety_checks": safety_checks,
        "safety_pass": safety_pass,
        "strong_signal_thresholds": {
            "f1_050_min": 0.80,
            "f1_075_min": 0.58,
        },
        "conditional_signal_thresholds": {
            "f1_050_min": 0.75,
            "f1_075_min": 0.55,
        },
        "continuation_complete": bool(continuation_complete),
        "verdict": verdict,
        "next_action": next_action,
        "full_validation_authorized": strong_signal,
        "long_run_authorized": False,
        "joint_training_status": "not_tested_by_frozen_head_gate",
        "joint_training_ruled_out": False,
        "test_split_closed": True,
    }


def main() -> None:
    args = _parse_args()
    gradient = _load(args.gradient_contract)
    payload = summarize(
        args.report,
        gradient,
        continuation_complete=bool(args.continuation_complete),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
