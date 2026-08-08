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
        description="Summarize the V7 joint 0-to-10k safety gate."
    )
    parser.add_argument("--report", action="append", type=_parse_spec, default=[])
    parser.add_argument("--preflight-contract", required=True)
    parser.add_argument("--final-gradient-contract", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(iteration: int, path: Path) -> dict[str, Any]:
    report = _load(path)
    methods = report["methods"]
    refined = methods.get("four_slot_refined") or methods[
        "four_slot_global_unique"
    ]
    unrefined = methods["four_slot_global_unique"]
    refined_050 = refined["0.50"]
    refined_075 = refined["0.75"]
    unrefined_050 = unrefined["0.50"]
    unrefined_075 = unrefined["0.75"]
    slot = report["four_slot_diagnostics"]
    cardinality = slot.get("cardinality", {})
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
        "oracle_recall_050": float(
            report["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        ),
        "oracle_recall_075": float(
            report["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        ),
        "cardinality_exact": float(cardinality.get("exact_fraction", 0.0)),
        "cardinality_mae": float(
            cardinality.get("mean_absolute_error", 4.0)
        ),
        "semantic_duplicate": float(
            slot["semantic_duplicate_cluster_fraction"]
        ),
        "close_pair_fraction_20px": float(
            refined_050["selected_curve_diversity"][
                "close_pair_fraction_below_20px"
            ]
        ),
        "repair_fraction": float(slot["global_assignment_repair_fraction"]),
        "route_entropy": float(slot["mean_route_entropy"]),
        "refinement_gain_050": float(
            refined_050["f1"] - unrefined_050["f1"]
        ),
        "refinement_gain_075": float(
            refined_075["f1"] - unrefined_075["f1"]
        ),
    }


def summarize(
    specs: list[tuple[int, Path]],
    preflight: dict[str, Any],
    final_gradient: dict[str, Any],
) -> dict[str, Any]:
    if not specs:
        raise ValueError("at least one joint trajectory report is required")
    trajectory = [_row(iteration, path) for iteration, path in sorted(specs)]
    first = trajectory[0]
    final = trajectory[-1]
    best_050 = max(float(row["f1_050"]) for row in trajectory)
    route_cosine = float(
        final_gradient.get("route_geometry_selection_cosine", 0.0)
    )
    route_ratio = float(
        final_gradient.get(
            "route_geometry_to_selection_norm_ratio",
            float("inf"),
        )
    )
    safety_checks = {
        "preflight_contract": bool(preflight.get("passed", False)),
        "final_gradient_contract": bool(final_gradient.get("passed", False)),
        "preflight_iteration_zero": int(preflight.get("iteration", -1)) == 0,
        "final_gradient_iteration_10k": (
            int(final_gradient.get("iteration", -1)) == 10000
        ),
        "trajectory_iterations_exact": (
            int(first["iteration"]) == 5000
            and int(final["iteration"]) == 10000
        ),
        "hard_assignment_at_final": bool(
            final_gradient.get("checks", {}).get("hard_slot_assignment", False)
        ),
        "oracle_050_floor": float(final["oracle_recall_050"]) >= 0.90,
        "oracle_075_floor": float(final["oracle_recall_075"]) >= 0.70,
        "oracle_050_no_collapse": (
            float(final["oracle_recall_050"])
            >= float(first["oracle_recall_050"]) - 0.02
        ),
        "oracle_075_no_collapse": (
            float(final["oracle_recall_075"])
            >= float(first["oracle_recall_075"]) - 0.03
        ),
        "final_near_best": float(final["f1_050"]) >= best_050 - 0.03,
        "mean_selected_valid": 2.5 <= float(final["mean_selected"]) <= 3.8,
        "cardinality_exact": float(final["cardinality_exact"]) >= 0.40,
        "semantic_duplicate": float(final["semantic_duplicate"]) <= 0.05,
        "close_pair_fraction": (
            float(final["close_pair_fraction_20px"]) <= 0.05
        ),
        "global_repair": float(final["repair_fraction"]) <= 0.05,
        "refinement_050_bounded": (
            float(final["refinement_gain_050"]) >= -0.03
        ),
        "route_gradient_cosine_bounded": route_cosine >= -0.35,
        "route_geometry_norm_bounded": route_ratio <= 2.0,
    }
    learning_checks = {
        "f1_050_minimum": float(final["f1_050"]) >= 0.55,
        "f1_075_minimum": float(final["f1_075"]) >= 0.35,
        "tail_not_regressing": (
            float(final["f1_050"]) >= float(first["f1_050"]) - 0.01
        ),
    }
    safety_pass = all(safety_checks.values())
    learning_pass = all(learning_checks.values())
    continuation_authorized = bool(safety_pass and learning_pass)
    return {
        "experiment": "V7 hard-ownership joint four-slot 0-to-10k gate",
        "diagnostic_only": True,
        "trajectory": trajectory,
        "preflight_contract": preflight,
        "final_gradient_contract": final_gradient,
        "route_gradient_diagnostics": {
            "geometry_selection_cosine": route_cosine,
            "geometry_to_selection_norm_ratio": route_ratio,
        },
        "safety_thresholds": {
            "oracle_recall_050_min": 0.90,
            "oracle_recall_075_min": 0.70,
            "mean_selected_range": [2.5, 3.8],
            "cardinality_exact_min": 0.40,
            "semantic_duplicate_max": 0.05,
            "close_pair_fraction_max": 0.05,
            "global_repair_max": 0.05,
            "route_gradient_cosine_min": -0.35,
            "route_geometry_to_selection_norm_ratio_max": 2.0,
        },
        "learning_thresholds": {
            "f1_050_min": 0.55,
            "f1_075_min": 0.35,
            "maximum_5k_to_10k_f1_050_regression": 0.01,
        },
        "safety_checks": safety_checks,
        "learning_checks": learning_checks,
        "safety_pass": safety_pass,
        "learning_pass": learning_pass,
        "verdict": "pass" if continuation_authorized else "fail",
        "next_action": (
            "resume_same_joint_run_10k_to25k"
            if continuation_authorized
            else "stop_and_audit_joint_training"
        ),
        "joint_25k_continuation_authorized": continuation_authorized,
        "full_validation_authorized": False,
        "long_run_authorized": False,
        "test_split_closed": True,
    }


def main() -> None:
    args = _parse_args()
    payload = summarize(
        args.report,
        _load(args.preflight_contract),
        _load(args.final_gradient_contract),
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
