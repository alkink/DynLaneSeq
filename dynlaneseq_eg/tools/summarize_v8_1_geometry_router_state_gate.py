from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed-64 or paired V8.1 router-state gate."
    )
    parser.add_argument("--mode", choices=("fixed64", "generalization"), required=True)
    parser.add_argument("--population-contract", required=True)
    parser.add_argument("--source-route", required=True)
    parser.add_argument("--source-coverage", required=True)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--control-coverage", required=True)
    parser.add_argument("--treatment-route", required=True)
    parser.add_argument("--treatment-coverage", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _route_summary(report: dict[str, Any]) -> dict[str, Any]:
    v7 = report["v7"]
    decomposition = v7["support_mass_decomposition"]
    total = int(decomposition["total_assigned_gt"])
    correct = int(decomposition["counts"]["CORRECT_ID"])
    policies = v7["reference_policies"]
    stages: dict[str, Any] = {}
    for stage in ("reference", "refined"):
        stages[stage] = {
            threshold: dict(
                policies["current_hard"][stage]["fixed_neural_active"][threshold]
            )
            for threshold in THRESHOLDS
        }
    target_hard = {
        stage: {
            threshold: dict(
                policies["target_hard"][stage]["fixed_neural_active"][threshold]
            )
            for threshold in THRESHOLDS
        }
        for stage in ("reference", "refined")
    }
    images = int(v7["images"])
    predictions = int(stages["refined"]["0.50"]["predictions"])
    return {
        "images": images,
        "assigned_gt": total,
        "target_id_top1_fraction": float(correct) / float(max(total, 1)),
        "mean_target_support_mass": float(
            decomposition["predicted_support_mass"]["mean"]
        ),
        "fraction_support_mass_ge_0p5": float(
            decomposition["fraction_support_mass_ge_0p5"]
        ),
        "fraction_support_mass_ge_0p8": float(
            decomposition["fraction_support_mass_ge_0p8"]
        ),
        "active_predictions_per_image": float(predictions) / float(max(images, 1)),
        "current_hard": stages,
        "target_hard": target_hard,
    }


def _coverage_summary(report: dict[str, Any]) -> dict[str, Any]:
    method = report["methods"]["four_slot_refined"]
    slot = report["four_slot_diagnostics"]
    return {
        "f1": {threshold: float(method[threshold]["f1"]) for threshold in THRESHOLDS},
        "tp": {threshold: int(method[threshold]["tp"]) for threshold in THRESHOLDS},
        "mean_selected": float(method["0.50"]["mean_selected_per_image"]),
        "cardinality_exact": float(slot["cardinality"]["exact_fraction"]),
        "semantic_duplicate": float(slot["semantic_duplicate_cluster_fraction"]),
        "proposal_oracle_recall": {
            threshold: float(
                report["capacity"][threshold]["all_candidate_oracle"]["recall"]
            )
            for threshold in THRESHOLDS
        },
    }


def _gap_closure(source: dict[str, Any], endpoint: dict[str, Any], threshold: str) -> float:
    source_tp = int(source["current_hard"]["reference"][threshold]["tp"])
    oracle_tp = int(source["target_hard"]["reference"][threshold]["tp"])
    endpoint_tp = int(endpoint["current_hard"]["reference"][threshold]["tp"])
    gap = oracle_tp - source_tp
    if gap <= 0:
        return 1.0 if endpoint_tp >= source_tp else 0.0
    return float(endpoint_tp - source_tp) / float(gap)


def _oracle_unchanged(source: dict[str, Any], endpoint: dict[str, Any]) -> bool:
    return all(
        abs(
            float(source["proposal_oracle_recall"][threshold])
            - float(endpoint["proposal_oracle_recall"][threshold])
        )
        <= 1.0e-12
        for threshold in THRESHOLDS
    )


def main() -> None:
    args = parse_args()
    population = _load(args.population_contract)
    source_route = _route_summary(_load(args.source_route))
    source_coverage = _coverage_summary(_load(args.source_coverage))
    control_route = _route_summary(_load(args.control_route))
    control_coverage = _coverage_summary(_load(args.control_coverage))
    treatment_route = _route_summary(_load(args.treatment_route))
    treatment_coverage = _coverage_summary(_load(args.treatment_coverage))

    payload: dict[str, Any] = {
        "experiment": "V8.1 geometry-to-global-router-state causal gate",
        "mode": args.mode,
        "iteration": int(args.iteration),
        "test_set_used": False,
        "population_contract_passed": bool(population.get("passed", False)),
        "source": {"route": source_route, "coverage": source_coverage},
        "control": {"route": control_route, "coverage": control_coverage},
        "treatment": {"route": treatment_route, "coverage": treatment_coverage},
    }

    if args.mode == "fixed64":
        gap_closure = {
            threshold: _gap_closure(source_route, treatment_route, threshold)
            for threshold in THRESHOLDS
        }
        checks = {
            "population_contract": bool(population.get("passed", False)),
            "mean_target_support_mass_ge_0p95": (
                treatment_route["mean_target_support_mass"] >= 0.95
            ),
            "target_id_top1_ge_0p95": (
                treatment_route["target_id_top1_fraction"] >= 0.95
            ),
            "gap_closure_050_ge_0p80": gap_closure["0.50"] >= 0.80,
            "gap_closure_075_ge_0p70": gap_closure["0.75"] >= 0.70,
            "cardinality_exact_ge_0p95": (
                treatment_coverage["cardinality_exact"] >= 0.95
            ),
            "semantic_duplicate_le_0p02": (
                treatment_coverage["semantic_duplicate"] <= 0.02
            ),
            "proposal_oracle_unchanged": _oracle_unchanged(
                source_coverage,
                treatment_coverage,
            ),
        }
        payload["gap_closure"] = gap_closure
        payload["thresholds"] = {
            "mean_target_support_mass": 0.95,
            "target_id_top1_fraction": 0.95,
            "gap_closure_050": 0.80,
            "gap_closure_075": 0.70,
            "cardinality_exact": 0.95,
            "semantic_duplicate_max": 0.02,
        }
    else:
        route_delta = {
            "mean_target_support_mass": (
                treatment_route["mean_target_support_mass"]
                - control_route["mean_target_support_mass"]
            ),
            "target_id_top1_fraction": (
                treatment_route["target_id_top1_fraction"]
                - control_route["target_id_top1_fraction"]
            ),
            "reference_tp_050": (
                int(treatment_route["current_hard"]["reference"]["0.50"]["tp"])
                - int(control_route["current_hard"]["reference"]["0.50"]["tp"])
            ),
            "reference_tp_075": (
                int(treatment_route["current_hard"]["reference"]["0.75"]["tp"])
                - int(control_route["current_hard"]["reference"]["0.75"]["tp"])
            ),
            "refined_tp_050": (
                int(treatment_route["current_hard"]["refined"]["0.50"]["tp"])
                - int(control_route["current_hard"]["refined"]["0.50"]["tp"])
            ),
            "refined_tp_075": (
                int(treatment_route["current_hard"]["refined"]["0.75"]["tp"])
                - int(control_route["current_hard"]["refined"]["0.75"]["tp"])
            ),
            "active_predictions_per_image": (
                treatment_route["active_predictions_per_image"]
                - control_route["active_predictions_per_image"]
            ),
            "coverage_f1_050": (
                treatment_coverage["f1"]["0.50"] - control_coverage["f1"]["0.50"]
            ),
            "coverage_f1_075": (
                treatment_coverage["f1"]["0.75"] - control_coverage["f1"]["0.75"]
            ),
        }
        checks = {
            "population_contract": bool(population.get("passed", False)),
            "support_mass_delta_ge_0p05": route_delta[
                "mean_target_support_mass"
            ] >= 0.05,
            "target_id_top1_delta_ge_0p05": route_delta[
                "target_id_top1_fraction"
            ] >= 0.05,
            "reference_tp_050_delta_ge_5": route_delta["reference_tp_050"] >= 5,
            "reference_tp_075_delta_ge_10": route_delta["reference_tp_075"] >= 10,
            "refined_tp_050_delta_ge_3": route_delta["refined_tp_050"] >= 3,
            "refined_tp_075_delta_ge_6": route_delta["refined_tp_075"] >= 6,
            "active_count_drift_le_0p05": abs(
                route_delta["active_predictions_per_image"]
            ) <= 0.05,
            "coverage_f1_050_not_worse": route_delta["coverage_f1_050"] >= 0.0,
            "coverage_f1_075_not_worse": route_delta["coverage_f1_075"] >= 0.0,
            "proposal_oracle_unchanged": _oracle_unchanged(
                source_coverage,
                treatment_coverage,
            )
            and _oracle_unchanged(source_coverage, control_coverage),
        }
        payload["treatment_minus_control"] = route_delta
        payload["thresholds"] = {
            "support_mass_delta": 0.05,
            "target_id_top1_delta": 0.05,
            "reference_tp_delta_050": 5,
            "reference_tp_delta_075": 10,
            "refined_tp_delta_050": 3,
            "refined_tp_delta_075": 6,
            "active_predictions_per_image_abs_delta_max": 0.05,
            "coverage_f1_deltas_min": 0.0,
        }

    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    payload["long_training_authorized"] = False
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": payload["mode"],
        "checks": checks,
        "passed": payload["passed"],
        "long_training_authorized": False,
    }, indent=2))
    print(f"output_json: {output}")
    if not payload["passed"]:
        raise SystemExit("V8.1 gate failed")


if __name__ == "__main__":
    main()

