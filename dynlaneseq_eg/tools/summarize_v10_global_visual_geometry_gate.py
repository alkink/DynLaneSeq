from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V10 fixed-64 or paired generalization gate."
    )
    parser.add_argument("--mode", choices=("fixed64", "generalization"), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--source-route", required=True)
    parser.add_argument("--source-coverage", required=True)
    parser.add_argument("--init-route", required=True)
    parser.add_argument("--init-coverage", required=True)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--control-coverage", required=True)
    parser.add_argument("--treatment-route", required=True)
    parser.add_argument("--treatment-coverage", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _route(report: dict[str, Any]) -> dict[str, Any]:
    data = report["v7"]
    policy = str(data.get("production_policy", "current_hard"))
    policies = data["reference_policies"]
    if policy not in policies:
        raise ValueError(f"production policy {policy!r} is missing")
    stages = {
        stage: {
            threshold: dict(
                policies[policy][stage]["fixed_neural_active"][threshold]
            )
            for threshold in THRESHOLDS
        }
        for stage in ("reference", "refined")
    }
    target = policies["target_hard"]
    route_invariant = all(
        target[stage]["fixed_neural_active"][threshold]
        == policies[policy][stage]["fixed_neural_active"][threshold]
        for stage in ("reference", "refined")
        for threshold in THRESHOLDS
    )
    images = int(data["images"])
    predictions = int(stages["refined"]["0.50"]["predictions"])
    return {
        "images": images,
        "production_policy": policy,
        "production_reference_mode": data.get("production_reference_mode"),
        "stages": stages,
        "active_predictions_per_image": float(predictions) / float(max(images, 1)),
        "target_route_invariant": route_invariant,
        "replay_max_abs_error": float(data.get("replay_max_abs_error", 0.0)),
    }


def _coverage(report: dict[str, Any]) -> dict[str, Any]:
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


def _oracle_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        abs(
            float(left["proposal_oracle_recall"][threshold])
            - float(right["proposal_oracle_recall"][threshold])
        )
        <= 1.0e-12
        for threshold in THRESHOLDS
    )


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    source = {
        "route": _route(_load(args.source_route)),
        "coverage": _coverage(_load(args.source_coverage)),
    }
    initialization = {
        "route": _route(_load(args.init_route)),
        "coverage": _coverage(_load(args.init_coverage)),
    }
    control = {
        "route": _route(_load(args.control_route)),
        "coverage": _coverage(_load(args.control_coverage)),
    }
    treatment = {
        "route": _route(_load(args.treatment_route)),
        "coverage": _coverage(_load(args.treatment_coverage)),
    }
    payload: dict[str, Any] = {
        "experiment": "V10 full-width global visual slot geometry gate",
        "mode": args.mode,
        "iteration": int(args.iteration),
        "test_set_used": False,
        "contract_passed": bool(contract.get("passed", False)),
        "source_v7": source,
        "v10_initialization": initialization,
        "legacy_control": control,
        "v10_treatment": treatment,
    }

    if args.mode == "fixed64":
        checks = {
            "zero_step_contract": bool(contract.get("passed", False)),
            "v10_route_identity_invariant": bool(
                treatment["route"]["target_route_invariant"]
            ),
            "memorized_f1_050_ge_0p95": (
                treatment["coverage"]["f1"]["0.50"] >= 0.95
            ),
            "memorized_f1_075_ge_0p85": (
                treatment["coverage"]["f1"]["0.75"] >= 0.85
            ),
            "memorized_tp_050_ge_source": (
                treatment["coverage"]["tp"]["0.50"]
                >= source["coverage"]["tp"]["0.50"]
            ),
            "memorized_tp_075_ge_source": (
                treatment["coverage"]["tp"]["0.75"]
                >= source["coverage"]["tp"]["0.75"]
            ),
            "cardinality_exact_ge_0p95": (
                treatment["coverage"]["cardinality_exact"] >= 0.95
            ),
            "semantic_duplicate_le_0p02": (
                treatment["coverage"]["semantic_duplicate"] <= 0.02
            ),
            "proposal_oracle_unchanged": _oracle_equal(
                source["coverage"], treatment["coverage"]
            ),
        }
        payload["thresholds"] = {
            "f1_050": 0.95,
            "f1_075": 0.85,
            "cardinality_exact": 0.95,
            "semantic_duplicate_max": 0.02,
        }
    else:
        delta = {
            "coverage_f1_050": (
                treatment["coverage"]["f1"]["0.50"]
                - control["coverage"]["f1"]["0.50"]
            ),
            "coverage_f1_075": (
                treatment["coverage"]["f1"]["0.75"]
                - control["coverage"]["f1"]["0.75"]
            ),
            "coverage_tp_050": (
                treatment["coverage"]["tp"]["0.50"]
                - control["coverage"]["tp"]["0.50"]
            ),
            "coverage_tp_075": (
                treatment["coverage"]["tp"]["0.75"]
                - control["coverage"]["tp"]["0.75"]
            ),
            "active_predictions_per_image": (
                treatment["route"]["active_predictions_per_image"]
                - control["route"]["active_predictions_per_image"]
            ),
            "treatment_refined_minus_reference_tp_050": (
                treatment["route"]["stages"]["refined"]["0.50"]["tp"]
                - treatment["route"]["stages"]["reference"]["0.50"]["tp"]
            ),
            "treatment_refined_minus_reference_tp_075": (
                treatment["route"]["stages"]["refined"]["0.75"]["tp"]
                - treatment["route"]["stages"]["reference"]["0.75"]["tp"]
            ),
        }
        checks = {
            "zero_step_contract": bool(contract.get("passed", False)),
            "v10_route_identity_invariant": bool(
                treatment["route"]["target_route_invariant"]
            ),
            "f1_050_delta_ge_0p005": delta["coverage_f1_050"] >= 0.005,
            "f1_075_delta_ge_0p005": delta["coverage_f1_075"] >= 0.005,
            "tp_050_delta_ge_5": delta["coverage_tp_050"] >= 5,
            "tp_075_delta_ge_8": delta["coverage_tp_075"] >= 8,
            "refiner_retains_reference_050": (
                delta["treatment_refined_minus_reference_tp_050"] >= -2
            ),
            "refiner_retains_reference_075": (
                delta["treatment_refined_minus_reference_tp_075"] >= -2
            ),
            "active_count_drift_le_0p05": abs(
                delta["active_predictions_per_image"]
            ) <= 0.05,
            "semantic_duplicate_le_0p02": (
                treatment["coverage"]["semantic_duplicate"] <= 0.02
            ),
            "proposal_oracle_unchanged": (
                _oracle_equal(source["coverage"], control["coverage"])
                and _oracle_equal(source["coverage"], treatment["coverage"])
            ),
        }
        payload["treatment_minus_legacy_control"] = delta
        payload["thresholds"] = {
            "f1_delta_each": 0.005,
            "tp_delta_050": 5,
            "tp_delta_075": 8,
            "refined_minus_reference_tp_min": -2,
            "active_predictions_per_image_abs_delta_max": 0.05,
            "semantic_duplicate_max": 0.02,
        }

    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    payload["long_training_authorized"] = False
    payload["next_action"] = (
        "paired_full_validation_only"
        if payload["passed"] and args.mode == "generalization"
        else (
            "run_short_generalization_gate"
            if payload["passed"]
            else "stop_v10_and_reaudit"
        )
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "checks": checks,
                "passed": payload["passed"],
                "long_training_authorized": False,
                "next_action": payload["next_action"],
            },
            indent=2,
        )
    )
    print(f"output_json: {output}")
    if payload["passed"] is not True:
        raise SystemExit("V10 gate failed")


if __name__ == "__main__":
    main()
