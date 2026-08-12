from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V11 fixed64/generalization gate."
    )
    parser.add_argument("--mode", choices=("fixed64", "generalization"), required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--source-coverage", required=True)
    parser.add_argument("--init-coverage", required=True)
    parser.add_argument("--treatment-coverage", required=True)
    parser.add_argument("--init-state", required=True)
    parser.add_argument("--treatment-state", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _coverage(payload: dict[str, Any]) -> dict[str, Any]:
    methods = payload["methods"]
    key = "four_slot_refined"
    if key not in methods:
        raise KeyError("coverage report has no four_slot_refined method")
    metric = {threshold: methods[key][threshold] for threshold in THRESHOLDS}
    capacity = {
        threshold: payload["capacity"][threshold]["all_candidate_oracle"]
        for threshold in THRESHOLDS
    }
    slots = payload.get("four_slot_diagnostics") or {}
    cardinality = slots.get("cardinality") or {}
    return {
        "metric": metric,
        "proposal_oracle": capacity,
        "cardinality_exact": float(cardinality.get("exact_fraction", 0.0)),
        "cardinality_mae": float(cardinality.get("mean_absolute_error", 0.0)),
        "semantic_duplicate_fraction": float(
            slots.get("semantic_duplicate_cluster_fraction", 0.0)
        ),
        "mean_selected": float(metric["0.50"]["mean_selected_per_image"]),
        "refinement": slots.get("refinement"),
    }


def _state(payload: dict[str, Any]) -> dict[str, float]:
    statistics = payload["statistics"]
    return {
        name: float(value["mean"]) for name, value in statistics.items()
    }


def _gap_closure(source_tp: int, treatment_tp: int, oracle_tp: int) -> float:
    denominator = int(oracle_tp) - int(source_tp)
    if denominator <= 0:
        return 1.0 if int(treatment_tp) >= int(source_tp) else -1.0
    return float(int(treatment_tp) - int(source_tp)) / float(denominator)


def main() -> None:
    args = parse_args()
    contract = _read(args.contract)
    source = _coverage(_read(args.source_coverage))
    initialization = _coverage(_read(args.init_coverage))
    treatment = _coverage(_read(args.treatment_coverage))
    init_state = _state(_read(args.init_state))
    treatment_state = _state(_read(args.treatment_state))

    initialization_vs_source = {
        threshold: {
            "tp_equal": int(initialization["metric"][threshold]["tp"])
            == int(source["metric"][threshold]["tp"]),
            "fp_equal": int(initialization["metric"][threshold]["fp"])
            == int(source["metric"][threshold]["fp"]),
            "fn_equal": int(initialization["metric"][threshold]["fn"])
            == int(source["metric"][threshold]["fn"]),
        }
        for threshold in THRESHOLDS
    }
    proposal_oracle_equal = all(
        int(treatment["proposal_oracle"][threshold]["hits"])
        == int(source["proposal_oracle"][threshold]["hits"])
        for threshold in THRESHOLDS
    )
    closure = {
        threshold: _gap_closure(
            int(source["metric"][threshold]["tp"]),
            int(treatment["metric"][threshold]["tp"]),
            int(source["proposal_oracle"][threshold]["hits"]),
        )
        for threshold in THRESHOLDS
    }
    common_checks = {
        "initialization_contract_passed": contract.get("passed") is True,
        # V11 intentionally changes continuous geometry at initialization:
        # it exposes predicted-soft proposal memory instead of the V7 hard
        # refiner.  Exact metric parity would reinstate the hard-ID contract
        # that this experiment is designed to remove.
        "initialization_metrics_finite": all(
            all(
                isinstance(initialization["metric"][threshold][name], (int, float))
                and math.isfinite(
                    float(initialization["metric"][threshold][name])
                )
                for name in ("precision", "recall", "f1")
            )
            for threshold in THRESHOLDS
        ),
        "proposal_oracle_unchanged": proposal_oracle_equal,
        "semantic_duplicate_at_most_2pct": treatment[
            "semantic_duplicate_fraction"
        ] <= 0.02,
        "active_count_drift_at_most_0p05": abs(
            treatment["mean_selected"] - source["mean_selected"]
        ) <= 0.05,
        "final_surrogate_not_worse_than_base": treatment_state[
            "quality_gain"
        ] >= -1.0e-3,
    }
    if args.mode == "fixed64":
        mode_checks = {
            "gap_closure_050_at_least_80pct": closure["0.50"] >= 0.80,
            "gap_closure_075_at_least_70pct": closure["0.75"] >= 0.70,
            "cardinality_exact_at_least_95pct": treatment[
                "cardinality_exact"
            ] >= 0.95,
            "f1_050_not_below_source": treatment["metric"]["0.50"]["f1"]
            >= source["metric"]["0.50"]["f1"],
            "f1_075_not_below_source": treatment["metric"]["0.75"]["f1"]
            >= source["metric"]["0.75"]["f1"],
        }
        next_action_pass = (
            "Run the predeclared 3k uniform-256 validation generalization gate; "
            "do not start long training."
        )
    else:
        mode_checks = {
            "target_support_mass_gain_at_least_5_points": (
                treatment_state["target_support_mass"]
                - init_state["target_support_mass"]
            )
            >= 0.05,
            "tp_050_gain_at_least_3": int(
                treatment["metric"]["0.50"]["tp"]
            )
            - int(source["metric"]["0.50"]["tp"])
            >= 3,
            "tp_075_gain_at_least_6": int(
                treatment["metric"]["0.75"]["tp"]
            )
            - int(source["metric"]["0.75"]["tp"])
            >= 6,
            "f1_050_not_below_source": treatment["metric"]["0.50"]["f1"]
            >= source["metric"]["0.50"]["f1"],
            "f1_075_not_below_source": treatment["metric"]["0.75"]["f1"]
            >= source["metric"]["0.75"]["f1"],
        }
        next_action_pass = (
            "Run exact full validation once. Long training remains unauthorized "
            "until that paired primary gate passes."
        )
    checks = {**common_checks, **mode_checks}
    passed = all(checks.values())
    report = {
        "experiment": "V11 unified proposal-visual slot-row gate",
        "mode": args.mode,
        "iteration": int(args.iteration),
        "test_set_used": False,
        "contract_passed": contract.get("passed") is True,
        "source": source,
        "initialization": initialization,
        "treatment": treatment,
        "initialization_state": init_state,
        "treatment_state": treatment_state,
        "initialization_vs_source_count_equality": initialization_vs_source,
        "proposal_oracle_gap_closure": closure,
        "checks": checks,
        "passed": passed,
        "long_training_authorized": False,
        "next_action": (
            next_action_pass
            if passed
            else (
                "STOP this V11 arm. Preserve artifacts and diagnose the failed "
                "predeclared check; do not relax the gate post hoc."
            )
        ),
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
