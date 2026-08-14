from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the immutable paired V18 two-domain endpoint gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout-treatment", required=True)
    parser.add_argument("--heldout-control", required=True)
    parser.add_argument("--validation-treatment", required=True)
    parser.add_argument("--validation-control", required=True)
    parser.add_argument("--heldout-source-coverage", required=True)
    parser.add_argument("--heldout-treatment-coverage", required=True)
    parser.add_argument("--heldout-control-coverage", required=True)
    parser.add_argument("--val-source-coverage", required=True)
    parser.add_argument("--val-treatment-coverage", required=True)
    parser.add_argument("--val-control-coverage", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _threshold(
    report: dict[str, Any], policy: str, mode: str, threshold: str
) -> dict[str, Any]:
    return report["metrics"][policy][mode]["thresholds"][threshold]


def _cardinality(report: dict[str, Any], policy: str) -> dict[str, Any]:
    return report["metrics"][policy]["writer_valid"]["cardinality"]


def _delta(
    report: dict[str, Any], comparison: str, threshold: str
) -> dict[str, Any]:
    return report["deltas"][comparison]["writer_valid"][threshold]


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


def _domain(
    treatment: dict[str, Any],
    control: dict[str, Any],
    source_coverage_payload: dict[str, Any],
    treatment_coverage_payload: dict[str, Any],
    control_coverage_payload: dict[str, Any],
) -> dict[str, Any]:
    source_coverage = _coverage(source_coverage_payload)
    treatment_coverage = _coverage(treatment_coverage_payload)
    control_coverage = _coverage(control_coverage_payload)
    source_050 = _threshold(treatment, "source_v7", "writer_valid", "0.50")
    source_075 = _threshold(treatment, "source_v7", "writer_valid", "0.75")
    endpoint_050 = _threshold(treatment, "v18_deployed", "writer_valid", "0.50")
    endpoint_075 = _threshold(treatment, "v18_deployed", "writer_valid", "0.75")
    control_050 = _threshold(control, "v18_deployed", "writer_valid", "0.50")
    control_075 = _threshold(control, "v18_deployed", "writer_valid", "0.75")
    source_neural_050 = _threshold(treatment, "source_v7", "neural_active", "0.50")
    endpoint_neural_050 = _threshold(treatment, "v18_deployed", "neural_active", "0.50")
    source_cardinality = _cardinality(treatment, "source_v7")
    endpoint_cardinality = _cardinality(treatment, "v18_deployed")
    wrong_image_050 = _delta(
        treatment, "correct_minus_cross_clip_wrong_image", "0.50"
    )
    wrong_image_075 = _delta(
        treatment, "correct_minus_cross_clip_wrong_image", "0.75"
    )
    wrong_memory_050 = _delta(
        treatment, "correct_minus_cross_clip_wrong_proposal_memory", "0.50"
    )
    wrong_memory_075 = _delta(
        treatment, "correct_minus_cross_clip_wrong_proposal_memory", "0.75"
    )
    paired_050 = treatment["paired_image_effects"]["0.50"]
    paired_075 = treatment["paired_image_effects"]["0.75"]
    source_invalid = int(source_neural_050["predictions"]) - int(
        source_050["predictions"]
    )
    endpoint_invalid = int(endpoint_neural_050["predictions"]) - int(
        endpoint_050["predictions"]
    )
    proposal_oracle_nonregression = all(
        treatment_coverage["proposal_oracle_hits"][threshold]
        >= source_coverage["proposal_oracle_hits"][threshold]
        and treatment_coverage["proposal_oracle_hits"][threshold]
        >= control_coverage["proposal_oracle_hits"][threshold]
        for threshold in THRESHOLDS
    )
    values = {
        "source": {
            "tp_050": int(source_050["tp"]),
            "tp_075": int(source_075["tp"]),
            "f1_050_points": 100.0 * float(source_050["f1"]),
            "f1_075_points": 100.0 * float(source_075["f1"]),
            "cardinality_mae": float(source_cardinality["mae"]),
        },
        "treatment": {
            "tp_050": int(endpoint_050["tp"]),
            "tp_075": int(endpoint_075["tp"]),
            "f1_050_points": 100.0 * float(endpoint_050["f1"]),
            "f1_075_points": 100.0 * float(endpoint_075["f1"]),
            "cardinality_mae": float(endpoint_cardinality["mae"]),
        },
        "control": {
            "tp_050": int(control_050["tp"]),
            "tp_075": int(control_075["tp"]),
            "f1_050_points": 100.0 * float(control_050["f1"]),
            "f1_075_points": 100.0 * float(control_075["f1"]),
        },
        "treatment_minus_source": {
            "tp_050": int(endpoint_050["tp"]) - int(source_050["tp"]),
            "tp_075": int(endpoint_075["tp"]) - int(source_075["tp"]),
            "f1_050_points": 100.0 * (
                float(endpoint_050["f1"]) - float(source_050["f1"])
            ),
            "f1_075_points": 100.0 * (
                float(endpoint_075["f1"]) - float(source_075["f1"])
            ),
        },
        "treatment_minus_control": {
            "tp_050": int(endpoint_050["tp"]) - int(control_050["tp"]),
            "tp_075": int(endpoint_075["tp"]) - int(control_075["tp"]),
            "f1_050_points": 100.0 * (
                float(endpoint_050["f1"]) - float(control_050["f1"])
            ),
            "f1_075_points": 100.0 * (
                float(endpoint_075["f1"]) - float(control_075["f1"])
            ),
        },
        "writer_invalid_increase": endpoint_invalid - source_invalid,
        "semantic_duplicate_fraction": treatment_coverage[
            "semantic_duplicate_fraction"
        ],
        "proposal_oracle_nonregression": proposal_oracle_nonregression,
        "chosen_set_regret_reduction": float(
            treatment["chosen_set_regret"]["relative_reduction"]
        ),
        "improved_images_050": int(paired_050["improved"]),
        "worsened_images_050": int(paired_050["worsened"]),
        "improved_images_075": int(paired_075["improved"]),
        "worsened_images_075": int(paired_075["worsened"]),
        "anchor_correct_degradation_050": float(
            treatment["source_correct_degradation"]["0.50"]["fraction"]
        ),
        "anchor_correct_degradation_075": float(
            treatment["source_correct_degradation"]["0.75"]["fraction"]
        ),
        "refine_precision": float(
            treatment["refine_policy"]["precision_improved"]
        ),
        "refine_decisions": int(
            treatment["refine_policy"]["matched_refine_decisions"]
        ),
        "correct_minus_wrong_image_tp_050": int(wrong_image_050["delta_tp"]),
        "correct_minus_wrong_image_tp_075": int(wrong_image_075["delta_tp"]),
        "correct_minus_wrong_memory_tp_050": int(wrong_memory_050["delta_tp"]),
        "correct_minus_wrong_memory_tp_075": int(wrong_memory_075["delta_tp"]),
    }
    delta_source = values["treatment_minus_source"]
    delta_control = values["treatment_minus_control"]
    checks = {
        "tp_050_at_least_source_plus_8": delta_source["tp_050"] >= 8,
        "f1_050_at_least_source_plus_0p8": delta_source["f1_050_points"] >= 0.8,
        "tp_075_at_least_source_plus_5": delta_source["tp_075"] >= 5,
        "f1_075_nonregression": delta_source["f1_075_points"] >= 0.0,
        "treatment_at_least_control_plus_5_tp_050": delta_control["tp_050"] >= 5,
        "treatment_at_least_control_plus_5_tp_075": delta_control["tp_075"] >= 5,
        "cardinality_mae_not_worse": values["treatment"]["cardinality_mae"]
        <= values["source"]["cardinality_mae"],
        "writer_invalid_increase_zero": values["writer_invalid_increase"] == 0,
        "semantic_duplicate_at_most_1pct": values[
            "semantic_duplicate_fraction"
        ]
        <= 0.01,
        "proposal_oracle_nonregression": proposal_oracle_nonregression,
        "chosen_set_regret_reduced_at_least_15pct": values[
            "chosen_set_regret_reduction"
        ]
        >= 0.15,
        "improved_images_at_least_1p5x_worsened_050": values[
            "improved_images_050"
        ]
        >= 1.5 * max(values["worsened_images_050"], 1),
        "improved_images_at_least_1p5x_worsened_075": values[
            "improved_images_075"
        ]
        >= 1.5 * max(values["worsened_images_075"], 1),
        "source_correct_degradation_below_2pct_050": values[
            "anchor_correct_degradation_050"
        ]
        < 0.02,
        "source_correct_degradation_below_2pct_075": values[
            "anchor_correct_degradation_075"
        ]
        < 0.02,
        "refine_precision_at_least_65pct": values["refine_decisions"] > 0
        and values["refine_precision"] >= 0.65,
        "correct_image_at_least_wrong_plus_10_tp_050": values[
            "correct_minus_wrong_image_tp_050"
        ]
        >= 10,
        "correct_image_at_least_wrong_plus_15_tp_075": values[
            "correct_minus_wrong_image_tp_075"
        ]
        >= 15,
        "correct_memory_at_least_wrong_plus_8_tp_050": values[
            "correct_minus_wrong_memory_tp_050"
        ]
        >= 8,
        "correct_memory_at_least_wrong_plus_8_tp_075": values[
            "correct_minus_wrong_memory_tp_075"
        ]
        >= 8,
        "cross_clip_runtime_exact": int(
            treatment.get("cross_clip_runtime_same_image", -1)
        )
        == 0
        and int(treatment.get("cross_clip_runtime_same_clip", -1)) == 0,
        "source_replay_matches_control_report": all(
            _threshold(treatment, "source_v7", "writer_valid", threshold)
            == _threshold(control, "source_v7", "writer_valid", threshold)
            for threshold in THRESHOLDS
        ),
        "test_closed": treatment.get("test_set_used") is False
        and control.get("test_set_used") is False,
    }
    return {"metrics": values, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout = _domain(
        _load(args.heldout_treatment),
        _load(args.heldout_control),
        _load(args.heldout_source_coverage),
        _load(args.heldout_treatment_coverage),
        _load(args.heldout_control_coverage),
    )
    validation = _domain(
        _load(args.validation_treatment),
        _load(args.validation_control),
        _load(args.val_source_coverage),
        _load(args.val_treatment_coverage),
        _load(args.val_control_coverage),
    )
    checks = {
        "gate0_contract_passed": contract.get("passed") is True,
        "heldout_clip_256_passed": heldout["passed"],
        "validation_256_passed": validation["passed"],
        "all_reports_test_closed": all(
            report.get("test_set_used") is False
            for report in (
                contract,
                _load(args.heldout_treatment),
                _load(args.heldout_control),
                _load(args.validation_treatment),
                _load(args.validation_control),
            )
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V18 joint exact set-energy paired fixed gate",
        "gate_thresholds": {
            "source_gain": {"tp_050": 8, "f1_050_points": 0.8, "tp_075": 5},
            "treatment_control_tp_gain": {"0.50": 5, "0.75": 5},
            "set_regret_relative_reduction": 0.15,
            "source_correct_degradation_max": 0.02,
            "refine_precision_min": 0.65,
            "correct_wrong_image_tp": {"0.50": 10, "0.75": 15},
            "correct_wrong_memory_tp": {"0.50": 8, "0.75": 8},
            "semantic_duplicate_max": 0.01,
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
            "v18_fixed_gate_pass_stop_for_review"
            if passed
            else "v18_fixed_gate_fail_stop_for_review"
        ),
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
