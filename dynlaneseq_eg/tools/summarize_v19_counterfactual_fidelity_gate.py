from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed two-domain V19 endpoint gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _threshold_metric(
    report: dict[str, Any], policy: str, mode: str, threshold: str
) -> dict[str, Any]:
    return report["metrics"][policy][mode]["thresholds"][threshold]


def _domain_checks(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for threshold in ("0.50", "0.75"):
        source = _threshold_metric(
            report, "source_v7", "writer_valid", threshold
        )
        treatment = _threshold_metric(
            report, "v19_deployed", "writer_valid", threshold
        )
        paired = report["paired_image_effects"][threshold]
        degradation = report["source_correct_degradation"][threshold]
        checks[f"tp_gain_{threshold}"] = int(treatment["tp"] - source["tp"])
        checks[f"tp_gate_{threshold}"] = (
            checks[f"tp_gain_{threshold}"] >= 5
        )
        checks[f"f1_gain_{threshold}"] = float(
            treatment["f1"] - source["f1"]
        )
        checks[f"f1_nonregression_{threshold}"] = (
            checks[f"f1_gain_{threshold}"] >= 0.0
        )
        checks[f"improved_gt_worsened_{threshold}"] = int(
            paired["improved"]
        ) > int(paired["worsened"])
        checks[f"source_correct_loss_fraction_{threshold}"] = float(
            degradation["fraction"]
        )
        checks[f"source_correct_protection_{threshold}"] = (
            float(degradation["fraction"]) < 0.01
        )

    source_neural = report["metrics"]["source_v7"]["neural_active"]
    source_writer = report["metrics"]["source_v7"]["writer_valid"]
    v19_neural = report["metrics"]["v19_deployed"]["neural_active"]
    v19_writer = report["metrics"]["v19_deployed"]["writer_valid"]
    source_invalid = int(
        source_neural["thresholds"]["0.50"]["predictions"]
        - source_writer["thresholds"]["0.50"]["predictions"]
    )
    v19_invalid = int(
        v19_neural["thresholds"]["0.50"]["predictions"]
        - v19_writer["thresholds"]["0.50"]["predictions"]
    )
    checks["source_writer_invalid"] = source_invalid
    checks["v19_writer_invalid"] = v19_invalid
    checks["writer_invalid_nonincrease"] = v19_invalid <= source_invalid
    checks["cardinality_exact"] = (
        float(report["cardinality"]["exact_fraction"]) == 1.0
    )
    checks["proposal_oracle_exact"] = bool(
        report["proposal_oracle"]["exact_nonregression"]
    ) and all(
        float(value) == 0.0
        for value in report["proposal_oracle"][
            "proposal_tensor_max_difference"
        ].values()
    )
    fidelity = report["official_counterfactual_fidelity"]
    checks["same_gt_pair_accuracy"] = float(
        fidelity["same_gt_pair_accuracy"]
    )
    checks["same_gt_pair_gate"] = checks["same_gt_pair_accuracy"] >= 0.70
    checks["threshold_pair_accuracy"] = float(
        fidelity["threshold_crossing_pair_accuracy"]
    )
    checks["threshold_pair_gate"] = checks["threshold_pair_accuracy"] >= 0.70
    checks["fidelity_pearson"] = float(fidelity["fidelity_pearson"])
    checks["fidelity_spearman"] = float(
        fidelity["fidelity_spearman_ordinal"]
    )
    checks["quality_correlation_gate"] = (
        checks["fidelity_pearson"] >= 0.30
        and checks["fidelity_spearman"] >= 0.30
    )
    checks["training_surrogate_pearson"] = float(
        fidelity["training_surrogate_pearson"]
    )
    checks["training_surrogate_spearman"] = float(
        fidelity["training_surrogate_spearman_ordinal"]
    )
    checks["training_surrogate_agreement_50"] = float(
        fidelity["training_surrogate_threshold_agreement_50"]
    )
    checks["training_surrogate_agreement_75"] = float(
        fidelity["training_surrogate_threshold_agreement_75"]
    )
    checks["training_target_audit_gate"] = (
        checks["training_surrogate_pearson"] >= 0.85
        and checks["training_surrogate_spearman"] >= 0.85
        and checks["training_surrogate_agreement_50"] >= 0.75
        and checks["training_surrogate_agreement_75"] >= 0.85
    )
    checks["regret_reduced"] = float(fidelity["v19_regret"]) < float(
        fidelity["source_regret"]
    )
    boolean_values = [
        value
        for key, value in checks.items()
        if isinstance(value, bool)
    ]
    checks["passed"] = bool(boolean_values) and all(boolean_values)
    return checks


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout = _load(args.heldout)
    validation = _load(args.validation)
    heldout_checks = _domain_checks(heldout)
    validation_checks = _domain_checks(validation)
    passed = bool(contract.get("passed")) and bool(
        heldout_checks["passed"] and validation_checks["passed"]
    )
    report = {
        "experiment": "V19 frozen counterfactual proposal fidelity fixed gate",
        "passed": passed,
        "gate0_passed": bool(contract.get("passed")),
        "heldout": heldout_checks,
        "validation": validation_checks,
        "decision": (
            "v19_fixed_gate_pass_stop_for_joint_planning"
            if passed
            else "v19_fixed_gate_fail_stop"
        ),
        "long_training_authorized": False,
        "full_validation_executed": False,
        "test_set_used": False,
        "checkpoint_selection_performed": False,
        "threshold_or_nms_search_performed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
