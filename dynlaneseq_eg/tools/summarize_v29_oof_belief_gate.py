from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file


DIRECTIONS = (
    ("support_a_to_fold_b", "b"),
    ("support_b_to_fold_a", "a"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the two independently trained V29 OOF belief directions "
            "without selecting checkpoints or thresholds."
        )
    )
    parser.add_argument("--direction-a-to-b", required=True)
    parser.add_argument("--direction-b-to-a", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _auc(report: dict[str, Any], key: str) -> float:
    value = report["margin_auc"][key]["auc"]
    return float(value) if value is not None else float("-inf")


def _direction(
    root: Path,
    *,
    expected_name: str,
    expected_belief_fold: str,
) -> dict[str, Any]:
    paired_path = root / "paired" / "official_val_report.json"
    confidence_path = (
        root / "switch_confidence" / "switch_confidence_report.json"
    )
    paired = _load(paired_path)
    confidence = _load(confidence_path)
    if root.name != expected_name:
        raise ValueError(f"V29 direction path mismatch: {root.name}")

    oof = paired.get("oof_support_contract") or {}
    full = paired["domains"]["full_validation"]
    metrics = full["metrics"]
    source = metrics["source_v7"]
    arm_b = metrics["arm_b"]
    arm_c = metrics["arm_c"]
    wrong = metrics["arm_c_wrong_image"]
    quality_auc = _auc(confidence, "quality_delta_0p01")
    threshold_50_auc = _auc(confidence, "threshold_0p50")
    threshold_75_auc = _auc(confidence, "threshold_0p75")
    risk_50 = confidence["risk_curve"]["0.50"][
        "best_validation_diagnostic_at_harmful_rate_le_1pct"
    ]
    risk_75 = confidence["risk_curve"]["0.75"][
        "best_validation_diagnostic_at_harmful_rate_le_1pct"
    ]

    checks = {
        "paired_gate_passed": paired.get("gate", {}).get("passed") is True,
        "belief_fold_exact": str(oof.get("belief_fold"))
        == expected_belief_fold,
        "support_contract_all_passed": all(
            bool(value) for value in oof.get("checks", {}).values()
        ),
        "confidence_oof_contract_present": bool(
            confidence.get("oof_support_contract")
        ),
        "quality_margin_auc_at_least_0p65": quality_auc >= 0.65,
        "threshold_0p50_margin_auc_at_least_0p60": threshold_50_auc >= 0.60,
        "threshold_0p75_margin_auc_at_least_0p60": threshold_75_auc >= 0.60,
        "positive_safe_net_0p50": int(risk_50["net"]) > 0,
        "positive_safe_net_0p75": int(risk_75["net"]) > 0,
        "test_unused": paired.get("contract", {}).get("test_set_used") is False
        and confidence.get("contract", {}).get("test_set_used") is False,
    }
    return {
        "direction": expected_name,
        "belief_fold": expected_belief_fold,
        "checks": checks,
        "passed": all(checks.values()),
        "paired_report": str(paired_path),
        "paired_report_sha256": sha256_file(paired_path),
        "switch_confidence_report": str(confidence_path),
        "switch_confidence_report_sha256": sha256_file(confidence_path),
        "full_validation_f1_percent": {
            "source_0p50": 100.0 * float(source["0.50"]["F1"]),
            "source_0p75": 100.0 * float(source["0.75"]["F1"]),
            "arm_b_0p50": 100.0 * float(arm_b["0.50"]["F1"]),
            "arm_b_0p75": 100.0 * float(arm_b["0.75"]["F1"]),
            "arm_c_0p50": 100.0 * float(arm_c["0.50"]["F1"]),
            "arm_c_0p75": 100.0 * float(arm_c["0.75"]["F1"]),
            "wrong_image_c_0p50": 100.0 * float(wrong["0.50"]["F1"]),
            "wrong_image_c_0p75": 100.0 * float(wrong["0.75"]["F1"]),
        },
        "deltas_points": paired["gate"]["deltas"]["full_validation"],
        "margin_auc": {
            "quality_delta_0p01": quality_auc,
            "threshold_0p50": threshold_50_auc,
            "threshold_0p75": threshold_75_auc,
        },
        "safe_risk_diagnostic": {"0.50": risk_50, "0.75": risk_75},
    }


def main() -> None:
    args = parse_args()
    roots = (
        Path(args.direction_a_to_b).expanduser().resolve(),
        Path(args.direction_b_to_a).expanduser().resolve(),
    )
    directions = {
        name: _direction(
            root,
            expected_name=name,
            expected_belief_fold=belief_fold,
        )
        for root, (name, belief_fold) in zip(roots, DIRECTIONS)
    }
    report = {
        "experiment": "V29 two-direction clip-disjoint OOF belief gate",
        "passed": all(value["passed"] for value in directions.values()),
        "directions": directions,
        "contract": {
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "full_official_validation": True,
            "test_set_used": False,
            "requires_both_oof_directions": True,
        },
        "recommendation": (
            "oof_belief_mechanism_passed_review_before_test"
            if all(value["passed"] for value in directions.values())
            else "stop_rbf_family_oof_gate_failed"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
