from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared V14 Stage-A identifiability gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _domain(report: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "support_mass_gain_over_v7": float(
            report["correct_minus_v7"]["support_mass"]
        ),
        "hard_target_top1_gain_over_v7": float(
            report["correct_minus_v7"]["hard_target_top1"]
        ),
        "support_mass_gain_over_cross_clip_wrong": float(
            report["correct_minus_cross_clip_wrong"]["support_mass"]
        ),
        "support_mass_gain_over_zero_content": float(
            report["correct_minus_zero_content"]["support_mass"]
        ),
        "support_mass_gain_over_position_only": float(
            report["correct_minus_position_only"]["support_mass"]
        ),
        "visual_dfl_correct_over_cross_clip_wrong": float(
            report["visual_dfl_ratios"][
                "correct_over_cross_clip_wrong"
            ]
        ),
        "visual_dfl_correct_over_position_only": float(
            report["visual_dfl_ratios"]["correct_over_position_only"]
        ),
    }
    checks = {
        "deployment_parity_exact": report["deployment"]["exact"] is True,
        "cross_clip_pairing_exact": report["cross_clip"]["exact"] is True,
        "support_mass_at_least_v7_plus_0p05": (
            metrics["support_mass_gain_over_v7"] >= 0.05
        ),
        "hard_top1_at_least_v7_plus_5_points": (
            metrics["hard_target_top1_gain_over_v7"] >= 0.05
        ),
        "correct_support_at_least_wrong_plus_0p05": (
            metrics["support_mass_gain_over_cross_clip_wrong"] >= 0.05
        ),
        "correct_visual_dfl_at_most_0p95_wrong": (
            metrics["visual_dfl_correct_over_cross_clip_wrong"] <= 0.95
        ),
        "correct_visual_dfl_at_most_0p95_position_only": (
            metrics["visual_dfl_correct_over_position_only"] <= 0.95
        ),
        # These two checks prevent a proposal/anchor shortcut from passing
        # merely because the correct-P2 branch improved nominal metrics.
        "zero_content_does_not_preserve_association_gain": (
            metrics["support_mass_gain_over_zero_content"] >= 0.05
        ),
        "position_only_does_not_preserve_association_gain": (
            metrics["support_mass_gain_over_position_only"] >= 0.05
        ),
        "test_closed": report.get("test_set_used") is False,
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout_report = _load(args.heldout_end)
    val_report = _load(args.val_end)
    heldout = _domain(heldout_report)
    validation = _domain(val_report)
    checks = {
        "gate0_contract_passed": contract.get("passed") is True,
        "heldout_clip_256_passed": heldout["passed"],
        "validation_256_passed": validation["passed"],
        "all_reports_test_closed": all(
            report.get("test_set_used") is False
            for report in (contract, heldout_report, val_report)
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V14 corrected visual-first Stage-A gate",
        "gate_thresholds": {
            "support_mass_gain_over_v7": 0.05,
            "hard_target_top1_gain_over_v7": 0.05,
            "support_mass_gain_over_each_negative_control": 0.05,
            "visual_dfl_correct_over_wrong_max": 0.95,
            "visual_dfl_correct_over_position_only_max": 0.95,
        },
        "heldout_clip_256": heldout,
        "validation_256": validation,
        "checks": checks,
        "passed": passed,
        "stage_b_authorized": passed,
        "long_training_authorized": False,
        "test_set_used": False,
        "decision": (
            "stage_a_pass_authorize_v14_stage_b"
            if passed
            else "stage_a_fail_stop_v14_and_package_for_sol"
        ),
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
