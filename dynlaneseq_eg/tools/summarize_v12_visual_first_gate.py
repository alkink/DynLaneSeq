from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V12 Stage-A association gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--heldout-init", required=True)
    parser.add_argument("--heldout-end", required=True)
    parser.add_argument("--val-init", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(report: dict[str, Any], key: str) -> float:
    return float(report["summaries"][key]["mean"])


def _domain(
    initialization: dict[str, Any], endpoint: dict[str, Any]
) -> dict[str, Any]:
    metrics = {
        "support_mass_gain_over_v7": float(
            endpoint["correct_minus_v7"]["support_mass"]
        ),
        "target_top1_gain_over_v7": float(
            endpoint["correct_minus_v7"]["target_top1"]
        ),
        "correct_wrong_support_gap": float(
            endpoint["correct_minus_wrong_p2"]["support_mass"]
        ),
        "correct_wrong_visual_loss_advantage": float(
            endpoint["correct_minus_wrong_p2"][
                "final_visual_loss_advantage"
            ]
        ),
        "correct_zero_support_gap": float(
            endpoint["correct_minus_zero_p2"]["support_mass"]
        ),
        "endpoint_support_mass": _mean(
            endpoint, "correct_p2_support_mass"
        ),
        "endpoint_target_top1": _mean(
            endpoint, "correct_p2_target_top1"
        ),
        "endpoint_visual_mae_px": _mean(
            endpoint, "correct_p2_final_visual_mae_px"
        ),
        "support_mass_change_from_initialization": (
            _mean(endpoint, "correct_p2_support_mass")
            - _mean(initialization, "correct_p2_support_mass")
        ),
        "target_top1_change_from_initialization": (
            _mean(endpoint, "correct_p2_target_top1")
            - _mean(initialization, "correct_p2_target_top1")
        ),
        "visual_mae_change_from_initialization_px": (
            _mean(endpoint, "correct_p2_final_visual_mae_px")
            - _mean(initialization, "correct_p2_final_visual_mae_px")
        ),
    }
    checks = {
        "support_mass_gain_at_least_0p05": (
            metrics["support_mass_gain_over_v7"] >= 0.05
        ),
        "target_top1_gain_at_least_5_points": (
            metrics["target_top1_gain_over_v7"] >= 0.05
        ),
        "correct_p2_support_advantage_at_least_0p05": (
            metrics["correct_wrong_support_gap"] >= 0.05
        ),
        "correct_p2_visual_loss_advantage_at_least_0p05": (
            metrics["correct_wrong_visual_loss_advantage"] >= 0.05
        ),
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    contract = _load(args.contract)
    heldout = _domain(_load(args.heldout_init), _load(args.heldout_end))
    validation = _domain(_load(args.val_init), _load(args.val_end))
    checks = {
        "zero_step_contract_passed": contract.get("passed") is True,
        "heldout_clip_gate_passed": heldout["passed"],
        "validation_gate_passed": validation["passed"],
        "test_closed": all(
            report.get("test_set_used") is False
            for report in (
                contract,
                _load(args.heldout_init),
                _load(args.heldout_end),
                _load(args.val_init),
                _load(args.val_end),
            )
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "V12 visual-first Stage-A bridge gate",
        "contract": contract,
        "heldout_clip": heldout,
        "validation": validation,
        "checks": checks,
        "passed": passed,
        "stage_b_authorized": passed,
        "long_training_authorized": False,
        "test_set_used": False,
        "decision": (
            "stage_a_pass_authorize_parity_anchored_geometry_stage_b"
            if passed
            else "stage_a_fail_stop_visual_first_family"
        ),
    }
    destination = Path(args.output_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {destination}")


if __name__ == "__main__":
    main()
