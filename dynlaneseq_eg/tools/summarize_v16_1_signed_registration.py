from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the fixed V16.1 gate.")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _delta(report: dict[str, Any], comparison: str, threshold: str) -> int:
    return int(
        report["deltas"][comparison]["writer_valid"][threshold]["delta_tp"]
    )


def _prediction_count(report: dict[str, Any], policy: str) -> int:
    return int(
        report["metrics"][policy]["writer_valid"]["thresholds"]["0.50"][
            "predictions"
        ]
    )


def _domain(report: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "complete_fixed_list": report.get("complete_fixed_list") is True,
        "correct_vs_v7_anchor_tp_050_at_least_3": _delta(
            report, "correct_minus_v7_anchor_reference", "0.50"
        )
        >= 3,
        "correct_vs_v7_anchor_tp_075_at_least_6": _delta(
            report, "correct_minus_v7_anchor_reference", "0.75"
        )
        >= 6,
        "correct_vs_wrong_tp_050_at_least_3": _delta(
            report, "correct_minus_cross_clip_wrong_p2", "0.50"
        )
        >= 3,
        "correct_vs_wrong_tp_075_at_least_3": _delta(
            report, "correct_minus_cross_clip_wrong_p2", "0.75"
        )
        >= 3,
        "correct_vs_zero_tp_050_at_least_3": _delta(
            report, "correct_minus_zero_content_p2", "0.50"
        )
        >= 3,
        "correct_vs_zero_tp_075_at_least_3": _delta(
            report, "correct_minus_zero_content_p2", "0.75"
        )
        >= 3,
        "prediction_count_exact_v7": _prediction_count(
            report, "registration_correct_p2"
        )
        == _prediction_count(report, "v7_anchor_reference"),
        "selected_duplicate_count_zero": all(
            int(value) == 0
            for value in report["selected_duplicate_count"].values()
        ),
        "selected_outside_group_count_zero": int(
            report["selected_outside_group_count"]
        )
        == 0,
        "whole_proposal_gather_exact": float(
            report["exact_whole_proposal_gather_max_abs"]
        )
        == 0.0,
        "optimizer_steps_zero": int(report["optimizer_steps_during_audit"])
        == 0,
        "test_closed": report.get("test_set_used") is False,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "delta_tp": {
            "correct_minus_v7_anchor": {
                threshold: _delta(
                    report,
                    "correct_minus_v7_anchor_reference",
                    threshold,
                )
                for threshold in ("0.50", "0.75")
            },
            "correct_minus_cross_clip_wrong": {
                threshold: _delta(
                    report,
                    "correct_minus_cross_clip_wrong_p2",
                    threshold,
                )
                for threshold in ("0.50", "0.75")
            },
            "correct_minus_zero_content": {
                threshold: _delta(
                    report,
                    "correct_minus_zero_content_p2",
                    threshold,
                )
                for threshold in ("0.50", "0.75")
            },
            "correct_minus_position_only": {
                threshold: _delta(
                    report,
                    "correct_minus_position_only_p2",
                    threshold,
                )
                for threshold in ("0.50", "0.75")
            },
        },
    }


def main() -> None:
    args = parse_args()
    heldout = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    heldout_result = _domain(heldout)
    validation_result = _domain(validation)
    passed = bool(heldout_result["passed"] and validation_result["passed"])
    report = {
        "experiment": "V16.1 signed-registration fixed training-free gate",
        "gate_thresholds": {
            "correct_minus_v7_anchor_tp_050_min": 3,
            "correct_minus_v7_anchor_tp_075_min": 6,
            "correct_minus_each_wrong_zero_tp_each_threshold_min": 3,
            "prediction_count_exact": True,
            "duplicates_zero": True,
            "whole_proposal_gather_exact": True,
        },
        "heldout_clip_256": heldout_result,
        "validation_256": validation_result,
        "passed": passed,
        "decision": (
            "v16_1_pass_stop_for_review_before_v17"
            if passed
            else "v16_1_fail_stop_for_review"
        ),
        "training_authorized": False,
        "full_validation_authorized": False,
        "long_training_authorized": False,
        "checkpoint_selection_performed": False,
        "test_set_used": False,
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
