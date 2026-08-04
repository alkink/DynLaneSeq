from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V4.4 set-oracle pointer gate."
    )
    parser.add_argument("--metric-summary", required=True)
    parser.add_argument("--target-alignment", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def summarize_gate(
    metric_summary: dict[str, Any],
    target_alignment: dict[str, Any],
) -> dict[str, Any]:
    pointer = metric_summary["pointer"]
    base_checks = metric_summary["checks"]
    analysis = target_alignment["analysis"]
    rollout = analysis["pointer_rollout_dynamics"]
    first_step = rollout["lane_images_first_step"]
    checks = {
        "gradient_isolation_passed": bool(
            base_checks["gradient_isolation_passed"]
        ),
        "geometry_oracle_050_preserved": bool(
            base_checks["geometry_oracle_050_preserved"]
        ),
        "geometry_oracle_075_preserved": bool(
            base_checks["geometry_oracle_075_preserved"]
        ),
        "duplicate_fp_fraction_below_25pct": bool(
            base_checks["duplicate_fp_fraction_below_25pct"]
        ),
        "direct_recall_050_at_least_75pct": float(pointer["recall_050"])
        >= 0.75,
        "direct_recall_075_at_least_60pct": float(pointer["recall_075"])
        >= 0.60,
        "first_step_any_target_at_least_40pct": float(
            first_step["any_target_representative_rate"]
        )
        >= 0.40,
        "pointer_target_jaccard_at_least_035": float(
            analysis["pointer_target_set_jaccard"]
        )
        >= 0.35,
        "exact_target_set_rate_at_least_20pct": float(
            analysis["exact_pointer_target_set_rate"]
        )
        >= 0.20,
        "variable_cardinality_active": bool(
            base_checks["variable_cardinality_active"]
        ),
    }
    passed = all(checks.values())
    return {
        "experiment": "V4.4 permutation-invariant set-oracle pointer gate",
        "metrics": {
            "f1_050": float(pointer["f1_050"]),
            "recall_050": float(pointer["recall_050"]),
            "f1_075": float(pointer["f1_075"]),
            "recall_075": float(pointer["recall_075"]),
            "mean_selected_per_image": float(
                pointer["mean_selected_per_image"]
            ),
            "first_step_any_target_representative_rate": float(
                first_step["any_target_representative_rate"]
            ),
            "pointer_target_set_jaccard": float(
                analysis["pointer_target_set_jaccard"]
            ),
            "exact_pointer_target_set_rate": float(
                analysis["exact_pointer_target_set_rate"]
            ),
            "fraction_of_failures_starting_at_step1": float(
                rollout["fraction_of_unordered_failures_starting_at_step1"]
            ),
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "set_oracle_pointer_passed_run_full_validation"
            if passed
            else "set_oracle_pointer_failed_do_not_run_full_validation_or_278k"
        ),
        "warning": (
            "Uniform-256 diagnostic gate only. Test remains sealed; a pass "
            "authorizes full validation, not a benchmark claim."
        ),
    }


def main() -> None:
    args = parse_args()
    metric_summary = json.loads(
        Path(args.metric_summary).read_text(encoding="utf-8")
    )
    target_alignment = json.loads(
        Path(args.target_alignment).read_text(encoding="utf-8")
    )
    report = summarize_gate(metric_summary, target_alignment)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
