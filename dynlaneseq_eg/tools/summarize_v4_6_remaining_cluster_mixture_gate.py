from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.tools.summarize_v4_5_pointer_trajectory import (
    _trajectory_row,
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_mapping(items: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for item in items:
        iteration, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"invalid ITERATION=REPORT entry: {item!r}")
        result[int(iteration)] = Path(path)
    return result


def _policy_row(report: dict[str, Any]) -> dict[str, Any]:
    audit = report["audit"]
    prefix = audit["teacher_prefix_policy"]
    step_support_hit = {
        step: float(values["candidate_support_hit_rate"])
        for step, values in prefix.items()
    }
    return {
        "target_mode": report["teacher_contract"].get("target_mode"),
        "step_support_hit": step_support_hit,
        # The remaining-cluster mixture changes steps 1--3, where more than
        # one GT cluster can remain.  At step 4 only one cluster remains, so
        # its target is identical to the sampled-cluster V4.5 target.
        "early_step_support_hit_mean": sum(
            step_support_hit[str(step)] for step in range(1, 4)
        )
        / 3.0,
        "step_target_mass": {
            step: float(values["mean_probability_mass_on_candidate_support"])
            for step, values in prefix.items()
        },
        "step_soft_cross_entropy": {
            step: float(values["mean_soft_target_cross_entropy"])
            for step, values in prefix.items()
        },
        "mean_support_size": float(
            audit["teacher_support"]["support_size"]["mean"]
        ),
        "mean_target_entropy": float(
            audit["teacher_support"]["target_entropy"]["mean"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the V4.6 remaining-cluster mixture pointer gate."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--gradient-audit", required=True)
    parser.add_argument("--trajectory", nargs="+", required=True)
    parser.add_argument("--policy-trajectory", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_report = _load(args.source)
    source_metric = _trajectory_row(105000, source_report)
    source_policy = _policy_row(_load(args.source_policy))
    gradient = _load(args.gradient_audit)
    metric_paths = _parse_mapping(args.trajectory)
    policy_paths = _parse_mapping(args.policy_trajectory)
    if set(metric_paths) != set(policy_paths):
        raise ValueError("metric and policy trajectory iterations differ")

    rows: list[dict[str, Any]] = []
    for iteration in sorted(metric_paths):
        row = _trajectory_row(iteration, _load(metric_paths[iteration]))
        row["policy"] = _policy_row(_load(policy_paths[iteration]))
        row["delta"] = {
            "f1_050": row["f1_050"] - source_metric["f1_050"],
            "f1_075": row["f1_075"] - source_metric["f1_075"],
            "early_step_support_hit_mean": (
                row["policy"]["early_step_support_hit_mean"]
                - source_policy["early_step_support_hit_mean"]
            ),
            "step4_support_hit": (
                row["policy"]["step_support_hit"]["4"]
                - source_policy["step_support_hit"]["4"]
            ),
        }
        row["checks"] = {
            "geometry_oracle_050_preserved": abs(
                row["oracle_recall_050"] - source_metric["oracle_recall_050"]
            )
            <= 1e-6,
            "geometry_oracle_075_preserved": abs(
                row["oracle_recall_075"] - source_metric["oracle_recall_075"]
            )
            <= 1e-6,
            "f1_050_gain_at_least_0p30": row["delta"]["f1_050"] >= 0.003,
            "f1_075_not_down_more_than_0p30": row["delta"]["f1_075"] >= -0.003,
            "early_step_support_hit_gain_at_least_5pp": row["delta"][
                "early_step_support_hit_mean"
            ]
            >= 0.05,
            "duplicate_fp_fraction_below_25pct": row[
                "duplicate_fp_fraction_050"
            ]
            < 0.25,
            "variable_cardinality_active": row["mean_selected_per_image"] < 3.95,
        }
        row["passed_metric_gate"] = all(row["checks"].values())
        rows.append(row)

    best = max(
        rows,
        key=lambda row: (float(row["f1_050"]), float(row["f1_075"])),
    )
    passing = [row for row in rows if row["passed_metric_gate"]]
    gradient_passed = bool(gradient.get("passed", False))
    passed = gradient_passed and bool(passing)
    payload = {
        "experiment": (
            "V4.6 teacher-only balanced remaining-cluster mixture pointer gate"
        ),
        "source_iteration": 105000,
        "source_metric": source_metric,
        "source_policy": source_policy,
        "gradient_isolation_passed": gradient_passed,
        "trajectory": rows,
        "best_diagnostic_checkpoint": best,
        "passing_iterations": [row["iteration"] for row in passing],
        "passed": passed,
        "decision": (
            "run_one_full_validation_for_best_passing_checkpoint"
            if passed
            else "do_not_run_test_or_free_rollout; inspect_teacher_policy"
        ),
        "warning": (
            "Uniform-256 validation is a causal gate only. Select at most one "
            "checkpoint for full validation and do not tune on CULane test."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
