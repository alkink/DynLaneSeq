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
        "early_step_support_hit_mean": sum(
            step_support_hit[str(step)] for step in range(1, 4)
        )
        / 3.0,
        "step_target_mass": {
            step: float(values["mean_probability_mass_on_candidate_support"])
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
        description="Summarize the V4.6.1 low-LR pointer stability gate."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--aggressive-summary", required=True)
    parser.add_argument("--gradient-audit", required=True)
    parser.add_argument("--trajectory", nargs="+", required=True)
    parser.add_argument("--policy-trajectory", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_metric = _trajectory_row(105000, _load(args.source))
    source_policy = _policy_row(_load(args.source_policy))
    aggressive = _load(args.aggressive_summary)
    aggressive_best = aggressive["best_diagnostic_checkpoint"]
    gradient = _load(args.gradient_audit)
    metric_paths = _parse_mapping(args.trajectory)
    policy_paths = _parse_mapping(args.policy_trajectory)
    if set(metric_paths) != set(policy_paths):
        raise ValueError("metric and policy trajectory iterations differ")
    if sorted(metric_paths) != list(range(106000, 115001, 1000)):
        raise ValueError(
            "V4.6.1 requires the complete 106k--115k trajectory at 1k steps"
        )

    rows: list[dict[str, Any]] = []
    for iteration in sorted(metric_paths):
        row = _trajectory_row(iteration, _load(metric_paths[iteration]))
        row["policy"] = _policy_row(_load(policy_paths[iteration]))
        row["delta_from_source"] = {
            "f1_050": row["f1_050"] - source_metric["f1_050"],
            "f1_075": row["f1_075"] - source_metric["f1_075"],
            "early_step_support_hit_mean": (
                row["policy"]["early_step_support_hit_mean"]
                - source_policy["early_step_support_hit_mean"]
            ),
        }
        rows.append(row)

    best = max(
        rows,
        key=lambda row: (float(row["f1_050"]), float(row["f1_075"])),
    )
    final = rows[-1]
    tail = rows[-3:]
    aggressive_f1_050 = float(aggressive_best["f1_050"])
    aggressive_f1_075 = float(aggressive_best["f1_075"])
    best_f1_050 = float(best["f1_050"])

    checks = {
        "gradient_isolation_passed": bool(gradient.get("passed", False)),
        "all_geometry_oracles_preserved": all(
            abs(row["oracle_recall_050"] - source_metric["oracle_recall_050"])
            <= 1e-6
            and abs(
                row["oracle_recall_075"] - source_metric["oracle_recall_075"]
            )
            <= 1e-6
            for row in rows
        ),
        # A stable recipe may trade at most 0.30 F1 point against the short,
        # aggressive V4.6 peak; it must not buy stability by giving up the
        # causal gain.
        "final_f1_050_within_0p30_of_aggressive_peak": (
            float(final["f1_050"]) >= aggressive_f1_050 - 0.003
        ),
        "final_f1_075_within_0p50_of_aggressive_peak": (
            float(final["f1_075"]) >= aggressive_f1_075 - 0.005
        ),
        "final_early_support_hit_gain_at_least_3pp": (
            final["delta_from_source"]["early_step_support_hit_mean"] >= 0.03
        ),
        # The final 3k steps must form a genuine plateau instead of merely
        # containing another narrow checkpoint spike.
        "last_three_checkpoints_within_0p30_of_best": all(
            float(row["f1_050"]) >= best_f1_050 - 0.003 for row in tail
        ),
        "duplicates_controlled_throughout": all(
            float(row["duplicate_fp_fraction_050"]) < 0.25 for row in rows
        ),
        "variable_cardinality_active_throughout": all(
            float(row["mean_selected_per_image"]) < 3.95 for row in rows
        ),
    }
    passed = all(checks.values())
    payload = {
        "experiment": "V4.6.1 low-LR local-cosine pointer stability gate",
        "source_iteration": 105000,
        "source_metric": source_metric,
        "source_policy": source_policy,
        "aggressive_v4_6_reference": {
            "iteration": int(aggressive_best["iteration"]),
            "f1_050": aggressive_f1_050,
            "f1_075": aggressive_f1_075,
        },
        "optimizer_contract": {
            "fresh_adamw": True,
            "set_selection_peak_lr": 3e-5,
            "semantic_peak_lr": 1e-5,
            "local_warmup_steps": 500,
            "local_cosine_steps": 10000,
            "min_lr_ratio": 0.10,
        },
        "trajectory": rows,
        "best_diagnostic_checkpoint": best,
        "final_checkpoint": final,
        "tail_plateau_iterations": [int(row["iteration"]) for row in tail],
        "checks": checks,
        "passed": passed,
        # If the stability contract passes, use the predeclared final
        # checkpoint rather than selecting a narrow uniform-subset maximum.
        "selected_full_validation_iteration": 115000 if passed else None,
        "decision": (
            "run_one_full_validation_for_final_115k_checkpoint"
            if passed
            else "do_not_run_test; diagnose_lr_or_teacher_policy"
        ),
        "warning": (
            "Uniform-256 is a stability diagnostic. A pass authorizes one "
            "full-validation run of the predeclared final checkpoint, never "
            "test-set tuning."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
