from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pointer_metrics(report: dict[str, Any]) -> dict[str, float]:
    row_050 = report["methods"]["pointer_greedy"]["0.50"]
    row_075 = report["methods"]["pointer_greedy"]["0.75"]
    return {
        "f1_050": float(row_050["f1"]),
        "precision_050": float(row_050["precision"]),
        "recall_050": float(row_050["recall"]),
        "f1_075": float(row_075["f1"]),
        "precision_075": float(row_075["precision"]),
        "recall_075": float(row_075["recall"]),
        "mean_selected_per_image": float(row_050["mean_selected_per_image"]),
        "duplicate_fp_fraction_050": float(
            row_050["false_positive_breakdown"]["duplicate_fp"]["fraction_of_fp"]
        ),
        "oracle_recall_050": float(
            report["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        ),
        "oracle_recall_075": float(
            report["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        ),
    }


def summarize(
    baseline_pointer: dict[str, Any],
    geometry_trajectory: dict[str, Any],
    mature_pointer: dict[str, Any],
) -> dict[str, Any]:
    baseline = _pointer_metrics(baseline_pointer)
    geometry_rows = sorted(
        geometry_trajectory["rows"], key=lambda row: int(row["iteration"])
    )
    if len(geometry_rows) < 2:
        raise ValueError("geometry trajectory must contain source and mature rows")
    source_geometry = geometry_rows[0]
    mature_geometry = geometry_rows[-1]
    source_iteration = int(source_geometry["iteration"])
    mature_iteration = int(mature_geometry["iteration"])
    source_oracle_050 = float(source_geometry["all_candidates_recall_050"])
    source_oracle_075 = float(source_geometry["all_candidates_recall_075"])
    mature_oracle_050 = float(mature_geometry["all_candidates_recall_050"])
    mature_oracle_075 = float(mature_geometry["all_candidates_recall_075"])
    geometry_checks = {
        "source_iteration_is_50k": source_iteration == 50000,
        "mature_iteration_is_at_least_125k": mature_iteration >= 125000,
        "oracle_050_has_no_catastrophic_regression": (
            mature_oracle_050 >= source_oracle_050 - 0.01
        ),
        "oracle_075_has_no_catastrophic_regression": (
            mature_oracle_075 >= source_oracle_075 - 0.01
        ),
    }

    rows: list[dict[str, Any]] = []
    for raw in mature_pointer["trajectory"]:
        row = dict(raw)
        row["delta_f1_050_vs_50k_geometry_pointer"] = (
            float(row["f1_050"]) - baseline["f1_050"]
        )
        row["delta_f1_075_vs_50k_geometry_pointer"] = (
            float(row["f1_075"]) - baseline["f1_075"]
        )
        checks = {
            "mature_geometry_oracle_050_preserved": abs(
                float(row["oracle_recall_050"]) - mature_oracle_050
            )
            <= 0.005,
            "mature_geometry_oracle_075_preserved": abs(
                float(row["oracle_recall_075"]) - mature_oracle_075
            )
            <= 0.005,
            "f1_050_not_regressed_vs_50k_pointer": (
                float(row["f1_050"]) >= baseline["f1_050"] - 0.002
            ),
            "f1_075_not_regressed_vs_50k_pointer": (
                float(row["f1_075"]) >= baseline["f1_075"] - 0.002
            ),
            "material_pointer_gain": (
                row["delta_f1_050_vs_50k_geometry_pointer"] >= 0.003
                or row["delta_f1_075_vs_50k_geometry_pointer"] >= 0.003
            ),
            "duplicate_fp_fraction_below_25pct": (
                float(row["duplicate_fp_fraction_050"]) < 0.25
            ),
            "variable_cardinality_active": (
                float(row["mean_selected_per_image"]) < 3.95
            ),
        }
        row["long_geometry_checks"] = checks
        row["passed_long_geometry_pointer_gate"] = all(checks.values())
        rows.append(row)

    eligible = [row for row in rows if row["passed_long_geometry_pointer_gate"]]
    winner = (
        max(
            eligible,
            key=lambda row: (
                float(row["delta_f1_050_vs_50k_geometry_pointer"])
                + float(row["delta_f1_075_vs_50k_geometry_pointer"]),
                float(row["f1_050"]),
            ),
        )
        if eligible
        else None
    )
    gradient_passed = bool(mature_pointer.get("gradient_isolation_passed", False))
    passed = all(geometry_checks.values()) and gradient_passed and winner is not None
    return {
        "experiment": "V4 long-geometry 50k-to-125k plus fresh V4.5 pointer gate",
        "baseline_50k_geometry_v4_5_pointer": baseline,
        "geometry": {
            "source_iteration": source_iteration,
            "mature_iteration": mature_iteration,
            "source_oracle_recall_050": source_oracle_050,
            "source_oracle_recall_075": source_oracle_075,
            "mature_oracle_recall_050": mature_oracle_050,
            "mature_oracle_recall_075": mature_oracle_075,
            "delta_oracle_recall_050": mature_oracle_050 - source_oracle_050,
            "delta_oracle_recall_075": mature_oracle_075 - source_oracle_075,
            "checks": geometry_checks,
            "trajectory": geometry_rows,
        },
        "mature_pointer_gradient_isolation_passed": gradient_passed,
        "mature_pointer_trajectory": rows,
        "winner": winner,
        "passed": passed,
        "decision": (
            "run_one_full_validation_for_mature_geometry_pointer"
            if passed
            else "do_not_run_test_review_geometry_and_pointer_trajectory"
        ),
        "warning": (
            "Uniform-256 is a validation diagnostic. A pass authorizes one "
            "full-validation run, never checkpoint selection on test."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a mature-geometry V4.5 pointer against the 50k baseline."
    )
    parser.add_argument("--baseline-pointer-report", required=True)
    parser.add_argument("--geometry-trajectory", required=True)
    parser.add_argument("--mature-pointer-summary", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline-pointer-checkpoint", required=True)
    parser.add_argument("--mature-geometry-checkpoint", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        _load(args.baseline_pointer_report),
        _load(args.geometry_trajectory),
        _load(args.mature_pointer_summary),
    )
    payload["provenance"] = {
        "git_commit": args.git_commit,
        "pointer_seed": args.seed,
        "baseline_pointer_checkpoint": args.baseline_pointer_checkpoint,
        "mature_geometry_checkpoint": args.mature_geometry_checkpoint,
        "baseline_pointer_report": args.baseline_pointer_report,
        "geometry_trajectory": args.geometry_trajectory,
        "mature_pointer_summary": args.mature_pointer_summary,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
