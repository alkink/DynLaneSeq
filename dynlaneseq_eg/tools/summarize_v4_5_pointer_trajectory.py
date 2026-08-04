from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _method(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    return report["methods"]["pointer_greedy"][threshold]


def _oracle(report: dict[str, Any], threshold: str) -> float:
    return float(
        report["capacity"][threshold]["all_candidate_oracle"]["recall"]
    )


def _trajectory_row(iteration: int, report: dict[str, Any]) -> dict[str, Any]:
    row_050 = _method(report, "0.50")
    row_075 = _method(report, "0.75")
    return {
        "iteration": int(iteration),
        "f1_050": float(row_050["f1"]),
        "precision_050": float(row_050["precision"]),
        "recall_050": float(row_050["recall"]),
        "f1_075": float(row_075["f1"]),
        "precision_075": float(row_075["precision"]),
        "recall_075": float(row_075["recall"]),
        "mean_selected_per_image": float(row_050["mean_selected_per_image"]),
        "duplicate_fp_fraction_050": float(
            row_050["false_positive_breakdown"]["duplicate_fp"][
                "fraction_of_fp"
            ]
        ),
        "oracle_recall_050": _oracle(report, "0.50"),
        "oracle_recall_075": _oracle(report, "0.75"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V4.5 cluster-soft trajectory."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--gradient-audit", required=True)
    parser.add_argument(
        "--trajectory",
        nargs="+",
        required=True,
        help="Entries formatted as ITERATION=REPORT.json",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _load(args.source)
    gradient = _load(args.gradient_audit)
    rows: list[dict[str, Any]] = []
    for item in args.trajectory:
        iteration_text, separator, report_path = item.partition("=")
        if not separator:
            raise ValueError(f"invalid trajectory item: {item!r}")
        rows.append(_trajectory_row(int(iteration_text), _load(report_path)))
    rows.sort(key=lambda row: int(row["iteration"]))
    source_oracle_050 = _oracle(source, "0.50")
    source_oracle_075 = _oracle(source, "0.75")

    for row in rows:
        row["checks"] = {
            "geometry_oracle_050_preserved": abs(
                float(row["oracle_recall_050"]) - source_oracle_050
            )
            <= 0.005,
            "geometry_oracle_075_preserved": abs(
                float(row["oracle_recall_075"]) - source_oracle_075
            )
            <= 0.005,
            "f1_050_at_least_081": float(row["f1_050"]) >= 0.81,
            "f1_075_at_least_0605": float(row["f1_075"]) >= 0.605,
            "duplicate_fp_fraction_below_25pct": float(
                row["duplicate_fp_fraction_050"]
            )
            < 0.25,
            "variable_cardinality_active": float(
                row["mean_selected_per_image"]
            )
            < 3.95,
        }
        row["passed_metric_gate"] = all(row["checks"].values())

    eligible = [row for row in rows if bool(row["passed_metric_gate"])]
    best = max(
        rows,
        key=lambda row: (float(row["f1_050"]), float(row["f1_075"])),
    )
    passed = bool(gradient.get("passed", False)) and bool(eligible)
    payload = {
        "experiment": "V4.5 GT-cluster soft randomized-teacher pointer gate",
        "source_geometry_oracle": {
            "recall_050": source_oracle_050,
            "recall_075": source_oracle_075,
        },
        "gradient_isolation_passed": bool(gradient.get("passed", False)),
        "trajectory": rows,
        "best_diagnostic_checkpoint": best,
        "passing_iterations": [
            int(row["iteration"]) for row in eligible
        ],
        "passed": passed,
        "decision": (
            "cluster_soft_pointer_passed_run_full_validation"
            if passed
            else "cluster_soft_pointer_failed_do_not_run_test_or_278k"
        ),
        "warning": (
            "Uniform-256 validation diagnostic only. A pass authorizes one "
            "full-validation run of the selected checkpoint, never test-set "
            "tuning or a 278k claim."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
