from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(report: dict[str, Any], method: str, threshold: str) -> dict[str, Any]:
    return report["methods"][method][threshold]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen-geometry V4.3 pointer/STOP gate."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--pointer", required=True)
    parser.add_argument("--gradient-audit", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    source = _load(args.source)
    pointer = _load(args.pointer)
    gradient = _load(args.gradient_audit)

    source_row_050 = _row(source, "score_top4", "0.50")
    pointer_row_050 = _row(pointer, "pointer_greedy", "0.50")
    pointer_row_075 = _row(pointer, "pointer_greedy", "0.75")
    source_oracle_050 = float(
        source["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
    )
    pointer_oracle_050 = float(
        pointer["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
    )
    source_oracle_075 = float(
        source["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
    )
    pointer_oracle_075 = float(
        pointer["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
    )
    duplicate_fraction = float(
        pointer_row_050["false_positive_breakdown"]["duplicate_fp"][
            "fraction_of_fp"
        ]
    )
    checks = {
        "gradient_isolation_passed": bool(gradient.get("passed", False)),
        "geometry_oracle_050_preserved": abs(
            pointer_oracle_050 - source_oracle_050
        ) <= 0.005,
        "geometry_oracle_075_preserved": abs(
            pointer_oracle_075 - source_oracle_075
        ) <= 0.005,
        "pointer_recall_050_at_least_70pct": float(
            pointer_row_050["recall"]
        ) >= 0.70,
        "pointer_recall_075_at_least_60pct": float(
            pointer_row_075["recall"]
        ) >= 0.60,
        "duplicate_fp_fraction_below_25pct": duplicate_fraction < 0.25,
        "variable_cardinality_active": float(
            pointer_row_050["mean_selected_per_image"]
        ) < 3.95,
    }
    passed = all(checks.values())
    payload = {
        "experiment": "V4.3 frozen-geometry sequential pointer + STOP gate",
        "source": {
            "f1_050": float(source_row_050["f1"]),
            "recall_050": float(source_row_050["recall"]),
            "oracle_recall_050": source_oracle_050,
            "oracle_recall_075": source_oracle_075,
        },
        "pointer": {
            "f1_050": float(pointer_row_050["f1"]),
            "precision_050": float(pointer_row_050["precision"]),
            "recall_050": float(pointer_row_050["recall"]),
            "f1_075": float(pointer_row_075["f1"]),
            "recall_075": float(pointer_row_075["recall"]),
            "mean_selected_per_image": float(
                pointer_row_050["mean_selected_per_image"]
            ),
            "duplicate_fp_fraction_050": duplicate_fraction,
            "oracle_recall_050": pointer_oracle_050,
            "oracle_recall_075": pointer_oracle_075,
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "pointer_stop_contract_passed_run_full_validation"
            if passed
            else "pointer_stop_contract_failed_do_not_start_278k"
        ),
        "warning": (
            "Uniform diagnostic subset only. A pass authorizes full-validation "
            "evaluation, not a benchmark claim or a 278k run."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
