from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pointer_row(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    return report["methods"]["pointer_greedy"][threshold]


def _oracle(report: dict[str, Any], threshold: str) -> float:
    return float(report["capacity"][threshold]["all_candidate_oracle"]["recall"])


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    row_050 = _pointer_row(report, "0.50")
    row_075 = _pointer_row(report, "0.75")
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
        "oracle_recall_050": _oracle(report, "0.50"),
        "oracle_recall_075": _oracle(report, "0.75"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V4.5.1 quality-policy Q1/Q2 gate."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline-gradient-audit", required=True)
    parser.add_argument(
        "--gradient-audit",
        nargs="+",
        required=True,
        help="Entries formatted as ARM=REPORT.json",
    )
    parser.add_argument(
        "--trajectory",
        nargs="+",
        required=True,
        help="Entries formatted as ARM:ITERATION=REPORT.json",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_report = _load(args.source)
    source = _metrics(source_report)
    baseline_audit = _load(args.baseline_gradient_audit)
    audits: dict[str, dict[str, Any]] = {}
    for item in args.gradient_audit:
        arm, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"invalid gradient-audit item: {item!r}")
        audits[arm] = _load(path)

    trajectories: dict[str, list[dict[str, Any]]] = {}
    for item in args.trajectory:
        label, separator, path = item.partition("=")
        arm, colon, iteration_text = label.partition(":")
        if not separator or not colon:
            raise ValueError(f"invalid trajectory item: {item!r}")
        row = {"iteration": int(iteration_text), **_metrics(_load(path))}
        row["delta_f1_050_vs_source"] = row["f1_050"] - source["f1_050"]
        row["delta_f1_075_vs_source"] = row["f1_075"] - source["f1_075"]
        row["checks"] = {
            "geometry_oracle_050_preserved": abs(
                row["oracle_recall_050"] - source["oracle_recall_050"]
            )
            <= 0.005,
            "geometry_oracle_075_preserved": abs(
                row["oracle_recall_075"] - source["oracle_recall_075"]
            )
            <= 0.005,
            "f1_050_not_regressed": row["f1_050"] >= source["f1_050"] - 0.002,
            "f1_075_not_regressed": row["f1_075"] >= source["f1_075"] - 0.002,
            "material_gain_on_at_least_one_iou": (
                row["delta_f1_050_vs_source"] >= 0.003
                or row["delta_f1_075_vs_source"] >= 0.003
            ),
            "duplicate_fp_fraction_below_25pct": row[
                "duplicate_fp_fraction_050"
            ]
            < 0.25,
            "variable_cardinality_active": row["mean_selected_per_image"] < 3.95,
        }
        row["passed_metric_gate"] = all(row["checks"].values())
        trajectories.setdefault(arm, []).append(row)

    arm_summaries: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[str, dict[str, Any]]] = []
    for arm, rows in trajectories.items():
        rows.sort(key=lambda row: int(row["iteration"]))
        best = max(
            rows,
            key=lambda row: (
                float(row["f1_050"]),
                float(row["f1_075"]),
            ),
        )
        audit_passed = bool(audits.get(arm, {}).get("passed", False))
        passed = audit_passed and any(
            bool(row["passed_metric_gate"]) for row in rows
        )
        arm_summaries[arm] = {
            "gradient_contract_passed": audit_passed,
            "trajectory": rows,
            "best_checkpoint": best,
            "passed": passed,
        }
        if passed:
            for row in rows:
                if bool(row["passed_metric_gate"]):
                    eligible.append((arm, row))

    winner = (
        max(
            eligible,
            key=lambda item: (
                float(item[1]["f1_050"]),
                float(item[1]["f1_075"]),
            ),
        )
        if eligible
        else None
    )
    baseline_audit_passed = bool(baseline_audit.get("passed", False))
    passed = baseline_audit_passed and winner is not None
    payload = {
        "experiment": "V4.5.1 detached quality-prior and cluster-listwise gate",
        "source_v4_5": source,
        "baseline_v4_5_gradient_audit": {
            "passed": baseline_audit_passed,
            "sequence_vs_quality_weighted_cosine": baseline_audit.get(
                "gradient_cosines", {}
            ).get("sequence_vs_quality_weighted"),
            "gradient_groups": baseline_audit.get("gradient_groups", {}),
        },
        "arms": arm_summaries,
        "winner": (
            {"arm": winner[0], **winner[1]} if winner is not None else None
        ),
        "passed": passed,
        "decision": (
            "run_one_full_validation_for_selected_checkpoint"
            if passed
            else "keep_v4_5_and_do_not_run_test"
        ),
        "warning": (
            "Uniform-256 is a causal gate. It may select one checkpoint for "
            "full validation, but must not be used for further test-set tuning."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output.resolve()}")


if __name__ == "__main__":
    main()
