from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], threshold: float) -> dict[str, float | int]:
    row = report["results"][str(float(threshold))]
    return {
        "f1": float(row["F1"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "tp": int(row["TP"]),
        "fp": int(row["FP"]),
        "fn": int(row["FN"]),
    }


def _oracle_hits(report: dict[str, Any], threshold: float) -> int:
    return int(
        report["capacity"][f"{threshold:.2f}"]["all_candidate_oracle"]["hits"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize exact-paired V7 versus V30 Field-only at 50K."
    )
    parser.add_argument("--v7-metrics", required=True)
    parser.add_argument("--v30-metrics", required=True)
    parser.add_argument("--historical-v7-metrics", default="")
    parser.add_argument("--v7-coverage", required=True)
    parser.add_argument("--v30-coverage", required=True)
    parser.add_argument("--paired-audit", required=True)
    parser.add_argument("--transition-audit", required=True)
    parser.add_argument("--pair-contract", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    v7_metrics = _read(args.v7_metrics)
    v30_metrics = _read(args.v30_metrics)
    historical = _read(args.historical_v7_metrics) if args.historical_v7_metrics else None
    v7_coverage = _read(args.v7_coverage)
    v30_coverage = _read(args.v30_coverage)
    paired = _read(args.paired_audit)
    transitions = _read(args.transition_audit)
    contract = _read(args.pair_contract)

    thresholds: dict[str, Any] = {}
    for threshold in (0.50, 0.75):
        key = str(float(threshold))
        v7 = _metric(v7_metrics, threshold)
        v30 = _metric(v30_metrics, threshold)
        delta = {
            name: float(v30[name]) - float(v7[name])
            for name in ("f1", "precision", "recall", "tp", "fp", "fn")
        }
        transition = transitions["thresholds"][key]
        thresholds[key] = {
            "exact_v7": v7,
            "v30_field_only": v30,
            "v30_minus_v7": delta,
            "v30_minus_v7_f1_points": 100.0 * float(delta["f1"]),
            "historical_v7_diagnostic": (
                _metric(historical, threshold) if historical is not None else None
            ),
            "paired_clip_bootstrap": paired["thresholds"][key][
                "full_validation"
            ]["paired_clip_bootstrap_f1_delta"],
            "gt_transitions": transition,
            "uniform256_all32_oracle_hits": {
                "exact_v7": _oracle_hits(v7_coverage, threshold),
                "v30_field_only": _oracle_hits(v30_coverage, threshold),
            },
        }

    checks = {
        "pair_contract_passed": contract.get("passed") is True,
        "f1_050_gain_at_least_0p50_points": thresholds["0.5"][
            "v30_minus_v7"
        ]["f1"] >= 0.005,
        "f1_075_gain_at_least_0p30_points": thresholds["0.75"][
            "v30_minus_v7"
        ]["f1"] >= 0.003,
        "clip_bootstrap_050_lower_bound_positive": thresholds["0.5"][
            "paired_clip_bootstrap"
        ]["ci_2p5"] > 0.0,
        "oracle_050_non_regression": thresholds["0.5"][
            "uniform256_all32_oracle_hits"
        ]["v30_field_only"] >= thresholds["0.5"]["uniform256_all32_oracle_hits"]["exact_v7"],
        "oracle_075_non_regression": thresholds["0.75"][
            "uniform256_all32_oracle_hits"
        ]["v30_field_only"] >= thresholds["0.75"]["uniform256_all32_oracle_hits"]["exact_v7"],
        "paired_audit_reproduces_reports": all(
            row["source_exact"] and row["candidate_exact"]
            for row in paired["report_reproduction"].values()
        ),
        "test_split_closed": not bool(paired.get("test_set_used", False))
        and not bool(transitions.get("test_set_used", False))
        and not bool(contract.get("test_split_used", False)),
    }
    passed = all(checks.values())
    output = {
        "experiment": "V30 Field-only versus exact-paired V7 at global 50K",
        "contract": {
            "common_v7_30k_ancestor": True,
            "paired_30k_to35k_history": True,
            "identical_35k_to50k_sample_and_augmentation_stream": True,
            "only_training_objective_difference": "joint slot-row field auxiliary supervision",
            "test_split_used": False,
        },
        "thresholds": thresholds,
        "checks": checks,
        "passed": passed,
        "decision": (
            "field_only_authorized_for_longer_validation"
            if passed
            else "field_only_not_authorized_as_50k_replacement_for_v7"
        ),
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
