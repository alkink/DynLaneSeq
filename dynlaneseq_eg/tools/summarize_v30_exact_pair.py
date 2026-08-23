from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and gate the V30 field-only exact-paired run."
    )
    parser.add_argument("--control-metrics", required=True)
    parser.add_argument("--treatment-metrics", required=True)
    parser.add_argument("--control-coverage", required=True)
    parser.add_argument("--treatment-coverage", required=True)
    parser.add_argument("--paired-audit", required=True)
    parser.add_argument("--transition-audit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], threshold: float) -> dict[str, float]:
    results = report.get("results", report)
    row = results[str(float(threshold))]
    return {
        "f1": float(row.get("F1", row.get("f1"))),
        "precision": float(row.get("Precision", row.get("precision"))),
        "recall": float(row.get("Recall", row.get("recall"))),
        "tp": int(row.get("TP", row.get("tp"))),
        "fp": int(row.get("FP", row.get("fp"))),
        "fn": int(row.get("FN", row.get("fn"))),
    }


def _oracle_hits(report: dict[str, Any], threshold: float) -> int:
    row = report["capacity"][f"{threshold:.2f}"]["all_candidate_oracle"]
    return int(row["hits"])


def main() -> None:
    args = parse_args()
    control_metrics = _read(args.control_metrics)
    treatment_metrics = _read(args.treatment_metrics)
    control_coverage = _read(args.control_coverage)
    treatment_coverage = _read(args.treatment_coverage)
    paired = _read(args.paired_audit)
    transitions = _read(args.transition_audit)

    thresholds: dict[str, Any] = {}
    for threshold in (0.5, 0.75):
        key = str(float(threshold))
        control = _metric(control_metrics, threshold)
        treatment = _metric(treatment_metrics, threshold)
        clip_bootstrap = paired["thresholds"][key]["full_validation"][
            "paired_clip_bootstrap_f1_delta"
        ]
        thresholds[key] = {
            "control": control,
            "treatment": treatment,
            "delta": {
                name: float(treatment[name]) - float(control[name])
                for name in ("f1", "precision", "recall", "tp", "fp", "fn")
            },
            "clip_bootstrap": clip_bootstrap,
            "gt_transitions": transitions["thresholds"][key],
            "uniform256_all32_oracle_hits": {
                "control": _oracle_hits(control_coverage, threshold),
                "treatment": _oracle_hits(treatment_coverage, threshold),
            },
        }

    checks = {
        "f1_050_gain_at_least_0p60_points": thresholds["0.5"]["delta"][
            "f1"
        ]
        >= 0.006,
        "f1_075_non_regression": thresholds["0.75"]["delta"]["f1"]
        >= 0.0,
        "clip_bootstrap_050_lower_bound_positive": thresholds["0.5"][
            "clip_bootstrap"
        ]["ci_2p5"]
        > 0.0,
        "oracle_050_non_regression": thresholds["0.5"][
            "uniform256_all32_oracle_hits"
        ]["treatment"]
        >= thresholds["0.5"]["uniform256_all32_oracle_hits"]["control"],
        "oracle_075_non_regression": thresholds["0.75"][
            "uniform256_all32_oracle_hits"
        ]["treatment"]
        >= thresholds["0.75"]["uniform256_all32_oracle_hits"]["control"],
        "paired_audit_reproduces_reports": all(
            row["source_exact"] and row["candidate_exact"]
            for row in paired["report_reproduction"].values()
        ),
        "test_split_closed": not bool(paired.get("test_set_used", False))
        and not bool(transitions.get("test_set_used", False)),
    }
    report = {
        "experiment": "V30 field-only exact-paired 30K-to-35K causal gate",
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "resume_exact_pair_to_long_horizon"
            if all(checks.values())
            else "stop_and_reassess_field_only_effect"
        ),
        "test_split_used": False,
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
