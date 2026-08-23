from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = (0.50, 0.75)


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], threshold: float) -> dict[str, float | int]:
    row = report.get("results", report)[str(float(threshold))]
    return {
        "f1": float(row.get("F1", row.get("f1"))),
        "precision": float(row.get("Precision", row.get("precision"))),
        "recall": float(row.get("Recall", row.get("recall"))),
        "tp": int(row.get("TP", row.get("tp"))),
        "fp": int(row.get("FP", row.get("fp"))),
        "fn": int(row.get("FN", row.get("fn"))),
    }


def _support(report: dict[str, Any], threshold: float) -> dict[str, float | int]:
    row = report["capacity"][f"{threshold:.2f}"]["all_candidate_oracle"]
    return {"hits": int(row["hits"]), "recall": float(row["recall"])}


def _comparison(
    *,
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    paired: dict[str, Any],
    transitions: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    key = str(float(threshold))
    source = _metric(source_metrics, threshold)
    candidate = _metric(candidate_metrics, threshold)
    delta = {
        name: float(candidate[name]) - float(source[name])
        for name in ("f1", "precision", "recall", "tp", "fp", "fn")
    }
    return {
        "source": source,
        "candidate": candidate,
        "delta": delta,
        "delta_f1_points": 100.0 * float(delta["f1"]),
        "paired_clip_bootstrap": paired["thresholds"][key][
            "full_validation"
        ]["paired_clip_bootstrap_f1_delta"],
        "gt_transitions": transitions["thresholds"][key],
    }


def _paired_reproduction_exact(report: dict[str, Any]) -> bool:
    return all(
        bool(row["source_exact"]) and bool(row["candidate_exact"])
        for row in report["report_reproduction"].values()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the fixed-endpoint V32 field and field+bridge pulse "
            "consolidation experiment at global iteration 50K."
        )
    )
    parser.add_argument("--v7-metrics", required=True)
    parser.add_argument("--field-pulse-metrics", required=True)
    parser.add_argument("--bridge-pulse-metrics", required=True)
    parser.add_argument("--v7-coverage", required=True)
    parser.add_argument("--field-pulse-coverage", required=True)
    parser.add_argument("--bridge-pulse-coverage", required=True)
    parser.add_argument("--v7-vs-field-paired", required=True)
    parser.add_argument("--v7-vs-bridge-paired", required=True)
    parser.add_argument("--field-vs-bridge-paired", required=True)
    parser.add_argument("--v7-vs-field-transitions", required=True)
    parser.add_argument("--v7-vs-bridge-transitions", required=True)
    parser.add_argument("--field-vs-bridge-transitions", required=True)
    parser.add_argument("--pair-contract", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = {
        "v7": _read(args.v7_metrics),
        "field_pulse": _read(args.field_pulse_metrics),
        "bridge_pulse": _read(args.bridge_pulse_metrics),
    }
    coverage = {
        "v7": _read(args.v7_coverage),
        "field_pulse": _read(args.field_pulse_coverage),
        "bridge_pulse": _read(args.bridge_pulse_coverage),
    }
    paired = {
        "v7_vs_field": _read(args.v7_vs_field_paired),
        "v7_vs_bridge": _read(args.v7_vs_bridge_paired),
        "field_vs_bridge": _read(args.field_vs_bridge_paired),
    }
    transitions = {
        "v7_vs_field": _read(args.v7_vs_field_transitions),
        "v7_vs_bridge": _read(args.v7_vs_bridge_transitions),
        "field_vs_bridge": _read(args.field_vs_bridge_transitions),
    }
    contract = _read(args.pair_contract)

    threshold_report: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = str(float(threshold))
        threshold_report[key] = {
            "metrics": {
                name: _metric(report, threshold)
                for name, report in metrics.items()
            },
            "all32_official_raster_support": {
                name: _support(report, threshold)
                for name, report in coverage.items()
            },
            "comparisons": {
                "field_pulse_minus_v7": _comparison(
                    source_metrics=metrics["v7"],
                    candidate_metrics=metrics["field_pulse"],
                    paired=paired["v7_vs_field"],
                    transitions=transitions["v7_vs_field"],
                    threshold=threshold,
                ),
                "bridge_pulse_minus_v7": _comparison(
                    source_metrics=metrics["v7"],
                    candidate_metrics=metrics["bridge_pulse"],
                    paired=paired["v7_vs_bridge"],
                    transitions=transitions["v7_vs_bridge"],
                    threshold=threshold,
                ),
                "bridge_pulse_minus_field_pulse": _comparison(
                    source_metrics=metrics["field_pulse"],
                    candidate_metrics=metrics["bridge_pulse"],
                    paired=paired["field_vs_bridge"],
                    transitions=transitions["field_vs_bridge"],
                    threshold=threshold,
                ),
            },
        }

    field_050 = threshold_report["0.5"]["comparisons"][
        "field_pulse_minus_v7"
    ]
    field_075 = threshold_report["0.75"]["comparisons"][
        "field_pulse_minus_v7"
    ]
    bridge_v7_050 = threshold_report["0.5"]["comparisons"][
        "bridge_pulse_minus_v7"
    ]
    bridge_v7_075 = threshold_report["0.75"]["comparisons"][
        "bridge_pulse_minus_v7"
    ]
    bridge_field_050 = threshold_report["0.5"]["comparisons"][
        "bridge_pulse_minus_field_pulse"
    ]
    bridge_field_075 = threshold_report["0.75"]["comparisons"][
        "bridge_pulse_minus_field_pulse"
    ]

    common_checks = {
        "pair_contract_passed": contract.get("passed") is True,
        "paired_reports_reproduce_metrics": all(
            _paired_reproduction_exact(report) for report in paired.values()
        ),
        "test_split_closed": not bool(contract.get("test_split_used", False))
        and all(not bool(report.get("test_set_used", False)) for report in paired.values())
        and all(
            not bool(report.get("test_set_used", False))
            for report in transitions.values()
        ),
    }
    field_checks = {
        "f1_050_gain_at_least_0p30_points": field_050["delta"]["f1"]
        >= 0.003,
        "f1_075_non_regression": field_075["delta"]["f1"] >= 0.0,
        "clip_positive_probability_at_least_0p90_both": min(
            field_050["paired_clip_bootstrap"]["probability_delta_positive"],
            field_075["paired_clip_bootstrap"]["probability_delta_positive"],
        )
        >= 0.90,
        "recovered_to_lost_above_one_both": min(
            field_050["gt_transitions"]["recovered_to_lost_ratio"],
            field_075["gt_transitions"]["recovered_to_lost_ratio"],
        )
        > 1.0,
        "all32_support_non_regression_both": all(
            threshold_report[str(float(threshold))][
                "all32_official_raster_support"
            ]["field_pulse"]["hits"]
            >= threshold_report[str(float(threshold))][
                "all32_official_raster_support"
            ]["v7"]["hits"]
            for threshold in THRESHOLDS
        ),
    }
    bridge_checks = {
        "versus_field_f1_050_non_regression": bridge_field_050["delta"][
            "f1"
        ]
        >= 0.0,
        "versus_field_f1_075_gain_at_least_0p30_points": bridge_field_075[
            "delta"
        ]["f1"]
        >= 0.003,
        "versus_v7_f1_050_gain_at_least_0p30_points": bridge_v7_050[
            "delta"
        ]["f1"]
        >= 0.003,
        "versus_v7_f1_075_gain_at_least_0p50_points": bridge_v7_075[
            "delta"
        ]["f1"]
        >= 0.005,
        "versus_v7_clip_positive_probability_at_least_0p90_both": min(
            bridge_v7_050["paired_clip_bootstrap"][
                "probability_delta_positive"
            ],
            bridge_v7_075["paired_clip_bootstrap"][
                "probability_delta_positive"
            ],
        )
        >= 0.90,
        "versus_v7_recovered_to_lost_above_one_both": min(
            bridge_v7_050["gt_transitions"]["recovered_to_lost_ratio"],
            bridge_v7_075["gt_transitions"]["recovered_to_lost_ratio"],
        )
        > 1.0,
        "all32_support_non_regression_both": all(
            threshold_report[str(float(threshold))][
                "all32_official_raster_support"
            ]["bridge_pulse"]["hits"]
            >= threshold_report[str(float(threshold))][
                "all32_official_raster_support"
            ]["v7"]["hits"]
            for threshold in THRESHOLDS
        ),
    }
    field_pass = all(common_checks.values()) and all(field_checks.values())
    bridge_pass = all(common_checks.values()) and all(bridge_checks.values())
    if bridge_pass:
        decision = "bridge_pulse_requires_independent_seed_confirmation"
    elif field_pass:
        decision = "field_pulse_requires_independent_seed_confirmation"
    else:
        decision = "close_shared_gradient_pulse_family_and_test_dual_state"

    report = {
        "experiment": (
            "V32 exact-paired auxiliary-pulse consolidation at global 50K"
        ),
        "scientific_status": (
            "exploratory_same_seed_schedule_hypothesis; any pass requires "
            "an independent seed"
        ),
        "contract": {
            "common_v7_30k_ancestor": True,
            "auxiliary_pulse_interval": "30K-to-35K",
            "consolidation_interval": "35K-to-50K",
            "field_loss_during_consolidation": 0.0,
            "selection_bridge_during_consolidation": 0.0,
            "field_route_residual_during_consolidation": 0.0,
            "fixed_decision_endpoint": 50000,
            "test_split_used": False,
        },
        "thresholds": threshold_report,
        "checks": {
            "common": common_checks,
            "field_pulse": field_checks,
            "bridge_pulse": bridge_checks,
        },
        "field_pulse_passed": field_pass,
        "bridge_pulse_passed": bridge_pass,
        "decision": decision,
        "test_split_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
