from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ARM_ORDER = (
    "source_v4",
    "a_mlp_shared",
    "b_mlp_unique",
    "c_set_shared",
    "d_set_unique",
)


def _metric(report: Mapping[str, Any], threshold: str) -> dict[str, Any]:
    return dict(report["methods"]["score_top4"][threshold])


def _capacity(report: Mapping[str, Any], threshold: str) -> float:
    return float(
        report["capacity"][threshold]["all_candidate_oracle"]["recall"]
    )


def _flatten(report: Mapping[str, Any]) -> dict[str, Any]:
    direct_050 = _metric(report, "0.50")
    direct_075 = _metric(report, "0.75")
    score = report.get("score_diagnostics", {})
    unique_ap = score.get("unique_candidate_ap", {})
    return {
        "score_mode": score.get("score_mode"),
        "precision_050": float(direct_050["precision"]),
        "recall_050": float(direct_050["recall"]),
        "f1_050": float(direct_050["f1"]),
        "precision_075": float(direct_075["precision"]),
        "recall_075": float(direct_075["recall"]),
        "f1_075": float(direct_075["f1"]),
        "all32_oracle_recall_050": _capacity(report, "0.50"),
        "all32_oracle_recall_075": _capacity(report, "0.75"),
        "unique_candidate_ap_050": (
            None
            if unique_ap.get("0.50") is None
            else float(unique_ap["0.50"])
        ),
        "unique_candidate_ap_075": (
            None
            if unique_ap.get("0.75") is None
            else float(unique_ap["0.75"])
        ),
        "mean_foreground_probability_mass": float(
            score.get("mean_foreground_probability_mass", 0.0)
        ),
        "mean_target_lane_count": float(score.get("mean_target_lane_count", 0.0)),
        "duplicate_fp_fraction_050": float(
            direct_050["false_positive_breakdown"]["duplicate_fp"][
                "fraction_of_fp"
            ]
        ),
    }


def _delta(
    rows: Mapping[str, Mapping[str, Any]],
    candidate: str,
    control: str,
) -> dict[str, float | None]:
    keys = (
        "precision_050",
        "recall_050",
        "f1_050",
        "recall_075",
        "f1_075",
        "unique_candidate_ap_050",
        "mean_foreground_probability_mass",
    )
    result: dict[str, float | None] = {}
    for key in keys:
        left = rows[candidate].get(key)
        right = rows[control].get(key)
        result[key] = (
            None if left is None or right is None else float(left) - float(right)
        )
    return result


def summarize(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    geometry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    missing = [name for name in ARM_ORDER if name not in reports]
    if missing:
        raise ValueError("missing V4.1 gate reports: " + ", ".join(missing))
    rows = {name: _flatten(reports[name]) for name in ARM_ORDER}
    source = rows["source_v4"]
    geometry_checks: dict[str, dict[str, Any]] = {}
    for name in ARM_ORDER[1:]:
        delta_050 = abs(
            float(rows[name]["all32_oracle_recall_050"])
            - float(source["all32_oracle_recall_050"])
        )
        delta_075 = abs(
            float(rows[name]["all32_oracle_recall_075"])
            - float(source["all32_oracle_recall_075"])
        )
        geometry_checks[name] = {
            "absolute_delta_050": delta_050,
            "absolute_delta_075": delta_075,
            "preserved": max(delta_050, delta_075) <= float(geometry_tolerance),
        }

    factor_effects = {
        "unique_loss_on_independent": _delta(
            rows, "b_mlp_unique", "a_mlp_shared"
        ),
        "unique_loss_on_set": _delta(rows, "d_set_unique", "c_set_shared"),
        "set_interaction_with_shared_loss": _delta(
            rows, "c_set_shared", "a_mlp_shared"
        ),
        "set_interaction_with_unique_loss": _delta(
            rows, "d_set_unique", "b_mlp_unique"
        ),
        "combined_vs_source": _delta(rows, "d_set_unique", "source_v4"),
    }

    combined = rows["d_set_unique"]
    combined_ap = combined["unique_candidate_ap_050"]
    combined_gate_checks = {
        "geometry_preserved": bool(geometry_checks["d_set_unique"]["preserved"]),
        "direct_top4_recall_050_at_least_070": (
            float(combined["recall_050"]) >= 0.70
        ),
        "direct_top4_recall_075_at_least_060": (
            float(combined["recall_075"]) >= 0.60
        ),
        "diagnostic_f1_050_at_least_065": float(combined["f1_050"]) >= 0.65,
        "unique_ap_050_at_least_050": (
            combined_ap is not None and float(combined_ap) >= 0.50
        ),
        "foreground_mass_in_3_to_4p5": (
            3.0
            <= float(combined["mean_foreground_probability_mass"])
            <= 4.5
        ),
    }
    combined_pass = all(combined_gate_checks.values())
    best_arm = max(
        ARM_ORDER[1:],
        key=lambda name: (
            float(rows[name]["f1_050"]),
            float(rows[name]["recall_075"]),
            float(rows[name]["unique_candidate_ap_050"] or -1.0),
        ),
    )
    if combined_pass:
        verdict = "combined_v4_1_gate_passed_continue_to_full_validation"
    elif best_arm != "d_set_unique":
        verdict = "combined_hypothesis_failed_inspect_winning_factor_arm"
    else:
        verdict = "combined_improved_but_did_not_clear_score_gate"
    return {
        "diagnostic_only": True,
        "warning": (
            "This 2x2 frozen-geometry experiment identifies score architecture "
            "and supervision effects on a diagnostic validation subset. It is "
            "not an official test result and does not predict the exact F1 of "
            "a from-scratch V4.1 training run."
        ),
        "design": {
            "a_mlp_shared": "independent scorer + matcher-shared target",
            "b_mlp_unique": "independent scorer + score-independent unique target",
            "c_set_shared": "candidate-set scorer + matcher-shared target",
            "d_set_unique": "candidate-set scorer + score-independent unique target",
        },
        "arms": rows,
        "geometry_invariance": {
            "tolerance": float(geometry_tolerance),
            "checks": geometry_checks,
            "all_preserved": all(
                bool(row["preserved"]) for row in geometry_checks.values()
            ),
        },
        "factor_effects": factor_effects,
        "combined_gate": {
            "checks": combined_gate_checks,
            "passed": combined_pass,
        },
        "best_diagnostic_arm": best_arm,
        "verdict": verdict,
    }


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the V4.1 score-only 2x2 gate.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--d", required=True)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        {
            "source_v4": _load(args.source),
            "a_mlp_shared": _load(args.a),
            "b_mlp_unique": _load(args.b),
            "c_set_shared": _load(args.c),
            "d_set_unique": _load(args.d),
        },
        geometry_tolerance=args.geometry_tolerance,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
