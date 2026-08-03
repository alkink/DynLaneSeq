from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .summarize_v4_1_score_gate import _delta, _flatten


ARM_ORDER = (
    "source_v4",
    "r0_generic_frozen",
    "r1_generic_semantic",
    "r2_relation_semantic",
    "r3_relation_setloss",
)


def _mmr_f1(report: Mapping[str, Any], threshold: str = "0.50") -> float | None:
    method = report.get("methods", {}).get("mmr_sigma20_penalty0p5")
    if not isinstance(method, Mapping) or threshold not in method:
        return None
    return float(method[threshold]["f1"])


def summarize(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    geometry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    missing = [name for name in ARM_ORDER if name not in reports]
    if missing:
        raise ValueError("missing V4.2 reports: " + ", ".join(missing))
    rows: dict[str, dict[str, Any]] = {}
    for name in ARM_ORDER:
        row = _flatten(reports[name])
        mmr_050 = _mmr_f1(reports[name], "0.50")
        mmr_075 = _mmr_f1(reports[name], "0.75")
        row["mmr_f1_050"] = mmr_050
        row["mmr_f1_075"] = mmr_075
        row["mmr_gain_050"] = (
            None if mmr_050 is None else mmr_050 - float(row["f1_050"])
        )
        rows[name] = row

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
        "longer_generic_control_vs_source": _delta(
            rows, "r0_generic_frozen", "source_v4"
        ),
        "semantic_adapter_effect": _delta(
            rows, "r1_generic_semantic", "r0_generic_frozen"
        ),
        "explicit_relation_effect": _delta(
            rows, "r2_relation_semantic", "r1_generic_semantic"
        ),
        "set_loss_effect": _delta(
            rows, "r3_relation_setloss", "r2_relation_semantic"
        ),
        "complete_v4_2_vs_control": _delta(
            rows, "r3_relation_setloss", "r0_generic_frozen"
        ),
    }
    final = rows["r3_relation_setloss"]
    final_ap = final["unique_candidate_ap_050"]
    final_mmr_gain = final["mmr_gain_050"]
    checks = {
        "geometry_preserved": geometry_checks["r3_relation_setloss"]["preserved"],
        "direct_top4_recall_050_at_least_070": float(final["recall_050"]) >= 0.70,
        "direct_top4_recall_075_at_least_060": float(final["recall_075"]) >= 0.60,
        "duplicate_fp_fraction_below_025": (
            float(final["duplicate_fp_fraction_050"]) < 0.25
        ),
        "unique_ap_050_at_least_050": (
            final_ap is not None and float(final_ap) >= 0.50
        ),
        "foreground_mass_in_3_to_4p5": (
            3.0 <= float(final["mean_foreground_probability_mass"]) <= 4.5
        ),
        "residual_mmr_gain_below_005": (
            final_mmr_gain is not None and float(final_mmr_gain) < 0.05
        ),
    }
    passed = all(bool(value) for value in checks.values())
    best_arm = max(
        ARM_ORDER[1:],
        key=lambda name: (
            float(rows[name]["f1_050"]),
            float(rows[name]["recall_075"]),
            float(rows[name]["unique_candidate_ap_050"] or -1.0),
        ),
    )
    return {
        "diagnostic_only": True,
        "warning": (
            "Frozen-geometry validation-subset results localize score/set "
            "selection behavior; they are not official CULane test scores."
        ),
        "design": {
            "r0_generic_frozen": "longer V4.1-D control",
            "r1_generic_semantic": "R0 + live detached semantic score adapters",
            "r2_relation_semantic": "R1 + explicit pairwise curve attention bias",
            "r3_relation_setloss": "R2 + coverage/duplicate/winner/count losses",
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
        "relation_gate": {"checks": checks, "passed": passed},
        "best_diagnostic_arm": best_arm,
        "verdict": (
            "relation_aware_scalar_selector_passed"
            if passed
            else "relation_gate_not_yet_sufficient"
        ),
    }


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V4.2 relation gate.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--r0", required=True)
    parser.add_argument("--r1", required=True)
    parser.add_argument("--r2", required=True)
    parser.add_argument("--r3", required=True)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        {
            "source_v4": _load(args.source),
            "r0_generic_frozen": _load(args.r0),
            "r1_generic_semantic": _load(args.r1),
            "r2_relation_semantic": _load(args.r2),
            "r3_relation_setloss": _load(args.r3),
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

