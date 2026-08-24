from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared V39 pattern-query 65K gate."
    )
    parser.add_argument("--official-pair", required=True)
    parser.add_argument("--control-autopsy", required=True)
    parser.add_argument("--treatment-autopsy", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    pair = _read(args.official_pair)
    control = _read(args.control_autopsy)
    treatment = _read(args.treatment_autopsy)
    if pair.get("test_set_used") is not False:
        raise ValueError("V39 gate refuses test-set input")
    for payload in (control, treatment):
        if payload.get("test_set_used") is not False:
            raise ValueError("V39 autopsy used the test split")
        if not payload.get("contract", {}).get(
            "deployed_counts_exact_official_report", False
        ):
            raise ValueError("V39 autopsy did not reproduce official counts")

    f1_delta = pair["treatment_minus_control_f1_points"]
    support_control = int(control["policies"]["cardinality_oracle"]["0.75"]["TP"])
    support_treatment = int(
        treatment["policies"]["cardinality_oracle"]["0.75"]["TP"]
    )
    support_delta = support_treatment - support_control
    quality_control = float(
        control["row_and_range"]["all_gt_best_of_four"]["quality"]["p50"]
    )
    quality_treatment = float(
        treatment["row_and_range"]["all_gt_best_of_four"]["quality"]["p50"]
    )
    bottom_key = "bottom_120_159"
    bottom_control = float(
        control["row_and_range"]["unsupported_075"]["bands"][bottom_key][
            "p50_absolute_error_px"
        ]
    )
    bottom_treatment = float(
        treatment["row_and_range"]["unsupported_075"]["bands"][bottom_key][
            "p50_absolute_error_px"
        ]
    )
    bottom_reduction = (
        (bottom_control - bottom_treatment) / bottom_control
        if bottom_control > 0.0
        else 0.0
    )
    geometry_control = control["geometry_health"]["deployed"]
    geometry_treatment = treatment["geometry_health"]["deployed"]
    duplicate_delta = float(geometry_treatment["duplicate_image_fraction"]) - float(
        geometry_control["duplicate_image_fraction"]
    )
    crossing_delta = float(geometry_treatment["crossing_image_fraction"]) - float(
        geometry_control["crossing_image_fraction"]
    )
    checks = {
        "all4_support_tp_075_at_least_plus_300": support_delta >= 300,
        "best_of_four_median_iou_at_least_plus_0p01": (
            quality_treatment - quality_control
        )
        >= 0.01,
        "unsupported_075_bottom_median_x_error_reduced_15_percent": (
            bottom_reduction >= 0.15
        ),
        "official_f1_075_at_least_plus_0p50": float(f1_delta["0.75"]) >= 0.50,
        "official_f1_050_non_regression": float(f1_delta["0.5"]) >= 0.0,
        "duplicate_fraction_increase_at_most_1pp": duplicate_delta <= 0.01,
        "crossing_fraction_increase_at_most_1pp": crossing_delta <= 0.01,
    }
    core = (
        checks["official_f1_075_at_least_plus_0p50"]
        and checks["official_f1_050_non_regression"]
        and checks["all4_support_tp_075_at_least_plus_300"]
    )
    decision = (
        "V39_PATTERN_QUERY_PASS"
        if all(checks.values())
        else (
            "V39_PATTERN_QUERY_PARTIAL_GEOMETRY_SIGNAL"
            if core
            else "V39_PATTERN_QUERY_FAIL"
        )
    )
    payload = {
        "experiment": "V39 image-conditioned full-curve query initialization gate",
        "decision": decision,
        "checks": checks,
        "official_f1_delta_points": f1_delta,
        "strict_support": {
            "control_tp": support_control,
            "treatment_tp": support_treatment,
            "delta_tp": support_delta,
        },
        "best_of_four_median_iou": {
            "control": quality_control,
            "treatment": quality_treatment,
            "delta": quality_treatment - quality_control,
        },
        "unsupported_075_bottom_median_x_error_px": {
            "control": bottom_control,
            "treatment": bottom_treatment,
            "relative_reduction": bottom_reduction,
        },
        "geometry_health_delta": {
            "duplicate_image_fraction": duplicate_delta,
            "crossing_image_fraction": crossing_delta,
        },
        "source_artifacts": {
            "official_pair": args.official_pair,
            "control_autopsy": args.control_autopsy,
            "treatment_autopsy": args.treatment_autopsy,
        },
        "checkpoint_selection_performed": False,
        "threshold_selection_performed": False,
        "test_set_used": False,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
