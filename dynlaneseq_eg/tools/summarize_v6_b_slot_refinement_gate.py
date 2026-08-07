from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V6-B bounded slot-refinement gate."
    )
    parser.add_argument("--source-report", required=True)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="ITERATION=PATH",
    )
    parser.add_argument("--max-f1-050-drop", type=float, default=0.005)
    parser.add_argument("--min-f1-075-gain", type=float, default=0.020)
    parser.add_argument("--max-boundary-mass", type=float, default=0.45)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _method(payload: dict) -> dict:
    method = payload.get("methods", {}).get("four_slot_refined")
    if not isinstance(method, dict):
        raise ValueError("V6-B report has no four_slot_refined method")
    return method


def main() -> None:
    args = parse_args()
    if not args.report:
        raise ValueError("at least one V6-B trajectory report is required")
    source = _load(args.source_report)
    source_method = _method(source)
    source_050 = float(source_method["0.50"]["f1"])
    source_075 = float(source_method["0.75"]["f1"])
    source_oracle_050 = float(
        source["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
    )
    source_oracle_075 = float(
        source["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
    )
    rows = []
    for value in args.report:
        iteration_text, path = value.split("=", 1)
        payload = _load(path)
        method = _method(payload)
        metric_050 = method["0.50"]
        metric_075 = method["0.75"]
        diagnostics = payload.get("four_slot_diagnostics") or {}
        refinement = diagnostics.get("refinement") or {}
        oracle_050 = float(
            payload["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        )
        oracle_075 = float(
            payload["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        )
        rows.append(
            {
                "iteration": int(iteration_text),
                "report": str(Path(path)),
                "f1_050": float(metric_050["f1"]),
                "precision_050": float(metric_050["precision"]),
                "recall_050": float(metric_050["recall"]),
                "f1_075": float(metric_075["f1"]),
                "precision_075": float(metric_075["precision"]),
                "recall_075": float(metric_075["recall"]),
                "mean_selected": float(metric_050["mean_selected_per_image"]),
                "f1_050_delta": float(metric_050["f1"]) - source_050,
                "f1_075_delta": float(metric_075["f1"]) - source_075,
                "oracle_050": oracle_050,
                "oracle_075": oracle_075,
                "oracle_050_delta": oracle_050 - source_oracle_050,
                "oracle_075_delta": oracle_075 - source_oracle_075,
                "mean_abs_delta_px": float(
                    refinement.get("mean_abs_delta_px", 0.0)
                ),
                "mean_max_abs_delta_px": float(
                    refinement.get("mean_max_abs_delta_px", 0.0)
                ),
                "mean_boundary_probability_mass": float(
                    refinement.get("mean_boundary_probability_mass", 1.0)
                ),
                "slot_diagnostics": diagnostics,
            }
        )
    rows.sort(key=lambda row: row["iteration"])
    eligible = [
        row
        for row in rows
        if row["f1_050"] >= source_050 - float(args.max_f1_050_drop)
    ]
    best = max(
        eligible or rows,
        key=lambda row: (row["f1_075"], row["f1_050"]),
    )
    checks = {
        "f1_050_preserved": best["f1_050"]
        >= source_050 - float(args.max_f1_050_drop),
        "strict_f1_materially_improved": best["f1_075"]
        >= source_075 + float(args.min_f1_075_gain),
        "frozen_proposal_oracle_exact": all(
            abs(float(row["oracle_050_delta"])) < 1.0e-12
            and abs(float(row["oracle_075_delta"])) < 1.0e-12
            for row in rows
        ),
        "bounded_refinement_active": best["mean_abs_delta_px"] > 0.05,
        "boundary_mass_controlled": best["mean_boundary_probability_mass"]
        <= float(args.max_boundary_mass),
        "deployment_count_preserved": 3.0 <= best["mean_selected"] <= 3.5,
    }
    passed = all(checks.values())
    payload = {
        "experiment": "V6-B slot-owned bounded geometry refinement",
        "diagnostic_only": True,
        "source_report": str(Path(args.source_report)),
        "source": {
            "f1_050": source_050,
            "f1_075": source_075,
            "oracle_050": source_oracle_050,
            "oracle_075": source_oracle_075,
        },
        "gate": {
            "max_f1_050_drop": float(args.max_f1_050_drop),
            "min_f1_075_gain": float(args.min_f1_075_gain),
            "max_boundary_probability_mass": float(args.max_boundary_mass),
            "mean_selected_band": [3.0, 3.5],
        },
        "trajectory": rows,
        "best": best,
        "checks": checks,
        "passed": passed,
        "next_step": (
            "full_validation_then_low_lr_router_refiner_coadaptation"
            if passed
            else "stop_and_inspect_slot_refinement_quality_and_bounds"
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
