from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the V6-A slot gate.")
    parser.add_argument("--source-report", required=True)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="ITERATION=PATH",
    )
    parser.add_argument("--min-f1-050", type=float, default=0.795)
    parser.add_argument("--min-f1-075", type=float, default=0.56)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not args.report:
        raise ValueError("at least one V6-A trajectory report is required")
    source = _load(args.source_report)
    source_oracle = float(
        source["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
    )
    rows = []
    for value in args.report:
        iteration_text, path = value.split("=", 1)
        payload = _load(path)
        metric_050 = payload["methods"]["four_slot_global_unique"]["0.50"]
        metric_075 = payload["methods"]["four_slot_global_unique"]["0.75"]
        oracle_050 = float(
            payload["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        )
        rows.append(
            {
                "iteration": int(iteration_text),
                "report": str(Path(path)),
                "f1_050": float(metric_050["f1"]),
                "precision_050": float(metric_050["precision"]),
                "recall_050": float(metric_050["recall"]),
                "f1_075": float(metric_075["f1"]),
                "mean_selected": float(metric_050["mean_selected_per_image"]),
                "oracle_050": oracle_050,
                "oracle_delta": oracle_050 - source_oracle,
                "slot_diagnostics": payload.get("four_slot_diagnostics"),
            }
        )
    rows.sort(key=lambda row: row["iteration"])
    best = max(rows, key=lambda row: (row["f1_050"], row["f1_075"]))
    checks = {
        "reproduces_frozen_probe_050": best["f1_050"]
        >= float(args.min_f1_050),
        "strict_signal_retained": best["f1_075"] >= float(args.min_f1_075),
        "deployment_count_in_expected_band": 3.0
        <= best["mean_selected"]
        <= 3.5,
        "frozen_candidate_oracle_exact": all(
            abs(float(row["oracle_delta"])) < 1.0e-12 for row in rows
        ),
    }
    payload = {
        "experiment": "V6-A frozen V5.1 proposals -> four lane-object slots",
        "diagnostic_only": True,
        "source_report": str(Path(args.source_report)),
        "gate": {
            "min_f1_050": float(args.min_f1_050),
            "min_f1_075": float(args.min_f1_075),
            "mean_selected_band": [3.0, 3.5],
        },
        "source_oracle_050": source_oracle,
        "trajectory": rows,
        "best": best,
        "checks": checks,
        "passed": all(checks.values()),
        "next_step": (
            "full_validation_then_v6_b_slot_owned_bounded_refinement"
            if all(checks.values())
            else "stop_and_audit_production_probe_mismatch"
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()

