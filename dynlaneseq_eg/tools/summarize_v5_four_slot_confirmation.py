from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-seed four-slot confirmations against a "
            "parameter-matched 32-query control."
        )
    )
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--min-mean-gain-050", type=float, default=5.0)
    parser.add_argument("--min-mean-gain-075", type=float, default=2.0)
    parser.add_argument("--min-positive-seeds", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


def summarize_reports(
    reports: list[dict[str, Any]],
    report_paths: list[str],
    *,
    min_mean_gain_050: float,
    min_mean_gain_075: float,
    min_positive_seeds: int,
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("multi-seed confirmation requires at least two reports")
    if len(report_paths) != len(reports):
        raise ValueError("report paths and payloads must have the same length")
    checkpoint_hashes = {str(row["checkpoint_sha256"]) for row in reports}
    train_hashes = {
        str(row["caches"]["train"]["list_sha256"]) for row in reports
    }
    val_hashes = {
        str(row["caches"]["val"]["list_sha256"]) for row in reports
    }
    if len(checkpoint_hashes) != 1 or len(train_hashes) != 1 or len(val_hashes) != 1:
        raise ValueError("confirmation reports do not share checkpoint/data provenance")

    rows = []
    gains_050 = []
    gains_075 = []
    slot_f1_050 = []
    slot_f1_075 = []
    positive = 0
    for path, report in zip(report_paths, reports):
        strategies = report["evaluation"]["strategies"]
        baseline_name = str(report["decision"]["best_32_query_baseline"])
        baseline = strategies[baseline_name]
        slot = strategies["learned_4_slots"]
        gain_050 = 100.0 * (
            _finite(slot["f1_050"], "slot F1@.50")
            - _finite(baseline["f1_050"], "baseline F1@.50")
        )
        gain_075 = 100.0 * (
            _finite(slot["f1_075"], "slot F1@.75")
            - _finite(baseline["f1_075"], "baseline F1@.75")
        )
        seed_positive = bool(
            gain_050 >= float(min_mean_gain_050)
            and gain_075 >= float(min_mean_gain_075)
        )
        positive += int(seed_positive)
        gains_050.append(gain_050)
        gains_075.append(gain_075)
        slot_f1_050.append(float(slot["f1_050"]))
        slot_f1_075.append(float(slot["f1_075"]))
        model = report["models"]
        matched_parameters = int(
            model["parameter_matched_proposal_set_scorer_parameters"]
        )
        slot_parameters = int(model["four_slot_router_parameters"])
        rows.append(
            {
                "report": path,
                "seed": int(report["training"]["seed"]),
                "best_32_query_baseline": baseline_name,
                "baseline_f1_050": float(baseline["f1_050"]),
                "baseline_f1_075": float(baseline["f1_075"]),
                "slot_f1_050": float(slot["f1_050"]),
                "slot_f1_075": float(slot["f1_075"]),
                "gain_f1_050_points": gain_050,
                "gain_f1_075_points": gain_075,
                "slot_predictions": int(slot["pred"]),
                "slot_tp_050": int(slot["tp_050"]),
                "slot_tp_075": int(slot["tp_075"]),
                "slot_duplicate_candidate_assignments": int(
                    report["evaluation"][
                        "slot_duplicate_candidate_assignments"
                    ]
                ),
                "matched_32_parameters": matched_parameters,
                "slot_parameters": slot_parameters,
                "parameter_ratio_slot_over_matched_32": slot_parameters
                / float(max(matched_parameters, 1)),
                "seed_positive": seed_positive,
            }
        )

    mean_gain_050 = mean(gains_050)
    mean_gain_075 = mean(gains_075)
    parameter_ratios = [
        float(row["parameter_ratio_slot_over_matched_32"]) for row in rows
    ]
    parameter_matched = all(0.80 <= ratio <= 1.25 for ratio in parameter_ratios)
    duplicate_free = all(
        int(row["slot_duplicate_candidate_assignments"]) == 0 for row in rows
    )
    confirmed = bool(
        mean_gain_050 >= float(min_mean_gain_050)
        and mean_gain_075 >= float(min_mean_gain_075)
        and positive >= int(min_positive_seeds)
        and parameter_matched
        and duplicate_free
    )
    result = {
        "diagnostic_only": True,
        "question": (
            "Does the four-final-slot advantage survive seed changes and a "
            "parameter-matched 32-query set-scorer control?"
        ),
        "provenance": {
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
            "train_list_sha256": next(iter(train_hashes)),
            "val_list_sha256": next(iter(val_hashes)),
            "reports": list(report_paths),
        },
        "seeds": rows,
        "aggregate": {
            "seed_count": len(rows),
            "positive_seed_count": int(positive),
            "mean_slot_f1_050": mean(slot_f1_050),
            "std_slot_f1_050": pstdev(slot_f1_050),
            "mean_slot_f1_075": mean(slot_f1_075),
            "std_slot_f1_075": pstdev(slot_f1_075),
            "mean_gain_f1_050_points": mean_gain_050,
            "std_gain_f1_050_points": pstdev(gains_050),
            "min_gain_f1_050_points": min(gains_050),
            "mean_gain_f1_075_points": mean_gain_075,
            "std_gain_f1_075_points": pstdev(gains_075),
            "min_gain_f1_075_points": min(gains_075),
            "parameter_matched": parameter_matched,
            "duplicate_free": duplicate_free,
        },
        "gate": {
            "min_mean_gain_f1_050_points": float(min_mean_gain_050),
            "min_mean_gain_f1_075_points": float(min_mean_gain_075),
            "min_positive_seeds": int(min_positive_seeds),
            "four_slot_structure_confirmed": confirmed,
        },
        "recommendation": (
            "build_v6_32_proposals_to_4_final_slots"
            if confirmed
            else "slot_advantage_not_yet_confirmed"
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    reports = [_load(path) for path in args.reports]
    result = summarize_reports(
        reports,
        list(args.reports),
        min_mean_gain_050=float(args.min_mean_gain_050),
        min_mean_gain_075=float(args.min_mean_gain_075),
        min_positive_seeds=int(args.min_positive_seeds),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
