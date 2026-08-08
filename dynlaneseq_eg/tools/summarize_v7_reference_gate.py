from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_spec(value: str) -> tuple[int, Path]:
    iteration, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("report must be ITERATION=PATH")
    return int(iteration), Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the paired V7 hard/direct slot-reference gate."
    )
    parser.add_argument("--hard-memorization", required=True)
    parser.add_argument("--direct-memorization", required=True)
    parser.add_argument("--hard-report", action="append", type=_parse_spec, default=[])
    parser.add_argument("--direct-report", action="append", type=_parse_spec, default=[])
    parser.add_argument("--hard-gradient", required=True)
    parser.add_argument("--direct-gradient", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _method(report: dict[str, Any]) -> dict[str, Any]:
    methods = report["methods"]
    return methods.get("four_slot_refined") or methods["four_slot_global_unique"]


def _row(iteration: int, path: Path) -> dict[str, Any]:
    report = _load(path)
    method = _method(report)
    slot = report["four_slot_diagnostics"]
    cardinality = slot.get("cardinality", {})
    return {
        "iteration": int(iteration),
        "report": str(path),
        "f1_050": float(method["0.50"]["f1"]),
        "f1_075": float(method["0.75"]["f1"]),
        "precision_050": float(method["0.50"]["precision"]),
        "recall_050": float(method["0.50"]["recall"]),
        "mean_selected": float(method["0.50"]["mean_selected_per_image"]),
        "close_pair_fraction_20px": float(
            method["0.50"]["selected_curve_diversity"][
                "close_pair_fraction_below_20px"
            ]
        ),
        "cardinality_exact": float(cardinality.get("exact_fraction", 0.0)),
        "cardinality_mae": float(cardinality.get("mean_absolute_error", 4.0)),
        "dustbin_fraction": float(slot["dustbin_fraction"]),
        "semantic_duplicate_fraction": float(
            slot["semantic_duplicate_cluster_fraction"]
        ),
        "global_repair_fraction": float(slot["global_assignment_repair_fraction"]),
        "oracle_050": float(
            report["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        ),
        "oracle_075": float(
            report["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        ),
    }


def _memorization(path: str | Path) -> dict[str, Any]:
    row = _row(-1, Path(path))
    row["passed"] = bool(
        row["f1_050"] >= 0.90
        and row["cardinality_exact"] >= 0.90
        and row["semantic_duplicate_fraction"] <= 0.02
        and row["close_pair_fraction_20px"] <= 0.02
    )
    return row


def _trajectory(specs: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    if not specs:
        raise ValueError("at least one trajectory report is required per arm")
    return [_row(iteration, path) for iteration, path in sorted(specs)]


def _stable(rows: list[dict[str, Any]]) -> bool:
    final = rows[-1]
    best_050 = max(float(row["f1_050"]) for row in rows)
    return bool(
        float(final["f1_050"]) >= best_050 - 0.02
        and 2.8 <= float(final["mean_selected"]) <= 3.6
        and float(final["semantic_duplicate_fraction"]) <= 0.03
        and float(final["close_pair_fraction_20px"]) <= 0.03
        and float(final["oracle_050"])
        >= float(rows[0]["oracle_050"]) - 1.0e-9
    )


def main() -> None:
    args = _parse_args()
    hard_memory = _memorization(args.hard_memorization)
    direct_memory = _memorization(args.direct_memorization)
    hard = _trajectory(args.hard_report)
    direct = _trajectory(args.direct_report)
    hard_gradient = _load(args.hard_gradient)
    direct_gradient = _load(args.direct_gradient)
    hard_final = hard[-1]
    direct_final = direct[-1]
    hard_stable = _stable(hard)
    direct_stable = _stable(direct)

    margin = 0.005
    strict_tolerance = 0.005
    winner = "inconclusive"
    if (
        direct_stable
        and float(direct_final["f1_050"]) >= float(hard_final["f1_050"]) + margin
        and float(direct_final["f1_075"]) >= float(hard_final["f1_075"]) - strict_tolerance
    ):
        winner = "direct_soft_reference"
    elif (
        hard_stable
        and float(hard_final["f1_050"]) >= float(direct_final["f1_050"]) + margin
        and float(hard_final["f1_075"]) >= float(direct_final["f1_075"]) - strict_tolerance
    ):
        winner = "structured_hard_reference"

    payload = {
        "experiment": "V7 parameter-matched hard versus direct slot reference",
        "diagnostic_only": True,
        "memorization": {
            "hard": hard_memory,
            "direct": direct_memory,
            "both_passed": bool(hard_memory["passed"] and direct_memory["passed"]),
        },
        "hard_reference": {
            "trajectory": hard,
            "stable": hard_stable,
            "gradient_contract": hard_gradient,
        },
        "direct_reference": {
            "trajectory": direct,
            "stable": direct_stable,
            "gradient_contract": direct_gradient,
        },
        "final_delta_direct_minus_hard": {
            "f1_050": float(direct_final["f1_050"]) - float(hard_final["f1_050"]),
            "f1_075": float(direct_final["f1_075"]) - float(hard_final["f1_075"]),
            "cardinality_exact": float(direct_final["cardinality_exact"])
            - float(hard_final["cardinality_exact"]),
            "semantic_duplicate_fraction": float(
                direct_final["semantic_duplicate_fraction"]
            ) - float(hard_final["semantic_duplicate_fraction"]),
        },
        "winner": winner,
        "long_run_authorized": bool(
            hard_memory["passed"]
            and direct_memory["passed"]
            and winner != "inconclusive"
        ),
        "decision_rule": {
            "primary_f1_050_margin": margin,
            "maximum_f1_075_regression": strict_tolerance,
            "no_seed_sweep": True,
            "test_split_closed": True,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
