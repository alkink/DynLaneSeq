from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the matched 25k from-scratch matcher intervention using "
            "threshold-free official-raster set metrics and query ownership."
        )
    )
    parser.add_argument("--ranking-summary", required=True)
    parser.add_argument("--assignment-report", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same_optional_float(left: Any, right: Any, atol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= atol


def _ranking_row(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    iou_threshold: float,
    quality_power: float | None,
    score_threshold: float | None,
    top_k: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row.get("strategy")) == strategy
        and int(row.get("top_k", 0)) == int(top_k)
        and _same_optional_float(row.get("iou_threshold"), iou_threshold)
        and _same_optional_float(row.get("quality_power"), quality_power)
        and _same_optional_float(row.get("score_threshold"), score_threshold)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one ranking row for "
            f"{strategy=}, {iou_threshold=}, {quality_power=}, "
            f"{score_threshold=}, {top_k=}; found {len(matches)}"
        )
    return matches[0]


def _query_summary(arm: dict[str, Any]) -> dict[str, Any]:
    per_query = list(arm.get("per_query", []))
    assignment_rates = [float(row.get("assignment_rate_per_image", 0.0)) for row in per_query]
    useful_050 = [float(row.get("useful_iou050_image_fraction", 0.0)) for row in per_query]
    return {
        "mean_active_queries_iou030_per_image": float(
            arm.get("mean_active_queries_iou030_per_image", 0.0)
        ),
        "mean_active_queries_iou050_per_image": float(
            arm.get("mean_active_queries_iou050_per_image", 0.0)
        ),
        "mean_query_supporters_iou050_per_lane": float(
            arm.get("mean_query_supporters_iou050_per_lane", 0.0)
        ),
        "assigned_queries_over_5pct": sum(value >= 0.05 for value in assignment_rates),
        "useful_queries_iou050_over_5pct": sum(value >= 0.05 for value in useful_050),
        "assignment_rate_per_query": assignment_rates,
        "useful_iou050_fraction_per_query": useful_050,
    }


def summarize(
    ranking: dict[str, Any], assignment: dict[str, Any]
) -> dict[str, Any]:
    comparability = ranking.get("comparability", {})
    if not bool(comparability.get("all_checks_pass", False)):
        raise ValueError(f"Ranking inputs are not paired: {comparability}")

    rows = list(ranking.get("rows", []))
    metrics: dict[str, dict[str, Any]] = {}
    for threshold in (0.5, 0.75):
        key = f"{threshold:.2f}"
        threshold_rows = {
            "all_32_capacity": _ranking_row(
                rows,
                strategy="all_raw",
                iou_threshold=threshold,
                quality_power=None,
                score_threshold=None,
                top_k=0,
            ),
            "oracle_top4_capacity": _ranking_row(
                rows,
                strategy="oracle_topk",
                iou_threshold=threshold,
                quality_power=None,
                score_threshold=None,
                top_k=4,
            ),
            "raw_model_top4_q0p25": _ranking_row(
                rows,
                strategy="model_topk",
                iou_threshold=threshold,
                quality_power=0.25,
                score_threshold=None,
                top_k=4,
            ),
            "nms_model_top4_q0p25_no_threshold": _ranking_row(
                rows,
                strategy="model_topk_nms",
                iou_threshold=threshold,
                quality_power=0.25,
                score_threshold=-1.0,
                top_k=4,
            ),
        }
        metrics[key] = {
            name: {
                "control_recall": float(row["base_recall"]),
                "candidate_recall": float(row["candidate_recall"]),
                "delta_recall_points": float(row["delta_recall_points"]),
            }
            for name, row in threshold_rows.items()
        }

    assignment_specialization = assignment.get("query_specialization", {})
    if "r34" not in assignment_specialization or "dla34" not in assignment_specialization:
        raise ValueError(
            "Assignment report must contain the analyzer's paired r34/dla34 arms"
        )
    query_ownership = {
        "control": _query_summary(assignment_specialization["r34"]),
        "candidate": _query_summary(assignment_specialization["dla34"]),
    }

    capacity_deltas = [
        metrics[key]["all_32_capacity"]["delta_recall_points"]
        for key in ("0.50", "0.75")
    ]
    deployed_deltas = [
        metrics[key]["nms_model_top4_q0p25_no_threshold"]["delta_recall_points"]
        for key in ("0.50", "0.75")
    ]
    mean_capacity_delta = sum(capacity_deltas) / len(capacity_deltas)
    mean_deployed_delta = sum(deployed_deltas) / len(deployed_deltas)

    # These gates are intentionally conservative for a 256-image diagnostic.
    # They are not benchmark claims; the full validation/test protocol remains
    # necessary before promoting a candidate to the paper model.
    if mean_deployed_delta >= 1.0 and min(capacity_deltas) >= -0.5:
        verdict = "positive_continue_training"
    elif mean_capacity_delta >= 1.0 and mean_deployed_delta > -1.0:
        verdict = "capacity_signal_ranking_not_yet_converted"
    elif mean_capacity_delta <= -1.0 and mean_deployed_delta <= -1.0:
        verdict = "negative_stop"
    else:
        verdict = "mixed_or_inconclusive"

    return {
        "diagnostic_only": True,
        "question": (
            "Does lowering Hungarian object cost from 2.0 to 0.5 from the "
            "first optimizer step improve candidate ownership/capacity before "
            "score threshold selection?"
        ),
        "primary_setting": {
            "iteration": 25000,
            "sample": "uniform validation subset",
            "official_raster_iou": True,
            "top_k": 4,
            "quality_power": 0.25,
            "score_threshold": None,
            "nms_distance_px": 20.0,
        },
        "metrics": metrics,
        "query_ownership": query_ownership,
        "aggregate_delta_points": {
            "mean_all_32_capacity": mean_capacity_delta,
            "mean_nms_top4": mean_deployed_delta,
        },
        "verdict": verdict,
        "verdict_note": (
            "A positive short gate licenses continuation; it does not establish "
            "final CULane F1 or justify test-set evaluation."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize(_load(args.ranking_summary), _load(args.assignment_report))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
