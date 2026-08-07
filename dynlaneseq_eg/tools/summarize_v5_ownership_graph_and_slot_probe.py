from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V5 ownership graph audit and four-slot probe."
    )
    parser.add_argument("--graph-audit", required=True)
    parser.add_argument("--slot-probe", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    graph = _load(args.graph_audit)
    slot = _load(args.slot_probe)
    final_checkpoint = graph["checkpoints"][-1]
    layer_rows = final_checkpoint["within_forward"]
    max_target_disagreement = max(
        float(row["binary_owner_target_disagreement_fraction"])
        for row in layer_rows
    )
    min_layer_final_agreement = min(
        float(row["assignment"]["same_query_given_matched_both"])
        for row in layer_rows
    )
    temporal_rows = graph.get("cross_checkpoint", [])
    conditional_retention = (
        sum(
            float(row["training_owner_retention_given_official_best_fixed"])
            * int(row["training_comparable_when_official_fixed"])
            for row in temporal_rows
        )
        / float(
            max(
                sum(
                    int(row["training_comparable_when_official_fixed"])
                    for row in temporal_rows
                ),
                1,
            )
        )
        if temporal_rows
        else 0.0
    )
    gradient = graph.get("gradient_audit") or {}
    cosine = (
        gradient.get("gradient_cosine_by_layer", {})
        .get("all_ownership", [])
    )
    final_index = len(cosine) - 1
    minimum_intermediate_final_cosine = (
        min(float(cosine[index][final_index]) for index in range(final_index))
        if final_index > 0
        else 0.0
    )
    slot_decision = slot["decision"]
    strong_slot = bool(slot_decision["strong_early_slot_signal"])
    strong_layer_conflict = bool(
        max_target_disagreement >= 0.05
        or min_layer_final_agreement < 0.85
        or minimum_intermediate_final_cosine < 0.0
    )
    if strong_slot:
        recommendation = (
            "prioritize_four_slot_decoder_before_long_v5_candidate_runs"
        )
    elif strong_layer_conflict:
        recommendation = "run_v5_2a_layer_consistent_ownership_gate"
    else:
        recommendation = "skip_v5_2a_and_test_v5_2b_post_geometry_competition"
    result = {
        "diagnostic_only": True,
        "graph_audit": args.graph_audit,
        "slot_probe": args.slot_probe,
        "key_findings": {
            "max_binary_owner_target_disagreement": max_target_disagreement,
            "min_layer_to_final_owner_agreement": min_layer_final_agreement,
            "minimum_intermediate_to_final_gradient_cosine": (
                minimum_intermediate_final_cosine
            ),
            "weighted_training_owner_retention_given_official_best_fixed": (
                conditional_retention
            ),
            "four_slot_gain_f1_050_points": float(
                slot_decision["gain_f1_050_points"]
            ),
            "four_slot_gain_f1_075_points": float(
                slot_decision["gain_f1_075_points"]
            ),
        },
        "decisions": {
            "strong_layer_target_or_gradient_conflict": strong_layer_conflict,
            "strong_early_four_slot_signal": strong_slot,
            "recommendation": recommendation,
            "negative_slot_probe_is_conclusive": False,
        },
        "interpretation": {
            "positive_slot_probe": (
                "Four final object slots outperform a same-memory 32-query "
                "scorer before any backbone or geometry co-adaptation. This "
                "is strong evidence against treating all 32 proposals as "
                "final lane objects."
            ),
            "negative_slot_probe": (
                "The frozen-memory lower-bound did not win. This does not "
                "falsify a from-scratch V6 because routing and slot geometry "
                "were not allowed to co-adapt upstream features."
            ),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
