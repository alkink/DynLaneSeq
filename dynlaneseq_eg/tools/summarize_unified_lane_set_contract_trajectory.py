from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize frozen unified lane-set contract audits over checkpoints."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def flatten(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration": int(payload["checkpoint_iteration"]),
        "checkpoint": payload["checkpoint"],
        "images": int(payload.get("images", 0)),
        "gradient_images": int(payload.get("gradient_images", 0)),
        "f1_050_at_score_0p20": _nested(
            payload,
            "deployed_operating_points",
            "score_0.20",
            "0.50",
            "f1",
        ),
        "precision_050_at_score_0p20": _nested(
            payload,
            "deployed_operating_points",
            "score_0.20",
            "0.50",
            "precision",
        ),
        "recall_050_at_score_0p20": _nested(
            payload,
            "deployed_operating_points",
            "score_0.20",
            "0.50",
            "recall",
        ),
        "f1_075_at_score_0p20": _nested(
            payload,
            "deployed_operating_points",
            "score_0.20",
            "0.75",
            "f1",
        ),
        "f1_050_at_score_0p30": _nested(
            payload,
            "deployed_operating_points",
            "score_0.30",
            "0.50",
            "f1",
        ),
        "precision_050_at_score_0p30": _nested(
            payload,
            "deployed_operating_points",
            "score_0.30",
            "0.50",
            "precision",
        ),
        "recall_050_at_score_0p30": _nested(
            payload,
            "deployed_operating_points",
            "score_0.30",
            "0.50",
            "recall",
        ),
        "all_candidates_recall_050": _nested(
            payload, "capacity", "0.50", "all_candidates_recall"
        ),
        "direct_top4_recall_050": _nested(
            payload, "capacity", "0.50", "direct_topk_recall"
        ),
        "oracle_top4_recall_050": _nested(
            payload, "capacity", "0.50", "oracle_topk_recall"
        ),
        "oracle_minus_direct_050_points": _nested(
            payload,
            "capacity",
            "0.50",
            "oracle_minus_direct_recall_points",
        ),
        "all_candidates_recall_075": _nested(
            payload, "capacity", "0.75", "all_candidates_recall"
        ),
        "direct_top4_recall_075": _nested(
            payload, "capacity", "0.75", "direct_topk_recall"
        ),
        "oracle_top4_recall_075": _nested(
            payload, "capacity", "0.75", "oracle_topk_recall"
        ),
        "score_iou_pearson": _nested(
            payload,
            "score_official_iou_alignment",
            "pearson_score_vs_best_official_iou",
        ),
        "matched_mean_score": _nested(
            payload,
            "score_official_iou_alignment",
            "matched_mean_score",
        ),
        "unmatched_mean_score": _nested(
            payload,
            "score_official_iou_alignment",
            "unmatched_mean_score",
        ),
        "matched_mean_best_official_iou": _nested(
            payload,
            "score_official_iou_alignment",
            "matched_mean_best_official_iou",
        ),
        "unique_candidate_ap_050": _nested(
            payload,
            "score_official_iou_alignment",
            "unique_candidate_ap_050",
        ),
        "mean_foreground_probability_mass": _nested(
            payload,
            "count_calibration",
            "mean_foreground_probability_mass",
        ),
        "mean_training_target_lane_count": _nested(
            payload,
            "count_calibration",
            "mean_training_target_lane_count",
        ),
        "probability_count_mae": _nested(
            payload,
            "count_calibration",
            "probability_mass_vs_training_count_mae",
        ),
        "backbone_geometry_score_cosine": _nested(
            payload,
            "gradient_alignment",
            "shared_backbone",
            "cosine_geometry_vs_score",
            "mean",
        ),
        "fpn_c4_c5_geometry_score_cosine": _nested(
            payload,
            "gradient_alignment",
            "shared_fpn_lateral_c4_c5",
            "cosine_geometry_vs_score",
            "mean",
        ),
        "lane_state_geometry_score_cosine": _nested(
            payload,
            "gradient_alignment",
            "unified_lane_state_core",
            "cosine_geometry_vs_score",
            "mean",
        ),
        "primary_signal": _nested(payload, "verdict", "primary_signal"),
    }
    return row


def main() -> None:
    args = parse_args()
    rows = sorted(
        (flatten(_load(path)) for path in args.inputs),
        key=lambda row: int(row["iteration"]),
    )
    if not rows:
        raise ValueError("at least one trajectory input is required")
    iterations = [int(row["iteration"]) for row in rows]
    if len(iterations) != len(set(iterations)):
        raise ValueError(f"duplicate checkpoint iterations: {iterations}")

    payload = {
        "diagnostic_only": True,
        "selection_rule": (
            "Two frozen points (score=0.20 and the historical score=0.30), "
            "quality=0, Top-4, NMS-free on identical uniform validation "
            "indices; no per-checkpoint threshold selection."
        ),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output_json}")
    print(f"output_csv: {output_csv}")


if __name__ == "__main__":
    main()
