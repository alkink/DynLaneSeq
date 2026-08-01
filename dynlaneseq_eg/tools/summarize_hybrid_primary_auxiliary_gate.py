from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.tools.summarize_train_many_infer_one_gate import (
    _best_point,
    _candidate_count,
    _comparability,
    _operating_points,
    _row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the paired 25k full-primary plus train-only auxiliary "
            "query gate using exact CULane raster metrics."
        )
    )
    parser.add_argument("--control-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _recall(
    report: dict[str, Any],
    strategy: str,
    iou_threshold: float,
    top_k: int,
) -> float:
    return float(
        _row(
            report,
            strategy=strategy,
            iou_threshold=iou_threshold,
            top_k=top_k,
            quality_power=None,
            score_threshold=None,
        )["recall"]
    )


def summarize(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    comparability = _comparability([control, candidate])
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Reports are not paired: {comparability}")
    counts = {
        "control_primary": _candidate_count(control),
        "candidate_primary": _candidate_count(candidate),
    }
    if counts != {"control_primary": 32, "candidate_primary": 32}:
        raise ValueError(f"Unexpected primary candidate-count contract: {counts}")

    capacity: dict[str, Any] = {}
    for iou_threshold in (0.5, 0.75):
        key = f"{iou_threshold:.2f}"
        control_all = _recall(control, "all_raw", iou_threshold, 0)
        candidate_all = _recall(candidate, "all_raw", iou_threshold, 0)
        control_oracle = _recall(control, "oracle_topk", iou_threshold, 4)
        candidate_oracle = _recall(candidate, "oracle_topk", iou_threshold, 4)
        control_quality = _recall(control, "quality_topk", iou_threshold, 4)
        candidate_quality = _recall(candidate, "quality_topk", iou_threshold, 4)
        capacity[key] = {
            "control_primary_all32_recall": control_all,
            "candidate_primary_all32_recall": candidate_all,
            "all32_delta_points": 100.0 * (candidate_all - control_all),
            "control_oracle_top4_recall": control_oracle,
            "candidate_oracle_top4_recall": candidate_oracle,
            "oracle_top4_delta_points": 100.0
            * (candidate_oracle - control_oracle),
            "control_quality_top4_recall": control_quality,
            "candidate_quality_top4_recall": candidate_quality,
            "quality_top4_delta_points": 100.0
            * (candidate_quality - control_quality),
            "candidate_quality_ranking_gap_points": 100.0
            * (candidate_oracle - candidate_quality),
        }

    control_points = _operating_points(
        control,
        strategy="model_topk_nms",
        quality_power=0.25,
    )
    candidate_points = _operating_points(
        candidate,
        strategy="model_topk_nms",
        quality_power=0.25,
    )
    control_best = _best_point(control_points)
    candidate_best = _best_point(candidate_points)
    f1_deltas = {
        key: 100.0
        * (
            float(candidate_best["metrics"][key]["f1"])
            - float(control_best["metrics"][key]["f1"])
        )
        for key in ("0.50", "0.75")
    }
    geometry_preserved = min(
        float(capacity[key]["all32_delta_points"])
        for key in ("0.50", "0.75")
    ) >= -1.0
    selection_positive = (
        f1_deltas["0.50"] >= 0.0 and f1_deltas["0.75"] >= 0.5
    )
    if geometry_preserved and selection_positive:
        verdict = "positive_continue_same_checkpoint"
    elif geometry_preserved:
        verdict = "geometry_preserved_but_auxiliary_supervision_not_helpful"
    elif selection_positive:
        verdict = "selection_signal_but_primary_geometry_regressed"
    else:
        verdict = "negative_stop"

    return {
        "diagnostic_only": True,
        "question": (
            "Can a full 32-query one-to-one primary set preserve proposal "
            "capacity while three train-only 8-query groups improve selection?"
        ),
        "protocol": {
            "iteration": 25000,
            "sample": "paired uniform validation subset",
            "images": int(
                control.get("metadata", {}).get("num_records", 0)
            ),
            "official_raster_iou": True,
            "control_training_queries": 32,
            "candidate_training_queries": 56,
            "candidate_inference_queries": 32,
            "candidate_auxiliary_group_sizes": [8, 8, 8],
            "score_mode": "exist_quality",
            "quality_power": 0.25,
            "lane_nms_distance_px": 20.0,
            "threshold_selection": (
                "validation F1@0.50 with strict-IoU and precision tie-break"
            ),
        },
        "comparability": comparability,
        "candidate_counts": counts,
        "capacity": capacity,
        "validation_operating_points": {
            "control": control_points,
            "candidate": candidate_points,
            "control_best": control_best,
            "candidate_best": candidate_best,
            "candidate_minus_control_f1_points": f1_deltas,
        },
        "gate": {
            "geometry_preserved": geometry_preserved,
            "selection_positive": selection_positive,
            "requirements": {
                "min_all32_capacity_delta_each_iou_points": -1.0,
                "min_f1_050_gain_points": 0.0,
                "min_f1_075_gain_points": 0.5,
            },
            "verdict": verdict,
            "note": (
                "A positive subset gate licenses continuation of this exact "
                "checkpoint; it is not a CULane test-set result."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    payload = summarize(_load(args.control_json), _load(args.candidate_json))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()

