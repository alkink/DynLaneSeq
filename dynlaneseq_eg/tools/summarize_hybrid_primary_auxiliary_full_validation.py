from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.tools.summarize_train_many_infer_one_gate import (
    _candidate_count,
    _comparability,
    _official,
    _row,
)


IOU_THRESHOLDS = (0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the full-validation confirmation of the primary32 "
            "plus train-only auxiliary-query experiment at predeclared and "
            "historical operating points."
        )
    )
    parser.add_argument("--control-q025-json", required=True)
    parser.add_argument("--candidate-q025-json", required=True)
    parser.add_argument("--control-historical-json", required=True)
    parser.add_argument("--candidate-historical-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_full_validation(reports: list[dict[str, Any]]) -> int:
    counts = []
    for report in reports:
        metadata = report.get("metadata", {})
        if str(metadata.get("split")) != "val":
            raise ValueError("full-validation confirmation requires split=val")
        if int(metadata.get("max_batches", -1)) != 0:
            raise ValueError(
                "full-validation confirmation refuses a max-batches subset"
            )
        records = int(metadata.get("num_records", 0))
        sampled = metadata.get("sampled_dataset_indices")
        if records <= 0 or not isinstance(sampled, list) or len(sampled) != records:
            raise ValueError(
                "full-validation report must cover every recorded dataset index"
            )
        counts.append(records)
    if len(set(counts)) != 1:
        raise ValueError(f"full-validation record counts differ: {counts}")
    return counts[0]


def _point(
    report: dict[str, Any],
    *,
    quality_power: float,
    score_threshold: float,
) -> dict[str, Any]:
    metrics = {}
    for iou_threshold in IOU_THRESHOLDS:
        metrics[f"{iou_threshold:.2f}"] = _official(
            _row(
                report,
                strategy="model_topk_nms",
                iou_threshold=iou_threshold,
                top_k=4,
                quality_power=quality_power,
                score_threshold=score_threshold,
            ),
            iou_threshold,
        )
    return {
        "quality_power": float(quality_power),
        "score_threshold": float(score_threshold),
        "metrics": metrics,
    }


def _delta(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ("0.50", "0.75"):
        candidate_metric = candidate["metrics"][key]
        control_metric = control["metrics"][key]
        out[key] = {
            "f1_points": 100.0
            * (float(candidate_metric["f1"]) - float(control_metric["f1"])),
            "precision_points": 100.0
            * (
                float(candidate_metric["precision"])
                - float(control_metric["precision"])
            ),
            "recall_points": 100.0
            * (
                float(candidate_metric["recall"])
                - float(control_metric["recall"])
            ),
            "tp": int(candidate_metric["tp"]) - int(control_metric["tp"]),
            "fp": int(candidate_metric["fp"]) - int(control_metric["fp"]),
            "fn": int(candidate_metric["fn"]) - int(control_metric["fn"]),
        }
    return out


def _capacity(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    capacity = {}
    for iou_threshold in IOU_THRESHOLDS:
        key = f"{iou_threshold:.2f}"

        def recall(report: dict[str, Any], strategy: str, top_k: int) -> float:
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

        control_raw = recall(control, "all_raw", 0)
        candidate_raw = recall(candidate, "all_raw", 0)
        control_quality = recall(control, "quality_topk", 4)
        candidate_quality = recall(candidate, "quality_topk", 4)
        candidate_oracle = recall(candidate, "oracle_topk", 4)
        capacity[key] = {
            "control_primary_all32_recall": control_raw,
            "candidate_primary_all32_recall": candidate_raw,
            "all32_delta_points": 100.0 * (candidate_raw - control_raw),
            "control_quality_top4_recall": control_quality,
            "candidate_quality_top4_recall": candidate_quality,
            "quality_top4_delta_points": 100.0
            * (candidate_quality - control_quality),
            "candidate_oracle_top4_recall": candidate_oracle,
            "candidate_quality_ranking_gap_points": 100.0
            * (candidate_oracle - candidate_quality),
        }
    return capacity


def summarize(
    control_q025: dict[str, Any],
    candidate_q025: dict[str, Any],
    control_historical: dict[str, Any],
    candidate_historical: dict[str, Any],
) -> dict[str, Any]:
    reports = [
        control_q025,
        candidate_q025,
        control_historical,
        candidate_historical,
    ]
    comparability = _comparability(reports)
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Reports are not paired: {comparability}")
    images = _assert_full_validation(reports)
    candidate_counts = {
        "control_primary": _candidate_count(control_q025),
        "candidate_primary": _candidate_count(candidate_q025),
        "control_historical": _candidate_count(control_historical),
        "candidate_historical": _candidate_count(candidate_historical),
    }
    if set(candidate_counts.values()) != {32}:
        raise ValueError(
            f"Every inference arm must emit 32 primary queries: {candidate_counts}"
        )

    control_selected = _point(
        control_q025,
        quality_power=0.25,
        score_threshold=0.15,
    )
    candidate_selected = _point(
        candidate_q025,
        quality_power=0.25,
        score_threshold=0.20,
    )
    selected_delta = _delta(candidate_selected, control_selected)

    matched_points = {}
    for threshold in (0.15, 0.20):
        control_point = _point(
            control_q025,
            quality_power=0.25,
            score_threshold=threshold,
        )
        candidate_point = _point(
            candidate_q025,
            quality_power=0.25,
            score_threshold=threshold,
        )
        matched_points[f"q0.25_score{threshold:.2f}"] = {
            "control": control_point,
            "candidate": candidate_point,
            "candidate_minus_control": _delta(candidate_point, control_point),
        }

    control_historical_point = _point(
        control_historical,
        quality_power=0.50,
        score_threshold=0.30,
    )
    candidate_historical_point = _point(
        candidate_historical,
        quality_power=0.50,
        score_threshold=0.30,
    )
    historical_delta = _delta(
        candidate_historical_point,
        control_historical_point,
    )
    capacity = _capacity(control_q025, candidate_q025)

    geometry_preserved = min(
        float(capacity[key]["all32_delta_points"])
        for key in ("0.50", "0.75")
    ) >= -1.0
    primary_preserved = float(selected_delta["0.50"]["f1_points"]) >= 0.0
    strict_improved = float(selected_delta["0.75"]["f1_points"]) >= 1.0
    confirmed = geometry_preserved and primary_preserved and strict_improved
    verdict = (
        "confirmed_continue_exact_checkpoint_to_50k"
        if confirmed
        else "not_confirmed_stop_before_long_training"
    )

    return {
        "diagnostic_only": True,
        "question": (
            "Does the 25k primary32+aux3x8 checkpoint preserve primary F1 "
            "and improve strict localization on the complete validation set?"
        ),
        "protocol": {
            "iteration": 25000,
            "split": "val",
            "images": images,
            "official_raster_iou": True,
            "top_k": 4,
            "lane_nms_distance_px": 20.0,
            "control_selected": {"quality_power": 0.25, "score_threshold": 0.15},
            "candidate_selected": {"quality_power": 0.25, "score_threshold": 0.20},
            "historical_shared": {"quality_power": 0.50, "score_threshold": 0.30},
            "selection_note": (
                "The q0.25 operating points were frozen from the preceding "
                "uniform-256 diagnostic; q0.50/score0.30 is the pre-existing "
                "historical project setting. No full-validation sweep is used "
                "for the continuation verdict."
            ),
        },
        "comparability": comparability,
        "candidate_counts": candidate_counts,
        "capacity": capacity,
        "selected_operating_points": {
            "control": control_selected,
            "candidate": candidate_selected,
            "candidate_minus_control": selected_delta,
        },
        "matched_q0.25_operating_points": matched_points,
        "historical_q0.50_score0.30": {
            "control": control_historical_point,
            "candidate": candidate_historical_point,
            "candidate_minus_control": historical_delta,
            "role": "robustness check only; it does not alter the gate",
        },
        "gate": {
            "geometry_preserved": geometry_preserved,
            "primary_f1_preserved": primary_preserved,
            "strict_f1_improved": strict_improved,
            "requirements": {
                "min_all32_capacity_delta_each_iou_points": -1.0,
                "min_selected_f1_050_gain_points": 0.0,
                "min_selected_f1_075_gain_points": 1.0,
            },
            "verdict": verdict,
            "note": (
                "Only a confirmed full-validation result licenses continuation "
                "of this exact 25k checkpoint; this is not a test-set claim."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    payload = summarize(
        _load(args.control_q025_json),
        _load(args.candidate_q025_json),
        _load(args.control_historical_json),
        _load(args.candidate_historical_json),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
