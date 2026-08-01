from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the matched 25k coherent lane-state gate from exact "
            "ranking/capacity and query-ownership diagnostics."
        )
    )
    parser.add_argument("--ranking-summary", required=True)
    parser.add_argument("--ownership-report", required=True)
    parser.add_argument("--control-calibration", required=True)
    parser.add_argument("--candidate-calibration", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same_optional(left: Any, right: Any, atol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= atol


def _row(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    iou_threshold: float,
    quality_power: float | None = None,
    score_threshold: float | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if str(row.get("strategy")) == strategy
        and int(row.get("top_k", 0)) == int(top_k)
        and _same_optional(row.get("iou_threshold"), iou_threshold)
        and _same_optional(row.get("quality_power"), quality_power)
        and _same_optional(row.get("score_threshold"), score_threshold)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ranking row for "
            f"{strategy=}, {iou_threshold=}, {quality_power=}, "
            f"{score_threshold=}, {top_k=}; found {len(matches)}"
        )
    return matches[0]


def _report_stage(report: dict[str, Any]) -> str:
    stages = {str(row["stage"]) for row in report.get("rows", [])}
    for preferred in ("main", "final", "stage2", "stage1", "coarse"):
        if preferred in stages:
            return preferred
    if not stages:
        raise ValueError("calibration report contains no prediction stage")
    return sorted(stages)[-1]


def _calibration_comparable(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    left = control.get("metadata", {})
    right = candidate.get("metadata", {})
    return all(
        left.get(field) == right.get(field)
        for field in (
            "split",
            "list_sha256",
            "max_batches",
            "num_records",
            "iou_space",
            "sample_strategy",
            "sampled_dataset_indices",
        )
    )


def _operating_points(report: dict[str, Any]) -> list[dict[str, Any]]:
    stage = _report_stage(report)
    thresholds = sorted(
        {
            float(row["score_threshold"])
            for row in report.get("rows", [])
            if str(row.get("stage")) == stage
            and str(row.get("strategy")) == "model_topk_nms"
            and int(row.get("top_k", 0)) == 4
            and _same_optional(row.get("quality_power"), 0.0)
            and row.get("score_threshold") is not None
            and float(row["score_threshold"]) >= 0.0
        }
    )
    points = []
    for score_threshold in thresholds:
        metrics = {}
        for iou_threshold in (0.5, 0.75):
            matches = [
                row
                for row in report.get("rows", [])
                if str(row.get("stage")) == stage
                and str(row.get("strategy")) == "model_topk_nms"
                and int(row.get("top_k", 0)) == 4
                and _same_optional(row.get("iou_threshold"), iou_threshold)
                and _same_optional(row.get("quality_power"), 0.0)
                and _same_optional(
                    row.get("score_threshold"), score_threshold
                )
            ]
            if len(matches) != 1:
                raise ValueError("calibration operating point is incomplete")
            key = f"official_iou_{iou_threshold:g}"
            if key not in matches[0]:
                raise ValueError(f"calibration row is missing {key}")
            metrics[f"{iou_threshold:.2f}"] = matches[0][key]
        points.append(
            {
                "score_threshold": score_threshold,
                "metrics": metrics,
            }
        )
    if not points:
        raise ValueError("no NMS-free direct-existence operating points found")
    return points


def _best_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        points,
        key=lambda point: (
            float(point["metrics"]["0.50"]["f1"]),
            float(point["metrics"]["0.75"]["f1"]),
            float(point["metrics"]["0.50"]["precision"]),
            float(point["score_threshold"]),
        ),
    )


def summarize(
    ranking: dict[str, Any],
    ownership: dict[str, Any],
    control_calibration: dict[str, Any],
    candidate_calibration: dict[str, Any],
) -> dict[str, Any]:
    comparability = ranking.get("comparability", {})
    if not bool(comparability.get("all_checks_pass", False)):
        raise ValueError(f"ranking inputs are not paired: {comparability}")
    if str(ownership.get("score_mode")) != "exist":
        raise ValueError("ownership trajectory must use direct existence scores")
    if not _calibration_comparable(
        control_calibration,
        candidate_calibration,
    ):
        raise ValueError("calibration reports are not paired")
    for name, report in (
        ("control", control_calibration),
        ("candidate", candidate_calibration),
    ):
        nms_distance = float(
            report.get("metadata", {}).get("nms_distance_thresh_px", -1.0)
        )
        if nms_distance != 0.0:
            raise ValueError(f"{name} calibration is not NMS-free")

    rows = list(ranking.get("rows", []))
    metrics: dict[str, Any] = {}
    capacity_deltas: list[float] = []
    direct_top4_deltas: list[float] = []
    control_nms_gains: list[float] = []
    candidate_nms_gains: list[float] = []
    oracle_gap_shrink: list[float] = []
    for threshold in (0.5, 0.75):
        all_raw = _row(
            rows,
            strategy="all_raw",
            iou_threshold=threshold,
            quality_power=None,
            score_threshold=None,
            top_k=0,
        )
        oracle = _row(
            rows,
            strategy="oracle_topk",
            iou_threshold=threshold,
            quality_power=None,
            score_threshold=None,
        )
        direct = _row(
            rows,
            strategy="model_topk",
            iou_threshold=threshold,
            quality_power=0.0,
            score_threshold=None,
        )
        nms = _row(
            rows,
            strategy="model_topk_nms",
            iou_threshold=threshold,
            quality_power=0.0,
            score_threshold=-1.0,
        )

        capacity_delta = float(all_raw["delta_recall_points"])
        direct_delta = float(direct["delta_recall_points"])
        control_nms_gain = 100.0 * (
            float(nms["base_recall"]) - float(direct["base_recall"])
        )
        candidate_nms_gain = 100.0 * (
            float(nms["candidate_recall"])
            - float(direct["candidate_recall"])
        )
        control_oracle_gap = 100.0 * (
            float(oracle["base_recall"]) - float(direct["base_recall"])
        )
        candidate_oracle_gap = 100.0 * (
            float(oracle["candidate_recall"])
            - float(direct["candidate_recall"])
        )

        capacity_deltas.append(capacity_delta)
        direct_top4_deltas.append(direct_delta)
        control_nms_gains.append(control_nms_gain)
        candidate_nms_gains.append(candidate_nms_gain)
        oracle_gap_shrink.append(control_oracle_gap - candidate_oracle_gap)
        metrics[f"{threshold:.2f}"] = {
            "all_32_capacity": {
                "control_recall": float(all_raw["base_recall"]),
                "candidate_recall": float(all_raw["candidate_recall"]),
                "delta_points": capacity_delta,
            },
            "oracle_top4_capacity": {
                "control_recall": float(oracle["base_recall"]),
                "candidate_recall": float(oracle["candidate_recall"]),
                "delta_points": float(oracle["delta_recall_points"]),
            },
            "direct_exist_top4": {
                "control_recall": float(direct["base_recall"]),
                "candidate_recall": float(direct["candidate_recall"]),
                "delta_points": direct_delta,
            },
            "nms_gain_over_direct_top4_points": {
                "control": control_nms_gain,
                "candidate": candidate_nms_gain,
                "reduction": control_nms_gain - candidate_nms_gain,
            },
            "oracle_to_direct_top4_gap_points": {
                "control": control_oracle_gap,
                "candidate": candidate_oracle_gap,
                "shrink": control_oracle_gap - candidate_oracle_gap,
            },
        }

    owner_gate = ownership.get("gate", {})
    owner_retention = float(
        owner_gate.get(
            "weighted_consecutive_training_owner_retention_recoverable",
            0.0,
        )
    )
    mean_capacity_delta = sum(capacity_deltas) / len(capacity_deltas)
    mean_direct_delta = sum(direct_top4_deltas) / len(direct_top4_deltas)
    mean_nms_reduction = sum(
        control - candidate
        for control, candidate in zip(control_nms_gains, candidate_nms_gains)
    ) / len(control_nms_gains)
    mean_oracle_gap_shrink = sum(oracle_gap_shrink) / len(oracle_gap_shrink)

    control_points = _operating_points(control_calibration)
    candidate_points = _operating_points(candidate_calibration)
    control_best = _best_point(control_points)
    candidate_best = _best_point(candidate_points)
    best_f1_deltas = {
        key: 100.0
        * (
            float(candidate_best["metrics"][key]["f1"])
            - float(control_best["metrics"][key]["f1"])
        )
        for key in ("0.50", "0.75")
    }

    if (
        best_f1_deltas["0.50"] >= 0.5
        and best_f1_deltas["0.75"] >= -0.5
        and mean_direct_delta >= 0.5
        and min(capacity_deltas) >= -0.5
        and owner_retention >= 0.75
    ):
        verdict = "positive_continue_training"
    elif (
        best_f1_deltas["0.50"] >= 0.0
        and mean_direct_delta >= 0.25
        and min(capacity_deltas) >= -1.0
        and (mean_nms_reduction >= 0.5 or mean_oracle_gap_shrink >= 0.5)
    ):
        verdict = "positive_contract_signal_needs_longer_gate"
    elif mean_capacity_delta <= -1.0 and mean_direct_delta <= -1.0:
        verdict = "negative_stop_architecture"
    else:
        verdict = "mixed_or_inconclusive"

    return {
        "diagnostic_only": True,
        "question": (
            "Does a persistent same-query lane state with direct existence "
            "supervision and detached iterative references improve NMS-free "
            "Top-4 selection without sacrificing geometric capacity?"
        ),
        "matched_setting": {
            "iteration": 25000,
            "sample": "uniform 256-image CULane validation subset",
            "official_raster_iou": True,
            "candidate_score": "direct existence",
            "top_k": 4,
            "candidate_deployment_nms": False,
            "scheduler_horizon": 278000,
            "seed": 3407,
        },
        "metrics": metrics,
        "nms_free_direct_existence_calibration": {
            "threshold_grid": [
                float(point["score_threshold"])
                for point in candidate_points
            ],
            "control_points": control_points,
            "candidate_points": candidate_points,
            "control_best": control_best,
            "candidate_best": candidate_best,
            "candidate_minus_control_best_f1_points": best_f1_deltas,
            "selection_rule": (
                "best validation F1@0.50; F1@0.75, precision, and the "
                "higher threshold break ties"
            ),
        },
        "aggregate": {
            "mean_all_32_capacity_delta_points": mean_capacity_delta,
            "mean_direct_exist_top4_delta_points": mean_direct_delta,
            "mean_nms_dependency_reduction_points": mean_nms_reduction,
            "mean_oracle_gap_shrink_points": mean_oracle_gap_shrink,
            "weighted_training_owner_retention": owner_retention,
            "best_nms_free_f1_050_delta_points": best_f1_deltas["0.50"],
            "best_nms_free_f1_075_delta_points": best_f1_deltas["0.75"],
        },
        "ownership_gate": owner_gate,
        "verdict": verdict,
        "verdict_note": (
            "This short validation gate decides whether the contract deserves "
            "longer training; it is not a test-set or final-F1 claim."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = summarize(
        _load(args.ranking_summary),
        _load(args.ownership_report),
        _load(args.control_calibration),
        _load(args.candidate_calibration),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
