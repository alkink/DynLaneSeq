from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the matched 25k row-reference train-many/infer-one "
            "causal gate from exact official-raster validation reports."
        )
    )
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--candidate-all-json", required=True)
    parser.add_argument("--candidate-group-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same_optional_float(left: Any, right: Any, atol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= atol


def _stage(report: dict[str, Any]) -> str:
    stages = {str(row["stage"]) for row in report.get("rows", [])}
    for preferred in ("main", "final", "stage2", "stage1", "coarse"):
        if preferred in stages:
            return preferred
    if not stages:
        raise ValueError("Report has no prediction stage")
    return sorted(stages)[-1]


def _comparability(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [report.get("metadata", {}) for report in reports]
    fields = (
        "split",
        "list_sha256",
        "max_batches",
        "num_records",
        "iou_space",
        "sample_strategy",
        "sampled_dataset_indices",
    )
    checks = {
        field: all(meta.get(field) == metadata[0].get(field) for meta in metadata[1:])
        for field in fields
    }
    return {"all_checks_pass": all(checks.values()), "checks": checks}


def _row(
    report: dict[str, Any],
    *,
    strategy: str,
    iou_threshold: float,
    top_k: int,
    quality_power: float | None,
    score_threshold: float | None,
) -> dict[str, Any]:
    stage = _stage(report)
    matches = [
        row
        for row in report.get("rows", [])
        if str(row.get("stage")) == stage
        and str(row.get("strategy")) == strategy
        and int(row.get("top_k", 0)) == int(top_k)
        and _same_optional_float(row.get("iou_threshold"), iou_threshold)
        and _same_optional_float(row.get("quality_power"), quality_power)
        and _same_optional_float(row.get("score_threshold"), score_threshold)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one row for "
            f"{strategy=}, {iou_threshold=}, {top_k=}, {quality_power=}, "
            f"{score_threshold=}; found {len(matches)}"
        )
    return matches[0]


def _official(row: dict[str, Any], iou_threshold: float) -> dict[str, Any]:
    key = f"official_iou_{float(iou_threshold):g}"
    if key not in row:
        raise ValueError(f"Row is missing exact official metric {key!r}")
    return row[key]


def _operating_points(
    report: dict[str, Any],
    *,
    strategy: str,
    quality_power: float,
) -> list[dict[str, Any]]:
    stage = _stage(report)
    thresholds = sorted(
        {
            float(row["score_threshold"])
            for row in report.get("rows", [])
            if str(row.get("stage")) == stage
            and str(row.get("strategy")) == strategy
            and int(row.get("top_k", 0)) == 4
            and _same_optional_float(row.get("quality_power"), quality_power)
            and row.get("score_threshold") is not None
        }
    )
    points = []
    for threshold in thresholds:
        metrics = {}
        for iou_threshold in (0.5, 0.75):
            metrics[f"{iou_threshold:.2f}"] = _official(
                _row(
                    report,
                    strategy=strategy,
                    iou_threshold=iou_threshold,
                    top_k=4,
                    quality_power=quality_power,
                    score_threshold=threshold,
                ),
                iou_threshold,
            )
        points.append({"score_threshold": threshold, "metrics": metrics})
    if not points:
        raise ValueError(f"No operating points found for strategy={strategy!r}")
    return points


def _best_point(points: list[dict[str, Any]]) -> dict[str, Any]:
    # CULane's primary comparison is F1@0.50.  Strict-IoU F1 and precision
    # break ties, and the threshold remains one shared value for both IoUs.
    return max(
        points,
        key=lambda point: (
            float(point["metrics"]["0.50"]["f1"]),
            float(point["metrics"]["0.75"]["f1"]),
            float(point["metrics"]["0.50"]["precision"]),
            float(point["score_threshold"]),
        ),
    )


def _candidate_count(report: dict[str, Any]) -> int:
    stage = _stage(report)
    counts = report.get("metadata", {}).get("candidate_counts_by_stage", {})
    if stage not in counts:
        raise ValueError("Report does not record candidate count")
    return int(counts[stage])


def summarize(
    base: dict[str, Any],
    candidate_all: dict[str, Any],
    candidate_group: dict[str, Any],
) -> dict[str, Any]:
    reports = [base, candidate_all, candidate_group]
    comparability = _comparability(reports)
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Reports are not paired: {comparability}")
    counts = {
        "base": _candidate_count(base),
        "candidate_all": _candidate_count(candidate_all),
        "candidate_group": _candidate_count(candidate_group),
    }
    if counts != {"base": 32, "candidate_all": 32, "candidate_group": 8}:
        raise ValueError(f"Unexpected candidate-count contract: {counts}")

    capacity: dict[str, Any] = {}
    for iou_threshold in (0.5, 0.75):
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

        base_all = recall(base, "all_raw", 0)
        candidate_all_raw = recall(candidate_all, "all_raw", 0)
        candidate_group_raw = recall(candidate_group, "all_raw", 0)
        base_oracle = recall(base, "oracle_topk", 4)
        candidate_all_oracle = recall(candidate_all, "oracle_topk", 4)
        candidate_group_oracle = recall(candidate_group, "oracle_topk", 4)
        candidate_quality = recall(candidate_group, "quality_topk", 4)
        capacity[key] = {
            "base_all32_recall": base_all,
            "candidate_all32_recall": candidate_all_raw,
            "candidate_group0_all8_recall": candidate_group_raw,
            "base_oracle_top4_recall": base_oracle,
            "candidate_all32_oracle_top4_recall": candidate_all_oracle,
            "candidate_group0_oracle_top4_recall": candidate_group_oracle,
            "candidate_group0_quality_top4_recall": candidate_quality,
            "candidate_all32_gain_over_base_points": 100.0
            * (candidate_all_raw - base_all),
            "group0_oracle_gap_from_candidate_all32_points": 100.0
            * (candidate_group_oracle - candidate_all_oracle),
            "group0_quality_ranking_gap_points": 100.0
            * (candidate_group_oracle - candidate_quality),
        }

    base_points = _operating_points(
        base,
        strategy="model_topk_nms",
        quality_power=0.25,
    )
    candidate_points = _operating_points(
        candidate_group,
        strategy="quality_topk_nms",
        quality_power=1.0,
    )
    base_best = _best_point(base_points)
    candidate_best = _best_point(candidate_points)
    f1_deltas = {
        key: 100.0
        * (
            float(candidate_best["metrics"][key]["f1"])
            - float(base_best["metrics"][key]["f1"])
        )
        for key in ("0.50", "0.75")
    }
    all32_capacity_deltas = [
        float(capacity[key]["candidate_all32_gain_over_base_points"])
        for key in ("0.50", "0.75")
    ]

    geometry_preserved = min(all32_capacity_deltas) >= -1.0
    selection_positive = (
        f1_deltas["0.50"] >= 1.0 and f1_deltas["0.75"] >= -0.25
    )
    if selection_positive and geometry_preserved:
        verdict = "positive_continue_full_validation"
    elif selection_positive:
        verdict = "selection_signal_but_geometry_regressed"
    elif geometry_preserved:
        verdict = "geometry_preserved_selection_not_fixed"
    else:
        verdict = "negative_stop"

    return {
        "diagnostic_only": True,
        "question": (
            "Does four-group one-to-many training make row-reference candidates "
            "selectable by one predeclared eight-query group without lane NMS?"
        ),
        "protocol": {
            "iteration": 25000,
            "sample": "paired uniform validation subset",
            "official_raster_iou": True,
            "candidate_group": 0,
            "candidate_score_mode": "quality",
            "candidate_nms_distance_px": 0.0,
            "base_score_mode": "exist_quality",
            "base_quality_power": 0.25,
            "base_nms_distance_px": 20.0,
            "threshold_selection": "validation F1@0.50, strict F1 tie-break",
        },
        "comparability": comparability,
        "candidate_counts": counts,
        "capacity": capacity,
        "validation_operating_points": {
            "base": base_points,
            "candidate_group0": candidate_points,
            "base_best": base_best,
            "candidate_group0_best": candidate_best,
            "candidate_minus_base_f1_points": f1_deltas,
        },
        "gate": {
            "geometry_preserved": geometry_preserved,
            "selection_positive": selection_positive,
            "requirements": {
                "min_f1_050_gain_points": 1.0,
                "min_f1_075_gain_points": -0.25,
                "min_all32_capacity_delta_each_iou_points": -1.0,
            },
            "verdict": verdict,
            "note": (
                "A positive 256-image gate licenses a complete validation "
                "evaluation; it is not a CULane test-set claim."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    payload = summarize(
        _load(args.base_json),
        _load(args.candidate_all_json),
        _load(args.candidate_group_json),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
