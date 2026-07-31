from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.tools.analyze_oracle_topk import _official_metric_key
from dynlaneseq_eg.tools.summarize_nms_ranking_pair import (
    _common_stage,
    _comparability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched official CULane precision-recall frontiers from "
            "two analyze_oracle_topk reports."
        )
    )
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _grid_key(point: dict[str, Any]) -> tuple[float, float]:
    return (
        round(float(point["quality_power"]), 12),
        round(float(point["score_threshold"]), 12),
    )


def _extract_points(
    report: dict[str, Any],
    *,
    stage: str,
    iou_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    metric_key = _official_metric_key(iou_threshold)
    points: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        if (
            str(row.get("stage")) != stage
            or str(row.get("strategy")) != "model_topk_nms"
            or int(row.get("top_k", 0)) != int(top_k)
            or abs(float(row.get("iou_threshold", -1.0)) - iou_threshold) > 1e-9
        ):
            continue
        metrics = row.get(metric_key)
        if not isinstance(metrics, dict):
            raise KeyError(
                f"Row is missing exact metric {metric_key}: "
                f"q={row.get('quality_power')}, "
                f"threshold={row.get('score_threshold')}"
            )
        points.append(
            {
                "quality_power": float(row["quality_power"]),
                "score_threshold": float(row["score_threshold"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            }
        )
    if not points:
        raise ValueError(
            f"No official PR points for stage={stage}, IoU={iou_threshold}, "
            f"Top-K={top_k}"
        )
    keys = [_grid_key(point) for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"Duplicate quality/threshold settings for IoU={iou_threshold}"
        )
    return sorted(points, key=_grid_key)


def _best(points: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        max(
            points,
            key=lambda point: (
                point["f1"],
                point["recall"],
                point["precision"],
                -point["quality_power"],
                -point["score_threshold"],
            ),
        )
    )


def _deduplicate_outcomes(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for point in points:
        key = (int(point["tp"]), int(point["fp"]), int(point["fn"]))
        grouped.setdefault(key, []).append(point)
    unique: list[dict[str, Any]] = []
    for settings in grouped.values():
        representative = dict(sorted(settings, key=_grid_key)[0])
        representative["equivalent_settings"] = [
            {
                "quality_power": float(point["quality_power"]),
                "score_threshold": float(point["score_threshold"]),
            }
            for point in sorted(settings, key=_grid_key)
        ]
        unique.append(representative)
    return unique


def _strictly_dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    eps: float = 1e-12,
) -> bool:
    precision_ge = float(left["precision"]) + eps >= float(right["precision"])
    recall_ge = float(left["recall"]) + eps >= float(right["recall"])
    one_strict = (
        float(left["precision"]) > float(right["precision"]) + eps
        or float(left["recall"]) > float(right["recall"]) + eps
    )
    return precision_ge and recall_ge and one_strict


def _weakly_covers(
    left: dict[str, Any],
    right: dict[str, Any],
    eps: float = 1e-12,
) -> bool:
    return (
        float(left["precision"]) + eps >= float(right["precision"])
        and float(left["recall"]) + eps >= float(right["recall"])
    )


def _frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = _deduplicate_outcomes(points)
    frontier = [
        point
        for point in unique
        if not any(
            _strictly_dominates(other, point)
            for other in unique
            if other is not point
        )
    ]
    return sorted(
        frontier,
        key=lambda point: (
            point["recall"],
            point["precision"],
            point["f1"],
        ),
    )


def _best_by_quality_power(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    powers = sorted({float(point["quality_power"]) for point in points})
    return [
        {
            "quality_power": power,
            "best": _best(
                [
                    point
                    for point in points
                    if abs(float(point["quality_power"]) - power) <= 1e-12
                ]
            ),
        }
        for power in powers
    ]


def _paired_grid(
    base_points: list[dict[str, Any]],
    candidate_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = {_grid_key(point): point for point in base_points}
    candidate = {_grid_key(point): point for point in candidate_points}
    if set(base) != set(candidate):
        raise ValueError(
            "Base and candidate PR grids differ: "
            f"base_only={sorted(set(base) - set(candidate))}, "
            f"candidate_only={sorted(set(candidate) - set(base))}"
        )
    paired: list[dict[str, Any]] = []
    for key in sorted(base):
        base_point = base[key]
        candidate_point = candidate[key]
        paired.append(
            {
                "quality_power": key[0],
                "score_threshold": key[1],
                "base": dict(base_point),
                "candidate": dict(candidate_point),
                "delta_precision_points": 100.0
                * (
                    float(candidate_point["precision"])
                    - float(base_point["precision"])
                ),
                "delta_recall_points": 100.0
                * (
                    float(candidate_point["recall"])
                    - float(base_point["recall"])
                ),
                "delta_f1_points": 100.0
                * (float(candidate_point["f1"]) - float(base_point["f1"])),
            }
        )
    return paired


def _at_setting(
    points: list[dict[str, Any]],
    setting: dict[str, Any],
) -> dict[str, Any]:
    key = _grid_key(setting)
    matches = [point for point in points if _grid_key(point) == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one point at setting={key}, got {len(matches)}")
    return dict(matches[0])


def _summarize_iou(
    base_points: list[dict[str, Any]],
    candidate_points: list[dict[str, Any]],
) -> dict[str, Any]:
    paired = _paired_grid(base_points, candidate_points)
    base_best = _best(base_points)
    candidate_best = _best(candidate_points)
    base_frontier = _frontier(base_points)
    candidate_frontier = _frontier(candidate_points)
    base_at_candidate_best = _at_setting(base_points, candidate_best)
    candidate_at_base_best = _at_setting(candidate_points, base_best)
    base_covered = sum(
        any(_weakly_covers(candidate, base) for candidate in candidate_frontier)
        for base in base_frontier
    )
    candidate_covered = sum(
        any(_weakly_covers(base, candidate) for base in base_frontier)
        for candidate in candidate_frontier
    )
    return {
        "base": {
            "best": base_best,
            "best_by_quality_power": _best_by_quality_power(base_points),
            "frontier": base_frontier,
        },
        "candidate": {
            "best": candidate_best,
            "best_by_quality_power": _best_by_quality_power(candidate_points),
            "frontier": candidate_frontier,
        },
        "best_f1_delta_points": 100.0
        * (float(candidate_best["f1"]) - float(base_best["f1"])),
        "base_at_candidate_best_setting": base_at_candidate_best,
        "candidate_at_base_best_setting": candidate_at_base_best,
        "base_frontier_points_covered_by_candidate": base_covered,
        "base_frontier_point_count": len(base_frontier),
        "candidate_frontier_points_covered_by_base": candidate_covered,
        "candidate_frontier_point_count": len(candidate_frontier),
        "candidate_has_higher_observed_max_f1": bool(
            float(candidate_best["f1"]) > float(base_best["f1"]) + 1e-12
        ),
        "paired_grid": paired,
    }


def summarize_pair(
    base: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    stage = _common_stage(base, candidate)
    comparability = _comparability(base, candidate)
    base_meta = base.get("metadata", {})
    candidate_meta = candidate.get("metadata", {})
    extra_checks = {
        "top_k_values": base_meta.get("top_k_values")
        == candidate_meta.get("top_k_values"),
        "iou_thresholds": base_meta.get("iou_thresholds")
        == candidate_meta.get("iou_thresholds"),
        "quality_powers": base_meta.get("quality_powers")
        == candidate_meta.get("quality_powers"),
        "score_thresholds": base_meta.get("score_thresholds")
        == candidate_meta.get("score_thresholds"),
        "line_width": base_meta.get("line_width")
        == candidate_meta.get("line_width"),
        "nms_distance_thresh_px": base_meta.get("nms_distance_thresh_px")
        == candidate_meta.get("nms_distance_thresh_px"),
        "nms_min_overlap_points": base_meta.get("nms_min_overlap_points")
        == candidate_meta.get("nms_min_overlap_points"),
    }
    comparability["checks"].update(extra_checks)
    comparability["all_checks_pass"] = all(comparability["checks"].values())
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Reports are not directly comparable: {comparability}")
    if str(base_meta.get("iou_space")) != "official_raster":
        raise ValueError("PR frontier requires iou_space='official_raster'")

    iou_thresholds = [
        float(value) for value in base_meta.get("iou_thresholds", [])
    ]
    if not iou_thresholds:
        raise ValueError("Report metadata does not contain IoU thresholds")
    summaries: dict[str, Any] = {}
    for iou_threshold in iou_thresholds:
        base_points = _extract_points(
            base,
            stage=stage,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        candidate_points = _extract_points(
            candidate,
            stage=stage,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        summaries[f"{iou_threshold:g}"] = _summarize_iou(
            base_points,
            candidate_points,
        )
    return {
        "diagnostic_only": True,
        "stage": stage,
        "top_k": int(top_k),
        "comparability": comparability,
        "iou": summaries,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"matched official PR frontier, stage={payload['stage']}")
    print(
        f"{'IoU':>5} {'base F1':>9} {'candidate F1':>13} {'delta':>9} "
        f"{'base q/thr':>15} {'candidate q/thr':>19} {'coverage B<-C':>14}"
    )
    for iou_threshold, summary in payload["iou"].items():
        base = summary["base"]["best"]
        candidate = summary["candidate"]["best"]
        base_setting = (
            f"{base['quality_power']:.2f}/{base['score_threshold']:.3f}"
        )
        candidate_setting = (
            f"{candidate['quality_power']:.2f}/"
            f"{candidate['score_threshold']:.3f}"
        )
        print(
            f"{float(iou_threshold):>5.2f} "
            f"{100.0 * float(base['f1']):>9.3f} "
            f"{100.0 * float(candidate['f1']):>13.3f} "
            f"{float(summary['best_f1_delta_points']):>9.3f} "
            f"{base_setting:>15} "
            f"{candidate_setting:>19} "
            f"{summary['base_frontier_points_covered_by_candidate']:>5}/"
            f"{summary['base_frontier_point_count']:<5}"
        )


def main() -> None:
    args = parse_args()
    payload = summarize_pair(
        _load(args.base_json),
        _load(args.candidate_json),
        top_k=args.top_k,
    )
    payload["base_json"] = str(Path(args.base_json))
    payload["candidate_json"] = str(Path(args.candidate_json))
    _print_summary(payload)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
