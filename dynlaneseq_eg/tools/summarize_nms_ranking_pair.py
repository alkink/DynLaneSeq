from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired all-proposal, Top-K, and NMS diagnostics from "
            "two analyze_oracle_topk JSON reports."
        )
    )
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _common_stage(base: dict[str, Any], candidate: dict[str, Any]) -> str:
    base_stages = {str(row["stage"]) for row in base.get("rows", [])}
    candidate_stages = {str(row["stage"]) for row in candidate.get("rows", [])}
    common = base_stages & candidate_stages
    if not common:
        raise ValueError(
            f"No common prediction stage: base={sorted(base_stages)}, "
            f"candidate={sorted(candidate_stages)}"
        )
    for preferred in ("main", "final", "stage2", "stage1", "coarse"):
        if preferred in common:
            return preferred
    return sorted(common)[-1]


def _same_optional_float(left: Any, right: Any, atol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= atol


def _find_row(
    report: dict[str, Any],
    *,
    stage: str,
    strategy: str,
    top_k: int,
    iou_threshold: float,
    quality_power: float | None,
    score_threshold: float | None,
) -> dict[str, Any]:
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
            "Expected exactly one row for "
            f"{stage=}, {strategy=}, {top_k=}, {iou_threshold=}, "
            f"{quality_power=}, {score_threshold=}; found {len(matches)}"
        )
    return matches[0]


def _comparability(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_meta = base.get("metadata", {})
    candidate_meta = candidate.get("metadata", {})
    checks = {
        "split": base_meta.get("split") == candidate_meta.get("split"),
        "list_sha256": base_meta.get("list_sha256")
        == candidate_meta.get("list_sha256"),
        "max_batches": base_meta.get("max_batches")
        == candidate_meta.get("max_batches"),
        "num_records": base_meta.get("num_records")
        == candidate_meta.get("num_records"),
        "iou_space": base_meta.get("iou_space") == candidate_meta.get("iou_space"),
    }
    return {
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "base_records": base_meta.get("num_records"),
        "candidate_records": candidate_meta.get("num_records"),
    }


def _specs(report: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    strategy_order = {
        "all_raw": 0,
        "oracle_topk": 1,
        "oracle_topk_nms": 2,
        "exist_topk": 3,
        "quality_topk": 4,
        "model_topk": 5,
        "model_topk_nms": 6,
    }
    rows = sorted(
        (row for row in report.get("rows", []) if str(row.get("stage")) == stage),
        key=lambda row: (
            float(row.get("iou_threshold", 0.0)),
            strategy_order.get(str(row.get("strategy")), 99),
            -1.0 if row.get("quality_power") is None else float(row["quality_power"]),
            -2.0
            if row.get("score_threshold") is None
            else float(row["score_threshold"]),
        ),
    )
    for row in rows:
        key = (
            str(row["strategy"]),
            int(row["top_k"]),
            float(row["iou_threshold"]),
            row.get("quality_power"),
            row.get("score_threshold"),
        )
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "strategy": key[0],
                "top_k": key[1],
                "iou_threshold": key[2],
                "quality_power": key[3],
                "score_threshold": key[4],
            }
        )
    return specs


def _paired_rows(
    base: dict[str, Any],
    candidate: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for spec in _specs(base, stage):
        base_row = _find_row(base, stage=stage, **spec)
        candidate_row = _find_row(candidate, stage=stage, **spec)
        base_recall = float(base_row["recall"])
        candidate_recall = float(candidate_row["recall"])
        paired.append(
            {
                **spec,
                "base_hits": int(base_row["hits"]),
                "candidate_hits": int(candidate_row["hits"]),
                "gt": int(base_row["gt"]),
                "base_recall": base_recall,
                "candidate_recall": candidate_recall,
                "delta_recall_points": 100.0 * (candidate_recall - base_recall),
                "base_mean_best_iou": float(base_row["mean_best_iou"]),
                "candidate_mean_best_iou": float(candidate_row["mean_best_iou"]),
            }
        )
    return paired


def _lookup(
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
        if row["strategy"] == strategy
        and int(row["top_k"]) == int(top_k)
        and _same_optional_float(row["iou_threshold"], iou_threshold)
        and _same_optional_float(row["quality_power"], quality_power)
        and _same_optional_float(row["score_threshold"], score_threshold)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one paired summary row, found {len(matches)}")
    return matches[0]


def _derived(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    iou_thresholds = sorted({float(row["iou_threshold"]) for row in rows})
    quality_powers = sorted(
        {
            float(row["quality_power"])
            for row in rows
            if row["strategy"] == "model_topk"
            and row["quality_power"] is not None
        }
    )
    for iou_threshold in iou_thresholds:
        all_raw = _lookup(
            rows,
            strategy="all_raw",
            iou_threshold=iou_threshold,
            quality_power=None,
            score_threshold=None,
            top_k=0,
        )
        oracle = _lookup(
            rows,
            strategy="oracle_topk",
            iou_threshold=iou_threshold,
            quality_power=None,
            score_threshold=None,
        )
        for quality_power in quality_powers:
            raw = _lookup(
                rows,
                strategy="model_topk",
                iou_threshold=iou_threshold,
                quality_power=quality_power,
                score_threshold=None,
            )
            nms = _lookup(
                rows,
                strategy="model_topk_nms",
                iou_threshold=iou_threshold,
                quality_power=quality_power,
                score_threshold=-1.0,
            )
            out.append(
                {
                    "iou_threshold": iou_threshold,
                    "quality_power": quality_power,
                    "base_all_to_raw_top4_gap_points": 100.0
                    * (all_raw["base_recall"] - raw["base_recall"]),
                    "candidate_all_to_raw_top4_gap_points": 100.0
                    * (all_raw["candidate_recall"] - raw["candidate_recall"]),
                    "base_oracle_to_raw_top4_gap_points": 100.0
                    * (oracle["base_recall"] - raw["base_recall"]),
                    "candidate_oracle_to_raw_top4_gap_points": 100.0
                    * (oracle["candidate_recall"] - raw["candidate_recall"]),
                    "base_nms_gain_over_raw_top4_points": 100.0
                    * (nms["base_recall"] - raw["base_recall"]),
                    "candidate_nms_gain_over_raw_top4_points": 100.0
                    * (nms["candidate_recall"] - raw["candidate_recall"]),
                }
            )
    return out


def _print_summary(stage: str, rows: list[dict[str, Any]]) -> None:
    print(f"paired NMS/ranking summary, stage={stage}")
    print(
        f"{'strategy':>18} {'IoU':>4} {'q':>5} {'thr':>5} "
        f"{'base':>8} {'candidate':>10} {'delta(pt)':>10}"
    )
    for row in rows:
        quality = "-" if row["quality_power"] is None else f"{row['quality_power']:.2f}"
        score = "-" if row["score_threshold"] is None else f"{row['score_threshold']:.2f}"
        print(
            f"{row['strategy']:>18} {row['iou_threshold']:>4.2f} "
            f"{quality:>5} {score:>5} "
            f"{100.0 * row['base_recall']:>8.2f} "
            f"{100.0 * row['candidate_recall']:>10.2f} "
            f"{row['delta_recall_points']:>10.2f}"
        )


def main() -> None:
    args = parse_args()
    base = _load(args.base_json)
    candidate = _load(args.candidate_json)
    stage = _common_stage(base, candidate)
    comparability = _comparability(base, candidate)
    if not comparability["all_checks_pass"]:
        raise ValueError(f"Reports are not directly comparable: {comparability}")
    rows = _paired_rows(base, candidate, stage)
    payload = {
        "diagnostic_only": True,
        "base_json": str(Path(args.base_json)),
        "candidate_json": str(Path(args.candidate_json)),
        "stage": stage,
        "comparability": comparability,
        "rows": rows,
        "derived_gaps": _derived(rows),
    }
    _print_summary(stage, rows)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
