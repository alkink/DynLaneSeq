from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the matched unified-selection 10k gate."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("ITERATION", "CONTROL_JSON", "CANDIDATE_JSON"),
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row(
    report: dict[str, Any],
    strategy: str,
    iou: float,
    *,
    quality_power: float | None = None,
    score_threshold: float | None = None,
) -> dict[str, Any]:
    matches = []
    for row in report["rows"]:
        if row.get("stage") != "main":
            continue
        if row.get("strategy") != strategy:
            continue
        if int(row.get("top_k", 0)) not in {0, 4}:
            continue
        if abs(float(row["iou_threshold"]) - float(iou)) > 1e-8:
            continue
        if quality_power is not None and row.get("quality_power") != quality_power:
            continue
        if score_threshold is not None and row.get("score_threshold") != score_threshold:
            continue
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {strategy} row at IoU={iou}, got {len(matches)}"
        )
    return matches[0]


def summarize(
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    trajectory: list[dict[str, Any]] = []
    for iteration, control, candidate in sorted(pairs):
        row: dict[str, Any] = {"iteration": int(iteration)}
        for iou in (0.5, 0.75):
            suffix = "050" if iou == 0.5 else "075"
            control_raw = float(_row(control, "all_raw", iou)["recall"])
            candidate_raw = float(_row(candidate, "all_raw", iou)["recall"])
            control_score = float(
                _row(
                    control,
                    "model_topk",
                    iou,
                    quality_power=0.5,
                )["recall"]
            )
            control_oracle = float(
                _row(control, "oracle_topk", iou)["recall"]
            )
            candidate_score = float(
                _row(candidate, "selection_topk", iou)["recall"]
            )
            candidate_nms = float(
                _row(
                    candidate,
                    "selection_topk_nms",
                    iou,
                    score_threshold=0.0,
                )["recall"]
            )
            candidate_oracle = float(
                _row(candidate, "oracle_topk", iou)["recall"]
            )
            row.update(
                {
                    f"control_raw_recall_{suffix}": control_raw,
                    f"candidate_raw_recall_{suffix}": candidate_raw,
                    f"raw_gain_points_{suffix}": 100.0
                    * (candidate_raw - control_raw),
                    f"control_score_top4_recall_{suffix}": control_score,
                    f"control_oracle_top4_recall_{suffix}": control_oracle,
                    f"control_oracle_gap_points_{suffix}": 100.0
                    * (control_oracle - control_score),
                    f"candidate_selection_top4_recall_{suffix}": candidate_score,
                    f"selection_gain_points_{suffix}": 100.0
                    * (candidate_score - control_score),
                    f"candidate_selection_nms_top4_recall_{suffix}": candidate_nms,
                    f"nms_gain_points_{suffix}": 100.0
                    * (candidate_nms - candidate_score),
                    f"candidate_oracle_top4_recall_{suffix}": candidate_oracle,
                    f"candidate_oracle_gap_points_{suffix}": 100.0
                    * (candidate_oracle - candidate_score),
                    f"oracle_gap_shrink_fraction_{suffix}": (
                        (control_oracle - control_score)
                        - (candidate_oracle - candidate_score)
                    )
                    / max(control_oracle - control_score, 1e-8),
                }
            )
        trajectory.append(row)

    final = trajectory[-1]
    checks = {
        "raw_geometry_retained_050": final["raw_gain_points_050"] >= -2.0,
        "raw_geometry_retained_075": final["raw_gain_points_075"] >= -2.0,
        "ranking_gain_050": final["selection_gain_points_050"] >= 8.0,
        "ranking_gain_075": final["selection_gain_points_075"] >= 5.0,
        "oracle_gap_shrinks_050": final["oracle_gap_shrink_fraction_050"]
        >= (1.0 / 3.0),
        "oracle_gap_shrinks_075": final["oracle_gap_shrink_fraction_075"]
        >= (1.0 / 3.0),
        "low_nms_dependence_050": final["nms_gain_points_050"] <= 3.0,
    }
    return {
        "diagnostic_only": True,
        "contract": {
            "control": "legacy exist*quality^0.5 scorer with NMS-era supervision",
            "candidate": "single range-aware matcher and unified final-set scorer",
            "sample": "uniform validation subset",
            "ranking": "threshold-free Top-4",
            "nms_probe_distance_px": 20.0,
        },
        "positive_gate_definition": (
            "At the final checkpoint: retain raw geometry within 2 points at "
            "IoU 0.50/0.75, improve threshold-free Top-4 ranking by at least "
            "8/5 points, shrink both oracle gaps by one third, and need no "
            "more than 3 points of NMS recovery at IoU 0.50."
        ),
        "trajectory": trajectory,
        "checks": checks,
        "positive_gate": all(checks.values()),
    }


def main() -> None:
    args = parse_args()
    pairs = [
        (int(iteration), _load(control), _load(candidate))
        for iteration, control, candidate in args.pair
    ]
    payload = summarize(pairs)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = payload["trajectory"]
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"output_json: {output_json}")
    if args.output_csv:
        print(f"output_csv: {args.output_csv}")


if __name__ == "__main__":
    main()
