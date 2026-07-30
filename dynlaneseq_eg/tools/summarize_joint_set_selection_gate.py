from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the matched 70k cooldown control with the 70k joint "
            "set-selection candidate on the exact same diagnostic subset."
        )
    )
    parser.add_argument("--control-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--quality-power", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=-1.0)
    parser.add_argument("--min-gain-050-points", type=float, default=1.0)
    parser.add_argument("--min-gain-070-points", type=float, default=0.5)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _close(left: Any, right: float, tolerance: float = 1e-8) -> bool:
    return left is not None and abs(float(left) - float(right)) <= tolerance


def _find_row(
    payload: dict[str, Any],
    *,
    strategy: str,
    top_k: int,
    iou_threshold: float,
    quality_power: float | None,
    score_threshold: float | None,
) -> dict[str, Any]:
    candidates = []
    for row in payload.get("rows", []):
        if str(row.get("strategy")) != str(strategy):
            continue
        if int(row.get("top_k", -1)) != int(top_k):
            continue
        if not _close(row.get("iou_threshold"), iou_threshold):
            continue
        if quality_power is None:
            if row.get("quality_power") is not None:
                continue
        elif not _close(row.get("quality_power"), quality_power):
            continue
        if score_threshold is None:
            if row.get("score_threshold") is not None:
                continue
        elif not _close(row.get("score_threshold"), score_threshold):
            continue
        candidates.append(row)
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one diagnostic row for "
            f"{strategy=} {top_k=} {iou_threshold=} {quality_power=} "
            f"{score_threshold=}; found {len(candidates)}"
        )
    return candidates[0]


def _strategy_pair(
    payload: dict[str, Any],
    *,
    strategy: str,
    top_k: int,
    quality_power: float | None,
    score_threshold: float | None,
) -> dict[str, Any]:
    return {
        f"{threshold:.2f}": _find_row(
            payload,
            strategy=strategy,
            top_k=top_k,
            iou_threshold=threshold,
            quality_power=quality_power,
            score_threshold=score_threshold,
        )
        for threshold in (0.5, 0.7)
    }


def _recall_gain(
    candidate: dict[str, Any],
    control: dict[str, Any],
    threshold: float,
) -> float:
    key = f"{threshold:.2f}"
    return 100.0 * (
        float(candidate[key]["recall"]) - float(control[key]["recall"])
    )


def summarize(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top_k: int,
    quality_power: float,
    score_threshold: float,
    min_gain_050_points: float,
    min_gain_070_points: float,
) -> dict[str, Any]:
    control_current = _strategy_pair(
        control,
        strategy="model_topk_nms",
        top_k=top_k,
        quality_power=quality_power,
        score_threshold=score_threshold,
    )
    candidate_current = _strategy_pair(
        candidate,
        strategy="model_topk_nms",
        top_k=top_k,
        quality_power=quality_power,
        score_threshold=score_threshold,
    )
    candidate_selection = _strategy_pair(
        candidate,
        strategy="selection_topk_nms",
        top_k=top_k,
        quality_power=None,
        score_threshold=score_threshold,
    )
    candidate_oracle = _strategy_pair(
        candidate,
        strategy="oracle_topk",
        top_k=top_k,
        quality_power=None,
        score_threshold=None,
    )
    gain_050 = _recall_gain(candidate_selection, control_current, 0.5)
    gain_070 = _recall_gain(candidate_selection, control_current, 0.7)
    selection_over_candidate = {
        "gain_recall_050_points": _recall_gain(
            candidate_selection,
            candidate_current,
            0.5,
        ),
        "gain_recall_070_points": _recall_gain(
            candidate_selection,
            candidate_current,
            0.7,
        ),
    }
    positive = bool(
        gain_050 >= float(min_gain_050_points)
        and gain_070 >= float(min_gain_070_points)
    )
    if positive:
        recommendation = "joint_selection_positive_prepare_full_schedule"
    elif (
        selection_over_candidate["gain_recall_050_points"] > 0.0
        or selection_over_candidate["gain_recall_070_points"] > 0.0
    ):
        recommendation = "ranking_signal_present_but_gate_not_met"
    else:
        recommendation = "joint_selection_negative_shift_focus_to_proposal_decoder"
    return {
        "diagnostic_only": True,
        "comparison": "matched_70k_control_vs_joint_set_selection",
        "settings": {
            "top_k": int(top_k),
            "quality_power": float(quality_power),
            "score_threshold": float(score_threshold),
            "iou_space": "official_raster",
        },
        "control_current": control_current,
        "candidate_current": candidate_current,
        "candidate_selection": candidate_selection,
        "candidate_oracle": candidate_oracle,
        "gains_over_control_points": {
            "recall_050": gain_050,
            "recall_070": gain_070,
        },
        "selection_over_candidate_current_points": selection_over_candidate,
        "gate": {
            "min_gain_050_points": float(min_gain_050_points),
            "min_gain_070_points": float(min_gain_070_points),
            "positive": positive,
        },
        "recommendation": recommendation,
    }


def main() -> None:
    args = parse_args()
    result = summarize(
        _load(args.control_json),
        _load(args.candidate_json),
        top_k=args.top_k,
        quality_power=args.quality_power,
        score_threshold=args.score_threshold,
        min_gain_050_points=args.min_gain_050_points,
        min_gain_070_points=args.min_gain_070_points,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
