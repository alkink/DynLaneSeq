from __future__ import annotations

from dynlaneseq_eg.tools.summarize_unified_selection_gate import summarize


def _row(
    strategy: str,
    iou: float,
    recall: float,
    *,
    quality_power: float | None = None,
    score_threshold: float | None = None,
) -> dict[str, object]:
    return {
        "stage": "main",
        "strategy": strategy,
        "top_k": 0 if strategy == "all_raw" else 4,
        "iou_threshold": iou,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "recall": recall,
    }


def _report(*, candidate: bool) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for iou, control_raw, control_score in (
        (0.5, 0.80, 0.60),
        (0.75, 0.60, 0.40),
    ):
        raw = control_raw - (0.01 if candidate else 0.0)
        rows.append(_row("all_raw", iou, raw))
        rows.append(
            _row(
                "model_topk",
                iou,
                control_score,
                quality_power=0.5,
            )
        )
        rows.append(_row("oracle_topk", iou, control_score + 0.20))
        if candidate:
            selection = control_score + (0.10 if iou == 0.5 else 0.08)
            rows.append(_row("selection_topk", iou, selection))
            rows.append(
                _row(
                    "selection_topk_nms",
                    iou,
                    selection + 0.01,
                    score_threshold=0.0,
                )
            )
    return {"rows": rows}


def test_unified_selection_gate_accepts_ranking_without_nms_dependency() -> None:
    payload = summarize([(10000, _report(candidate=False), _report(candidate=True))])
    assert payload["positive_gate"] is True
    row = payload["trajectory"][0]
    assert row["selection_gain_points_050"] > 9.9
    assert row["nms_gain_points_075"] < 1.1
