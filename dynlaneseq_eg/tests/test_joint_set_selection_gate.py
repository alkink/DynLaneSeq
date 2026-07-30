from __future__ import annotations

from dynlaneseq_eg.tools.summarize_joint_set_selection_gate import summarize


def _payload(
    *,
    current_050: float,
    current_070: float,
    selection_050: float | None = None,
    selection_070: float | None = None,
) -> dict:
    rows = []
    for threshold, recall in ((0.5, current_050), (0.7, current_070)):
        rows.append(
            {
                "stage": "main",
                "strategy": "model_topk_nms",
                "top_k": 4,
                "iou_threshold": threshold,
                "quality_power": 0.5,
                "score_threshold": -1.0,
                "recall": recall,
            }
        )
        rows.append(
            {
                "stage": "main",
                "strategy": "oracle_topk",
                "top_k": 4,
                "iou_threshold": threshold,
                "quality_power": None,
                "score_threshold": None,
                "recall": min(1.0, recall + 0.15),
            }
        )
    if selection_050 is not None and selection_070 is not None:
        for threshold, recall in (
            (0.5, selection_050),
            (0.7, selection_070),
        ):
            rows.append(
                {
                    "stage": "main",
                    "strategy": "selection_topk_nms",
                    "top_k": 4,
                    "iou_threshold": threshold,
                    "quality_power": None,
                    "score_threshold": -1.0,
                    "recall": recall,
                }
            )
    return {"rows": rows}


def test_joint_set_selection_gate_detects_matched_positive_signal() -> None:
    result = summarize(
        _payload(current_050=0.80, current_070=0.67),
        _payload(
            current_050=0.805,
            current_070=0.672,
            selection_050=0.815,
            selection_070=0.678,
        ),
        top_k=4,
        quality_power=0.5,
        score_threshold=-1.0,
        min_gain_050_points=1.0,
        min_gain_070_points=0.5,
    )

    assert result["gate"]["positive"] is True
    assert result["recommendation"] == (
        "joint_selection_positive_prepare_full_schedule"
    )
    assert abs(result["gains_over_control_points"]["recall_050"] - 1.5) < 1e-8
    assert abs(result["gains_over_control_points"]["recall_070"] - 0.8) < 1e-8
