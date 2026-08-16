from __future__ import annotations

from dynlaneseq_eg.tools.analyze_v23_source_raw_binary_oracle import (
    align_raw_to_source,
    summarize,
)


def _lane(x: float):
    return [(x, 100.0), (x, 200.0)]


def test_align_raw_to_source_restores_geometric_pairing() -> None:
    source = [_lane(100.0), _lane(400.0)]
    raw = [_lane(410.0), _lane(110.0)]
    aligned = align_raw_to_source(source, raw)
    assert aligned[0][0][0] == 110.0
    assert aligned[1][0][0] == 410.0


def _score(tp: int, predictions: int = 2, gt: int = 2):
    return {
        "total_iou": float(tp),
        "matched_iou": [],
        "0.50": {"TP": tp, "FP": predictions - tp, "FN": gt - tp},
        "0.75": {"TP": tp, "FP": predictions - tp, "FN": gt - tp},
    }


def test_summary_reports_binary_oracle_gain() -> None:
    rows = [
        {
            "source": _score(1),
            "raw": _score(0),
            "whole_image_binary_oracle": _score(1),
            "per_lane_binary_oracle": _score(2),
            "per_lane_edit_count": 1,
            "whole_image_edit_count": 0,
            "prediction_count": 2,
            "gt_count": 2,
        }
    ]
    result = summarize(rows)
    assert result["tp_gain_vs_source"]["per_lane_binary_oracle"]["0.50"] == 1
    assert result["per_lane_edit_histogram"] == {1: 1}
