from __future__ import annotations

from dynlaneseq_eg.tools.summarize_nms_ranking_pair import (
    _comparability,
    _derived,
    _paired_rows,
)


def _row(
    strategy: str,
    *,
    recall: float,
    top_k: int,
    quality_power: float | None = None,
    score_threshold: float | None = None,
) -> dict:
    return {
        "stage": "main",
        "strategy": strategy,
        "top_k": top_k,
        "iou_threshold": 0.5,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "hits": int(round(10 * recall)),
        "gt": 10,
        "recall": recall,
        "mean_best_iou": recall,
        "images": 4,
    }


def _report(offset: float = 0.0) -> dict:
    return {
        "metadata": {
            "split": "val",
            "list_sha256": "same",
            "max_batches": 4,
            "num_records": 4,
            "iou_space": "row_space",
        },
        "rows": [
            _row("all_raw", recall=0.8 + offset, top_k=0),
            _row("oracle_topk", recall=0.8 + offset, top_k=4),
            _row("oracle_topk_nms", recall=0.8 + offset, top_k=4),
            _row(
                "exist_topk",
                recall=0.4 + offset,
                top_k=4,
                quality_power=0.0,
            ),
            _row("quality_topk", recall=0.5 + offset, top_k=4),
            _row(
                "model_topk",
                recall=0.4 + offset,
                top_k=4,
                quality_power=0.0,
            ),
            _row(
                "model_topk_nms",
                recall=0.7 + offset,
                top_k=4,
                quality_power=0.0,
                score_threshold=-1.0,
            ),
            _row(
                "model_topk_nms",
                recall=0.6 + offset,
                top_k=4,
                quality_power=0.0,
                score_threshold=0.3,
            ),
        ],
    }


def test_paired_summary_computes_nms_gain() -> None:
    base = _report()
    candidate = _report(0.1)
    assert _comparability(base, candidate)["all_checks_pass"]
    rows = _paired_rows(base, candidate, "main")
    derived = _derived(rows)
    assert len(derived) == 1
    assert abs(derived[0]["base_nms_gain_over_raw_top4_points"] - 30.0) < 1e-6
    assert (
        abs(derived[0]["candidate_nms_gain_over_raw_top4_points"] - 30.0)
        < 1e-6
    )
