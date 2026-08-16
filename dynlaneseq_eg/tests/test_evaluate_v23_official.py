from __future__ import annotations

from dynlaneseq_eg.tools.evaluate_v23_official import (
    _gate,
    summarize_image_results,
)


def _row(source, endpoint, wrong, counts=(3, 3, 3)):
    def metric(tp, matched):
        return (tp, counts[0] - tp, 3 - tp, frozenset(matched))

    return {
        "prediction_count": {
            "source_v7": counts[0],
            "v23": counts[1],
            "cross_clip_wrong_image": counts[2],
        },
        "thresholds": {
            "0.50": {
                "source_v7": metric(*source[0]),
                "v23": metric(*endpoint[0]),
                "cross_clip_wrong_image": metric(*wrong[0]),
            },
            "0.75": {
                "source_v7": metric(*source[1]),
                "v23": metric(*endpoint[1]),
                "cross_clip_wrong_image": metric(*wrong[1]),
            },
        },
    }


def test_summary_tracks_paired_effects_and_owned_gt_loss() -> None:
    rows = [
        _row(
            ((2, {0, 1}), (1, {0})),
            ((3, {0, 1, 2}), (2, {0, 2})),
            ((1, {0}), (1, {0})),
        ),
        _row(
            ((2, {0, 1}), (2, {0, 1})),
            ((1, {1}), (2, {0, 1})),
            ((0, set()), (1, {1})),
        ),
    ]
    summary = summarize_image_results(rows)
    assert summary["paired_image_effects"]["0.50"] == {
        "improved": 1,
        "worsened": 1,
        "tied": 0,
    }
    assert summary["source_correct_degradation"]["0.50"] == {
        "source_correct": 4,
        "lost_by_v23": 1,
        "fraction": 0.25,
    }
    assert summary["cardinality"]["exact_fraction"] == 1.0


def test_gate_requires_real_f1_gain_and_causal_image_advantage() -> None:
    rows = []
    for _ in range(200):
        rows.append(
            _row(
                ((2, {0, 1}), (1, {0})),
                ((3, {0, 1, 2}), (2, {0, 1})),
                ((1, {0}), (0, set())),
            )
        )
    gate = _gate(summarize_image_results(rows))
    assert gate["passed"] is True
    assert gate["full_validation_target"]["met"] is True


def test_gate_rejects_cardinality_change() -> None:
    row = _row(
        ((1, {0}), (1, {0})),
        ((2, {0, 1}), (2, {0, 1})),
        ((0, set()), (0, set())),
        counts=(3, 2, 3),
    )
    gate = _gate(summarize_image_results([row] * 200))
    assert gate["passed"] is False
    assert gate["checks"]["prediction_count_exact_source_all_images"] is False
