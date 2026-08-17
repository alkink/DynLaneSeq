from __future__ import annotations

import numpy as np

from dynlaneseq_eg.evaluation.culane_metric import discrete_cross_iou, interp
from dynlaneseq_eg.tools.evaluate_v25_v7_top3_union_oracle import (
    official_iou_matrix,
    select_slot_union_oracle,
)


def _lane(x: float) -> list[tuple[float, float]]:
    return [(x, 100.0), (x + 1.0, 300.0), (x + 2.0, 580.0)]


def _empty_bank() -> list[tuple[str, list[tuple[float, float]] | None]]:
    return [
        ("v7", None),
        ("g0_path0", None),
        ("g0_path1", None),
        ("g0_path2", None),
        ("dustbin", None),
    ]


def test_crop_iou_matches_official_full_canvas() -> None:
    predictions = [_lane(101.4), _lane(300.9)]
    targets = [_lane(105.8), _lane(330.2)]
    expected = discrete_cross_iou(
        [interp(lane, n=5) for lane in predictions],
        [interp(lane, n=5) for lane in targets],
        width=30,
        img_shape=(590, 1640),
    )
    actual = official_iou_matrix(predictions, targets)
    np.testing.assert_array_equal(actual, expected)


def test_union_oracle_is_exact_v7_on_threshold_neutral_tie() -> None:
    source = _lane(100.0)
    banks = [
        [
            ("v7", source),
            ("g0_path0", list(source)),
            ("g0_path1", _lane(140.0)),
            ("g0_path2", _lane(180.0)),
            ("dustbin", None),
        ],
        _empty_bank(),
        _empty_bank(),
        _empty_bank(),
    ]
    result = select_slot_union_oracle(banks, [source], fixed_v7_count=True)
    assert result["labels"] == ("v7", "dustbin", "dustbin", "dustbin")
    assert result["edit_count"] == 0


def test_union_oracle_selects_immutable_alternative_for_new_tp() -> None:
    source = _lane(200.0)
    target = _lane(100.0)
    alternative = _lane(100.0)
    banks = [
        [
            ("v7", source),
            ("g0_path0", alternative),
            ("g0_path1", _lane(240.0)),
            ("g0_path2", _lane(260.0)),
            ("dustbin", None),
        ],
        _empty_bank(),
        _empty_bank(),
        _empty_bank(),
    ]
    result = select_slot_union_oracle(banks, [target], fixed_v7_count=True)
    assert result["labels"][0] == "g0_path0"
    assert result["lanes"][0] == alternative
    assert result["tp50"] == 1
    assert result["edit_count"] == 1


def test_fixed_oracle_preserves_v7_activity_count() -> None:
    banks = []
    for slot in range(4):
        source = _lane(100.0 + 200.0 * slot) if slot < 3 else None
        banks.append(
            [
                ("v7", source),
                ("g0_path0", _lane(110.0 + 200.0 * slot)),
                ("g0_path1", _lane(120.0 + 200.0 * slot)),
                ("g0_path2", _lane(130.0 + 200.0 * slot)),
                ("dustbin", None),
            ]
        )
    result = select_slot_union_oracle(
        banks,
        [_lane(100.0), _lane(300.0), _lane(500.0)],
        fixed_v7_count=True,
    )
    assert len(result["lanes"]) == 3
    assert result["labels"][3] == "dustbin"
