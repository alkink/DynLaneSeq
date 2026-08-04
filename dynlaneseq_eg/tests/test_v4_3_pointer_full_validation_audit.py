from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.audit_v4_3_pointer_full_validation import (
    PointerAuditAccumulator,
    analyze_pointer_image,
)


def _logits(indices: list[int], candidates: int = 4) -> torch.Tensor:
    logits = torch.full((4, candidates + 1), -5.0)
    for step, index in enumerate(indices):
        logits[step, index if index >= 0 else candidates] = 5.0
    return logits


def test_audit_decomposes_wrong_representative_at_fixed_count() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.80, 0.05, 0.00],
            [0.05, 0.00, 0.85, 0.70],
        ]
    )
    row = analyze_pointer_image(
        iou,
        torch.ones(4, dtype=torch.bool),
        torch.tensor([1, 3, -1, -1]),
        _logits([1, 3, -1, -1]),
        torch.zeros(4),
    )
    strict = row["thresholds"]["0.75"]
    assert row["selected_count"] == 2
    assert strict["pointer_tp"] == 1
    assert strict["same_count_oracle_tp"] == 2
    assert strict["top4_oracle_tp"] == 2
    assert strict["representative_gap"] == 1
    assert strict["stop_cardinality_gap"] == 0
    assert [item["chosen_rank"] for item in row["representatives"]] == [2, 2]


def test_audit_decomposes_early_stop_from_representative_error() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.80, 0.05, 0.00],
            [0.05, 0.00, 0.85, 0.70],
        ]
    )
    row = analyze_pointer_image(
        iou,
        torch.ones(4, dtype=torch.bool),
        torch.tensor([0, -1, -1, -1]),
        _logits([0, -1, -1, -1]),
        torch.zeros(4),
    )
    strict = row["thresholds"]["0.75"]
    assert strict["pointer_tp"] == 1
    assert strict["same_count_oracle_tp"] == 1
    assert strict["top4_oracle_tp"] == 2
    assert strict["representative_gap"] == 0
    assert strict["stop_cardinality_gap"] == 1


def test_empty_scene_stop_and_count_groups_are_reported() -> None:
    empty = analyze_pointer_image(
        torch.zeros((0, 4)),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([-1, -1, -1, -1]),
        _logits([-1, -1, -1, -1]),
        torch.full((4,), -4.0),
    )
    accumulator = PointerAuditAccumulator((0.50, 0.75), top_k=4)
    accumulator.update(empty)
    report = accumulator.finish()
    zero = report["cardinality_by_gt_count"]["0"]
    assert zero["images"] == 1
    assert zero["mean_emitted_lanes"] == 0.0
    assert zero["initial_stop_rate"] == 1.0
    assert report["verdict"]["empty_scene_stop"] == (
        "empty_scene_stop_is_well_controlled"
    )


def test_invalid_and_repeated_pointer_indices_do_not_count_as_emissions() -> None:
    row = analyze_pointer_image(
        torch.tensor([[0.9, 0.8, 0.0, 0.0]]),
        torch.tensor([True, True, False, True]),
        torch.tensor([0, 0, 2, -1]),
        _logits([0, 0, 2, -1]),
        torch.zeros(4),
    )
    assert row["selected_ids"] == [0]
    assert row["invalid_or_repeat_selections"] == 2


def test_first_stop_probability_uses_explicit_stop_class() -> None:
    row = analyze_pointer_image(
        torch.zeros((0, 4)),
        torch.ones(4, dtype=torch.bool),
        torch.tensor([-1, -1, -1, -1]),
        _logits([-1, -1, -1, -1]),
        torch.zeros(4),
    )
    assert row["first_stop_probability"] == pytest.approx(0.9998, abs=2e-4)
