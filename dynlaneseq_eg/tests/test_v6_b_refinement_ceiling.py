from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v6_b_slot_refinement_ceiling import (
    MetricAccumulator,
    build_counterfactual_geometries,
)


def _stage() -> dict[str, torch.Tensor]:
    return {
        "pred_x_rows": torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                [80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            ]
        ),
        "range_norm": torch.tensor([[0.0, 0.8], [0.0, 0.8]]),
        "selection_slot_indices": torch.tensor([0, -1]),
        "selection_slot_pred_x_rows": torch.tensor(
            [
                [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "selection_slot_range_norm": torch.tensor(
            [[0.0, 0.8], [0.0, 0.0]]
        ),
    }


def test_counterfactuals_separate_bounded_x_and_gt_range() -> None:
    target = {
        "x_rows": torch.tensor(
            [[40.0, 40.0, 40.0, 40.0, 40.0, 40.0]]
        ),
        "valid_mask": torch.tensor(
            [[False, True, True, True, True, True]]
        ),
    }
    methods, pairs, diagnostics = build_counterfactual_geometries(
        _stage(),
        target,
        input_h=60,
        input_w=100,
        line_width=30.0,
        min_valid_rows=5,
        match_min_quality=0.0,
        delta_bound_px=24.0,
    )
    assert [(slot, gt) for slot, gt, _quality in pairs] == [(0, 0)]
    bounded_x, current_range = methods[
        "bounded_x_oracle_current_range"
    ]
    assert torch.equal(
        bounded_x[0],
        torch.tensor([10.0, 34.0, 34.0, 34.0, 34.0, 34.0]),
    )
    exact_x, gt_range = methods["exact_x_plus_gt_range_oracle"]
    assert torch.equal(
        exact_x[0],
        torch.tensor([10.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
    )
    assert torch.allclose(gt_range[0], torch.tensor([1.0 / 6.0, 5.0 / 6.0]))
    assert torch.equal(current_range[0], torch.tensor([0.0, 0.8]))
    assert diagnostics["active_slots"] == 1
    assert diagnostics["matched_slots"] == 1
    assert diagnostics["target_residual_fraction_over"]["24px"] == 1.0


def test_metric_accumulator_uses_official_hungarian_counts() -> None:
    accumulator = MetricAccumulator()
    accumulator.update(
        torch.tensor([[0.8, 0.1], [0.2, 0.7]]),
        torch.tensor([True, True]),
        (0.5, 0.75),
    )
    summary = accumulator.summary((0.5, 0.75))
    assert summary["0.50"]["tp"] == 2
    assert summary["0.50"]["f1"] == 1.0
    assert summary["0.75"]["tp"] == 1
    assert summary["0.75"]["fp"] == 1
    assert summary["0.75"]["fn"] == 1
