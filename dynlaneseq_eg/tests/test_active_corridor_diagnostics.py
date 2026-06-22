from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.evaluation.active_corridor_diagnostics import (
    ActiveCorridorAccumulator,
    build_discrete_offset_oracle_stage,
    matched_index_pairs,
)


def test_matched_index_pairs_keeps_lowest_mae_proposal_per_gt():
    center = torch.tensor([[1.0, 1.0], [8.0, 8.0], [19.0, 19.0]])
    target = {
        "x_rows": torch.tensor([[10.0, 10.0], [20.0, 20.0]]),
        "valid_mask": torch.ones((2, 2), dtype=torch.bool),
    }
    match = {
        "pred_indices": torch.tensor([0, 1, 2]),
        "gt_indices": torch.tensor([0, 0, 1]),
    }

    pred_idx, gt_idx = matched_index_pairs(center, target, match, best_per_gt=True)

    assert pred_idx.tolist() == [1, 2]
    assert gt_idx.tolist() == [0, 1]


def test_active_corridor_accumulator_reports_geometry_and_intervention_effects():
    accumulator = ActiveCorridorAccumulator(bin_edges=(0.0, 4.0, 8.0, float("inf")))
    center = torch.tensor([[10.0, 20.0]])
    target = torch.tensor([[12.0, 26.0]])
    pred_delta = torch.tensor([[2.0, 4.0]])
    final = torch.tensor([[12.0, 25.0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    offsets = torch.tensor([-8.0, 0.0, 8.0])
    logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
    shuffled_delta = torch.zeros_like(pred_delta)
    shuffled_final = center.clone()

    accumulator.update(
        center_x=center,
        pred_delta=pred_delta,
        final_x=final,
        target_x=target,
        valid_mask=valid,
        offsets=offsets,
        logits=logits,
        interventions={"offset_reverse": (shuffled_delta, shuffled_final)},
    )
    result = accumulator.as_dict()

    assert result["corridor_coverage"] == 1.0
    assert result["coarse_mae_px"] == 4.0
    assert result["active_mae_px"] == 1.0
    assert result["final_mae_px"] == 0.5
    assert result["active_lane_improvement_rate"] == 1.0
    assert result["final_lane_improvement_rate"] == 1.0
    assert result["evidence_interventions"]["offset_reverse"]["shuffled_final_mae_px"] == 4.0
    assert result["evidence_interventions"]["offset_reverse"]["normal_final_better_rate"] == 1.0
    assert result["offset_target_pearson"] == pytest.approx(1.0)


def test_discrete_offset_oracle_uses_only_available_offsets_on_valid_rows():
    stage = {
        "pred_x_rows": torch.tensor([[10.0, 10.0, 10.0], [50.0, 50.0, 50.0]]),
        "exist_logits": torch.zeros((2, 2)),
        "quality_logits": torch.zeros(2),
        "range_norm": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    }
    target = {
        "x_rows": torch.tensor([[16.0, 19.0, 100.0]]),
        "valid_mask": torch.tensor([[True, True, False]]),
    }
    match = {"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}

    oracle, stats = build_discrete_offset_oracle_stage(stage, target, match, torch.tensor([-8.0, 0.0, 8.0]))

    assert oracle["pred_x_rows"][0].tolist() == [18.0, 18.0, 10.0]
    assert oracle["pred_x_rows"][1].tolist() == [50.0, 50.0, 50.0]
    assert torch.equal(oracle["quality_pred_x_rows"], oracle["pred_x_rows"])
    assert stats == {"matched_slots": 1, "valid_rows": 2, "clamped_rows": 1}
