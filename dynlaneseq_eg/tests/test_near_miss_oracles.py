import torch

from dynlaneseq_eg.evaluation.near_miss_oracles import (
    best_constant_shift,
    bounded_row_correction,
    row_lane_iou,
)


def test_constant_shift_oracle_recovers_uniform_lateral_error():
    gt_x = torch.full((8,), 100.0)
    pred_x = gt_x + 12.0
    mask = torch.ones(8, dtype=torch.bool)
    result = best_constant_shift(
        pred_x,
        mask,
        gt_x,
        mask,
        torch.arange(-20.0, 21.0),
        input_w=200,
        line_width=30.0,
    )
    assert result.parameter == -12.0
    assert torch.allclose(result.pred_x, gt_x)
    assert torch.isclose(row_lane_iou(result.pred_x, result.pred_mask, gt_x, mask), torch.tensor(1.0))


def test_range_oracle_exposes_extra_rows_as_union_penalty():
    pred_x = torch.full((8,), 50.0)
    gt_x = pred_x.clone()
    pred_mask = torch.ones(8, dtype=torch.bool)
    gt_mask = torch.tensor([False, False, True, True, True, True, False, False])
    baseline = row_lane_iou(pred_x, pred_mask, gt_x, gt_mask)
    oracle = row_lane_iou(pred_x, gt_mask, gt_x, gt_mask)
    assert baseline < 1.0
    assert torch.isclose(oracle, torch.tensor(1.0))


def test_bounded_row_correction_handles_nonuniform_curve_error():
    gt_x = torch.tensor([50.0, 52.0, 55.0, 59.0, 64.0, 70.0])
    pred_x = gt_x + torch.tensor([-8.0, -4.0, 0.0, 4.0, 8.0, 12.0])
    mask = torch.ones(6, dtype=torch.bool)
    corrected = bounded_row_correction(
        pred_x,
        mask,
        gt_x,
        mask,
        8.0,
        input_w=200,
    )
    before = row_lane_iou(pred_x, mask, gt_x, mask, line_width=10.0)
    after = row_lane_iou(corrected.pred_x, corrected.pred_mask, gt_x, mask, line_width=10.0)
    assert after > before
    assert torch.all((corrected.pred_x - pred_x).abs() <= 8.0)
    assert torch.allclose(corrected.pred_x[:5], gt_x[:5])
    assert corrected.pred_x[-1] == gt_x[-1] + 4.0

