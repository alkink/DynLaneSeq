from __future__ import annotations

import torch

from dynlaneseq_eg.losses import HungarianMatcherS0, S0Criterion


def _target():
    x_rows = torch.full((1, 72), -1.0)
    valid = torch.zeros((1, 72), dtype=torch.bool)
    x_rows[0, 10:20] = 100.0
    valid[0, 10:20] = True
    return {"x_rows": x_rows, "valid_mask": valid, "range_y": torch.tensor([[40.0, 76.0]]), "x_bins": torch.zeros((1, 72), dtype=torch.long)}


def test_matcher_and_loss_ignore_invalid_rows():
    outputs = {
        "exist_logits": torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]], requires_grad=True),
        "pred_x_rows": torch.full((1, 2, 72), 100.0, requires_grad=True),
        "range_norm": torch.tensor([[[0.13, 0.27], [0.0, 1.0]]], requires_grad=True),
        "row_x_logits": torch.randn(1, 2, 72, 200, requires_grad=True),
    }
    targets = [_target()]
    matcher = HungarianMatcherS0()
    matches = matcher(outputs, targets)
    assert matches[0]["pred_indices"].numel() == 1
    losses = S0Criterion()(outputs, targets, matches)
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    assert outputs["pred_x_rows"].grad is not None


def test_smoothness_uses_only_contiguous_valid_triplets():
    pred_x = torch.zeros((1, 1, 72), requires_grad=True)
    pred_x.data[0, 0, 0] = 0.0
    pred_x.data[0, 0, 1] = 100.0
    pred_x.data[0, 0, 4] = 400.0
    outputs = {
        "exist_logits": torch.zeros((1, 1, 2), requires_grad=True),
        "pred_x_rows": pred_x,
        "range_norm": torch.zeros((1, 1, 2), requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    valid = torch.zeros((1, 72), dtype=torch.bool)
    valid[0, [0, 1, 4]] = True
    targets = [
        {
            "x_rows": torch.zeros((1, 72)),
            "valid_mask": valid,
            "range_y": torch.zeros((1, 2)),
            "x_bins": torch.zeros((1, 72), dtype=torch.long),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    loss = S0Criterion().compute_smoothness_loss(outputs, targets, matches)
    assert loss.item() == 0.0


def test_line_iou_has_gradient_for_non_overlapping_intervals():
    pred_x = torch.full((1, 1, 72), 400.0, requires_grad=True)
    outputs = {
        "exist_logits": torch.zeros((1, 1, 2), requires_grad=True),
        "pred_x_rows": pred_x,
        "range_norm": torch.zeros((1, 1, 2), requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    valid = torch.ones((1, 72), dtype=torch.bool)
    targets = [
        {
            "x_rows": torch.full((1, 72), 100.0),
            "valid_mask": valid,
            "range_y": torch.zeros((1, 2)),
            "x_bins": torch.zeros((1, 72), dtype=torch.long),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    loss = S0Criterion().compute_line_iou_loss(outputs, targets, matches)
    loss.backward()
    assert pred_x.grad is not None
    assert pred_x.grad.abs().sum() > 0


def test_seg_loss_skips_missing_masks():
    outputs = {"seg_logits": torch.randn(1, 1, 288, 800, requires_grad=True)}
    targets = [
        {
            "x_rows": torch.ones((1, 72)),
            "seg_mask": torch.zeros((1, 288, 800)),
            "seg_valid": torch.tensor(False),
        }
    ]
    loss = S0Criterion().compute_seg_loss(outputs, targets)
    assert loss.item() == 0.0


def test_quality_loss_has_gradient():
    pred_x = torch.full((1, 1, 72), 130.0, requires_grad=True)
    quality_logits = torch.zeros((1, 1), dtype=torch.float16, requires_grad=True)
    outputs = {
        "exist_logits": torch.zeros((1, 1, 2), requires_grad=True),
        "quality_logits": quality_logits,
        "pred_x_rows": pred_x,
        "range_norm": torch.zeros((1, 1, 2), requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    valid = torch.ones((1, 72), dtype=torch.bool)
    targets = [
        {
            "x_rows": torch.full((1, 72), 100.0),
            "valid_mask": valid,
            "range_y": torch.zeros((1, 2)),
            "x_bins": torch.zeros((1, 72), dtype=torch.long),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    loss = S0Criterion().compute_quality_loss(outputs, targets, matches)
    assert torch.isfinite(loss)
    loss.backward()
    assert quality_logits.grad is not None
    assert quality_logits.grad.abs().sum() > 0
