from __future__ import annotations

import torch

from dynlaneseq_eg.losses import HungarianMatcherS0, S0Criterion
from dynlaneseq_eg.losses.loss_s0 import LossConfig
from dynlaneseq_eg.losses.loss_s2 import S2LossConfig
from dynlaneseq_eg.losses.loss_s3 import S3Criterion
from dynlaneseq_eg.losses.matcher_s0 import MatcherConfig


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


def test_matcher_line_iou_cost_is_finite_for_non_overlapping_lanes():
    target = _target()
    pred = torch.full((2, 72), 400.0)
    pred[1, 10:20] = 100.0
    matcher = HungarianMatcherS0(MatcherConfig(lambda_point=0.0, lambda_range=0.0, lambda_line_iou=1.0, line_iou_radius=7.5))
    cost, stats = matcher.compute_cost_for_image(
        torch.tensor([[5.0, -5.0], [5.0, -5.0]]),
        pred,
        torch.tensor([[0.13, 0.27], [0.13, 0.27]]),
        target,
    )
    assert torch.isfinite(cost).all()
    assert torch.isfinite(stats["mean_cost_line_iou"])
    assert cost[1, 0] < cost[0, 0]


def test_grouped_one_to_many_matcher_assigns_gt_once_per_group():
    target = _target()
    pred = torch.full((8, 72), 400.0)
    pred[0, 10:20] = 100.0
    pred[4, 10:20] = 100.0
    matcher = HungarianMatcherS0(
        MatcherConfig(
            assignment="grouped_one_to_many",
            num_groups=2,
            lambda_obj=0.0,
            lambda_point=1.0,
            lambda_range=0.0,
        )
    )
    outputs = {
        "exist_logits": torch.zeros((1, 8, 2)),
        "pred_x_rows": pred.unsqueeze(0),
        "range_norm": torch.zeros((1, 8, 2)),
    }
    matches = matcher(outputs, [target])
    assert matches[0]["pred_indices"].tolist() == [0, 4]
    assert matches[0]["gt_indices"].tolist() == [0, 0]


def test_focal_exist_loss_backprops_with_grouped_matches():
    logits = torch.zeros((1, 8, 2), requires_grad=True)
    outputs = {
        "exist_logits": logits,
        "pred_x_rows": torch.zeros((1, 8, 72), requires_grad=True),
        "range_norm": torch.zeros((1, 8, 2), requires_grad=True),
        "row_x_logits": torch.zeros((1, 8, 72, 200), requires_grad=True),
    }
    matches = [{"pred_indices": torch.tensor([0, 4]), "gt_indices": torch.tensor([0, 0])}]
    loss = S0Criterion(LossConfig(exist_loss_type="focal")).compute_exist_loss(outputs, matches)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


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


def test_centerline_loss_builds_soft_row_targets_and_backprops():
    logits = torch.zeros((1, 1, 72, 200), requires_grad=True)
    outputs = {"centerline_logits": logits, "exist_logits": torch.zeros((1, 1, 2))}
    valid = torch.zeros((2, 72), dtype=torch.bool)
    valid[0, 10] = True
    valid[1, 10] = True
    targets = [
        {
            "x_rows": torch.stack([torch.full((72,), 100.0), torch.full((72,), 104.0)]),
            "valid_mask": valid,
        }
    ]
    loss = S0Criterion().compute_centerline_loss(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_dynamic_proposal_seed_masked_losses_backprop():
    heatmap_logits = torch.zeros((1, 1, 18, 50), requires_grad=True)
    dense_x = torch.full((1, 72, 18, 50), 120.0, requires_grad=True)
    dense_range = torch.full((1, 2, 18, 50), 0.5, requires_grad=True)
    outputs = {
        "dynamic_proposals": {
            "dense": {
                "heatmap_logits": heatmap_logits,
                "x_rows": dense_x,
                "range_norm": dense_range,
            }
        },
        "exist_logits": torch.zeros((1, 1, 2)),
    }
    valid = torch.zeros((1, 72), dtype=torch.bool)
    valid[0, 10:20] = True
    targets = [
        {
            "x_rows": torch.full((1, 72), 100.0),
            "valid_mask": valid,
            "range_y": torch.tensor([[40.0, 80.0]]),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            dynamic_proposal_sigma_bins=1.5,
            dynamic_proposal_seed_radius_bins=2,
            dynamic_proposal_heatmap_pos_weight=4.0,
        )
    )
    losses = criterion.compute_dynamic_proposal_losses(outputs, targets)
    total = losses["heatmap"] + losses["x"] + losses["range"]
    assert torch.isfinite(total)
    total.backward()
    assert heatmap_logits.grad is not None and heatmap_logits.grad.abs().sum() > 0
    assert dense_x.grad is not None and dense_x.grad.abs().sum() > 0
    assert dense_range.grad is not None and dense_range.grad.abs().sum() > 0


def test_s0_lambda_coarse_adds_draft_supervision():
    target = _target()
    coarse = {
        "exist_logits": torch.tensor([[[5.0, -5.0]]], requires_grad=True),
        "pred_x_rows": torch.full((1, 1, 72), 100.0, requires_grad=True),
        "range_norm": torch.tensor([[[0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    final = {
        "exist_logits": torch.tensor([[[5.0, -5.0]]], requires_grad=True),
        "pred_x_rows": torch.full((1, 1, 72), 100.0, requires_grad=True),
        "range_norm": torch.tensor([[[0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    outputs = {"coarse": coarse, "final": final}
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    criterion = S0Criterion(LossConfig(lambda_coarse=0.25, w_line_iou=0.0, w_seg=0.0, w_quality=0.0))
    losses = criterion(outputs, [target], matches)
    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_coarse_total"])
    losses["loss_total"].backward()
    assert coarse["pred_x_rows"].grad is not None
    assert final["pred_x_rows"].grad is not None


def test_geometry_draft_supervision_backprops_to_sampler_draft():
    target = _target()
    geometry_draft = {
        "exist_logits": torch.tensor([[[5.0, -5.0]]], requires_grad=True),
        "pred_x_rows": torch.full((1, 1, 72), 130.0, requires_grad=True),
        "range_norm": torch.tensor([[[0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    final = {
        "exist_logits": torch.tensor([[[5.0, -5.0]]], requires_grad=True),
        "pred_x_rows": torch.full((1, 1, 72), 120.0, requires_grad=True),
        "range_norm": torch.tensor([[[0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 1, 72, 200), requires_grad=True),
    }
    outputs = {"s0_geometry_draft": geometry_draft, "final": final}
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    criterion = S0Criterion(LossConfig(lambda_geometry_draft=0.25, w_line_iou=0.0, w_seg=0.0, w_quality=0.0))
    losses = criterion(outputs, [target], matches)
    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_geometry_draft_total"])
    losses["loss_total"].backward()
    assert geometry_draft["pred_x_rows"].grad is not None
    assert geometry_draft["pred_x_rows"].grad.abs().sum() > 0


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


def test_s3_cascade_matching_uses_final_assignment():
    target = _target()
    coarse_x = torch.full((1, 2, 72), 400.0)
    coarse_x[0, 0, 10:20] = 100.0
    final_x = torch.full((1, 2, 72), 400.0)
    final_x[0, 1, 10:20] = 100.0
    coarse_exist = torch.tensor([[[5.0, -5.0], [5.0, -5.0]]], requires_grad=True)
    final_exist = coarse_exist.detach().clone().requires_grad_(True)
    coarse = {
        "exist_logits": coarse_exist,
        "range_norm": torch.tensor([[[0.13, 0.27], [0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 2, 72, 200), requires_grad=True),
        "pred_x_rows": coarse_x.clone().requires_grad_(True),
    }
    final = {
        "exist_logits": final_exist,
        "range_norm": torch.tensor([[[0.13, 0.27], [0.13, 0.27]]], requires_grad=True),
        "row_x_logits": torch.zeros((1, 2, 72, 200), requires_grad=True),
        "pred_x_rows": final_x.clone().requires_grad_(True),
        "quality_logits": torch.zeros((1, 2), requires_grad=True),
    }
    outputs = {"coarse": coarse, "final": final, "evidence": {}}
    matcher = HungarianMatcherS0()
    matches_coarse = matcher(coarse, [target])
    criterion = S3Criterion(
        S2LossConfig(cascade_matching=True, w_token=0.0, w_quality=0.0, w_line_iou=0.0, w_seg=0.0),
        matcher=matcher,
    )
    losses = criterion(outputs, [target], matches_coarse)
    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_exist_coarse"])
    assert losses["cascade_match_changed_ratio"].item() == 1.0
    losses["loss_total"].backward()
    assert coarse_exist.grad is not None
    assert coarse_exist.grad.abs().sum() > 0
