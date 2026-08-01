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


def test_row_dfl_loss_backprops_to_row_logits_with_soft_bin_target():
    row_x_logits = torch.zeros((1, 1, 4, 8), requires_grad=True)
    outputs = {
        "exist_logits": torch.zeros((1, 1, 2), requires_grad=True),
        "pred_x_rows": torch.zeros((1, 1, 4), requires_grad=True),
        "range_norm": torch.zeros((1, 1, 2), requires_grad=True),
        "row_x_logits": row_x_logits,
    }
    targets = [
        {
            "x_rows": torch.tensor([[0.0, 10.0, 15.0, -1.0]]),
            "valid_mask": torch.tensor([[True, True, True, False]]),
            "range_y": torch.zeros((1, 2)),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    criterion = S0Criterion(LossConfig(input_w=80, w_row_dfl=1.0))
    loss = criterion.compute_row_dfl_loss(outputs, targets, matches)
    assert torch.isfinite(loss)
    loss.backward()
    assert row_x_logits.grad is not None
    assert row_x_logits.grad.abs().sum() > 0


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


def test_match_many_preserves_independent_layer_assignments():
    target = _target()
    matcher = HungarianMatcherS0(
        MatcherConfig(
            assignment="hungarian",
            lambda_obj=0.0,
            lambda_point=1.0,
            lambda_range=0.0,
            lambda_line_iou=0.0,
        )
    )

    def prediction(first_x: float, second_x: float):
        pred = torch.full((2, 72), 400.0)
        pred[0, 10:20] = first_x
        pred[1, 10:20] = second_x
        return {
            "exist_logits": torch.zeros((1, 2, 2)),
            "pred_x_rows": pred.unsqueeze(0),
            "range_norm": torch.zeros((1, 2, 2)),
        }

    outputs = [prediction(100.0, 300.0), prediction(300.0, 100.0)]
    repeated = [matcher(output, [target]) for output in outputs]
    batched = matcher.match_many(outputs, [target])

    assert len(batched) == len(repeated)
    for actual, expected in zip(batched, repeated):
        assert actual[0]["pred_indices"].tolist() == expected[0]["pred_indices"].tolist()
        assert actual[0]["gt_indices"].tolist() == expected[0]["gt_indices"].tolist()


def test_negative_probability_matcher_cost_is_bounded():
    matcher = HungarianMatcherS0(
        MatcherConfig(
            object_cost_type="neg_probability",
            lambda_point=0.0,
            lambda_range=0.0,
            lambda_line_iou=0.0,
        )
    )
    logits = torch.tensor([[-4.595, 0.0], [4.595, 0.0]])
    cost, stats = matcher.compute_cost_for_image(
        logits,
        torch.full((2, 72), 100.0),
        torch.zeros((2, 2)),
        _target(),
    )
    assert torch.all(cost <= 0.0)
    assert torch.all(cost >= -matcher.cfg.lambda_obj)
    assert -1.0 <= float(stats["mean_cost_obj"]) <= 0.0


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


def test_s0_intermediate_supervision_matches_each_layer_and_backpropagates():
    target = _target()

    def prediction(offset: float):
        return {
            "exist_logits": torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]], requires_grad=True),
            "pred_x_rows": torch.full((1, 2, 72), 100.0 + offset, requires_grad=True),
            "range_norm": torch.tensor([[[0.13, 0.27], [0.0, 1.0]]], requires_grad=True),
            "row_x_logits": torch.randn(1, 2, 72, 200, requires_grad=True),
        }

    final = prediction(1.0)
    auxiliaries = [prediction(12.0), prediction(8.0), prediction(4.0)]
    outputs = dict(final)
    outputs["aux_outputs"] = auxiliaries
    matcher = HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=2.0,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            line_iou_radius=15.0,
        )
    )
    matches = matcher(final, [target])
    criterion = S0Criterion(
        LossConfig(
            input_w=800,
            input_h=288,
            w_exist=2.0,
            w_point=5.0,
            w_range=1.0,
            w_line_iou=2.0,
            line_iou_radius=15.0,
            w_row_dfl=0.5,
            lambda_intermediate=0.5,
            intermediate_layer_weights=(1.0, 2.0, 3.0),
        ),
        matcher=matcher,
    )
    losses = criterion(outputs, [target], matches)

    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_intermediate_total"])
    assert losses["weight_intermediate"].item() == 0.5
    losses["loss_total"].backward()
    for auxiliary in auxiliaries:
        assert auxiliary["pred_x_rows"].grad is not None
        assert auxiliary["pred_x_rows"].grad.abs().sum() > 0
        assert auxiliary["row_x_logits"].grad is not None
        assert auxiliary["row_x_logits"].grad.abs().sum() > 0


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


def test_set_selection_targets_keep_one_unique_proposal_per_lane() -> None:
    pred_x = torch.stack(
        (
            torch.full((8,), 10.0),
            torch.full((8,), 10.0),
            torch.full((8,), 50.0),
        ),
        dim=0,
    ).unsqueeze(0)
    outputs = {
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]
        ),
    }
    targets = [
        {
            "x_rows": torch.stack(
                (torch.full((8,), 10.0), torch.full((8,), 50.0)),
                dim=0,
            ),
            "valid_mask": torch.ones((2, 8), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 8.0], [0.0, 8.0]]),
            "x_bins": torch.zeros((2, 8), dtype=torch.long),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=8,
            set_selection_line_width=30.0,
            set_selection_min_valid_rows=5,
        )
    )
    selection_targets = criterion.compute_set_selection_targets(
        outputs,
        targets,
    )

    assert selection_targets.shape == (1, 3)
    assert int((selection_targets > 0.99).sum()) == 2
    assert float(selection_targets[0, 2]) > 0.99
    assert int((selection_targets[0, :2] > 0.99).sum()) == 1


def test_set_selection_loss_backpropagates_to_selection_logits() -> None:
    selection_logits = torch.zeros((1, 2), requires_grad=True)
    outputs = {
        "selection_logits": selection_logits,
        "selection_delta_logits": selection_logits,
        "pred_x_rows": torch.stack(
            (torch.full((8,), 10.0), torch.full((8,), 50.0)),
            dim=0,
        ).unsqueeze(0),
        "range_norm": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
    }
    targets = [
        {
            "x_rows": torch.full((1, 8), 10.0),
            "valid_mask": torch.ones((1, 8), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 8.0]]),
            "x_bins": torch.zeros((1, 8), dtype=torch.long),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=8,
            w_set_selection=1.0,
        )
    )
    losses = criterion.compute_set_selection_loss(outputs, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    assert selection_logits.grad is not None
    assert float(selection_logits.grad.abs().sum()) > 0.0
    assert float(losses["target_positive_fraction"]) == 0.5


def test_range_aware_matcher_and_selector_share_one_assignment() -> None:
    rows = 8
    # Both candidates have perfect x, but candidate zero hallucinates the lane
    # over the complete image height. Candidate one has the correct range.
    pred_x = torch.full((1, 2, rows), 20.0)
    outputs = {
        "exist_logits": torch.tensor([[[12.0, -12.0], [-12.0, 12.0]]]),
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.0, 1.0], [0.25, 0.70]]]),
        "selection_logits": torch.zeros((1, 2), requires_grad=True),
        "selection_delta_logits": torch.zeros((1, 2)),
    }
    valid = torch.zeros((1, rows), dtype=torch.bool)
    valid[:, 2:6] = True
    target_x = torch.full((1, rows), float("nan"))
    target_x[:, 2:6] = 20.0
    targets = [
        {
            "x_rows": target_x,
            "valid_mask": valid,
            "range_y": torch.tensor([[2.0, 5.0]]),
        }
    ]
    matcher = HungarianMatcherS0(
        MatcherConfig(
            input_w=64,
            input_h=8,
            cost_type="range_aware_iou",
            range_aware_line_width=30.0,
            range_aware_min_valid_rows=3,
        )
    )
    matches = matcher(outputs, targets)

    # Classification strongly favors candidate zero, proving that the new
    # geometry-only assignment is what selects candidate one.
    assert matches[0]["pred_indices"].tolist() == [1]
    assert matches[0]["gt_indices"].tolist() == [0]

    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=8,
            w_set_selection=1.0,
            set_selection_line_width=30.0,
            set_selection_min_valid_rows=3,
            set_selection_share_matcher_assignment=True,
            set_selection_positive_floor=0.5,
        )
    )
    selection_targets = criterion.compute_set_selection_targets(
        outputs,
        targets,
        matches,
    )
    assert selection_targets[0, 0].item() == 0.0
    assert selection_targets[0, 1].item() > 0.99


def test_selection_positive_floor_prevents_all_negative_cold_start() -> None:
    rows = 8
    outputs = {
        "pred_x_rows": torch.full((1, 2, rows), 60.0),
        "range_norm": torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
    }
    targets = [
        {
            "x_rows": torch.full((1, rows), 5.0),
            "valid_mask": torch.ones((1, rows), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 7.0]]),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([0]),
            "gt_indices": torch.tensor([0]),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=8,
            set_selection_share_matcher_assignment=True,
            set_selection_positive_floor=0.5,
        )
    )
    target = criterion.compute_set_selection_targets(outputs, targets, matches)
    torch.testing.assert_close(target, torch.tensor([[0.5, 0.0]]))


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
