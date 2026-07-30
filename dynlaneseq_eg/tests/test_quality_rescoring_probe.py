from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_row_reference_quality_rescoring import (
    _probe_verdict,
    all_proposal_quality_targets,
    geometry_aware_features,
    pairwise_lane_quality,
    pairwise_quality_ranking_loss,
    quality_focal_loss,
    unique_hungarian_quality_targets,
)


def test_pairwise_lane_quality_rewards_geometry_and_visible_range() -> None:
    gt_x = torch.full((1, 8), 40.0)
    gt_valid = torch.ones((1, 8), dtype=torch.bool)
    pred_x = torch.stack(
        (
            torch.full((8,), 40.0),
            torch.full((8,), 80.0),
            torch.full((8,), 40.0),
        )
    )
    ranges = torch.tensor(
        (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.5, 1.0),
        )
    )
    quality = pairwise_lane_quality(
        pred_x,
        ranges,
        gt_x,
        gt_valid,
        input_h=80,
        line_width=30.0,
    )
    assert quality.shape == (1, 3)
    assert quality[0, 0] > 0.99
    assert quality[0, 1] == 0.0
    assert 0.35 < quality[0, 2] < 0.65


def test_all_proposal_targets_score_unmatched_geometric_candidates() -> None:
    outputs = {
        "pred_x_rows": torch.tensor(
            [[
                [20.0, 20.0, 20.0, 20.0],
                [21.0, 21.0, 21.0, 21.0],
                [80.0, 80.0, 80.0, 80.0],
            ]]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[20.0, 20.0, 20.0, 20.0]]),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
        }
    ]
    target_quality, _ = all_proposal_quality_targets(
        outputs,
        targets,
        input_h=40,
        line_width=30.0,
    )
    assert target_quality.shape == (1, 3)
    assert target_quality[0, 0] > 0.99
    assert target_quality[0, 1] > 0.90
    assert target_quality[0, 2] == 0.0


def test_unique_hungarian_target_suppresses_duplicate_candidates() -> None:
    outputs = {
        "pred_x_rows": torch.tensor(
            [[
                [20.0, 20.0, 20.0, 20.0],
                [21.0, 21.0, 21.0, 21.0],
                [80.0, 80.0, 80.0, 80.0],
            ]]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[20.0, 20.0, 20.0, 20.0]]),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
        }
    ]
    target_quality, _ = unique_hungarian_quality_targets(
        outputs,
        targets,
        input_h=40,
        line_width=30.0,
    )
    assert target_quality.shape == (1, 3)
    assert target_quality[0, 0] > 0.99
    assert target_quality[0, 1] == 0.0
    assert target_quality[0, 2] == 0.0


def test_geometry_aware_features_are_finite_and_have_expected_shape() -> None:
    batch, candidates, rows, channels, bins = 2, 3, 6, 8, 10
    outputs = {
        "structured_row_tokens": torch.randn(batch, candidates, rows, channels),
        "queries": torch.randn(batch, candidates, channels),
        "range_norm": torch.tensor(
            [[[0.1, 0.9]] * candidates, [[0.2, 0.8]] * candidates]
        ),
        "pred_x_rows": torch.rand(batch, candidates, rows) * 100.0,
        "row_x_logits": torch.randn(batch, candidates, rows, bins),
        "input_reference_x_rows": torch.rand(batch, candidates, rows) * 100.0,
    }
    features = geometry_aware_features(outputs, input_w=100)
    assert features.shape == (batch, candidates, 2 * channels + 10)
    assert torch.isfinite(features).all()


def test_quality_losses_prefer_correct_order_and_calibration() -> None:
    targets = torch.tensor([[0.9, 0.1]])
    good_logits = torch.tensor([[3.0, -3.0]])
    bad_logits = torch.tensor([[-3.0, 3.0]])
    assert quality_focal_loss(good_logits, targets) < quality_focal_loss(
        bad_logits, targets
    )
    assert pairwise_quality_ranking_loss(good_logits, targets) < (
        pairwise_quality_ranking_loss(bad_logits, targets)
    )


def test_probe_verdict_distinguishes_target_from_representation() -> None:
    evaluation = {
        "strategies": {
            "current_exist_quality": {
                "top4_recall_050": 0.70,
                "top4_recall_070": 0.50,
            },
            "query_probe": {
                "top4_recall_050": 0.72,
                "top4_recall_070": 0.51,
            },
            "geometry_probe": {
                "top4_recall_050": 0.73,
                "top4_recall_070": 0.52,
            },
        }
    }
    verdict = _probe_verdict(
        evaluation,
        min_gain_050_points=1.0,
        min_gain_070_points=0.5,
    )
    assert verdict["recommendation"] == "quality_target_and_loss_are_sufficient"
