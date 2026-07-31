from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_row_reference_joint_score_recovery import (
    balanced_joint_loss,
    historical_joint_loss,
    training_targets,
    verdict,
)


def test_training_targets_reproduce_matched_quality() -> None:
    pred_x = torch.tensor([[[10.0, 20.0], [40.0, 50.0]]])
    outputs = {
        "pred_x_rows": pred_x,
        "quality_pred_x_rows": pred_x,
    }
    targets = [
        {
            "x_rows": torch.tensor([[10.0, 20.0]]),
            "valid_mask": torch.tensor([[True, True]]),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([0]),
            "gt_indices": torch.tensor([0]),
        }
    ]
    exist, quality = training_targets(outputs, targets, matches, radius=15.0)
    assert torch.equal(exist, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(quality, torch.tensor([[1.0, 0.0]]))


def test_balanced_loss_is_invariant_to_repeated_negatives() -> None:
    exist_logits = torch.tensor([[0.0, 0.0]])
    quality_logits = torch.tensor([[0.0, 0.0]])
    exist_targets = torch.tensor([[1.0, 0.0]])
    quality_targets = torch.tensor([[0.8, 0.0]])
    loss_two, _ = balanced_joint_loss(
        exist_logits,
        quality_logits,
        exist_targets,
        quality_targets,
        focal_gamma=2.0,
        rank_weight=0.0,
        rank_target_margin=0.1,
    )
    loss_many, _ = balanced_joint_loss(
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.8, 0.0, 0.0, 0.0]]),
        focal_gamma=2.0,
        rank_weight=0.0,
        rank_target_margin=0.1,
    )
    assert torch.allclose(loss_two, loss_many)


def test_historical_loss_changes_with_negative_multiplicity() -> None:
    loss_two, _ = historical_joint_loss(
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.8, 0.0]]),
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    loss_many, _ = historical_joint_loss(
        torch.tensor([[1.0, -1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.8, 0.0, 0.0, 0.0]]),
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    assert loss_many > loss_two


def _row(ap_match: float, ap050: float, ap075: float) -> dict[str, float]:
    return {
        "assignment_target_ap": ap_match,
        "candidate_row_iou_ap_050": ap050,
        "candidate_row_iou_ap_075": ap075,
        "top4_row_recall_050": 0.5,
        "top4_row_recall_075": 0.3,
    }


def test_verdict_separates_loss_from_representation() -> None:
    evaluation = {
        "strategies": {
            "current": _row(0.50, 0.50, 0.40),
            "historical_query": _row(0.51, 0.50, 0.40),
            "balanced_query": _row(0.60, 0.55, 0.44),
            "balanced_geometry": _row(0.61, 0.56, 0.45),
        }
    }
    assert verdict(evaluation)["diagnosis"] == "historical_score_loss_imbalance"
