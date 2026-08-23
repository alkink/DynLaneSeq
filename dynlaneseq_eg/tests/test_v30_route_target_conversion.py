from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v30_route_target_conversion import (
    LaneObservation,
    _official_match,
    _target_slot_assignment,
    _unique_target_policy,
    build_training_route_targets,
    summarize_observations,
)


def test_training_target_prefers_the_near_curve() -> None:
    record = {
        "target": {
            "x_rows": torch.full((1, 4), 10.0),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
        }
    }
    stage = {
        "pred_x_rows": torch.tensor(
            [[10.0, 10.0, 10.0, 10.0], [12.0, 12.0, 12.0, 12.0], [50.0, 50.0, 50.0, 50.0]]
        ),
        "range_norm": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    }
    targets, support, quality, supervised = build_training_route_targets(
        record,
        stage,
        input_h=4,
        line_width=10.0,
        min_valid_rows=1,
        cluster_min=0.0,
        cluster_delta=0.20,
        temperature=0.03,
    )
    assert targets.shape == (1, 3)
    assert int(targets.argmax(dim=-1)[0]) == 0
    assert bool(support[0, 0])
    assert quality[0, 0] > quality[0, 1] > quality[0, 2]
    assert torch.allclose(targets.sum(dim=-1), torch.ones(1))
    assert supervised.tolist() == [True]


def test_target_policy_and_official_matching_are_unique() -> None:
    targets = torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.7, 0.0]])
    policy = _unique_target_policy(targets)
    assert policy == (0, 1)
    official = torch.tensor([[0.9, 0.2, 0.0], [0.1, 0.8, 0.0]])
    iou, ids = _official_match(official, policy)
    assert torch.allclose(iou, torch.tensor([0.9, 0.8]))
    assert ids.tolist() == [0, 1]


def test_target_slot_assignment_respects_slot_preferences() -> None:
    active = torch.tensor([5.0, 5.0])
    conditional_log = torch.log(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert _target_slot_assignment(active, conditional_log, targets) == (0, 1)


def test_observation_summary_preserves_counts() -> None:
    base = dict(
        image_id="image",
        gt_index=0,
        support_hit=True,
        official_best_iou=0.9,
        official_best_proposal=1,
        target_supervised=True,
        target_support_has_good=True,
        target_top1_good=True,
        target_top1_is_official_best=True,
        target_top1_official_iou=0.9,
        target_good_mass=0.8,
        target_policy_hit=True,
        learned_target_slot_good=False,
        learned_target_slot_active=True,
        learned_target_slot_active_good=False,
        learned_target_slot_is_target_top1=False,
        learned_target_slot_official_iou=0.4,
        learned_good_mass=0.1,
        learned_any_slot_good=False,
        learned_any_active_slot_good=False,
        learned_official_match_hit=False,
        learned_official_match_iou=0.4,
    )
    summary = summarize_observations([LaneObservation(**base)])
    assert summary["lanes"] == 1
    assert summary["target_top1_good"]["fraction"] == 1.0
    assert summary["learned_official_match_hit"]["fraction"] == 0.0
    assert summary["target_good_mass"]["mean"] == 0.8
