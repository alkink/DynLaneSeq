from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v5_ownership_graph import (
    _finish_transition,
    _label_disagreement,
    _new_transition,
    _update_transition,
)
from dynlaneseq_eg.tools.probe_v5_four_slot_router import (
    _aggregate_proposal_targets,
    _decode_slots,
    _jointly_representable_targets,
    _permutation_marginal_slot_loss,
)


def test_layer_assignment_transition_separates_identity_and_presence() -> None:
    counts = _new_transition()
    _update_transition(counts, [1, 2, -1, 4, -1], [1, 3, 5, -1, -1])
    result = _finish_transition(counts)
    assert result["same_gt_same_query"] == 1
    assert result["same_gt_different_query"] == 1
    assert result["unmatched_to_matched"] == 1
    assert result["matched_to_unmatched"] == 1
    assert result["unmatched_both"] == 1
    assert result["same_query_given_matched_both"] == 0.5
    assert result["any_target_state_change"] == 0.6


def test_binary_owner_target_disagreement_counts_query_labels() -> None:
    different, total = _label_disagreement([1, 3], [1, 4], 8)
    assert different == 2
    assert total == 8


def test_jointly_representable_targets_are_soft_and_candidate_only() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.87, 0.05, 0.02],
            [0.01, 0.03, 0.80, 0.77],
        ]
    )
    targets = _jointly_representable_targets(
        iou,
        torch.tensor([True, True, True, True]),
        num_slots=4,
        representable_min=0.50,
        cluster_min=0.30,
        cluster_delta=0.05,
        temperature=0.03,
    )
    assert targets.shape == (2, 5)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(2))
    assert float(targets[:, -1].sum()) == 0.0
    assert bool((targets[0, :2] > 0).all())
    assert bool((targets[1, 2:4] > 0).all())


def test_permutation_marginal_slot_loss_is_slot_order_invariant() -> None:
    target = [
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
    ]
    logits = torch.tensor(
        [
            [
                [8.0, 0.0, 0.0, 0.0, -2.0],
                [0.0, 8.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 8.0],
            ]
        ]
    )
    loss = _permutation_marginal_slot_loss(
        logits, target, permutation_temperature=1.0
    )
    permuted = _permutation_marginal_slot_loss(
        logits[:, [2, 0, 3, 1]], target, permutation_temperature=1.0
    )
    assert torch.allclose(loss, permuted, atol=1e-6)
    bad = _permutation_marginal_slot_loss(
        torch.zeros_like(logits), target, permutation_temperature=1.0
    )
    assert float(loss) < float(bad)


def test_proposal_arm_receives_same_clusters_with_gt_axis_collapsed() -> None:
    rows = [
        torch.tensor(
            [
                [0.7, 0.3, 0.0, 0.0],
                [0.0, 0.2, 0.8, 0.0],
            ]
        )
    ]
    targets = _aggregate_proposal_targets(
        rows,
        candidate_count=3,
        device=torch.device("cpu"),
    )
    assert torch.allclose(targets, torch.tensor([[0.7, 0.3, 0.8]]))


def test_slot_decode_is_globally_unique_and_uses_dustbin() -> None:
    # Both first slots prefer candidate zero. The global assignment gives the
    # second slot its next-best candidate, while the last slot chooses dustbin.
    logits = torch.tensor(
        [
            [8.0, 1.0, 0.0, -2.0],
            [7.0, 6.0, 0.0, -2.0],
            [0.0, 0.0, 0.0, 5.0],
        ]
    )
    selected, scores = _decode_slots(
        logits, torch.tensor([True, True, False])
    )
    assert selected == [0, 1]
    assert len(selected) == len(set(selected))
    assert scores.shape == (3,)
