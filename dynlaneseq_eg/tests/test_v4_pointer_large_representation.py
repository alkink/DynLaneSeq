from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v4_pointer_large_representation import (
    NormalizedQualityProbe,
    build_cluster_pairs,
    evaluate_scores,
    make_image_disjoint_splits,
    probe_training_loss,
)


def _quality() -> torch.Tensor:
    return torch.tensor(
        [
            [0.90, 0.05],
            [0.80, 0.04],
            [0.03, 0.88],
            [0.02, 0.84],
        ],
        dtype=torch.float32,
    )


def test_cluster_pairs_are_ordered_within_natural_gt_owner() -> None:
    pairs, weights = build_cluster_pairs(
        _quality(),
        torch.ones(4, dtype=torch.bool),
        representable_min=0.50,
        support_min=0.20,
        support_delta=0.20,
        min_quality_gap=0.02,
    )

    assert pairs.tolist() == [[0, 1], [2, 3]]
    assert torch.all(weights > 0.0)


def test_cluster_pairs_exclude_invalid_and_ambiguous_candidates() -> None:
    pairs, _weights = build_cluster_pairs(
        _quality(),
        torch.tensor([True, False, True, True]),
        representable_min=0.50,
        support_min=0.20,
        support_delta=0.20,
        min_quality_gap=0.05,
    )

    assert pairs.numel() == 0


def test_image_splits_are_disjoint_complete_and_reproducible() -> None:
    first = make_image_disjoint_splits(20, 16, 4, 3, 1907)
    second = make_image_disjoint_splits(20, 16, 4, 3, 1907)

    assert first == second
    assert len(first["fit"]) == 13
    assert len(first["early_stop"]) == 3
    assert len(first["holdout"]) == 4
    assert set(first["fit"]).isdisjoint(first["early_stop"])
    assert set(first["development"]).isdisjoint(first["holdout"])
    assert set(first["development"] + first["holdout"]) == set(range(20))


def test_pairwise_training_loss_rewards_correct_representative_order() -> None:
    cache = {
        "pairs": [torch.tensor([[0, 1], [2, 3]], dtype=torch.long)],
        "pair_weights": [torch.ones(2)],
        "candidate_valid": torch.ones(1, 4, dtype=torch.bool),
        "quality_max": torch.tensor([[0.90, 0.80, 0.88, 0.84]]),
    }
    correct = torch.tensor([[4.0, 1.0, 3.0, 0.0]])
    reversed_order = -correct

    correct_loss, _, _ = probe_training_loss(
        correct,
        [0],
        cache,
        quality_aux_weight=0.0,
    )
    reversed_loss, _, _ = probe_training_loss(
        reversed_order,
        [0],
        cache,
        quality_aux_weight=0.0,
    )

    assert correct_loss < reversed_loss


def test_normalized_probe_preserves_candidate_axis() -> None:
    probe = NormalizedQualityProbe(
        torch.zeros(7),
        torch.ones(7),
        nonlinear=True,
        nonlinear_hidden=5,
        dropout=0.0,
    )

    assert probe(torch.randn(3, 11, 7)).shape == (3, 11)


def test_evaluation_counts_each_pair_once() -> None:
    quality = _quality()
    pairs, weights = build_cluster_pairs(
        quality,
        torch.ones(4, dtype=torch.bool),
        representable_min=0.50,
        support_min=0.20,
        support_delta=0.20,
        min_quality_gap=0.02,
    )
    cache = {
        "quality": [quality],
        "pairs": [pairs],
        "pair_weights": [weights],
        "candidate_valid": torch.ones(1, 4, dtype=torch.bool),
        "quality_max": quality.amax(dim=-1).unsqueeze(0),
    }
    metrics = evaluate_scores(
        cache,
        [0],
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        representable_min=0.50,
    )

    ranking = metrics["cluster_ranking"]
    assert ranking["pair_count"] == 2
    assert ranking["pairwise_accuracy"] == 1.0
    assert ranking["top1_rate"] == 1.0
