from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v4_pointer_root_causes import (
    ParallelHungarianProbe,
    _cluster_ranking_metrics,
    _pointer_losses,
    selection_metrics,
    unique_hungarian_target,
)


def _quality() -> torch.Tensor:
    return torch.tensor(
        [
            [0.90, 0.05],
            [0.80, 0.04],
            [0.03, 0.85],
            [0.02, 0.70],
        ],
        dtype=torch.float32,
    )


def test_unique_hungarian_target_selects_one_representative_per_gt() -> None:
    target, count = unique_hungarian_target(
        _quality(),
        torch.ones(4, dtype=torch.bool),
        representable_min=0.20,
        max_selections=4,
    )

    assert count == 2
    assert target.tolist() == [True, False, True, False]


def test_selection_metrics_are_duplicate_safe() -> None:
    record = {
        "quality": _quality(),
        "candidate_valid": torch.ones(4, dtype=torch.bool),
    }
    duplicate = selection_metrics([record], [[0, 1]])["iou_0.50"]
    diverse = selection_metrics([record], [[0, 2]])["iou_0.50"]

    assert duplicate["tp"] == 1
    assert duplicate["fp"] == 1
    assert diverse["tp"] == 2
    assert diverse["fp"] == 0


def test_cluster_ranking_reports_representative_regret() -> None:
    record = {
        "quality": _quality(),
        "candidate_valid": torch.ones(4, dtype=torch.bool),
    }
    good = _cluster_ranking_metrics(
        [record],
        [torch.tensor([4.0, 3.0, 2.0, 1.0])],
    )
    poor = _cluster_ranking_metrics(
        [record],
        [torch.tensor([3.0, 4.0, 1.0, 2.0])],
    )

    assert good["top1_rate"] == 1.0
    assert good["mean_regret"] == 0.0
    assert poor["top1_rate"] == 0.0
    assert poor["mean_regret"] > 0.0


def test_parallel_probe_shapes_candidate_and_count_outputs() -> None:
    probe = ParallelHungarianProbe(hidden_dim=16, max_selections=4)
    hidden = torch.randn(3, 7, 16)
    valid = torch.ones(3, 7, dtype=torch.bool)

    candidate, count = probe(hidden, valid)

    assert candidate.shape == (3, 7)
    assert count.shape == (3, 5)


def test_pointer_soft_loss_reaches_target_entropy_lower_bound() -> None:
    target = torch.tensor(
        [[[[0.75, 0.25, 0.0], [0.0, 0.0, 1.0]]]],
        dtype=torch.float32,
    ).reshape(1, 2, 3)
    logits = target.clamp_min(1e-8).log()
    teacher = {
        "probabilities": target,
        "active": torch.tensor([[True, True]]),
    }
    total, parts = _pointer_losses(
        logits,
        torch.zeros(1, 2),
        teacher,
        torch.zeros(1, 2),
        stop_weight=1.0,
        quality_weight=0.0,
        focal_beta=0.0,
    )
    entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1).mean()

    assert torch.allclose(parts["sequence"], entropy, atol=1e-6)
    assert torch.allclose(total, entropy, atol=1e-6)
