from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_hierarchical_cluster_selector import (
    ClusterExistenceProbe,
    RepresentativeQualityProbe,
    _representative_ids,
    cluster_and_representative_targets,
    hierarchy_gate,
    masked_quality_ranking_loss,
)


def test_targets_separate_unique_cluster_and_all_candidate_quality() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.85, 0.10, 0.00],
            [0.00, 0.05, 0.88, 0.80],
        ]
    )
    cluster_target, representative = cluster_and_representative_targets(
        iou,
        {0: [0, 1], 2: [2, 3]},
        positive_iou=0.30,
    )
    torch.testing.assert_close(cluster_target, torch.tensor([0.90, 0.88]))
    torch.testing.assert_close(
        representative,
        torch.tensor([0.90, 0.85, 0.88, 0.80]),
    )


def test_probes_return_expected_padded_shapes() -> None:
    torch.manual_seed(4)
    features = torch.randn(2, 4, 6)
    candidate_valid = torch.ones(2, 4, dtype=torch.bool)
    membership = torch.tensor(
        [
            [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
        ],
        dtype=torch.bool,
    )
    cluster_valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    source = torch.rand(2, 4)
    cluster = ClusterExistenceProbe(
        6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    representative = RepresentativeQualityProbe(
        6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    cluster_logits = cluster(
        features,
        candidate_valid,
        membership,
        cluster_valid,
        source,
    )
    representative_logits = representative(features, candidate_valid, source)
    assert cluster_logits.shape == (2, 3)
    assert representative_logits.shape == (2, 4)
    assert float(cluster_logits[0, 2]) == -1e4


def test_probes_remain_finite_for_an_all_invalid_image() -> None:
    torch.manual_seed(5)
    features = torch.randn(2, 4, 6)
    candidate_valid = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1]],
        dtype=torch.bool,
    )
    membership = torch.tensor(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 0, 0], [0, 0, 1, 1]],
        ],
        dtype=torch.bool,
    )
    cluster_valid = torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)
    source = torch.rand(2, 4)
    cluster = ClusterExistenceProbe(
        6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    representative = RepresentativeQualityProbe(
        6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    cluster_logits = cluster(
        features,
        candidate_valid,
        membership,
        cluster_valid,
        source,
    )
    representative_logits = representative(features, candidate_valid, source)
    assert torch.isfinite(cluster_logits).all()
    assert torch.isfinite(representative_logits).all()
    assert torch.equal(cluster_logits[0], torch.full((2,), -1e4))
    assert torch.equal(representative_logits[0], torch.full((4,), -1e4))


def test_representative_selection_can_replace_source_keeper() -> None:
    record = {
        "keepers": [0, 2],
        "members": [[0, 1], [2, 3]],
    }
    logits = torch.tensor([0.1, 0.9, 0.8, 0.2])
    assert _representative_ids([0, 1], record, logits, learned=False) == [0, 2]
    assert _representative_ids([0, 1], record, logits, learned=True) == [1, 2]


def test_grouped_ranking_loss_has_gradients() -> None:
    logits = torch.zeros((1, 4), requires_grad=True)
    targets = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    groups = torch.tensor([[0, 0, 1, 1]])
    losses = masked_quality_ranking_loss(
        logits,
        targets,
        valid,
        group_ids=groups,
        focal_beta=2.0,
        negative_weight=0.25,
        rank_weight=0.25,
        rank_margin=0.05,
        positive_iou=0.30,
    )
    losses["total"].backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    assert float(losses["ranking"]) > 0.0


def test_gate_requires_both_independent_arms_and_combination() -> None:
    def row(r50: float, r70: float) -> dict[str, float]:
        return {"recall_050": r50, "recall_070": r70}

    evaluation = {
        "modes": {
            "source_nms": row(0.73, 0.58),
            "learned_cluster_source_representative": row(0.80, 0.60),
            "source_cluster_learned_representative": row(0.75, 0.63),
            "learned_hierarchical": row(0.81, 0.65),
        }
    }
    gate = hierarchy_gate(
        evaluation,
        cluster_min_gain_050=5.0,
        representative_min_gain_070=3.0,
        hierarchical_min_gain_050=5.0,
        hierarchical_min_gain_070=3.0,
    )
    assert gate["cluster"]["positive"] is True
    assert gate["representative"]["positive"] is True
    assert gate["hierarchical"]["positive"] is True
    assert gate["dual_head_positive"] is True
    assert gate["interpretation"] == "decoupled_hierarchical_supervision_is_supported"
