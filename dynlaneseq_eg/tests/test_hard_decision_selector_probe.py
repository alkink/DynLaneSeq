from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_hard_decision_selector import (
    ClusterCoveragePointerProbe,
    coverage_pointer_loss,
    evaluate_source_nms,
    marginal_coverage_teacher,
    representative_listwise_loss,
    representative_target_statistics,
)


def test_marginal_teacher_selects_new_lane_instead_of_duplicate() -> None:
    cluster_iou = torch.tensor(
        [
            [0.90, 0.85, 0.00],
            [0.00, 0.00, 0.82],
        ]
    )
    distribution, sequence, active = marginal_coverage_teacher(
        cluster_iou,
        top_k=3,
        temperature=0.10,
        min_gain=1e-5,
        weight_050=1.0,
        weight_070=0.5,
        weight_iou=0.1,
    )
    assert sequence[:2].tolist() == [0, 2]
    assert active.tolist() == [True, True, False]
    assert float(distribution[1, 2]) == 1.0


def test_pointer_selects_without_replacement() -> None:
    torch.manual_seed(3)
    model = ClusterCoveragePointerProbe(
        6,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        top_k=4,
    ).eval()
    features = torch.randn(2, 4, 6)
    candidate_valid = torch.ones(2, 4, dtype=torch.bool)
    membership = torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
    cluster_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    source = torch.rand(2, 4)
    logits, selected = model(
        features,
        candidate_valid,
        membership,
        cluster_valid,
        source,
    )
    assert logits.shape == (2, 4, 4)
    assert len(set(selected[0].tolist())) == 4
    assert selected[1, :2].min() >= 0
    assert selected[1, 2:].tolist() == [-1, -1]


def test_coverage_pointer_loss_backpropagates() -> None:
    logits = torch.zeros((1, 2, 3), requires_grad=True)
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    active = torch.tensor([[1, 1]], dtype=torch.bool)
    loss = coverage_pointer_loss(logits, target, active)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_representative_listwise_loss_prefers_best_member() -> None:
    logits = torch.zeros((1, 3), requires_grad=True)
    targets = torch.tensor([[0.90, 0.50, 0.00]])
    membership = torch.tensor([[[1, 1, 0]]], dtype=torch.bool)
    cluster_valid = torch.tensor([[1]], dtype=torch.bool)
    loss, stats = representative_listwise_loss(
        logits,
        targets,
        membership,
        cluster_valid,
        temperature=0.05,
        positive_iou=0.30,
        min_spread=0.01,
    )
    loss.backward()
    assert stats["eligible_clusters"] == 1
    assert logits.grad is not None
    assert float(logits.grad[0, 0]) < 0.0
    assert float(logits.grad[0, 1]) > 0.0
    assert float(logits.grad[0, 2]) == 0.0


def test_source_preflight_uses_source_cluster_keepers() -> None:
    cache = {
        "official_iou": [
            torch.tensor(
                [
                    [0.9, 0.8, 0.0],
                    [0.0, 0.0, 0.9],
                ]
            )
        ]
    }
    hierarchy = {
        "records": [
            {
                "keepers": [0, 2],
                "source_selected_cluster_ids": [0, 1],
            }
        ]
    }
    result = evaluate_source_nms(cache, hierarchy)
    assert result["modes"]["source_nms"]["tp_050"] == 2
    assert result["modes"]["source_nms"]["tp_070"] == 2


def test_representative_statistics_count_only_hard_positive_clusters() -> None:
    hierarchy = {
        "membership": torch.tensor(
            [[
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
            ]],
            dtype=torch.bool,
        ),
        "cluster_valid": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "representative_targets": torch.tensor([[0.9, 0.5, 0.2, 0.1]]),
    }
    stats = representative_target_statistics(
        hierarchy,
        positive_iou=0.3,
        min_spread=0.01,
    )
    assert stats["valid_clusters"] == 2
    assert stats["multi_candidate_clusters"] == 2
    assert stats["positive_multi_candidate_clusters"] == 1
    assert stats["eligible_hard_clusters"] == 1
