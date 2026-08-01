from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability import (
    _compare,
)
from dynlaneseq_eg.tools.probe_frozen_unified_selector import _selection_loss
from dynlaneseq_eg.tools.summarize_frozen_selector_diagnosis import summarize


def _unified_head() -> StructuredLaneQueryHead:
    return StructuredLaneQueryHead(
        dim=32,
        num_instances=4,
        num_rows=8,
        x_bins=16,
        input_w=64,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        dropout=0.0,
        evidence_x_bins=12,
        num_groups=1,
        lane_pooling="mean",
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
        set_selection={
            "enabled": True,
            "unified_score": True,
            "prior_prob": 0.10,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "curve_samples": 4,
            "detach_geometry_features": True,
            "use_curve_evidence": True,
        },
    )


def test_cached_selection_feature_path_matches_normal_forward() -> None:
    torch.manual_seed(101)
    head = _unified_head().eval()
    outputs = head(torch.randn(2, 32, 8, 12))
    selector = head.set_selection_head
    assert selector is not None
    features = selector.build_selection_features(outputs)
    cached_logits = selector.score_selection_features(features)
    normal_logits, normal_delta = selector(outputs)
    torch.testing.assert_close(cached_logits, normal_logits)
    assert float(normal_delta.abs().max()) == 0.0


def test_frozen_selector_loss_only_updates_logits() -> None:
    logits = torch.zeros((2, 4), requires_grad=True)
    targets = torch.tensor(
        [[0.9, 0.0, 0.7, 0.0], [0.0, 0.8, 0.0, 0.6]],
        dtype=torch.float32,
    )
    total, quality, ranking = _selection_loss(
        logits,
        targets,
        focal_beta=2.0,
        negative_weight=0.25,
        rank_weight=0.25,
        rank_margin=0.10,
    )
    total.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
    assert float(quality) > 0.0
    assert float(ranking) > 0.0


def test_ownership_comparison_detects_near_equivalent_query_churn() -> None:
    matrix_before = torch.tensor(
        [[0.80, 0.10, 0.79], [0.05, 0.75, 0.10]],
        dtype=torch.float32,
    )
    matrix_after = torch.tensor(
        [[0.78, 0.10, 0.79], [0.05, 0.76, 0.10]],
        dtype=torch.float32,
    )
    common = {
        "dataset_indices": [10],
    }
    before = {
        **common,
        "iteration": 2500,
        "records": [
            {
                "image_path": "image.jpg",
                "owners": [0, 1],
                "owner_qualities": [0.80, 0.75],
                "official_iou": matrix_before,
            }
        ],
    }
    after = {
        **common,
        "iteration": 5000,
        "records": [
            {
                "image_path": "image.jpg",
                "owners": [2, 1],
                "owner_qualities": [0.79, 0.76],
                "official_iou": matrix_after,
            }
        ],
    }
    row = _compare(
        before,
        after,
        stable_iou_floor=0.30,
        near_tie_margin=0.02,
    )
    assert row["owner_retention_recoverable"] == 0.5
    assert row["owner_change_recoverable"] == 0.5
    assert row["changed_owner_near_equivalent_fraction"] == 1.0


def test_summary_separates_moving_targets_from_scalar_capacity() -> None:
    ownership = {"gate": {"ownership_unstable": True}}
    selector = {
        "gate": {
            "stationary_frozen_selector_positive": True,
            "best_arm": "continued",
        }
    }
    result = summarize(ownership, selector)
    assert result["diagnosis"] == "moving_owner_targets_are_primary"
    assert result["confidence"] == "strong"

    selector["gate"]["stationary_frozen_selector_positive"] = False
    result = summarize(ownership, selector)
    assert (
        result["diagnosis"]
        == "owner_churn_exists_but_scalar_selector_remains_insufficient"
    )
