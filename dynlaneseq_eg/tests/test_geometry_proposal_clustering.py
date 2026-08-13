from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_geometry_proposal_clustering import (
    POLICIES,
    _cluster_selection_ids,
    _pair_geometry,
    _pairwise_geometry,
    _policy_distance,
    _slot_cluster_mass_selection,
    build_cluster_prototypes,
    complete_link_clusters,
)


def _policy(name: str):
    return next(value for value in POLICIES if value.name == name)


def _curves(*values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.stack(values).float()
    masks = torch.ones_like(x, dtype=torch.bool)
    y = torch.arange(x.shape[-1], dtype=torch.float32) / float(x.shape[-1])
    return x, masks, y


def test_bottom_divergence_prevents_horizon_convergence_merge() -> None:
    rows = 40
    y = torch.arange(rows, dtype=torch.float32) / float(rows)
    left = torch.full((rows,), 500.0)
    # These lanes nearly coincide near the horizon, and their mean gap is
    # below 48 px, but the bottom gap approaches 100 px.
    right = left + 105.0 * y.pow(3.0)
    geometry = _pair_geometry(left, torch.ones(rows, dtype=torch.bool), right, torch.ones(rows, dtype=torch.bool), y)
    flat_distance, _ = _policy_distance(geometry, _policy("flat_complete_48"), 1600)
    perspective_distance, reason = _policy_distance(
        geometry, _policy("perspective_balanced_48"), 1600
    )
    assert geometry.unweighted_mean_px < 48.0
    assert flat_distance <= 1.0
    assert perspective_distance > 1.0
    assert reason in {
        "weighted_q90",
        "lower_q90",
        "bottom_endpoint",
        "lower_divergence",
        "polynomial_q90",
    }


def test_near_duplicate_lane_merges_under_perspective_contract() -> None:
    rows = 40
    y = torch.arange(rows, dtype=torch.float32) / float(rows)
    base = 420.0 + 180.0 * y + 25.0 * y.square()
    duplicate = base + 11.0
    geometry = _pair_geometry(
        base,
        torch.ones(rows, dtype=torch.bool),
        duplicate,
        torch.ones(rows, dtype=torch.bool),
        y,
    )
    distance, reason = _policy_distance(
        geometry, _policy("perspective_balanced_48"), 1600
    )
    assert distance <= 1.0
    assert reason == "compatible"


def test_upper_only_overlap_is_not_treated_as_same_lane() -> None:
    rows = 40
    y = torch.arange(rows, dtype=torch.float32) / float(rows)
    first = torch.full((rows,), 500.0)
    second = first + 5.0
    upper_only = y < 0.50
    geometry = _pair_geometry(first, upper_only, second, upper_only, y)
    distance, reason = _policy_distance(
        geometry, _policy("perspective_balanced_48"), 1600
    )
    assert distance == float("inf")
    assert reason == "no_reliable_lower_evidence"


def test_large_visible_range_start_gap_prevents_merge() -> None:
    rows = 40
    y = torch.arange(rows, dtype=torch.float32) / float(rows)
    first = torch.full((rows,), 500.0)
    second = first + 4.0
    first_mask = torch.ones(rows, dtype=torch.bool)
    second_mask = y >= 0.30
    geometry = _pair_geometry(first, first_mask, second, second_mask, y)
    distance, reason = _policy_distance(
        geometry, _policy("perspective_balanced_48"), 1600
    )
    assert geometry.range_start_gap >= 0.29
    assert distance > 1.0
    assert reason == "range_start"


def test_complete_link_does_not_chain_two_lanes_through_middle_curve() -> None:
    rows = 40
    first = torch.full((rows,), 300.0)
    middle = torch.full((rows,), 330.0)
    last = torch.full((rows,), 360.0)
    x, masks, y = _curves(first, middle, last)
    geometry = _pairwise_geometry(x, masks, y)
    clusters, _decisions = complete_link_clusters(
        [0, 1, 2], geometry, _policy("perspective_balanced_48"), 1600
    )
    assert clusters == [[0, 1], [2]]


def test_cluster_prototypes_include_robust_median_and_arithmetic_mean() -> None:
    rows = 40
    stage = {
        "pred_x_rows": torch.stack(
            (
                torch.full((rows,), 100.0),
                torch.full((rows,), 102.0),
                torch.full((rows,), 130.0),
            )
        ),
        "range_norm": torch.tensor([[0.0, 1.0]] * 3),
        "exist_logits": torch.tensor([[5.0, -5.0], [4.0, -4.0], [3.0, -3.0]]),
    }
    decisions = {
        (0, 1): (0.1, "compatible"),
        (0, 2): (0.2, "compatible"),
        (1, 2): (0.15, "compatible"),
    }
    output = build_cluster_prototypes(
        stage,
        [[0, 1, 2]],
        decisions,
        input_h=640,
        input_w=1600,
        min_valid_rows=5,
    )
    assert torch.allclose(output["median"]["pred_x_rows"], torch.full((1, rows), 102.0))
    assert torch.allclose(
        output["mean"]["pred_x_rows"],
        torch.full((1, rows), (100.0 + 102.0 + 130.0) / 3.0),
    )


def test_routed_cluster_selection_deduplicates_then_fills_to_count() -> None:
    selected = _cluster_selection_ids(
        [[0, 2], [1], [3], [4]],
        torch.tensor([0.8, 0.7, 0.9, 0.1]),
        [0, 2, 1],
        3,
    )
    assert selected["routed_consensus"] == [0, 1, 2]
    assert selected["score_topk"] == [2, 0, 1]


def test_slot_cluster_mass_recovers_probability_split_across_duplicates() -> None:
    # Slot 0's best single proposal is candidate 2, but candidates 0 and 1
    # jointly carry more probability and form one geometry cluster. Slot 1
    # strongly owns candidate 2. Cluster-level uniqueness therefore recovers
    # the intended two physical groups without GT.
    stage = {
        "selection_slot_indices": torch.tensor([2, 0, -1, -1]),
        "selection_slot_active": torch.tensor([True, True, False, False]),
        "selection_slot_official_candidate_valid": torch.tensor(
            [True, True, False, False]
        ),
        "selection_slot_logits": torch.tensor(
            [
                [1.9, 1.9, 2.2, -4.0, -8.0],
                [0.2, 0.1, 3.0, -4.0, -8.0],
                [-2.0, -2.0, -2.0, -2.0, 4.0],
                [-2.0, -2.0, -2.0, -2.0, 4.0],
            ]
        ),
    }
    result = _slot_cluster_mass_selection(
        [[0, 1], [2], [3]],
        stage,
        torch.tensor([True, True, True, True]),
    )
    assert result["available"] is True
    assert result["slot_ids"] == [0, 1]
    assert result["cluster_ids"] == [0, 1]
    assert len(result["selected_masses"]) == 2
