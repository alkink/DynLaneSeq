from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v8_route_neighborhood_oracle import (
    NeighborhoodPolicy,
    _max_binary_matching_hits,
    _neighbors_for_anchor,
    _pairwise_curve_geometry,
    _slot_neighborhood_oracle_hits,
)


def test_pairwise_curve_geometry_finds_same_lane_neighbor():
    rows = 12
    curves = torch.stack(
        (
            torch.full((rows,), 100.0),
            torch.full((rows,), 108.0),
            torch.full((rows,), 500.0),
        )
    )
    ranges = torch.tensor([[0.0, 1.0]] * 3)
    distance, overlap = _pairwise_curve_geometry(curves, ranges)
    assert torch.allclose(distance[0], torch.tensor([0.0, 8.0, 400.0]))
    assert torch.allclose(overlap, torch.ones_like(overlap))


def test_local_neighborhood_never_drops_its_anchor():
    distance = torch.tensor(
        [
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 10.0],
            [20.0, 10.0, 0.0],
        ]
    )
    overlap = torch.ones_like(distance)
    ids = _neighbors_for_anchor(
        2,
        candidate_valid=torch.tensor([True, True, True]),
        mean_distance=distance,
        common_fraction=overlap,
        policy=NeighborhoodPolicy("tiny", 1, 5.0, 0.5),
    )
    assert ids.tolist() == [2]


def test_binary_matching_respects_prediction_capacity():
    edge = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ]
    )
    assert _max_binary_matching_hits(edge) == 3
    assert _max_binary_matching_hits(edge, max_predictions=2) == 2


def test_neighborhood_oracle_enforces_unique_proposal_identity():
    # Both slots can see proposal 0, but only slot 1 can also see proposal 1.
    # The exact oracle must allocate the two different proposals to recover
    # both GT lanes.
    quality = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )
    hits = _slot_neighborhood_oracle_hits(
        quality,
        [torch.tensor([0]), torch.tensor([0, 1])],
        0.5,
    )
    assert hits == 2
