from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v12_visual_geometry_counterfactual import (
    _curve_distance,
    _gather_proposals,
    _hard_unique_indices,
    _soft_proposals,
)


def test_visual_distance_hard_route_is_global_and_unique():
    rows = 6
    proposal_x = torch.tensor(
        [[10.0, 30.0, 70.0, 90.0]]
    ).view(1, 4, 1).expand(-1, -1, rows)
    proposal_range = torch.tensor([[[0.0, 1.0]] * 4])
    candidate_valid = torch.ones(1, 4, dtype=torch.bool)
    visual_x = torch.tensor([[12.0, 28.0, 72.0, 88.0]]).view(
        1, 4, 1
    ).expand(-1, -1, rows)

    distance = _curve_distance(
        visual_x, proposal_x, proposal_range, candidate_valid
    )
    indices = _hard_unique_indices(distance, candidate_valid)
    assert torch.equal(indices, torch.tensor([[0, 1, 2, 3]]))
    assert len(set(indices.flatten().tolist())) == 4
    selected_x, selected_range = _gather_proposals(
        indices, proposal_x, proposal_range
    )
    assert torch.equal(selected_x, proposal_x)
    assert torch.equal(selected_range, proposal_range)


def test_soft_proposal_geometry_preserves_probability_expectation():
    proposal_x = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    proposal_range = torch.tensor([[[0.1, 0.7], [0.3, 0.9]]])
    probability = torch.tensor([[[0.25, 0.75]]])
    x, lane_range = _soft_proposals(
        probability, proposal_x, proposal_range
    )
    assert torch.allclose(x, torch.tensor([[[25.0, 35.0]]]))
    assert torch.allclose(lane_range, torch.tensor([[[0.25, 0.85]]]))
