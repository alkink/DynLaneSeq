from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.common import fixed_row_fractions
from dynlaneseq_eg.modeling.v16_candidate_groups import (
    V16CandidateGroupConfig,
    build_v16_anchor_candidate_groups,
)
from dynlaneseq_eg.tools.audit_geometry_proposal_clustering import _pairwise_geometry
from dynlaneseq_eg.tools.audit_v16_anchor_candidate_groups import (
    POLICIES,
    _fixed_assignment_selection,
    _json_floats,
    build_anchor_groups,
)
from dynlaneseq_eg.tools.summarize_v16_candidate_group_preflight import summarize


def _geometry(curves: torch.Tensor):
    masks = torch.ones_like(curves, dtype=torch.bool)
    y = fixed_row_fractions(curves.shape[-1], device=curves.device, dtype=curves.dtype)
    return _pairwise_geometry(curves, masks, y)


def test_adaptive_groups_are_variable_disjoint_and_do_not_pad_remote_candidates() -> None:
    rows = 16
    curves = torch.stack(
        (
            torch.full((rows,), 100.0),  # anchor 0
            torch.full((rows,), 122.0),  # same physical lane
            torch.full((rows,), 378.0),  # same physical lane as anchor 3
            torch.full((rows,), 400.0),  # anchor 3
            torch.full((rows,), 850.0),  # remote outer lane
        )
    )
    groups, diagnostic = build_anchor_groups(
        anchors=torch.tensor([0, 3]),
        candidate_valid=torch.ones(5, dtype=torch.bool),
        pair_geometry=_geometry(curves),
        input_w=1600,
        policy=POLICIES[0],
    )
    assert groups[0].tolist() == [0, 1]
    assert groups[1].tolist() == [3, 2]
    assert 4 not in {value for group in groups for value in group.tolist()}
    assert diagnostic["excluded_outside_corridor"] == 1
    assert sum(int(group.numel()) for group in groups) == 4


def test_overlap_upper_is_not_a_fixed_k_group() -> None:
    rows = 12
    curves = torch.stack(
        (
            torch.full((rows,), 100.0),
            torch.full((rows,), 130.0),
            torch.full((rows,), 160.0),
            torch.full((rows,), 500.0),
            torch.full((rows,), 540.0),
        )
    )
    groups, _ = build_anchor_groups(
        anchors=torch.tensor([0, 3]),
        candidate_valid=torch.ones(5, dtype=torch.bool),
        pair_geometry=_geometry(curves),
        input_w=1600,
        policy=POLICIES[1],
    )
    assert [int(group.numel()) for group in groups] == [3, 2]
    assert sorted(value for group in groups for value in group.tolist()) == list(range(5))
    assert _json_floats((float("inf"), 12.0)) == [None, 12.0]


def test_fixed_assignment_oracle_selects_one_coherent_candidate_per_group() -> None:
    quality = torch.tensor(
        (
            (0.55, 0.91, 0.05, 0.02),
            (0.01, 0.03, 0.60, 0.94),
        )
    )
    selected = _fixed_assignment_selection(
        quality,
        [torch.tensor([0, 1]), torch.tensor([2, 3])],
        torch.tensor([0, 2]),
        {0: 0, 1: 1},
    )
    assert selected.tolist() == [1, 3]


def test_tensor_group_builder_matches_audited_scalar_contract() -> None:
    rows = 21
    y = fixed_row_fractions(rows, device=torch.device("cpu"), dtype=torch.float32)
    curves = torch.stack(
        (
            80.0 + 18.0 * y.square(),
            92.0 + 22.0 * y.square(),
            119.0 + 26.0 * y.square(),
            350.0 - 35.0 * y + 20.0 * y.square(),
            375.0 - 28.0 * y + 18.0 * y.square(),
            720.0 + 10.0 * y,
        )
    )
    ranges = torch.tensor(
        (
            (0.00, 1.00),
            (0.10, 1.00),
            (0.25, 0.90),
            (0.00, 1.00),
            (0.20, 1.00),
            (0.00, 1.00),
        )
    )
    row_mask = (y[None] >= ranges[:, :1]) & (y[None] <= ranges[:, 1:])
    scalar_groups, scalar_diagnostic = build_anchor_groups(
        anchors=torch.tensor([0, 3]),
        candidate_valid=torch.ones(6, dtype=torch.bool),
        pair_geometry=_pairwise_geometry(curves, row_mask, y),
        input_w=1600,
        policy=POLICIES[0],
    )
    tensor = build_v16_anchor_candidate_groups(
        proposal_x_rows=curves.unsqueeze(0),
        proposal_range_norm=ranges.unsqueeze(0),
        candidate_valid=torch.ones(1, 6, dtype=torch.bool),
        anchor_indices=torch.tensor([[0, 3]]),
        anchor_active=torch.ones(1, 2, dtype=torch.bool),
        input_w=1600,
        config=V16CandidateGroupConfig(),
    )
    tensor_groups = [
        torch.nonzero(tensor["group_mask"][0, slot], as_tuple=False)
        .flatten()
        .tolist()
        for slot in range(2)
    ]
    assert [sorted(group.tolist()) for group in scalar_groups] == [
        sorted(group) for group in tensor_groups
    ]
    assert torch.allclose(
        tensor["corridor_px"][0],
        torch.tensor(scalar_diagnostic["corridor_px"]),
        atol=1.0e-4,
        rtol=0.0,
    )


def test_tensor_groups_use_only_active_anchors_and_retain_them() -> None:
    rows = 12
    curves = torch.stack(
        (
            torch.full((rows,), 100.0),
            torch.full((rows,), 120.0),
            torch.full((rows,), 500.0),
            torch.full((rows,), 520.0),
        )
    ).unsqueeze(0)
    result = build_v16_anchor_candidate_groups(
        proposal_x_rows=curves,
        proposal_range_norm=torch.tensor([[[0.0, 1.0]] * 4]),
        candidate_valid=torch.ones(1, 4, dtype=torch.bool),
        anchor_indices=torch.tensor([[0, 2, 3]]),
        anchor_active=torch.tensor([[True, True, False]]),
        input_w=1600,
    )
    mask = result["group_mask"][0]
    assert bool(mask[0, 0])
    assert bool(mask[1, 2])
    assert not bool(mask[2].any())
    assert bool((mask.sum(dim=0) <= 1).all())


def _report(*, support: float, near: float, gap50: float, gap75: float) -> dict:
    def metric(gap: float) -> dict:
        return {
            "tp": 10,
            "fp": 0,
            "fn": 0,
            "predictions": 10,
            "gt": 10,
            "f1": 1.0,
            "global_gap_closure": gap,
        }

    return {
        "scope": {
            "test_set_used": False,
            "training_performed": False,
            "proposal_coordinates_averaged": False,
            "fixed_k_or_padding_used": False,
        },
        "policies": {
            "adaptive_voronoi_060": {
                "anchor_retention": 1.0,
                "duplicate_memberships": 0,
                "target_support_any_coverage": support,
                "near_equivalent_coverage": near,
            }
        },
        "official_metrics": {
            "adaptive_voronoi_060/fixed_assignment_oracle/0.50": metric(gap50),
            "adaptive_voronoi_060/fixed_assignment_oracle/0.75": metric(gap75),
            "current_reference/0.50": metric(0.0),
            "current_reference/0.75": metric(0.0),
            "all32_same_count_oracle/0.50": metric(1.0),
            "all32_same_count_oracle/0.75": metric(1.0),
        },
        "metadata": {},
    }


def test_preflight_requires_both_domains() -> None:
    passed = _report(support=0.92, near=0.84, gap50=0.70, gap75=0.75)
    failed = _report(support=0.92, near=0.84, gap50=0.64, gap75=0.75)
    assert summarize(passed, passed)["passed"] is True
    assert summarize(passed, failed)["passed"] is False
