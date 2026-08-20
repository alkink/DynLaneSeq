from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_v27_pair_corridor_observability import (
    PairRibbonProbe,
    _clip_id,
    pair_ribbon_x,
    select_pair_specs,
)


def test_pair_ribbon_reverses_exactly_when_candidate_order_is_swapped() -> None:
    source = torch.tensor([[10.0, 11.0, 12.0], [30.0, 29.0, 28.0]])
    candidate = torch.tensor([[18.0, 19.0, 20.0], [22.0, 21.0, 20.0]])
    forward = pair_ribbon_x(source, candidate, points=25, margin_px=16.0)
    reverse = pair_ribbon_x(candidate, source, points=25, margin_px=16.0)
    torch.testing.assert_close(forward, reverse.flip(-1), atol=5.0e-6, rtol=0.0)


def test_clip_id_removes_only_the_frame_filename() -> None:
    assert (
        _clip_id("/driver_23_30frame/05160841_0471.MP4/00495.jpg")
        == "/driver_23_30frame/05160841_0471.MP4"
    )


def test_pair_population_keeps_same_owner_better_and_worse_members() -> None:
    rows = 10
    record = {
        "source_routes": torch.tensor([0]),
        "active": torch.tensor([True]),
        "source_quality": torch.tensor([[0.60]]),
        "source_valid": torch.tensor([True]),
        "refined_quality": torch.tensor([[[0.60, 0.80, 0.20]]]),
        "refined_valid": torch.tensor([[True, True, True]]),
    }
    source_x = torch.full((1, rows), 100.0)
    source_range = torch.tensor([[0.0, 1.0]])
    refined_x = torch.stack(
        (
            torch.full((rows,), 100.0),
            torch.full((rows,), 104.0),
            torch.full((rows,), 108.0),
        )
    ).unsqueeze(0)
    refined_range = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
    pairs = select_pair_specs(
        record,
        source_x,
        source_range,
        refined_x,
        refined_range,
        max_pairs_per_class_slot=3,
        max_mean_abs_dx=64.0,
        min_quality_gap=0.03,
    )
    assert {(row.candidate, row.label) for row in pairs} == {(1, 1), (2, 0)}


def test_probe_score_is_antisymmetric_under_pair_order_swap() -> None:
    torch.manual_seed(3)
    probe = PairRibbonProbe(channels=8, geometry_dim=5)
    profile = torch.randn(4, 8, 6, 9)
    geometry = torch.randn(4, 5)
    reverse_geometry = torch.randn(4, 5)
    forward = probe(profile, geometry, reverse_geometry)
    reverse = probe(profile.flip(-1), reverse_geometry, geometry)
    torch.testing.assert_close(forward, -reverse, atol=1.0e-6, rtol=1.0e-6)
