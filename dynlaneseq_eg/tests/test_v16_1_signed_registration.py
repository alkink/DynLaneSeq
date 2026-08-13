from __future__ import annotations

import torch

from dynlaneseq_eg.evaluation.signed_lane_registration import (
    _sample_row_values,
    gather_whole_proposals,
    signed_curve_registration,
)


def _logits_for_curve(
    curve_x: torch.Tensor,
    *,
    input_w: int,
    x_bins: int,
    sigma_bins: float = 0.8,
) -> torch.Tensor:
    feature_x = curve_x * float(x_bins - 1) / float(input_w - 1)
    bins = torch.arange(x_bins, dtype=torch.float32)
    return -0.5 * (
        (bins.view(1, 1, 1, -1) - feature_x[..., None]) / sigma_bins
    ).square()


def _register(
    logits: torch.Tensor,
    proposal_x: torch.Tensor,
    *,
    input_w: int,
) -> object:
    batch, slots, rows, _bins = logits.shape
    candidates = int(proposal_x.shape[1])
    return signed_curve_registration(
        visual_logits=logits,
        proposal_x_rows=proposal_x,
        proposal_range_norm=torch.tensor(
            [[[0.0, 1.0]] * candidates], dtype=torch.float32
        ).expand(batch, -1, -1),
        anchor_range_norm=torch.tensor(
            [[[0.0, 1.0]] * slots], dtype=torch.float32
        ).expand(batch, -1, -1),
        proposal_visible=torch.ones(
            batch, candidates, rows, dtype=torch.bool
        ),
        group_mask=torch.ones(batch, slots, candidates, dtype=torch.bool),
        writer_valid=torch.ones(batch, slots, dtype=torch.bool),
        input_w=input_w,
    )


def test_bilinear_sampling_preserves_exact_bins_and_midpoints() -> None:
    values = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    proposal_x = torch.tensor([[[0.0], [12.5], [25.0], [100.0]]])
    sampled = _sample_row_values(values, proposal_x, input_w=101)
    assert sampled.shape == (1, 1, 4, 1)
    assert torch.allclose(
        sampled.flatten(), torch.tensor([0.0, 0.5, 1.0, 4.0])
    )


def test_one_sided_proposal_cloud_selects_curve_on_visual_ridge() -> None:
    input_w = 161
    rows = 12
    visual_curve = torch.full((1, 1, rows), 40.0)
    logits = _logits_for_curve(visual_curve, input_w=input_w, x_bins=17)
    # The correct member is at the edge of a one-sided proposal cloud.
    proposal_x = torch.stack(
        (
            torch.full((rows,), 40.0),
            torch.full((rows,), 60.0),
            torch.full((rows,), 80.0),
        )
    ).unsqueeze(0)
    result = _register(logits, proposal_x, input_w=input_w)
    assert int(result.selected_indices[0, 0]) == 0
    assert float(result.mean_signed_displacement_px[0, 0, 0]) == 0.0
    assert float(result.mean_signed_displacement_px[0, 0, 1]) > 0.0


def test_lower_tail_prevents_local_large_error_from_being_hidden() -> None:
    input_w = 161
    rows = 20
    visual_curve = torch.full((1, 1, rows), 40.0)
    logits = _logits_for_curve(visual_curve, input_w=input_w, x_bins=17)
    consistently_close = torch.full((rows,), 45.0)
    mostly_exact_but_bad_bottom = torch.full((rows,), 40.0)
    mostly_exact_but_bad_bottom[-3:] = 100.0
    proposal_x = torch.stack(
        (consistently_close, mostly_exact_but_bad_bottom)
    ).unsqueeze(0)
    result = _register(logits, proposal_x, input_w=input_w)
    assert int(result.selected_indices[0, 0]) == 0
    assert (
        float(result.p90_absolute_displacement_px[0, 0, 1])
        > float(result.p90_absolute_displacement_px[0, 0, 0])
    )


def test_group_mask_and_writer_mask_control_eligibility() -> None:
    input_w = 161
    rows = 8
    visual_curve = torch.full((1, 1, rows), 80.0)
    logits = _logits_for_curve(visual_curve, input_w=input_w, x_bins=17)
    proposal_x = torch.stack(
        (torch.full((rows,), 40.0), torch.full((rows,), 80.0))
    ).unsqueeze(0)
    result = signed_curve_registration(
        visual_logits=logits,
        proposal_x_rows=proposal_x,
        proposal_range_norm=torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]),
        anchor_range_norm=torch.tensor([[[0.0, 1.0]]]),
        proposal_visible=torch.ones(1, 2, rows, dtype=torch.bool),
        group_mask=torch.tensor([[[True, False]]]),
        writer_valid=torch.tensor([[True]]),
        input_w=input_w,
    )
    assert int(result.selected_indices[0, 0]) == 0
    assert float(result.scores[0, 0, 1]) == -1.0e4


def test_whole_proposal_gather_never_mixes_rows_or_range() -> None:
    curves = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [10.0, 20.0, 30.0],
                [100.0, 200.0, 300.0],
            ]
        ]
    )
    ranges = torch.tensor([[[0.0, 0.5], [0.1, 0.7], [0.2, 0.9]]])
    ids = torch.tensor([[2, 0]])
    selected_curves = gather_whole_proposals(curves, ids)
    selected_ranges = gather_whole_proposals(ranges, ids)
    assert torch.equal(selected_curves[0, 0], curves[0, 2])
    assert torch.equal(selected_curves[0, 1], curves[0, 0])
    assert torch.equal(selected_ranges[0, 0], ranges[0, 2])
    assert torch.equal(selected_ranges[0, 1], ranges[0, 0])

