from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_frozen_residual_offset_probe import (
    ResidualOffsetProbe,
    _canonical_profile_input,
    _summarize_predictions,
)


def test_c2_and_p2_use_identical_probe_shape_and_parameter_count() -> None:
    c2 = torch.randn(12, 9, 64)
    p2 = torch.randn(12, 9, 256)
    c2_input = _canonical_profile_input(c2, center_index=4, common_channels=256)
    p2_input = _canonical_profile_input(p2, center_index=4, common_channels=256)
    assert c2_input.shape == p2_input.shape == (12, 9 * 256 * 2)

    torch.manual_seed(3)
    c2_probe = ResidualOffsetProbe(c2_input.shape[-1], 64, 9, 32.0)
    torch.manual_seed(3)
    p2_probe = ResidualOffsetProbe(p2_input.shape[-1], 64, 9, 32.0)
    assert sum(p.numel() for p in c2_probe.parameters()) == sum(
        p.numel() for p in p2_probe.parameters()
    )
    for c2_parameter, p2_parameter in zip(
        c2_probe.parameters(), p2_probe.parameters()
    ):
        torch.testing.assert_close(c2_parameter, p2_parameter)
    _logits, initial_residual = c2_probe(c2_input)
    torch.testing.assert_close(initial_residual, torch.zeros_like(initial_residual))


def test_residual_summary_rewards_a_correct_correction() -> None:
    offsets = torch.tensor([-8.0, 0.0, 8.0])
    labels = torch.tensor([0, 1, 2])
    residual = torch.tensor([-8.0, 0.0, 8.0])
    prediction = torch.tensor([0, 1, 2])
    mask = torch.ones(3, dtype=torch.bool)
    summary = _summarize_predictions(
        prediction,
        residual.clone(),
        labels,
        residual,
        offsets,
        mask,
    )
    assert summary["anchor_mae_px"] > 0.0
    assert summary["class_corrected_mae_px"] == 0.0
    assert summary["regression_corrected_mae_px"] == 0.0
    assert summary["regression_mae_gain_px"] > 0.0
