from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_curve_aligned_visual_verification import (
    CurveAlignedVisualVerifier,
    sample_curve_aligned_profiles,
    sampled_range_weights,
    verification_verdict,
)


def test_curve_aligned_sampler_hits_expected_horizontal_locations() -> None:
    feature = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4).expand(1, 1, 3, 4)
    pred_x = torch.tensor([[[0.0, 3.5, 7.0]]])
    sampled = sample_curve_aligned_profiles(
        feature,
        pred_x,
        row_indices=torch.tensor([0, 1, 2]),
        offsets_px=torch.tensor([0.0]),
        input_h=3,
        input_w=8,
    )
    assert sampled.shape == (1, 1, 3, 1, 1)
    torch.testing.assert_close(
        sampled.flatten(),
        torch.tensor([0.0, 1.5, 3.0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_range_weights_focus_on_predicted_visible_interval() -> None:
    weights = sampled_range_weights(
        torch.tensor([[[0.25, 0.75]]]),
        row_indices=torch.tensor([0, 1, 2, 3, 4]),
        total_rows=5,
        temperature=0.02,
    )
    assert weights.shape == (1, 1, 5)
    assert float(weights[0, 0, 2]) > 0.99
    assert float(weights[0, 0, 0]) < 1e-4
    assert float(weights[0, 0, 4]) < 1e-4


def test_visual_verifier_is_candidate_permutation_equivariant_and_zero_init() -> None:
    torch.manual_seed(11)
    verifier = CurveAlignedVisualVerifier(
        base_dim=14,
        feature_channels=12,
        row_state_dim=10,
        curve_samples=5,
        offsets=3,
        visual_dim=16,
        hidden_dim=24,
        row_layers=1,
        row_heads=4,
        set_layers=1,
        set_heads=4,
        set_ff_dim=48,
        dropout=0.0,
    ).eval()
    base = torch.randn(2, 4, 14)
    profiles = torch.randn(2, 4, 5, 3, 12)
    states = torch.randn(2, 4, 5, 10)
    weights = torch.rand(2, 4, 5).clamp_min(0.05)
    output = verifier(base, profiles, states, weights)
    assert torch.equal(output, torch.zeros_like(output))

    with torch.no_grad():
        verifier.output.weight.normal_(std=0.1)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = verifier(base, profiles, states, weights)[:, permutation]
    actual = verifier(
        base[:, permutation],
        profiles[:, permutation],
        states[:, permutation],
        weights[:, permutation],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_verdict_requires_visual_gain_beyond_geometry_control() -> None:
    def row(f1_050: float, f1_070: float) -> dict[str, float]:
        return {"f1_050": f1_050, "f1_070": f1_070}

    evaluation = {
        "modes": {
            "nms_top4": {
                "current_exist_quality": row(0.70, 0.50),
                "base_residual_set": row(0.705, 0.502),
                "curve_aligned_visual": row(0.72, 0.51),
            }
        }
    }
    verdict = verification_verdict(
        evaluation,
        min_gain_050_points=1.0,
        min_gain_070_points=0.5,
        min_visual_over_control_points=0.25,
    )
    assert verdict["visual_positive"] is True
    assert verdict["evidence_specific_positive"] is True
    assert verdict["recommendation"] == "curve_aligned_visual_evidence_positive_integrate_jointly"
