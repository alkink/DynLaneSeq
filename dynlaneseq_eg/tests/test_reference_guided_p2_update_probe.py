from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_reference_guided_p2_update import (
    ReferenceGuidedP2Probe,
    _sample_p2_profiles,
)


def test_reference_probe_is_exact_noop_at_initialization() -> None:
    probe = ReferenceGuidedP2Probe(
        state_dim=16,
        feature_dim=16,
        hidden_dim=16,
        num_rows=5,
        offsets_px=[-8, -4, 0, 4, 8],
        input_w=64,
        use_state=True,
        use_visual=True,
    )
    outputs = probe(
        torch.randn(3, 5, 16),
        torch.randn(3, 5, 5, 16),
        torch.full((3, 5), 31.5),
    )
    torch.testing.assert_close(
        outputs["residual"],
        torch.zeros_like(outputs["residual"]),
        atol=1e-6,
        rtol=0.0,
    )


def test_all_probe_modes_have_equal_parameter_count_and_initialization() -> None:
    modes = ((False, False), (True, False), (False, True), (True, True))
    probes = []
    for use_state, use_visual in modes:
        torch.manual_seed(9)
        probes.append(
            ReferenceGuidedP2Probe(
                state_dim=16,
                feature_dim=16,
                hidden_dim=16,
                num_rows=5,
                offsets_px=[-8, -4, 0, 4, 8],
                input_w=64,
                use_state=use_state,
                use_visual=use_visual,
            )
        )
    counts = [sum(parameter.numel() for parameter in probe.parameters()) for probe in probes]
    assert len(set(counts)) == 1
    reference = probes[0].state_dict()
    for probe in probes[1:]:
        for key, value in probe.state_dict().items():
            torch.testing.assert_close(value, reference[key])


def test_p2_profile_sampler_returns_lane_row_offset_channels() -> None:
    p2 = torch.randn(2, 8, 4, 6)
    anchor = torch.full((2, 3, 5), 31.5)
    profiles = _sample_p2_profiles(
        p2,
        anchor,
        torch.tensor([-8.0, 0.0, 8.0]),
        input_w=64,
        input_h=32,
    )
    assert profiles.shape == (2, 3, 5, 3, 8)
