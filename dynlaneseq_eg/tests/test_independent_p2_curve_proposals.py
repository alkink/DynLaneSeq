from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.probe_independent_p2_curve_proposals import (
    IndependentP2CurveProposalProbe,
    canonicalize_feature_channels,
    compute_probe_loss,
    extract_frozen_feature_source,
    ordered_target_tensors,
)


def _target(x_values: list[float]) -> dict[str, torch.Tensor]:
    rows = 6
    lanes = []
    for value in x_values:
        lanes.append(torch.linspace(value - 5.0, value, rows))
    x_rows = torch.stack(lanes) if lanes else torch.zeros((0, rows))
    return {
        "x_rows": x_rows,
        "valid_mask": torch.ones_like(x_rows, dtype=torch.bool),
    }


def test_ordered_targets_sort_lanes_left_to_right_and_pad() -> None:
    x, valid, exist, dropped = ordered_target_tensors(
        [_target([80.0, 20.0, 50.0])],
        num_proposals=4,
        num_rows=6,
        device=torch.device("cpu"),
    )
    assert dropped == 0
    assert exist.tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert x[0, :3, -1].tolist() == [20.0, 50.0, 80.0]
    assert bool(valid[0, :3].all())
    assert not bool(valid[0, 3].any())


def test_ordered_targets_report_dropped_lanes() -> None:
    _x, _valid, exist, dropped = ordered_target_tensors(
        [_target([10.0, 20.0, 30.0, 40.0, 50.0])],
        num_proposals=4,
        num_rows=6,
        device=torch.device("cpu"),
    )
    assert dropped == 1
    assert int(exist.sum()) == 4


def test_independent_probe_shapes_and_loss_backpropagate() -> None:
    probe = IndependentP2CurveProposalProbe(
        in_dim=8,
        feature_dim=8,
        hidden_dim=16,
        num_rows=6,
        probe_width=10,
        num_proposals=4,
        x_bins=20,
        input_w=100,
    )
    p2 = torch.randn((2, 8, 3, 5))
    outputs = probe(p2)
    assert outputs["row_x_logits"].shape == (2, 4, 6, 20)
    assert outputs["row_visibility_logits"].shape == (2, 4, 6)
    assert outputs["existence_logits"].shape == (2, 4)
    loss, components, dropped = compute_probe_loss(
        probe,
        outputs,
        [_target([25.0, 75.0]), _target([20.0, 50.0, 80.0])],
    )
    assert dropped == 0
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert probe.row_x.weight.grad is not None


def test_zero_p2_still_has_valid_coordinate_control_output() -> None:
    probe = IndependentP2CurveProposalProbe(
        in_dim=8,
        feature_dim=8,
        hidden_dim=16,
        num_rows=6,
        probe_width=10,
        num_proposals=4,
        x_bins=20,
        input_w=100,
    )
    outputs = probe(torch.zeros((1, 8, 3, 5)))
    decoded = probe.decode(outputs, method="expected")
    assert decoded.shape == (1, 4, 6)
    assert torch.isfinite(decoded).all()


def test_feature_channel_canonicalization_is_lossless_and_equal_size() -> None:
    c2 = torch.randn((2, 4, 3, 5))
    padded = canonicalize_feature_channels(c2, output_channels=8)
    assert padded.shape == (2, 8, 3, 5)
    torch.testing.assert_close(padded[:, :4], c2)
    assert int(torch.count_nonzero(padded[:, 4:])) == 0
    same = canonicalize_feature_channels(padded, output_channels=8)
    assert same.data_ptr() == padded.data_ptr()


def test_feature_channel_canonicalization_refuses_lossy_truncation() -> None:
    with pytest.raises(ValueError, match="Cannot losslessly"):
        canonicalize_feature_channels(torch.randn((1, 9, 2, 2)), output_channels=8)


class _TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = lambda images: {
            "c2": images[:, :2],
            "c3": images[:, :3, ::2, ::2],
        }
        self.fpn = lambda features: torch.cat(
            (features["c2"], features["c2"]),
            dim=1,
        )
        self.proj = torch.nn.Conv2d(4, 4, 1, bias=False)


class _TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()


def test_extract_frozen_source_returns_equal_channels_and_base_p2() -> None:
    model = _TinyBase()
    images = torch.randn((2, 3, 6, 8))
    c2, p2 = extract_frozen_feature_source(
        model,
        images,
        feature_source="c2",
        canonical_channels=4,
    )
    assert c2.shape == (2, 4, 6, 8)
    assert p2.shape == (2, 4, 6, 8)
    torch.testing.assert_close(c2[:, :2], images[:, :2])
    assert int(torch.count_nonzero(c2[:, 2:])) == 0
