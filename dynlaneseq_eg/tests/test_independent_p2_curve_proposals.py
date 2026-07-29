from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_independent_p2_curve_proposals import (
    IndependentP2CurveProposalProbe,
    compute_probe_loss,
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
