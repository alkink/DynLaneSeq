from __future__ import annotations

import torch

from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.v30_joint_slot_field import FourSlotJointBeliefField


def _module() -> FourSlotJointBeliefField:
    return FourSlotJointBeliefField(
        feature_dim=8,
        slot_dim=8,
        num_rows=6,
        input_w=80,
        hidden_dim=8,
        route_residual_scale=1.0,
    )


def test_joint_field_zero_gate_preserves_route_and_field_is_live() -> None:
    torch.manual_seed(3407)
    module = _module()
    row_features = torch.randn(2, 6, 10, 8, requires_grad=True)
    slots = torch.randn(2, 4, 8, requires_grad=True)
    proposal_x = torch.rand(2, 5, 6) * 79.0
    proposal_range = torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(2, 5, 2)
    valid = torch.ones(2, 5, dtype=torch.bool)

    out = module(
        slot_states=slots,
        row_value_features=row_features,
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        candidate_valid=valid,
    )

    assert out["field_logits"].shape == (2, 4, 6, 10)
    assert out["candidate_score"].shape == (2, 4, 5)
    assert torch.equal(
        out["route_residual"], torch.zeros_like(out["route_residual"])
    )
    # Dense field supervision must reach both the image and slot paths on the
    # very first step, despite exact zero deployment residual.
    loss = out["field_logits"].square().mean()
    loss.backward()
    assert row_features.grad is not None and row_features.grad.abs().sum() > 0
    assert slots.grad is not None and slots.grad.abs().sum() > 0


def test_joint_field_candidate_coordinates_are_detached() -> None:
    torch.manual_seed(3408)
    module = _module()
    with torch.no_grad():
        module.route_gate.fill_(0.5)
    row_features = torch.randn(1, 6, 10, 8, requires_grad=True)
    slots = torch.randn(1, 4, 8, requires_grad=True)
    proposal_x = (torch.rand(1, 5, 6) * 79.0).requires_grad_(True)
    proposal_range = torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(1, 5, 2)
    valid = torch.ones(1, 5, dtype=torch.bool)

    out = module(
        slot_states=slots,
        row_value_features=row_features,
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        candidate_valid=valid,
    )
    weights = torch.arange(20, dtype=torch.float32).view(1, 4, 5)
    (out["route_residual"] * weights).sum().backward()
    assert proposal_x.grad is None
    assert row_features.grad is not None and row_features.grad.abs().sum() > 0


def test_joint_field_loss_uses_slot_owned_gt_rows() -> None:
    cfg = LossConfig(
        input_w=80,
        input_h=60,
        four_slot_line_width=12.0,
        four_slot_min_valid_rows=2,
        four_slot_geometry_match_all_slots=True,
        w_four_slot_joint_field=1.0,
    )
    criterion = S0Criterion(cfg)
    rows, x_bins = 6, 10
    reference = torch.tensor(
        [[[16.0] * rows, [56.0] * rows]], dtype=torch.float32
    )
    field_logits = torch.zeros(1, 2, rows, x_bins, requires_grad=True)
    outputs = {
        "selection_slot_joint_field_logits": field_logits,
        "selection_slot_input_reference_x_rows": reference,
        "selection_slot_input_range_norm": torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0]]]
        ),
        "selection_slot_geometry_valid": torch.ones(1, 2, dtype=torch.bool),
    }
    targets = [
        {
            "x_rows": torch.tensor(
                [[16.0] * rows, [56.0] * rows], dtype=torch.float32
            ),
            "valid_mask": torch.ones(2, rows, dtype=torch.bool),
        }
    ]
    loss = criterion.compute_four_slot_joint_field_loss(outputs, targets)
    assert torch.isfinite(loss["total"])
    assert float(loss["mean_matched"]) == 2.0
    loss["total"].backward()
    assert field_logits.grad is not None and field_logits.grad.abs().sum() > 0
