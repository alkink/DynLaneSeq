from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotGlobalVisualGeometry,
    FourSlotLaneSelectionHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal_outputs(*, batch: int = 1) -> dict[str, torch.Tensor]:
    candidates, rows, dim = 8, 12, 16
    return {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, requires_grad=True
        ),
        "queries": torch.randn(batch, candidates, dim, requires_grad=True),
        "ownership_state": torch.randn(
            batch, candidates, dim, requires_grad=True
        ),
        "range_norm": torch.tensor(
            [[[0.0, 0.9]] * candidates] * batch,
            requires_grad=True,
        ),
        "pred_x_rows": (
            torch.rand(batch, candidates, rows) * 99.0
        ).requires_grad_(),
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, requires_grad=True
        ),
        "exist_logits": torch.randn(
            batch, candidates, 2, requires_grad=True
        ),
        "input_reference_x_rows": (
            torch.rand(batch, candidates, rows) * 99.0
        ).requires_grad_(),
    }


def _head() -> FourSlotLaneSelectionHead:
    return FourSlotLaneSelectionHead(
        16,
        input_w=100,
        hidden_dim=32,
        num_slots=4,
        proposal_layers=1,
        slot_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        curve_samples=8,
        min_valid_rows=5,
        factorized_routing=True,
        active_prior_prob=0.8,
        refinement_enabled=False,
        slot_owned_geometry_enabled=False,
        global_visual_geometry_enabled=True,
        global_visual_geometry_hidden_dim=32,
        global_visual_geometry_num_heads=4,
        global_visual_geometry_ff_dim=64,
        global_visual_geometry_vertical_layers=1,
        global_visual_geometry_dropout=0.0,
        global_visual_geometry_delta_offsets_px=(
            -24.0,
            -12.0,
            0.0,
            12.0,
            24.0,
        ),
        global_visual_geometry_detach_slot_states=False,
        global_visual_geometry_slot_gradient_scale=1.0,
        global_visual_geometry_zero_init_delta_head=True,
    )


def test_v10_direct_module_has_no_proposal_identity_input():
    torch.manual_seed(3407)
    module = FourSlotGlobalVisualGeometry(
        16,
        input_w=100,
        slot_dim=32,
        num_slots=4,
        hidden_dim=32,
        num_heads=4,
        ff_dim=64,
        vertical_layers=1,
        dropout=0.0,
        delta_offsets_px=(-24.0, -12.0, 0.0, 12.0, 24.0),
    )
    slots = torch.randn(2, 4, 32)
    active = torch.tensor([[True, True, False, True]] * 2)
    p2 = torch.randn(2, 12, 20, 16)
    result = module(
        slot_states=slots,
        slot_active=active,
        row_value_features=p2,
    )
    assert result["selection_slot_pred_x_rows"].shape == (2, 4, 12)
    assert result["selection_slot_visual_attention"].shape == (2, 4, 12, 20)
    assert torch.allclose(
        result["selection_slot_visual_attention"].sum(dim=-1),
        torch.ones(2, 4, 12),
        atol=1.0e-6,
    )
    assert result["selection_slot_geometry_valid"].all()
    assert torch.equal(result["selection_slot_active"], active)


def test_v10_zero_delta_starts_from_global_visual_expectation():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    row_features = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        result = head(outputs, row_value_features=row_features)
    assert head.slot_refinement is None
    assert head.slot_owned_geometry is None
    assert head.global_visual_geometry is not None
    assert torch.equal(
        result["selection_slot_pred_x_rows"],
        result["selection_slot_input_reference_x_rows"],
    )
    probability = result["selection_slot_visual_attention"]
    x_pixels = torch.linspace(0.0, 99.0, 20)
    expected = torch.einsum("bsrx,x->bsr", probability, x_pixels)
    assert torch.allclose(
        result["selection_slot_input_reference_x_rows"],
        expected,
        atol=1.0e-5,
    )
    assert torch.all(
        result["selection_slot_range_norm"][..., 0]
        < result["selection_slot_range_norm"][..., 1]
    )


def test_v10_geometry_gradient_reaches_visual_and_slot_graph_only():
    torch.manual_seed(3407)
    head = _head()
    outputs = _proposal_outputs()
    row_features = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(outputs, row_value_features=row_features)
    loss = (
        result["selection_slot_pred_x_rows"].square().mean()
        + result["selection_slot_range_norm"].square().mean()
    )
    loss.backward()
    assert head.active is not None
    assert head.active.weight.grad is None
    assert head.global_visual_geometry is not None
    assert head.global_visual_geometry.feature_key.weight.grad is not None
    assert float(
        head.global_visual_geometry.feature_key.weight.grad.abs().sum()
    ) > 0.0
    assert any(
        parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in head.slot_decoder.parameters()
    )
    assert any(
        parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in head.proposal_encoder.parameters()
    )
    for value in (*outputs.values(), row_features):
        assert value.grad is None


def test_v10_full_width_attention_is_not_hard_route_geometry():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    outputs["pred_x_rows"].data.fill_(0.0)
    outputs["input_reference_x_rows"].data.fill_(0.0)
    row_features = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        result = head(outputs, row_value_features=row_features)
    # Every routed proposal lies at x=0.  A positive direct reference proves
    # the geometry forward did not gather or average proposal coordinates.
    assert float(
        result["selection_slot_input_reference_x_rows"].mean()
    ) > 5.0


def test_v10_config_enables_only_global_visual_geometry():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v10_global_visual_geometry_225k_to228k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_refinement_enabled"] is False
    assert selection["four_slot_slot_owned_geometry_enabled"] is False
    assert selection["four_slot_global_visual_geometry_enabled"] is True
    assert selection[
        "four_slot_global_visual_geometry_detach_slot_states"
    ] is False
    assert cfg["training"]["frozen_detector_eval"] is True
    assert cfg["loss"]["w_four_slot_geometry"] == 1.0
    assert cfg["loss"]["four_slot_geometry_match_all_slots"] is True
