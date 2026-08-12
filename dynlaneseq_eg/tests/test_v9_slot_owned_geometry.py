from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotLaneSelectionHead,
    structured_unique_route_marginals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal_outputs(*, batch: int = 1):
    candidates, rows, dim = 8, 12, 16
    return {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, requires_grad=True
        ),
        "queries": torch.randn(
            batch, candidates, dim, requires_grad=True
        ),
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


def _head(*, detach_router_states: bool) -> FourSlotLaneSelectionHead:
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
        refinement_enabled=False,
        factorized_routing=True,
        geometry_detach_router_states=detach_router_states,
        range_refinement_enabled=True,
        range_delta_offsets_norm=(-0.2, -0.1, 0.0, 0.1, 0.2),
        slot_owned_geometry_enabled=True,
        slot_owned_geometry_hidden_dim=32,
        slot_owned_geometry_delta_offsets_px=(
            -24.0,
            -12.0,
            0.0,
            12.0,
            24.0,
        ),
        slot_owned_geometry_evidence_offsets_px=(
            -12.0,
            -6.0,
            0.0,
            6.0,
            12.0,
        ),
        slot_owned_geometry_vertical_layers=1,
        slot_owned_geometry_vertical_num_heads=4,
        slot_owned_geometry_vertical_ff_dim=64,
        slot_owned_geometry_vertical_dropout=0.0,
        slot_owned_geometry_zero_init_delta_heads=False,
    )


def test_v9_geometry_forward_uses_global_soft_memory_not_hard_id():
    torch.manual_seed(3407)
    head = _head(detach_router_states=False)
    outputs = _proposal_outputs()
    row_features = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(outputs, row_value_features=row_features)
    marginal = structured_unique_route_marginals(
        result["selection_slot_real_route_logits"],
        result["selection_slot_candidate_valid"],
    )
    expected = torch.einsum(
        "bsn,bnr->bsr",
        marginal,
        outputs["pred_x_rows"].detach().float(),
    )
    assert head.slot_refinement is None
    assert head.slot_owned_geometry is not None
    assert torch.allclose(
        result["selection_slot_input_reference_x_rows"],
        expected,
        atol=1.0e-5,
    )
    assert torch.allclose(
        result["selection_slot_owned_weight"],
        marginal,
        atol=1.0e-6,
    )
    hard = outputs["pred_x_rows"].detach().gather(
        1,
        result["selection_slot_geometry_route_indices"]
        .unsqueeze(-1)
        .expand(-1, -1, 12),
    )
    assert not torch.allclose(expected, hard, atol=1.0e-4)


def test_v9_treatment_geometry_reaches_slot_and_candidate_states_only():
    torch.manual_seed(3407)
    head = _head(detach_router_states=False)
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
    assert head.slot_query.weight.grad is not None
    assert float(head.slot_query.weight.grad.abs().sum()) > 0.0
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
    assert head.slot_owned_geometry is not None
    assert head.slot_owned_geometry.vertical_encoder is not None
    assert any(
        parameter.grad is not None
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in head.slot_owned_geometry.vertical_encoder.parameters()
    )
    for value in (*outputs.values(), row_features):
        assert value.grad is None


def test_v9_control_and_treatment_have_identical_forward_graph_values():
    torch.manual_seed(3407)
    control = _head(detach_router_states=True)
    treatment = _head(detach_router_states=False)
    treatment.load_state_dict(control.state_dict())
    outputs = _proposal_outputs()
    row_features = torch.randn(1, 12, 20, 16)
    control.eval()
    treatment.eval()
    with torch.no_grad():
        left = control(outputs, row_value_features=row_features)
        right = treatment(outputs, row_value_features=row_features)
    for name in (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_input_reference_x_rows",
        "selection_slot_input_range_norm",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_active",
    ):
        assert torch.equal(left[name], right[name])


def test_v9_configs_differ_only_by_geometry_router_state_edge():
    control = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_control_225k_to227k.yaml"
    )
    treatment = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v9_slot_owned_treatment_225k_to227k.yaml"
    )
    control_selection = control["model"]["structured_query"]["set_selection"]
    treatment_selection = treatment["model"]["structured_query"]["set_selection"]
    assert control_selection["four_slot_refinement_enabled"] is False
    assert control_selection["four_slot_slot_owned_geometry_enabled"] is True
    assert control_selection["four_slot_geometry_detach_router_states"] is True
    assert treatment_selection["four_slot_geometry_detach_router_states"] is False
    assert control["training"]["frozen_detector_eval"] is True
    assert control["loss"]["w_four_slot_selection"] == 1.0
    assert control["loss"]["w_four_slot_geometry"] == 1.0
