from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotLaneSelectionHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _head(*, detach: bool) -> FourSlotLaneSelectionHead:
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
        min_valid_rows=3,
        refinement_enabled=True,
        refinement_hidden_dim=32,
        refinement_straight_through_routing=True,
        refinement_detach_slot_states=True,
        refinement_structured_unique_routing=True,
        refinement_route_gradient_scale=0.10,
        refinement_reference_mode="hard_st",
        factorized_routing=True,
        geometry_detach_router_states=detach,
        range_refinement_enabled=True,
    )


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(3407)
    batch, candidates, rows, dim = 1, 6, 12, 16
    proposal_x = torch.linspace(10.0, 80.0, candidates).view(1, candidates, 1)
    proposal_x = proposal_x.expand(batch, candidates, rows).clone()
    outputs = {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, generator=generator
        ),
        "queries": torch.randn(batch, candidates, dim, generator=generator),
        "ownership_state": torch.randn(
            batch, candidates, dim, generator=generator
        ),
        "range_norm": torch.tensor([[[0.0, 0.9]] * candidates]),
        "pred_x_rows": proposal_x,
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, generator=generator
        ),
        "exist_logits": torch.randn(
            batch, candidates, 2, generator=generator
        ),
        "input_reference_x_rows": proposal_x.clone(),
    }
    row_features = torch.randn(batch, rows, 20, dim, generator=generator)
    return outputs, row_features


def _geometry_loss(result: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        result["selection_slot_pred_x_rows"].square().mean()
        + result["selection_slot_range_norm"].square().mean()
    )


def test_detach_removal_is_forward_exact_and_opens_only_router_state_edge() -> None:
    detached = _head(detach=True)
    attached = _head(detach=False)
    attached.load_state_dict(detached.state_dict())
    # Hard-ST geometry routing is intentionally a training-only backward
    # surrogate. Dropout is zero, so paired train-mode forwards stay exact.
    detached.train()
    attached.train()
    detached_inputs, detached_features = _inputs()
    attached_inputs, attached_features = _inputs()

    detached_result = detached(
        detached_inputs, row_value_features=detached_features
    )
    attached_result = attached(
        attached_inputs, row_value_features=attached_features
    )
    for name in (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_active_logits",
        "selection_slot_scores",
        "selection_slot_input_reference_x_rows",
        "selection_slot_input_range_norm",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    ):
        assert torch.equal(detached_result[name], attached_result[name]), name

    _geometry_loss(detached_result).backward()
    _geometry_loss(attached_result).backward()
    assert detached.input_projection.weight.grad is None
    assert detached.slot_decoder.layers[0].linear1.weight.grad is None
    assert attached.input_projection.weight.grad is not None
    assert float(attached.input_projection.weight.grad.abs().sum()) > 0.0
    assert attached.slot_decoder.layers[0].linear1.weight.grad is not None
    assert float(
        attached.slot_decoder.layers[0].linear1.weight.grad.abs().sum()
    ) > 0.0
    assert detached.slot_query.weight.grad is not None
    assert attached.slot_query.weight.grad is not None
    assert detached.active is not None and detached.active.weight.grad is None
    assert attached.active is not None and attached.active.weight.grad is None


def test_v8_1_configs_are_a_strict_paired_backward_fork() -> None:
    control = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_control_225k_to227k.yaml"
    )
    treatment = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_1_geometry_router_state_treatment_225k_to227k.yaml"
    )
    control_selection = control["model"]["structured_query"]["set_selection"]
    treatment_selection = treatment["model"]["structured_query"]["set_selection"]
    assert control_selection["four_slot_geometry_detach_router_states"] is True
    assert treatment_selection["four_slot_geometry_detach_router_states"] is False
    treatment_selection["four_slot_geometry_detach_router_states"] = True
    control["output_dir"] = treatment["output_dir"]
    control.pop("_config_path", None)
    treatment.pop("_config_path", None)
    assert control == treatment
