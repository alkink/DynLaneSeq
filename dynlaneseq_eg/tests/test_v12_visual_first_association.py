from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.four_slot_selection import FourSlotLaneSelectionHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal_outputs(*, batch: int = 1) -> dict[str, torch.Tensor]:
    candidates, rows, dim = 8, 12, 16
    base = torch.linspace(12.0, 88.0, candidates).view(1, candidates, 1)
    slope = torch.linspace(-3.0, 3.0, rows).view(1, 1, rows)
    return {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, requires_grad=True
        ),
        "queries": torch.randn(batch, candidates, dim, requires_grad=True),
        "ownership_state": torch.randn(
            batch, candidates, dim, requires_grad=True
        ),
        "range_norm": torch.tensor(
            [[[0.0, 0.95]] * candidates] * batch,
            requires_grad=True,
        ),
        "pred_x_rows": (base + slope)
        .expand(batch, -1, -1)
        .clone()
        .requires_grad_(),
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, requires_grad=True
        ),
        "exist_logits": torch.randn(batch, candidates, 2, requires_grad=True),
        "input_reference_x_rows": (base + slope)
        .expand(batch, -1, -1)
        .clone()
        .requires_grad_(),
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
        refinement_enabled=True,
        refinement_hidden_dim=32,
        refinement_delta_offsets_px=(-20.0, -10.0, 0.0, 10.0, 20.0),
        refinement_straight_through_routing=True,
        refinement_detach_slot_states=True,
        refinement_structured_unique_routing=True,
        refinement_route_gradient_scale=0.1,
        range_refinement_enabled=True,
        range_delta_offsets_norm=(-0.2, -0.1, 0.0, 0.1, 0.2),
        unified_slot_decoder_enabled=False,
        visual_first_association_enabled=True,
        visual_first_association_hidden_dim=32,
        visual_first_association_num_heads=4,
        visual_first_association_ff_dim=64,
        visual_first_association_vertical_layers=1,
        visual_first_association_dropout=0.0,
        visual_first_association_sinkhorn_iterations=64,
    )


def _targets() -> list[dict[str, torch.Tensor]]:
    rows = 12
    x = torch.stack(
        (
            torch.linspace(18.0, 24.0, rows),
            torch.linspace(45.0, 48.0, rows),
            torch.linspace(76.0, 72.0, rows),
        )
    )
    return [
        {
            "x_rows": x,
            "valid_mask": torch.ones_like(x, dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 49.0]] * 3),
        }
    ]


def test_v12_stage_a_is_exact_v7_deployment_sidecar():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    module = head.visual_first_association
    assert module is not None
    head.visual_first_association = None
    with torch.no_grad():
        source = head(outputs, row_value_features=p2)
    head.visual_first_association = module
    with torch.no_grad():
        treatment = head(outputs, row_value_features=p2)

    for name in (
        "selection_slot_real_route_logits",
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    ):
        assert torch.equal(source[name], treatment[name])
    assert torch.equal(
        treatment["selection_slot_v12_anchor_x_rows"],
        source["selection_slot_pred_x_rows"],
    )
    assert torch.equal(
        treatment["selection_slot_v12_anchor_range_norm"],
        source["selection_slot_range_norm"],
    )
    attention = treatment["selection_slot_v12_proposal_attention"]
    assert attention.shape == (1, 4, 8)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 4), atol=1e-6)
    assert bool((attention.sum(dim=1) <= 1.0 + 1.0e-5).all())


def test_v12_p2_is_before_and_causal_for_proposal_association():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        correct = head(outputs, row_value_features=p2)
        zero = head(outputs, row_value_features=torch.zeros_like(p2))
        wrong = head(outputs, row_value_features=p2.flip(2))
    correct_attention = correct["selection_slot_v12_proposal_attention"]
    assert not torch.equal(
        correct_attention,
        zero["selection_slot_v12_proposal_attention"],
    )
    assert not torch.equal(
        correct_attention,
        wrong["selection_slot_v12_proposal_attention"],
    )
    assert not torch.equal(
        correct["selection_slot_v12_first_visual_x_rows"],
        correct["selection_slot_v12_visual_x_rows"],
    )
    # No V7 route-logit tensor is accepted by the V12 module; its only V7
    # inputs are the slot state and final geometric anchor.
    assert "legacy_route_logits" not in module_forward_parameters(
        head.visual_first_association
    )


def module_forward_parameters(module: torch.nn.Module | None) -> set[str]:
    import inspect

    assert module is not None
    return set(inspect.signature(module.forward).parameters)


def test_v12_one_loss_reaches_visual_set_and_proposal_state_only():
    torch.manual_seed(3407)
    head = _head()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(outputs, row_value_features=p2)
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            four_slot_cluster_min=0.0,
            four_slot_cluster_delta=0.10,
            four_slot_cluster_temperature=0.03,
            w_four_slot_visual_first=1.0,
        )
    )
    merged = {**outputs, **result}
    losses = criterion.compute_four_slot_visual_first_loss(merged, _targets())
    losses["total"].backward()
    module = head.visual_first_association
    assert module is not None
    for parameter in (
        module.feature_key.weight,
        module.first_visual_query.weight,
        module.second_visual_query.weight,
        module.proposal_key.weight,
        module.proposal_query.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.cross_slot_attention.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.vertical_encoder.parameters()
    )
    assert all(value.grad is None for value in outputs.values())
    assert p2.grad is None
    assert head.active is not None and head.active.weight.grad is None
    assert all(
        parameter.grad is None
        for parameter in head.slot_refinement.parameters()
    )
    assert float(losses["mean_matched"]) == 3.0
    assert torch.isfinite(losses["total"])


def test_v12_config_has_one_association_objective_and_frozen_deployment():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v12_visual_first_association_stage_a_225k_to227k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_refinement_enabled"] is True
    assert selection["four_slot_visual_first_association_enabled"] is True
    assert selection["four_slot_unified_slot_decoder_enabled"] is False
    assert cfg["loss"]["w_four_slot_visual_first"] == 1.0
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and value != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    assert nonzero_objectives == {"w_four_slot_visual_first": 1.0}
    assert cfg["training"]["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.visual_first_association"
    ]
    assert cfg["training"]["max_iters"] == 2000
