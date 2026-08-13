from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.four_slot_selection import FourSlotLaneSelectionHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _proposal_outputs(*, batch: int = 1) -> dict[str, torch.Tensor]:
    candidates, rows, dim = 8, 12, 16
    y = torch.linspace(0.0, 1.0, rows).view(1, 1, rows)
    centers = torch.tensor([18.0, 20.0, 42.0, 45.0, 70.0, 73.0, 88.0, 90.0])
    base = centers.view(1, candidates, 1)
    curves = base + (y - 0.5) * torch.linspace(-4.0, 4.0, candidates).view(
        1, candidates, 1
    )
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
        "pred_x_rows": curves.expand(batch, -1, -1).clone().requires_grad_(),
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, requires_grad=True
        ),
        "exist_logits": torch.randn(batch, candidates, 2, requires_grad=True),
        "input_reference_x_rows": curves.expand(batch, -1, -1)
        .clone()
        .requires_grad_(),
    }


def _targets() -> list[dict[str, torch.Tensor]]:
    rows = 12
    x = torch.stack(
        (
            torch.linspace(18.0, 22.0, rows),
            torch.linspace(43.0, 46.0, rows),
            torch.linspace(72.0, 70.0, rows),
        )
    )
    return [
        {
            "x_rows": x,
            "valid_mask": torch.ones_like(x, dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 49.0]] * 3),
        }
    ]


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
        min_valid_rows=3,
        factorized_routing=True,
        active_prior_prob=0.99,
        refinement_enabled=True,
        refinement_hidden_dim=32,
        refinement_delta_offsets_px=(-20.0, -10.0, 0.0, 10.0, 20.0),
        refinement_straight_through_routing=True,
        refinement_detach_slot_states=True,
        refinement_structured_unique_routing=True,
        refinement_route_gradient_scale=0.1,
        range_refinement_enabled=True,
        range_delta_offsets_norm=(-0.2, -0.1, 0.0, 0.1, 0.2),
        bottom_aware_relational_geometry_enabled=True,
        bottom_aware_relational_geometry_hidden_dim=32,
        bottom_aware_relational_geometry_num_heads=4,
        bottom_aware_relational_geometry_ff_dim=64,
        bottom_aware_relational_geometry_visual_vertical_layers=1,
        bottom_aware_relational_geometry_fusion_vertical_layers=1,
        bottom_aware_relational_geometry_slot_interaction_layers=1,
        bottom_aware_relational_geometry_dropout=0.0,
        bottom_aware_relational_geometry_delta_offsets_px=(
            -100.0,
            -50.0,
            0.0,
            50.0,
            100.0,
        ),
        bottom_aware_relational_geometry_range_offsets_norm=(
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ),
    )


def test_v15_zero_step_is_exact_v7_public_geometry() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    module = head.bottom_aware_relational_geometry
    assert module is not None
    head.bottom_aware_relational_geometry = None
    with torch.no_grad():
        source = head(proposals, row_value_features=p2)
    head.bottom_aware_relational_geometry = module
    with torch.no_grad():
        treatment = head(proposals, row_value_features=p2)
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
        assert torch.equal(source[name], treatment[name]), name
    assert float(treatment["selection_slot_v15_delta_x_rows"].abs().max()) == 0.0


def test_v15_graph_is_soft_row_stochastic_and_bottom_aware() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    with torch.no_grad():
        result = head(
            proposals,
            row_value_features=torch.randn(1, 12, 20, 16),
        )
    graph = result["selection_slot_v15_graph_attention"]
    valid = result["selection_slot_v15_graph_pair_valid"]
    assert graph.shape == (1, 8, 8)
    assert torch.allclose(graph.sum(dim=-1), torch.ones(1, 8), atol=1.0e-6)
    invalid_values = graph.masked_select(~valid)
    assert invalid_values.numel() == 0 or float(invalid_values.abs().max()) == 0.0
    # Proposal 0 is geometrically much closer to proposal 1 than proposal 6.
    assert float(graph[0, 0, 1]) > float(graph[0, 0, 6])
    edge = result["selection_slot_v15_graph_edge_features"]
    # bottom-endpoint feature index 7 preserves the raw relational evidence.
    assert float(edge[0, 0, 1, 7]) < float(edge[0, 0, 6, 7])


def test_v15_graph_policies_preserve_nodes_without_prototypes() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    module = head.bottom_aware_relational_geometry
    assert module is not None
    with torch.no_grad():
        source = head(proposals, row_value_features=p2)
        identity = module(
            slot_states=torch.randn(1, 4, 32),
            anchor_x_rows=source["selection_slot_v15_anchor_x_rows"],
            anchor_range_norm=source["selection_slot_v15_anchor_range_norm"],
            anchor_geometry_valid=source["selection_slot_v15_geometry_valid"],
            anchor_active=source["selection_slot_active"],
            proposal_row_tokens=proposals["structured_row_tokens"],
            proposal_x_rows=proposals["pred_x_rows"],
            proposal_range_norm=proposals["range_norm"],
            candidate_valid=source["selection_slot_candidate_valid"],
            row_value_features=p2,
            graph_policy="identity",
        )
    graph = identity["selection_slot_v15_graph_attention"]
    assert torch.equal(graph, torch.eye(8).view(1, 8, 8))
    assert identity["selection_slot_v15_graph_state"].shape[:3] == (1, 8, 12)
    assert "cluster" not in " ".join(identity.keys()).lower()
    assert "prototype" not in " ".join(identity.keys()).lower()


def test_v15_no_context_replay_removes_proposal_geometry_path() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    module = head.bottom_aware_relational_geometry
    assert module is not None
    slot_states = torch.randn(1, 4, 32)
    with torch.no_grad():
        source = head(proposals, row_value_features=p2)
        kwargs = {
            "slot_states": slot_states,
            "anchor_x_rows": source["selection_slot_v15_anchor_x_rows"],
            "anchor_range_norm": source[
                "selection_slot_v15_anchor_range_norm"
            ],
            "anchor_geometry_valid": source[
                "selection_slot_v15_geometry_valid"
            ],
            "anchor_active": source["selection_slot_active"],
            "proposal_row_tokens": proposals["structured_row_tokens"],
            "proposal_x_rows": proposals["pred_x_rows"],
            "proposal_range_norm": proposals["range_norm"],
            "candidate_valid": source["selection_slot_candidate_valid"],
            "row_value_features": p2,
        }
        correct = module(**kwargs)
        no_context = module(**kwargs, context_policy="none")
    assert int(correct["selection_slot_v15_context_policy_id"][0]) == 0
    assert int(no_context["selection_slot_v15_context_policy_id"][0]) == 1
    # Proposal attention stays observable, but its values cannot enter the
    # context-free geometry trunk.
    assert torch.equal(
        correct["selection_slot_v15_proposal_attention"],
        no_context["selection_slot_v15_proposal_attention"],
    )
    assert torch.isfinite(no_context["selection_slot_pred_x_rows"]).all()


def test_v15_loss_trains_visual_graph_and_geometry_but_not_v7() -> None:
    torch.manual_seed(3407)
    head = _head()
    proposals = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(proposals, row_value_features=p2)
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v15=1.0,
        )
    )
    merged = {**proposals, **result}
    losses = criterion.compute_four_slot_v15_loss(merged, _targets())
    losses["total"].backward()
    module = head.bottom_aware_relational_geometry
    assert module is not None
    for parameter in (
        module.feature_key.weight,
        module.visual_query.weight,
        module.delta_head.weight,
        module.range_head.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
    # Exact zero-residual initialization intentionally delays final-geometry
    # gradients into value/graph/trunk parameters until the output head has
    # taken one update.  This is checked explicitly instead of hidden behind a
    # gradient-only straight-through term.
    assert module.feature_value.weight.grad is not None
    assert float(module.feature_value.weight.grad.abs().sum()) == 0.0
    assert module.graph_edge_mlp[-1].weight.grad is not None
    assert float(module.graph_edge_mlp[-1].weight.grad.abs().sum()) == 0.0

    with torch.no_grad():
        module.delta_head.weight.add_(-1.0e-3 * module.delta_head.weight.grad)
        module.range_head.weight.add_(-1.0e-3 * module.range_head.weight.grad)
    head.zero_grad(set_to_none=True)
    second = head(proposals, row_value_features=p2)
    second_losses = criterion.compute_four_slot_v15_loss(
        {**proposals, **second}, _targets()
    )
    second_losses["total"].backward()
    for parameter in (
        module.feature_value.weight,
        module.proposal_content.weight,
        module.graph_edge_mlp[-1].weight,
        module.graph_message.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
    assert all(value.grad is None for value in proposals.values())
    assert p2.grad is None
    assert head.active is not None and head.active.weight.grad is None
    assert all(
        parameter.grad is None for parameter in head.slot_refinement.parameters()
    )
    assert torch.isfinite(losses["total"])


def test_v15_config_is_single_frozen_upstream_arm() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v15_bottom_aware_relational_geometry_225k_to228k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_bottom_aware_relational_geometry_enabled"] is True
    assert selection["four_slot_corrected_visual_first_association_enabled"] is False
    assert selection["four_slot_corrected_visual_first_geometry_enabled"] is False
    assert cfg["loss"]["w_four_slot_v15"] == 1.0
    assert cfg["loss"]["w_four_slot_v14_stage_a"] == 0.0
    assert cfg["loss"]["w_four_slot_v14_stage_b"] == 0.0
    prefixes = cfg["training"]["trainable_parameter_prefixes"]
    assert prefixes == [
        "structured_query_head.set_selection_head.bottom_aware_relational_geometry"
    ]
    assert cfg["training"]["max_iters"] == 3000
    assert cfg["augmentation"]["horizontal_flip_prob"] == 0.0
