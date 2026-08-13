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
    curves = centers.view(1, candidates, 1) + (y - 0.5) * torch.linspace(
        -4.0, 4.0, candidates
    ).view(1, candidates, 1)
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
        iterative_slot_geometry_enabled=True,
        iterative_slot_geometry_hidden_dim=32,
        iterative_slot_geometry_num_heads=4,
        iterative_slot_geometry_ff_dim=64,
        iterative_slot_geometry_num_stages=3,
        iterative_slot_geometry_vertical_layers_per_stage=1,
        iterative_slot_geometry_dropout=0.0,
        iterative_slot_geometry_scale_names=("p2", "p3", "p4"),
        iterative_slot_geometry_visual_offsets_px=(-16.0, 0.0, 16.0),
        iterative_slot_geometry_delta_offsets_px=(-8.0, 0.0, 8.0),
        iterative_slot_geometry_range_offsets_norm=(-0.01, 0.0, 0.01),
    )


def _features(*, requires_grad: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p2 = torch.randn(1, 12, 20, 16, requires_grad=requires_grad)
    multi = {
        "p2": p2.permute(0, 3, 1, 2),
        "p3": torch.randn(1, 16, 6, 10, requires_grad=requires_grad),
        "p4": torch.randn(1, 16, 3, 5, requires_grad=requires_grad),
    }
    return p2, multi


def test_v17_zero_step_is_exact_v7_and_has_three_recentered_stages() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    p2, multi = _features()
    module = head.iterative_slot_geometry
    assert module is not None
    head.iterative_slot_geometry = None
    with torch.no_grad():
        source = head(
            proposals,
            row_value_features=p2,
            multi_scale_features=multi,
        )
    head.iterative_slot_geometry = module
    with torch.no_grad():
        treatment = head(
            proposals,
            row_value_features=p2,
            multi_scale_features=multi,
        )
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
    stage_x = treatment["selection_slot_v17_stage_x_rows"]
    stage_input = treatment["selection_slot_v17_stage_input_x_rows"]
    assert stage_x.shape[1] == 3
    assert torch.equal(stage_input[:, 1], stage_x[:, 0])
    assert torch.equal(stage_input[:, 2], stage_x[:, 1])
    assert float(
        treatment["selection_slot_v17_stage_delta_x_rows"].abs().max()
    ) == 0.0


def test_v17_never_uses_proposal_coordinates_as_output_geometry() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposals = _proposal_outputs()
    p2, multi = _features()
    with torch.no_grad():
        first = head(
            proposals,
            row_value_features=p2,
            multi_scale_features=multi,
        )
        shifted = dict(proposals)
        shifted["pred_x_rows"] = proposals["pred_x_rows"] + 35.0
        second = head(
            shifted,
            row_value_features=p2,
            multi_scale_features=multi,
        )
    # Proposal coordinates change contextual attention but exact zero output
    # heads leave public geometry on the immutable V7 refined anchor.
    assert torch.equal(
        first["selection_slot_pred_x_rows"],
        first["selection_slot_v17_anchor_x_rows"],
    )
    assert torch.equal(
        second["selection_slot_pred_x_rows"],
        second["selection_slot_v17_anchor_x_rows"],
    )


def test_v17_loss_reaches_visual_at_zero_and_full_trunk_after_head_update() -> None:
    torch.manual_seed(3407)
    head = _head()
    proposals = _proposal_outputs()
    p2, multi = _features(requires_grad=True)
    result = head(
        proposals,
        row_value_features=p2,
        multi_scale_features=multi,
    )
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v17=1.0,
        )
    )
    loss = criterion.compute_four_slot_v17_loss(
        {**proposals, **result}, _targets()
    )
    loss["total"].backward()
    module = head.iterative_slot_geometry
    assert module is not None
    first_stage = module.stages[0]
    assert first_stage.scale_keys["p2"].weight.grad is not None
    assert float(first_stage.scale_keys["p2"].weight.grad.abs().sum()) > 0.0
    assert first_stage.delta_head.weight.grad is not None
    assert float(first_stage.delta_head.weight.grad.abs().sum()) > 0.0
    # Later-stage visual DFL consumes the persistent state produced by earlier
    # proposal/image fusion, so the iterative trunk is live even while public
    # geometry remains exact V7.
    assert first_stage.proposal_value.weight.grad is not None
    assert float(first_stage.proposal_value.weight.grad.abs().sum()) > 0.0

    with torch.no_grad():
        for stage in module.stages:
            assert stage.delta_head.weight.grad is not None
            assert stage.range_head.weight.grad is not None
            stage.delta_head.weight.add_(-1.0e-3 * stage.delta_head.weight.grad)
            stage.range_head.weight.add_(-1.0e-3 * stage.range_head.weight.grad)
    head.zero_grad(set_to_none=True)
    second = head(
        proposals,
        row_value_features=p2,
        multi_scale_features=multi,
    )
    second_loss = criterion.compute_four_slot_v17_loss(
        {**proposals, **second}, _targets()
    )
    second_loss["total"].backward()
    for parameter in (
        first_stage.scale_values["p2"].weight,
        first_stage.proposal_key.weight,
        first_stage.proposal_value.weight,
        first_stage.slot_interaction.value.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
    assert all(value.grad is None for value in proposals.values())
    assert head.active is not None and head.active.weight.grad is None
    assert all(
        parameter.grad is None for parameter in head.slot_refinement.parameters()
    )
    assert torch.isfinite(second_loss["total"])


def test_v17_config_is_one_fixed_endpoint_multiscale_arm() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v17_iterative_multiscale_geometry_225k_to230k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_iterative_slot_geometry_enabled"] is True
    assert selection["four_slot_candidate_aligned_reranker_enabled"] is False
    assert cfg["model"]["multi_scale_evidence"] == {
        "enabled": True,
        "scales": ["p2", "p3", "p4", "p5"],
    }
    assert cfg["loss"]["w_four_slot_v17"] == 1.0
    assert cfg["loss"]["four_slot_v17_stage_weights"] == [0.25, 0.5, 1.0]
    assert cfg["training"]["max_iters"] == 5000
    assert cfg["training"]["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.iterative_slot_geometry",
        "encoder.ms_proj.p3",
    ]
    assert cfg["augmentation"]["horizontal_flip_prob"] == 0.0
