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
        "pred_x_rows": (base + slope).expand(batch, -1, -1).clone().requires_grad_(),
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
        refinement_enabled=False,
        range_refinement_enabled=True,
        range_delta_offsets_norm=(-0.2, -0.1, 0.0, 0.1, 0.2),
        unified_slot_decoder_enabled=True,
        unified_slot_decoder_hidden_dim=32,
        unified_slot_decoder_num_heads=4,
        unified_slot_decoder_ff_dim=64,
        unified_slot_decoder_vertical_layers=1,
        unified_slot_decoder_dropout=0.0,
        unified_slot_decoder_delta_offsets_px=(
            -100.0,
            -50.0,
            -20.0,
            0.0,
            20.0,
            50.0,
            100.0,
        ),
        unified_slot_decoder_range_delta_offsets_norm=(
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ),
        unified_slot_decoder_output_head_init_std=1.0e-5,
        unified_slot_decoder_activity_head_init_std=1.0e-7,
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


def test_v11_zero_step_is_soft_memory_geometry_with_post_geometry_activity():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        result = head(outputs, row_value_features=p2)
    assert head.slot_refinement is None
    assert head.unified_slot_decoder is not None
    assert float(
        (
            result["selection_slot_pred_x_rows"]
            - result["selection_slot_unified_aux_x_rows"]
        ).abs().max()
    ) < 0.1
    assert float(
        (
            result["selection_slot_range_norm"]
            - result["selection_slot_unified_aux_range_norm"]
        ).abs().max()
    ) < 1.0e-3
    expected_active = result["selection_slot_active_logits"] >= 0.0
    assert torch.equal(result["selection_slot_active"], expected_active)
    assert torch.equal(
        result["selection_slot_scores"],
        torch.sigmoid(result["selection_slot_active_logits"]),
    )


def test_v11_global_attention_reads_all_proposal_rows():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        first = head(outputs, row_value_features=p2)
    changed = {name: value.detach().clone() for name, value in outputs.items()}
    # Change every proposal row state.  Both attention and public geometry
    # must react because soft all-proposal memory owns the initial curve.
    changed["structured_row_tokens"][:, :, :, :] += torch.linspace(
        -2.0,
        2.0,
        8,
    ).view(1, 8, 1, 1)
    with torch.no_grad():
        second = head(changed, row_value_features=p2)
    attention = first["selection_slot_unified_proposal_attention"]
    assert attention.shape == (1, 4, 8)
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones(1, 4),
        atol=1.0e-6,
    )
    assert bool((attention.sum(dim=1) <= 1.0 + 1.0e-5).all())
    assert not torch.allclose(
        attention,
        second["selection_slot_unified_proposal_attention"],
    )
    assert not torch.allclose(
        first["selection_slot_unified_aux_x_rows"],
        second["selection_slot_unified_aux_x_rows"],
    )
    assert not torch.allclose(
        first["selection_slot_pred_x_rows"],
        second["selection_slot_pred_x_rows"],
    )


def test_v11_hard_route_id_is_provenance_only():
    torch.manual_seed(3407)
    head = _head().eval()
    unified = head.unified_slot_decoder
    assert unified is not None
    outputs = _proposal_outputs()
    slot_states = torch.randn(1, 4, 32)
    active_logits = torch.randn(1, 4)
    route_logits = torch.randn(1, 4, 8)
    candidate_valid = torch.ones(1, 8, dtype=torch.bool)
    p2 = torch.randn(1, 12, 20, 16)
    common = {
        "slot_states": slot_states,
        "legacy_active_logits": active_logits,
        "proposal_row_tokens": outputs["structured_row_tokens"],
        "proposal_x_rows": outputs["pred_x_rows"],
        "proposal_range_norm": outputs["range_norm"],
        "legacy_route_logits": route_logits,
        "candidate_valid": candidate_valid,
        "row_value_features": p2,
    }
    with torch.no_grad():
        first = unified(
            **common,
            route_indices=torch.tensor([[0, 1, 2, 3]]),
        )
        second = unified(
            **common,
            route_indices=torch.tensor([[7, 6, 5, 4]]),
        )
    for name in (
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
        "selection_slot_unified_proposal_attention",
    ):
        assert torch.equal(first[name], second[name])
    assert not torch.equal(
        first["selection_slot_indices"],
        second["selection_slot_indices"],
    )


def test_v11_single_loss_reaches_fresh_row_graph_but_not_frozen_sources():
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
            w_four_slot_unified=1.0,
        )
    )
    merged = {**outputs, **result}
    losses = criterion.compute_four_slot_unified_loss(merged, _targets())
    losses["total"].backward()
    unified = head.unified_slot_decoder
    assert unified is not None
    assert unified.delta_head.weight.grad is not None
    assert float(unified.delta_head.weight.grad.abs().sum()) > 0.0
    assert unified.feature_key.weight.grad is not None
    assert float(unified.feature_key.weight.grad.abs().sum()) > 0.0
    assert unified.proposal_key.weight.grad is not None
    assert float(unified.proposal_key.weight.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in unified.vertical_encoder.parameters()
    )
    assert unified.post_geometry_activity.weight.grad is not None
    assert head.active is not None and head.active.weight.grad is None
    assert all(value.grad is None for value in outputs.values())
    assert p2.grad is None
    assert float(losses["mean_matched"]) == 3.0


def test_v11_matching_and_non_activity_losses_ignore_activity_logits():
    torch.manual_seed(3407)
    head = _head().eval()
    outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
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
            w_four_slot_unified=1.0,
        )
    )
    merged = {**outputs, **result}
    low = criterion.compute_four_slot_unified_loss(
        {
            **merged,
            "selection_slot_active_logits": torch.full((1, 4), -8.0),
        },
        _targets(),
    )
    high = criterion.compute_four_slot_unified_loss(
        {
            **merged,
            "selection_slot_active_logits": torch.full((1, 4), 8.0),
        },
        _targets(),
    )
    for name in (
        "attention",
        "point",
        "range",
        "line_iou",
        "dfl",
        "aux_point",
        "aux_range",
        "aux_line_iou",
        "mean_final_quality",
    ):
        assert torch.equal(low[name], high[name])
    assert not torch.equal(low["active"], high["active"])


def test_v11_config_is_one_loss_one_fresh_trainable_subtree():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_unified_slot_row_225k_to228k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_refinement_enabled"] is False
    assert selection["four_slot_unified_slot_decoder_enabled"] is True
    assert (
        selection[
            "four_slot_unified_slot_decoder_proposal_attention_sinkhorn_iterations"
        ]
        == 64
    )
    assert selection["four_slot_slot_owned_geometry_enabled"] is False
    assert selection["four_slot_global_visual_geometry_enabled"] is False
    assert cfg["loss"]["w_four_slot_selection"] == 0.0
    assert cfg["loss"]["w_four_slot_geometry"] == 0.0
    assert cfg["loss"]["w_four_slot_unified"] == 1.0
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and value != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    assert nonzero_objectives == {"w_four_slot_unified": 1.0}
    assert cfg["training"]["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.unified_slot_decoder"
    ]
    assert cfg["training"]["frozen_detector_eval"] is True
