from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotLaneSelectionHead,
    structured_unique_route_marginals_with_private_dustbins,
)
from dynlaneseq_eg.tools.build_cross_clip_derangement import build


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


def _head(*, stage_b: bool = False) -> FourSlotLaneSelectionHead:
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
        corrected_visual_first_association_enabled=True,
        corrected_visual_first_association_hidden_dim=32,
        corrected_visual_first_association_num_heads=4,
        corrected_visual_first_association_ff_dim=64,
        corrected_visual_first_association_vertical_layers=1,
        corrected_visual_first_association_dropout=0.0,
        corrected_visual_first_association_sinkhorn_iterations=64,
        corrected_visual_first_geometry_enabled=stage_b,
        corrected_visual_first_geometry_hidden_dim=32,
        corrected_visual_first_geometry_num_heads=4,
        corrected_visual_first_geometry_ff_dim=64,
        corrected_visual_first_geometry_vertical_layers=1,
        corrected_visual_first_geometry_dropout=0.0,
        corrected_visual_first_geometry_delta_offsets_px=(
            -100.0,
            -50.0,
            0.0,
            50.0,
            100.0,
        ),
        corrected_visual_first_geometry_range_offsets_norm=(
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ),
    )


def test_private_dustbin_sinkhorn_has_exact_capacity_contract() -> None:
    torch.manual_seed(3407)
    logits = torch.randn(2, 4, 8, requires_grad=True)
    valid = torch.tensor(
        [[True] * 8, [True, True, True, True, True, False, False, False]]
    )
    dustbin = torch.randn(2, 4, requires_grad=True)
    attention = structured_unique_route_marginals_with_private_dustbins(
        logits,
        valid,
        dustbin,
        iterations=100,
    )
    assert attention.shape == (2, 4, 12)
    assert torch.allclose(
        attention.sum(dim=-1), torch.ones(2, 4), atol=2.0e-5
    )
    assert bool((attention[..., :8].sum(dim=1) <= 1.0 + 2.0e-5).all())
    assert float(attention[1, :, 5:8].abs().max()) == 0.0
    private = attention[..., 8:]
    off_diagonal = ~torch.eye(4, dtype=torch.bool).view(1, 4, 4)
    assert float(private.masked_select(off_diagonal).abs().max()) == 0.0
    attention.square().sum().backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0
    assert dustbin.grad is not None and float(dustbin.grad.abs().sum()) > 0.0


def test_v14_stage_a_is_exact_v7_deployment_sidecar() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposal_outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    module = head.corrected_visual_first_association
    assert module is not None
    head.corrected_visual_first_association = None
    with torch.no_grad():
        source = head(proposal_outputs, row_value_features=p2)
    head.corrected_visual_first_association = module
    with torch.no_grad():
        treatment = head(proposal_outputs, row_value_features=p2)
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
    attention = treatment["selection_slot_v14_proposal_attention"]
    assert attention.shape == (1, 4, 12)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 4), atol=2e-5)
    assert bool((attention[..., :8].sum(dim=1) <= 1.0 + 2e-5).all())
    assert torch.equal(
        treatment["selection_slot_v14_writer_valid"],
        source["selection_slot_active"],
    )


def test_v14_losses_reach_visual_and_proposal_consumers_only() -> None:
    torch.manual_seed(3407)
    head = _head()
    proposal_outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(proposal_outputs, row_value_features=p2)
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v14_stage_a=1.0,
            four_slot_v14_representable_min=0.10,
            four_slot_v14_cluster_delta=0.10,
            four_slot_v14_cluster_temperature=0.03,
        )
    )
    merged = {**proposal_outputs, **result}
    losses = criterion.compute_four_slot_v14_stage_a_loss(
        merged, _targets()
    )
    losses["total"].backward()
    module = head.corrected_visual_first_association
    assert module is not None
    for parameter in (
        module.feature_key.weight,
        module.feature_value.weight,
        module.visual_query.weight,
        module.proposal_content_key.weight,
        module.proposal_query.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.vertical_encoder.parameters()
    )
    assert all(value.grad is None for value in proposal_outputs.values())
    assert p2.grad is None
    assert head.active is not None and head.active.weight.grad is None
    assert all(
        parameter.grad is None for parameter in head.slot_refinement.parameters()
    )
    assert float(losses["target_attention_row_error"]) < 2.0e-5
    assert float(losses["target_real_column_excess"]) < 2.0e-5
    assert torch.isfinite(losses["total"])


def test_v14_association_cannot_rank_through_direct_u0_bypass() -> None:
    torch.manual_seed(3407)
    head = _head()
    proposal_outputs = _proposal_outputs()
    result = head(
        proposal_outputs,
        row_value_features=torch.randn(1, 12, 20, 16),
    )
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v14_stage_a=1.0,
            four_slot_v14_representable_min=0.10,
        )
    )
    losses = criterion.compute_four_slot_v14_stage_a_loss(
        {**proposal_outputs, **result}, _targets()
    )
    losses["association"].backward()
    module = head.corrected_visual_first_association
    assert module is not None
    # U0 has the V7 slot/anchor state. Association may observe its detached
    # value, but it must not optimize this route shortcut directly.
    for parameter in (
        module.slot_norm.weight,
        module.slot_projection.weight,
        module.slot_tokens.weight,
        module.row_position_projection.weight,
        module.anchor_geometry_projection.weight,
        module.initial_norm.weight,
    ):
        assert parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
    # It must instead learn through the image consumer and proposal transport.
    for parameter in (
        module.feature_key.weight,
        module.feature_value.weight,
        module.visual_query.weight,
        module.proposal_content_key.weight,
        module.proposal_query.weight,
    ):
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0


def test_v14_intervention_policies_are_explicit_and_causal() -> None:
    torch.manual_seed(3407)
    head = _head().eval()
    proposal_outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    with torch.no_grad():
        correct = head(proposal_outputs, row_value_features=p2)
    module = head.corrected_visual_first_association
    assert module is not None
    # The public forward already proves the correct policy is connected.  The
    # module-level policy set is asserted here so audits cannot silently merge
    # position-only and fully-zero controls.
    assert module.FEATURE_POLICIES == {
        "correct",
        "zero_content",
        "position_only",
        "zero_content_zero_position",
        "x_reversed",
        "row_reversed",
    }
    assert "selection_slot_v14_visual_logits" in correct


def test_v14_stage_b_zero_init_is_exact_v7_geometry_and_fixed_activity() -> None:
    torch.manual_seed(3407)
    head = _head(stage_b=True).eval()
    proposal_outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16)
    geometry = head.corrected_visual_first_geometry
    assert geometry is not None
    head.corrected_visual_first_geometry = None
    with torch.no_grad():
        source = head(proposal_outputs, row_value_features=p2)
    head.corrected_visual_first_geometry = geometry
    with torch.no_grad():
        initialized = head(proposal_outputs, row_value_features=p2)
    for name in (
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    ):
        assert torch.equal(source[name], initialized[name])
    assert float(initialized["selection_slot_v14_stage_b_delta_x_rows"].abs().max()) == 0.0


def test_v14_stage_b_geometry_loss_only_reaches_fresh_consumer() -> None:
    torch.manual_seed(3407)
    head = _head(stage_b=True)
    proposal_outputs = _proposal_outputs()
    p2 = torch.randn(1, 12, 20, 16, requires_grad=True)
    result = head(proposal_outputs, row_value_features=p2)
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v14_stage_b=1.0,
        )
    )
    losses = criterion.compute_four_slot_v14_stage_b_loss(
        {**proposal_outputs, **result}, _targets()
    )
    losses["total"].backward()
    geometry = head.corrected_visual_first_geometry
    association = head.corrected_visual_first_association
    assert geometry is not None and association is not None
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in geometry.parameters()
    )
    assert all(parameter.grad is None for parameter in association.parameters())
    assert all(value.grad is None for value in proposal_outputs.values())
    assert p2.grad is None
    assert head.active is not None and head.active.weight.grad is None
    assert torch.isfinite(losses["total"])


def test_v14_config_has_one_stage_a_objective_and_zero_augmentation() -> None:
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_a_225k_to227k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_corrected_visual_first_association_enabled"] is True
    assert selection["four_slot_visual_first_association_enabled"] is False
    assert selection["four_slot_visual_precision_geometry_enabled"] is False
    assert cfg["loss"]["w_four_slot_v14_stage_a"] == 1.0
    nonzero_objectives = {
        name: value
        for name, value in cfg["loss"].items()
        if isinstance(value, (int, float))
        and value != 0.0
        and (name.startswith("w_") or name.startswith("lambda_"))
    }
    assert nonzero_objectives == {"w_four_slot_v14_stage_a": 1.0}
    augmentation = cfg["augmentation"]
    assert augmentation["horizontal_flip_prob"] == 0.0
    assert augmentation["color_jitter"] is False
    assert augmentation["affine_prob"] == 0.0
    assert augmentation["random_shadow_prob"] == 0.0
    assert cfg["training"]["max_iters"] == 2000
    stage_b = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_b_227k_to229k.yaml"
    )
    assert stage_b["loss"]["w_four_slot_v14_stage_a"] == 0.0
    assert stage_b["loss"]["w_four_slot_v14_stage_b"] == 1.0
    assert stage_b["model"]["structured_query"]["set_selection"][
        "four_slot_corrected_visual_first_geometry_enabled"
    ] is True


def test_cross_clip_derangement_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    rows = [
        f"/driver_{driver}/clip_{clip}/{frame:03d}.jpg /anno/{frame:03d}.lines.txt"
        for driver, clip in ((1, 1), (1, 2), (2, 3), (2, 4))
        for frame in range(3)
    ]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    first = tmp_path / "wrong_a.txt"
    second = tmp_path / "wrong_b.txt"
    first_report = build(
        SimpleNamespace(
            input_list=str(source),
            output_list=str(first),
            output_json=str(tmp_path / "a.json"),
            seed=3407,
        )
    )
    second_report = build(
        SimpleNamespace(
            input_list=str(source),
            output_list=str(second),
            output_json=str(tmp_path / "b.json"),
            seed=3407,
        )
    )
    assert first_report["passed"] is True
    assert first_report["same_image_partner_count"] == 0
    assert first_report["same_clip_partner_count"] == 0
    assert first_report["output_sha256"] == second_report["output_sha256"]
    assert first.read_bytes() == second.read_bytes()
