from __future__ import annotations

import torch

from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.four_slot_selection import FourSlotLaneSelectionHead
from dynlaneseq_eg.modeling.v16_candidate_reranker import (
    FourSlotCandidateAlignedReranker,
)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch, slots, candidates, rows, x_bins, dim = 2, 2, 6, 16, 40, 32
    y = torch.linspace(0.0, 1.0, rows)
    curves = torch.stack(
        (
            90.0 + 20.0 * y,
            112.0 + 18.0 * y,
            145.0 + 15.0 * y,
            430.0 - 20.0 * y,
            458.0 - 18.0 * y,
            900.0 + 5.0 * y,
        )
    ).unsqueeze(0).expand(batch, -1, -1).clone()
    ranges = torch.tensor([[[0.0, 1.0]] * candidates]).expand(
        batch, -1, -1
    ).clone()
    anchors = torch.tensor([[0, 3], [0, 3]])
    anchor_x = curves.gather(
        1, anchors.unsqueeze(-1).expand(-1, -1, rows)
    )
    anchor_range = ranges.gather(
        1, anchors.unsqueeze(-1).expand(-1, -1, 2)
    )
    return {
        "slot_states": torch.randn(batch, slots, dim),
        "anchor_indices": anchors,
        "anchor_x_rows": anchor_x,
        "anchor_range_norm": anchor_range,
        "anchor_geometry_valid": torch.ones(batch, slots, dtype=torch.bool),
        "anchor_active": torch.ones(batch, slots, dtype=torch.bool),
        "proposal_row_tokens": torch.randn(batch, candidates, rows, dim),
        "proposal_x_rows": curves,
        "proposal_range_norm": ranges,
        "candidate_valid": torch.ones(batch, candidates, dtype=torch.bool),
        "row_value_features": torch.randn(batch, rows, x_bins, dim),
    }


def _module() -> FourSlotCandidateAlignedReranker:
    return FourSlotCandidateAlignedReranker(
        32,
        input_w=1600,
        slot_dim=32,
        num_slots=2,
        hidden_dim=32,
        row_dilations=(1, 2),
        evidence_offsets_px=(-24.0, 0.0, 24.0),
        dropout=0.0,
    )


def test_v16_selects_one_complete_member_without_coordinate_averaging() -> None:
    module = _module().eval()
    inputs = _inputs()
    with torch.no_grad():
        outputs = module(**inputs)
    group_mask = outputs["selection_slot_v16_group_mask"]
    selected = outputs["selection_slot_v16_selected_indices"]
    selected_x = outputs["selection_slot_v16_selected_x_rows"]
    expected_x = inputs["proposal_x_rows"].gather(
        1, selected.unsqueeze(-1).expand_as(selected_x)
    )
    assert torch.equal(selected_x, expected_x)
    assert bool(group_mask.gather(2, selected.unsqueeze(-1)).all())
    assert bool((group_mask.sum(dim=1) <= 1).all())
    assert outputs["selection_slot_v16_group_size"].min() >= 2
    assert outputs["selection_slot_v16_group_size"].max() < 6
    assert "selection_slot_pred_x_rows" not in outputs
    assert "selection_slot_range_norm" not in outputs


def test_v16_score_has_direct_gradients_to_visual_and_proposal_consumers() -> None:
    module = _module().train()
    outputs = module(**_inputs())
    mask = outputs["selection_slot_v16_group_mask"]
    loss = outputs["selection_slot_v16_candidate_scores"][mask].square().mean()
    loss.backward()
    assert module.feature_value.weight.grad is not None
    assert float(module.feature_value.weight.grad.abs().sum()) > 0.0
    assert module.proposal_projection.weight.grad is not None
    assert float(module.proposal_projection.weight.grad.abs().sum()) > 0.0
    assert module.row_blocks[0].depthwise.weight.grad is not None
    assert float(module.row_blocks[0].depthwise.weight.grad.abs().sum()) > 0.0


def test_v16_feature_intervention_changes_only_private_reranker_state() -> None:
    module = _module().eval()
    inputs = _inputs()
    with torch.no_grad():
        correct = module(**inputs)
        zero = module(**inputs, feature_policy="zero_content")
    assert torch.equal(
        correct["selection_slot_v16_group_mask"],
        zero["selection_slot_v16_group_mask"],
    )
    assert torch.equal(
        correct["selection_slot_v16_anchor_x_rows"],
        zero["selection_slot_v16_anchor_x_rows"],
    )
    assert not torch.equal(
        correct["selection_slot_v16_candidate_scores"],
        zero["selection_slot_v16_candidate_scores"],
    )


def _head_inputs() -> dict[str, torch.Tensor]:
    batch, candidates, rows, dim = 1, 8, 12, 16
    base = torch.linspace(12.0, 88.0, candidates).view(1, candidates, 1)
    slope = torch.linspace(-3.0, 3.0, rows).view(1, 1, rows)
    curves = (base + slope).clone().requires_grad_()
    return {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, requires_grad=True
        ),
        "queries": torch.randn(batch, candidates, dim, requires_grad=True),
        "ownership_state": torch.randn(
            batch, candidates, dim, requires_grad=True
        ),
        "range_norm": torch.tensor(
            [[[0.0, 0.95]] * candidates], requires_grad=True
        ),
        "pred_x_rows": curves,
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, requires_grad=True
        ),
        "exist_logits": torch.randn(
            batch, candidates, 2, requires_grad=True
        ),
        "input_reference_x_rows": curves.detach().clone().requires_grad_(),
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
        candidate_aligned_reranker_enabled=True,
        candidate_aligned_reranker_hidden_dim=32,
        candidate_aligned_reranker_row_dilations=(1, 2),
        candidate_aligned_reranker_evidence_offsets_px=(-8.0, 0.0, 8.0),
    )


def _head_targets() -> list[dict[str, torch.Tensor]]:
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


def test_v16_head_is_exact_v7_deployment_sidecar_and_loss_is_isolated() -> None:
    torch.manual_seed(3407)
    head = _head()
    proposal_outputs = _head_inputs()
    p2 = torch.randn(1, 12, 20, 16, requires_grad=True)
    module = head.candidate_aligned_reranker
    assert module is not None
    head.candidate_aligned_reranker = None
    with torch.no_grad():
        source = head(proposal_outputs, row_value_features=p2)
    head.candidate_aligned_reranker = module
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

    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v16=1.0,
            four_slot_v16_representable_min=0.10,
        )
    )
    losses = criterion.compute_four_slot_v16_loss(
        {**proposal_outputs, **treatment},
        _head_targets(),
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert module.feature_value.weight.grad is not None
    assert float(module.feature_value.weight.grad.abs().sum()) > 0.0
    assert module.proposal_projection.weight.grad is not None
    assert float(module.proposal_projection.weight.grad.abs().sum()) > 0.0
    assert p2.grad is None
    assert all(value.grad is None for value in proposal_outputs.values())
    assert head.active is not None and head.active.weight.grad is None
    assert all(
        parameter.grad is None for parameter in head.slot_refinement.parameters()
    )
