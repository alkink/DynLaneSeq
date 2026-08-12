from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.four_slot_selection import FourSlotBoundedRefinement


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _inputs():
    batch, slots, candidates, rows, dim = 1, 4, 6, 12, 16
    slot_states = torch.randn(batch, slots, 32, requires_grad=True)
    row_tokens = torch.randn(
        batch, candidates, rows, dim, requires_grad=True
    )
    proposal_x = torch.stack(
        (
            torch.full((rows,), 10.0),
            torch.full((rows,), 16.0),
            torch.full((rows,), 55.0),
            torch.full((rows,), 61.0),
            torch.full((rows,), 82.0),
            torch.full((rows,), 88.0),
        )
    ).unsqueeze(0).requires_grad_()
    proposal_range = torch.tensor(
        [[[0.0, 0.9]] * candidates], requires_grad=True
    )
    row_features = torch.randn(
        batch, rows, 20, dim, requires_grad=True
    )
    route = torch.tensor([[0, 2, 4, 5]])
    valid = torch.ones((batch, candidates), dtype=torch.bool)
    return {
        "slot_states": slot_states,
        "proposal_row_tokens": row_tokens,
        "proposal_x_rows": proposal_x,
        "proposal_range_norm": proposal_range,
        "route_indices": route,
        "candidate_valid": valid,
        "row_value_features": row_features,
    }


def _refiner(reference_mode: str):
    return FourSlotBoundedRefinement(
        16,
        input_w=100,
        slot_dim=32,
        hidden_dim=32,
        delta_offsets_px=(-12.0, -6.0, 0.0, 6.0, 12.0),
        reference_mode=reference_mode,
        neighborhood_max_candidates=4,
        neighborhood_max_mean_distance_px=48.0,
        neighborhood_min_common_fraction=0.5,
        neighborhood_distance_temperature_px=24.0,
        neighborhood_gradient_scale=0.1,
        range_refinement=True,
    )


def test_neighborhood_reference_starts_as_exact_v7_forward():
    torch.manual_seed(3407)
    hard = _refiner("hard_st")
    neighborhood = _refiner("neighborhood_soft")
    neighborhood.load_state_dict(hard.state_dict(), strict=False)
    inputs = _inputs()
    hard.eval()
    neighborhood.eval()
    with torch.no_grad():
        expected = hard(**inputs)
        actual = neighborhood(**inputs)
    assert torch.equal(
        actual["selection_slot_pred_x_rows"],
        expected["selection_slot_pred_x_rows"],
    )
    assert torch.equal(
        actual["selection_slot_range_norm"],
        expected["selection_slot_range_norm"],
    )
    assert float(actual["selection_slot_neighborhood_mix"]) == 0.0
    support = actual["selection_slot_neighborhood_support"]
    assert bool(((support >= 1) & (support <= 4)).all())


def test_neighborhood_geometry_gradient_is_local_and_input_detached():
    refiner = _refiner("neighborhood_soft")
    inputs = _inputs()
    refiner.train()
    result = refiner(**inputs)
    loss = (
        result["selection_slot_pred_x_rows"].square().mean()
        + result["selection_slot_range_norm"].square().mean()
    )
    loss.backward()
    assert refiner.neighborhood_mix is not None
    assert refiner.neighborhood_mix.grad is not None
    assert torch.isfinite(refiner.neighborhood_mix.grad)
    for module in (
        refiner.neighborhood_anchor_projection,
        refiner.neighborhood_candidate_projection,
        refiner.neighborhood_slot_projection,
    ):
        assert module is not None
        assert module.weight.grad is not None
        assert float(module.weight.grad.abs().sum()) > 0.0
    assert refiner.delta_head.weight.grad is not None
    assert float(refiner.delta_head.weight.grad.abs().sum()) > 0.0
    for source in inputs.values():
        if isinstance(source, torch.Tensor) and source.requires_grad:
            assert source.grad is None


def test_signed_neighborhood_residual_has_no_negative_dead_zone():
    refiner = _refiner("neighborhood_soft")
    assert refiner.neighborhood_mix is not None
    refiner.neighborhood_mix.data.fill_(-0.25)
    inputs = _inputs()
    refiner.eval()
    result = refiner(**inputs)
    assert float(result["selection_slot_neighborhood_mix"]) < 0.0
    result["selection_slot_pred_x_rows"].sum().backward()
    assert refiner.neighborhood_mix.grad is not None
    assert float(refiner.neighborhood_mix.grad.abs()) > 0.0


def test_v8_gate_trains_only_geometry_owned_neighborhood_and_refiner():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_anchor_neighborhood_gate_225k_to227k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    training = cfg["training"]
    assert selection["four_slot_refinement_reference_mode"] == "neighborhood_soft"
    assert selection["four_slot_refinement_neighborhood_max_candidates"] == 4
    assert selection["four_slot_refinement_neighborhood_max_mean_distance_px"] == 48.0
    assert loss["w_four_slot_selection"] == 0.0
    assert loss["w_four_slot_geometry"] == 1.0
    assert training["frozen_detector_eval"] is True
    assert training["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.slot_refinement"
    ]
    assert training["checkpoint_model_prefixes"] == [
        "structured_query_head.set_selection_head.slot_refinement"
    ]
