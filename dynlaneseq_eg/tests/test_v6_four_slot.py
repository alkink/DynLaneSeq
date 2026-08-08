from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.evaluation.four_slot_decode import decode_four_slot_logits
from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes
from dynlaneseq_eg.losses.loss_s0 import (
    LossConfig,
    S0Criterion,
    build_four_slot_cluster_targets,
    four_slot_permutation_loss,
)
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotBoundedRefinement,
    FourSlotLaneSelectionHead,
    decode_unique_four_slot_routes,
)
from dynlaneseq_eg.tools.audit_v6_a_target_distribution import (
    _finish_accumulator,
    _new_accumulator,
)
from dynlaneseq_eg.tools.probe_v5_four_slot_router import FourSlotRouter
from dynlaneseq_eg.tools.summarize_v6_a_probe_mismatch import _log_integer
from dynlaneseq_eg.tools.summarize_v6_c_router_refiner_gate import (
    _method as v6_c_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _head_outputs(*, batch: int = 2, candidates: int = 6, rows: int = 12):
    dim = 16
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
        "pred_x_rows": (torch.rand(batch, candidates, rows) * 99.0)
        .requires_grad_(),
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


def test_four_slot_head_matches_probe_parameter_count_and_detaches_inputs():
    head = FourSlotLaneSelectionHead(
        16,
        input_w=100,
        hidden_dim=32,
        num_slots=4,
        proposal_layers=2,
        slot_layers=2,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        curve_samples=8,
        min_valid_rows=5,
    )
    outputs = _head_outputs()
    result = head(outputs)
    assert result["selection_slot_logits"].shape == (2, 4, 7)
    assert result["selection_slot_candidate_valid"].shape == (2, 6)
    assert torch.isfinite(result["selection_slot_logits"]).all()
    result["selection_slot_logits"].sum().backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    for value in outputs.values():
        assert value.grad is None


def test_v6_b_config_trains_only_zero_initialized_slot_refinement():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_b_four_slot_refinement_25k_to29k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    training = cfg["training"]
    loss = cfg["loss"]
    assert selection["four_slot_refinement_enabled"] is True
    assert loss["w_four_slot_selection"] == 0.0
    assert loss["w_four_slot_geometry"] == 1.0
    assert training["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.slot_refinement"
    ]
    assert training["checkpoint_model_prefixes"] == [
        "structured_query_head.set_selection_head.slot_refinement"
    ]


def test_v6_c_config_opens_only_straight_through_router_refiner_subtree():
    cfg = load_config(
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v6_c_router_refiner_cotrain_28k_to32k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    training = cfg["training"]
    loss = cfg["loss"]
    assert selection["four_slot_refinement_enabled"] is True
    assert selection["four_slot_refinement_straight_through_routing"] is True
    assert selection["four_slot_refinement_detach_slot_states"] is False
    assert selection["four_slot_refinement_route_temperature"] == 1.0
    assert loss["w_four_slot_selection"] == 0.25
    assert loss["w_four_slot_geometry"] == 1.0
    assert training["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head"
    ]
    assert training["checkpoint_model_prefixes"] == [
        "structured_query_head.set_selection_head"
    ]


def test_production_four_slot_state_is_checkpoint_compatible_with_probe():
    production = FourSlotLaneSelectionHead(
        16,
        input_w=100,
        hidden_dim=32,
        num_slots=4,
        proposal_layers=2,
        slot_layers=2,
        num_heads=4,
        ff_dim=64,
        dropout=0.1,
        curve_samples=8,
        min_valid_rows=5,
    )
    probe = FourSlotRouter(
        3 * 16 + 11 + 2 * 8,
        hidden_dim=32,
        num_slots=4,
        proposal_layers=2,
        slot_layers=2,
        num_heads=4,
        ff_dim=64,
        dropout=0.1,
    )
    assert production.state_dict().keys() == probe.state_dict().keys()
    assert {
        name: tuple(value.shape)
        for name, value in production.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in probe.state_dict().items()
    }


def test_four_slot_head_handles_an_all_invalid_candidate_set():
    head = FourSlotLaneSelectionHead(
        16,
        input_w=100,
        hidden_dim=32,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        curve_samples=8,
        min_valid_rows=5,
    )
    outputs = _head_outputs(batch=1)
    outputs["range_norm"] = torch.zeros((1, 6, 2))
    result = head(outputs)
    assert not result["selection_slot_candidate_valid"].any()
    assert torch.isfinite(result["selection_slot_logits"]).all()
    assert torch.all(result["selection_slot_logits"][..., :-1] == -1.0e4)


def _geometry_outputs():
    rows = 10
    pred = torch.stack(
        (
            torch.full((rows,), 10.0),
            torch.full((rows,), 12.0),
            torch.full((rows,), 80.0),
        )
    ).unsqueeze(0)
    return {
        "pred_x_rows": pred,
        "range_norm": torch.tensor([[[0.0, 0.9]] * 3]),
    }


def _targets():
    rows = 10
    return [
        {
            "x_rows": torch.stack(
                (
                    torch.full((rows,), 10.0),
                    torch.full((rows,), 80.0),
                )
            ),
            "valid_mask": torch.ones((2, rows), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 90.0], [0.0, 90.0]]),
        }
    ]


def test_four_slot_targets_preserve_two_gt_cluster_rows():
    built = build_four_slot_cluster_targets(
        _geometry_outputs(),
        _targets(),
        num_slots=4,
        input_h=100,
        line_width=30.0,
        min_valid_rows=5,
        representable_min=0.5,
        cluster_min=0.3,
        cluster_delta=0.1,
        temperature=0.03,
    )
    rows = built["rows"][0]
    assert rows.shape == (2, 4)
    assert torch.allclose(rows.sum(dim=-1), torch.ones(2))
    assert torch.all(rows[:, -1] == 0.0)
    assert int((rows[0, :3] > 0).sum()) >= 1
    assert int((rows[1, :3] > 0).sum()) >= 1


def test_four_slot_permutation_loss_is_gt_order_invariant():
    logits = torch.randn(1, 4, 4, requires_grad=True)
    rows = torch.tensor(
        [[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    forward = four_slot_permutation_loss(logits, [rows])
    reverse = four_slot_permutation_loss(logits, [rows.flip(0)])
    assert torch.allclose(forward, reverse, atol=1.0e-6)
    forward.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_global_decode_uses_private_dustbins_and_unique_proposals():
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0],
            [8.0, 7.0, 0.0],
            [6.0, 5.0, 10.0],
            [4.0, 3.0, 10.0],
        ]
    )
    decoded = decode_four_slot_logits(logits, torch.tensor([True, True]))
    active = decoded["indices"][decoded["indices"] >= 0]
    assert active.numel() == 2
    assert active.unique().numel() == 2
    assert int((decoded["indices"] < 0).sum()) == 2
    assert int(decoded["raw_collision_count"]) == 1
    assert int(decoded["repair_count"]) >= 1


def test_device_unique_decoder_matches_scipy_assignment():
    generator = torch.Generator().manual_seed(3407)
    logits = torch.randn(8, 4, 33, generator=generator)
    valid = torch.rand(8, 32, generator=generator) > 0.15
    device_decode = decode_unique_four_slot_routes(logits, valid)
    scipy_decode = decode_four_slot_logits(logits, valid)
    assert torch.equal(device_decode["indices"], scipy_decode["indices"])
    assert torch.allclose(
        device_decode["scores"],
        scipy_decode["scores"],
        atol=1.0e-7,
    )


def test_bounded_slot_refinement_starts_as_exact_identity_and_detaches_inputs():
    batch, slots, candidates, rows, dim = 1, 4, 6, 12, 16
    refiner = FourSlotBoundedRefinement(
        dim,
        input_w=100,
        slot_dim=32,
        hidden_dim=32,
        delta_offsets_px=(-12.0, -6.0, 0.0, 6.0, 12.0),
    )
    slot_states = torch.randn(batch, slots, 32, requires_grad=True)
    row_tokens = torch.randn(
        batch, candidates, rows, dim, requires_grad=True
    )
    proposal_x = (torch.rand(batch, candidates, rows) * 99.0).requires_grad_()
    proposal_range = torch.tensor(
        [[[0.0, 0.9]] * candidates], requires_grad=True
    )
    row_features = torch.randn(
        batch, rows, 20, dim, requires_grad=True
    )
    route = torch.tensor([[0, 2, -1, 5]])
    result = refiner(
        slot_states=slot_states,
        proposal_row_tokens=row_tokens,
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        route_indices=route,
        row_value_features=row_features,
    )
    safe = route.clamp_min(0).unsqueeze(-1).expand(-1, -1, rows)
    reference = proposal_x.detach().gather(1, safe)
    reference[:, 2] = 0.0
    assert torch.allclose(
        result["selection_slot_pred_x_rows"],
        reference,
        atol=1.0e-6,
    )
    assert float(result["selection_slot_delta_max_abs"].max()) < 1.0e-6
    result["selection_slot_pred_x_rows"].sum().backward()
    assert refiner.delta_head.weight.grad is not None
    assert float(refiner.delta_head.weight.grad.abs().sum()) > 0.0
    for source in (
        slot_states,
        row_tokens,
        proposal_x,
        proposal_range,
        row_features,
    ):
        assert source.grad is None


def test_straight_through_refinement_preserves_hard_forward_and_routes_gradient():
    batch, slots, candidates, rows, dim = 1, 4, 6, 12, 16
    refiner = FourSlotBoundedRefinement(
        dim,
        input_w=100,
        slot_dim=32,
        hidden_dim=32,
        delta_offsets_px=(-12.0, -6.0, 0.0, 6.0, 12.0),
        straight_through_routing=True,
        detach_slot_states=False,
    )
    torch.nn.init.normal_(refiner.delta_head.weight, std=0.01)
    slot_states = torch.randn(batch, slots, 32, requires_grad=True)
    row_tokens = torch.randn(
        batch, candidates, rows, dim, requires_grad=True
    )
    proposal_x = (torch.rand(batch, candidates, rows) * 99.0).requires_grad_()
    proposal_range = torch.tensor(
        [[[0.0, 0.9]] * candidates], requires_grad=True
    )
    row_features = torch.randn(
        batch, rows, 20, dim, requires_grad=True
    )
    route = torch.tensor([[0, 2, -1, 5]])
    route_logits = torch.randn(
        batch,
        slots,
        candidates + 1,
        requires_grad=True,
    )
    candidate_valid = torch.ones((batch, candidates), dtype=torch.bool)

    result = refiner(
        slot_states=slot_states,
        proposal_row_tokens=row_tokens,
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        route_indices=route,
        route_logits=route_logits,
        candidate_valid=candidate_valid,
        row_value_features=row_features,
    )
    refiner.straight_through_routing = False
    expected = refiner(
        slot_states=slot_states.detach(),
        proposal_row_tokens=row_tokens.detach(),
        proposal_x_rows=proposal_x.detach(),
        proposal_range_norm=proposal_range.detach(),
        route_indices=route,
        row_value_features=row_features.detach(),
    )
    assert torch.allclose(
        result["selection_slot_pred_x_rows"],
        expected["selection_slot_pred_x_rows"],
        atol=1.0e-5,
    )

    result["selection_slot_pred_x_rows"].sum().backward()
    assert route_logits.grad is not None
    assert float(route_logits.grad.abs().sum()) > 0.0
    assert slot_states.grad is not None
    assert float(slot_states.grad.abs().sum()) > 0.0
    for source in (row_tokens, proposal_x, proposal_range, row_features):
        assert source.grad is None


def test_postprocess_skips_dustbin_between_active_slots():
    rows = 8
    outputs = {
        "exist_logits": torch.zeros((1, 3, 2)),
        "quality_logits": torch.zeros((1, 3)),
        "pred_x_rows": torch.tensor(
            [[[10.0] * rows, [50.0] * rows, [90.0] * rows]]
        ),
        "range_norm": torch.tensor([[[0.0, 0.875]] * 3]),
        "selection_slot_candidate_valid": torch.ones(
            (1, 3), dtype=torch.bool
        ),
        "selection_slot_logits": torch.tensor(
            [
                [
                    [9.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 9.0],
                    [0.0, 9.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 9.0],
                ]
            ]
        ),
    }
    lanes = predictions_to_lanes(
        outputs,
        score_mode="four_slot",
        score_thresh=0.0,
        min_pred_points=5,
        input_w=100,
        input_h=100,
        top_k=4,
    )
    assert len(lanes[0]) == 2


def test_postprocess_uses_refined_slot_geometry_when_available():
    rows = 8
    outputs = {
        "exist_logits": torch.zeros((1, 3, 2)),
        "quality_logits": torch.zeros((1, 3)),
        "pred_x_rows": torch.tensor(
            [[[10.0] * rows, [50.0] * rows, [70.0] * rows]]
        ),
        "range_norm": torch.tensor([[[0.0, 0.875]] * 3]),
        "selection_slot_candidate_valid": torch.ones(
            (1, 3), dtype=torch.bool
        ),
        "selection_slot_logits": torch.tensor(
            [
                [
                    [9.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 9.0],
                    [0.0, 9.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, 9.0],
                ]
            ]
        ),
        "selection_slot_indices": torch.tensor([[0, -1, 1, -1]]),
        "selection_slot_scores": torch.ones((1, 4)),
        "selection_slot_pred_x_rows": torch.tensor(
            [[[90.0] * rows, [0.0] * rows, [30.0] * rows, [0.0] * rows]]
        ),
        "selection_slot_range_norm": torch.tensor(
            [[[0.0, 0.875], [0.0, 0.0], [0.0, 0.875], [0.0, 0.0]]]
        ),
    }
    lanes = predictions_to_lanes(
        outputs,
        score_mode="four_slot",
        score_thresh=0.0,
        min_pred_points=5,
        input_w=100,
        input_h=100,
        top_k=4,
    )[0]
    assert len(lanes) == 2
    assert {round(lane[0][0]) for lane in lanes} == {30, 90}


def test_criterion_backpropagates_only_through_slot_logits():
    outputs = _geometry_outputs()
    outputs["selection_slot_logits"] = torch.randn(
        1, 4, 4, requires_grad=True
    )
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=100,
            w_exist=0.0,
            w_point=0.0,
            w_range=0.0,
            w_four_slot_selection=1.0,
            four_slot_cluster_delta=0.1,
        )
    )
    losses = criterion(outputs, _targets(), matches=[{}])
    losses["loss_total"].backward()
    assert outputs["selection_slot_logits"].grad is not None
    assert torch.isfinite(outputs["selection_slot_logits"].grad).all()


def test_slot_geometry_loss_backpropagates_through_local_delta_logits():
    outputs = _geometry_outputs()
    rows = int(outputs["pred_x_rows"].shape[-1])
    reference = torch.stack(
        (
            torch.full((rows,), 10.0),
            torch.full((rows,), 80.0),
            torch.zeros(rows),
            torch.zeros(rows),
        )
    ).unsqueeze(0)
    offsets = torch.tensor([-6.0, 0.0, 6.0])
    delta_logits = torch.zeros(
        1,
        4,
        rows,
        3,
        requires_grad=True,
    )
    delta = (torch.softmax(delta_logits, dim=-1) * offsets).sum(dim=-1)
    outputs.update(
        {
            "selection_slot_input_reference_x_rows": reference,
            "selection_slot_pred_x_rows": reference + delta,
            "selection_slot_range_norm": torch.tensor(
                [[[0.0, 0.9], [0.0, 0.9], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "selection_slot_active": torch.tensor(
                [[True, True, False, False]]
            ),
            "selection_slot_row_delta_logits": delta_logits,
            "selection_slot_row_delta_offsets_px": offsets,
        }
    )
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=100,
            w_exist=0.0,
            w_point=0.0,
            w_range=0.0,
            w_four_slot_geometry=1.0,
            four_slot_geometry_match_min_quality=0.2,
        )
    )
    losses = criterion(outputs, _targets(), matches=[{}])
    assert float(losses["four_slot_geometry_mean_matched"]) == 2.0
    assert torch.isfinite(losses["loss_four_slot_geometry"])
    losses["loss_total"].backward()
    assert delta_logits.grad is not None
    assert float(delta_logits.grad.abs().sum()) > 0.0


def test_v6_target_distribution_accumulator_reports_dustbin_contract():
    accumulator = _new_accumulator()
    accumulator.update(
        {
            "images": 2,
            "gt_lanes": 7.0,
            "representable": 5.0,
            "support_weighted_sum": 10.0,
            "entropy_weighted_sum": 2.5,
            "quality_weighted_sum": 4.0,
        }
    )
    accumulator["count_histogram"].update((2, 3))
    result = _finish_accumulator(accumulator, slots=4)
    assert result["mean_gt_lanes"] == 3.5
    assert result["mean_representable_lanes"] == 2.5
    assert result["expected_dustbin_fraction"] == 0.375
    assert result["mean_support_size"] == 2.0
    assert result["representable_count_histogram"] == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 1,
        "4": 0,
    }


def test_v6_mismatch_summary_reads_training_exposure(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "{'train_images': 88880, 'effective_batch_size': 16, "
        "'iters': 4000}\n",
        encoding="utf-8",
    )
    assert _log_integer(str(log), "train_images") == 88880
    assert _log_integer(str(log), "effective_batch_size") == 16
    assert _log_integer(str(log), "iters") == 4000
    assert _log_integer(str(log), "missing") is None


def test_v6_c_summary_requires_refined_four_slot_method():
    method = {"0.50": {"f1": 0.81}, "0.75": {"f1": 0.59}}
    assert v6_c_method({"methods": {"four_slot_refined": method}}) is method
