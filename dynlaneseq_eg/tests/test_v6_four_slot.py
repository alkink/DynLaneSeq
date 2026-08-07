from __future__ import annotations

import torch

from dynlaneseq_eg.evaluation.four_slot_decode import decode_four_slot_logits
from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes
from dynlaneseq_eg.losses.loss_s0 import (
    LossConfig,
    S0Criterion,
    build_four_slot_cluster_targets,
    four_slot_permutation_loss,
)
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotLaneSelectionHead,
)


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
