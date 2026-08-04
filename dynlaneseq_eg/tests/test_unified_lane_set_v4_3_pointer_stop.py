from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import _apply_pointer_teacher_forcing
from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes
from dynlaneseq_eg.losses.loss_s0 import (
    LossConfig,
    S0Criterion,
    build_pointer_sequence_targets,
)
from dynlaneseq_eg.modeling.structured_queries import (
    SetAwareLaneSelectionHead,
    StructuredLaneQueryHead,
)


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml"
)


def _selector() -> SetAwareLaneSelectionHead:
    return SetAwareLaneSelectionHead(
        8,
        input_w=64,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        curve_samples=4,
        unified_score=True,
        detach_geometry_features=True,
        candidate_interaction="sequential_pointer",
        relation_hidden_dim=8,
        pointer_max_selections=4,
        pointer_min_valid_rows=2,
        pointer_similarity_prior=0.5,
    )


def test_pointer_is_without_replacement_and_stop_is_absorbing() -> None:
    torch.manual_seed(701)
    selector = _selector().eval()
    hidden = torch.randn(2, 3, 16)
    relations = torch.zeros(2, 3, 3, 6)
    unary = torch.tensor([[4.0, 3.0, 2.0], [4.0, 3.0, 2.0]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        selector.pointer_stop[-1].bias.fill_(-20.0)
    result = selector.decode_pointer(hidden, relations, unary, valid)
    for row in result["selection_pointer_indices"]:
        emitted = [int(value) for value in row if int(value) >= 0]
        assert len(emitted) == len(set(emitted))

    with torch.no_grad():
        selector.pointer_stop[-1].bias.fill_(20.0)
        selector.output.weight.zero_()
        selector.output.bias.fill_(-20.0)
    stopped = selector.decode_pointer(
        hidden,
        relations,
        torch.full_like(unary, -20.0),
        valid,
    )["selection_pointer_indices"]
    assert bool((stopped == -1).all())


def test_pointer_decisions_are_candidate_permutation_equivariant() -> None:
    torch.manual_seed(703)
    selector = _selector().eval()
    with torch.no_grad():
        selector.pointer_stop[-1].bias.fill_(-20.0)
    hidden = torch.randn(1, 4, 16)
    relations = torch.randn(1, 4, 4, 6)
    unary = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    original = selector.decode_pointer(hidden, relations, unary, valid)[
        "selection_pointer_indices"
    ][0]
    permuted = selector.decode_pointer(
        hidden[:, permutation],
        relations[:, permutation][:, :, permutation],
        unary[:, permutation],
        valid[:, permutation],
    )["selection_pointer_indices"][0]
    mapped = torch.tensor(
        [int(permutation[index]) if int(index) >= 0 else -1 for index in permuted]
    )
    torch.testing.assert_close(mapped, original)


def test_pointer_targets_are_left_to_right_unique_then_stop() -> None:
    outputs = {
        "pred_x_rows": torch.tensor(
            [[[50.0] * 8, [10.0] * 8, [30.0] * 8]]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[10.0] * 8, [50.0] * 8]),
            "valid_mask": torch.ones(2, 8, dtype=torch.bool),
        }
    ]
    sequence = build_pointer_sequence_targets(
        outputs,
        targets,
        max_selections=4,
        input_h=32,
        line_width=12.0,
        min_valid_rows=2,
    )
    assert sequence.tolist() == [[1, 0, 3, -100]]


def test_pointer_loss_backpropagates_through_sequence_and_unary_quality() -> None:
    pointer_logits = torch.randn(1, 4, 4, requires_grad=True)
    unary_logits = torch.randn(1, 3, requires_grad=True)
    outputs = {
        "selection_pointer_logits": pointer_logits,
        "selection_pointer_teacher_indices": torch.tensor([[0, 3, -100, -100]]),
        "selection_logits": unary_logits,
        "pred_x_rows": torch.tensor([[[10.0] * 8, [30.0] * 8, [50.0] * 8]]),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[10.0] * 8]),
            "valid_mask": torch.ones(1, 8, dtype=torch.bool),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            input_h=32,
            input_w=64,
            set_selection_line_width=12.0,
            set_selection_min_valid_rows=2,
            w_pointer_selection=1.0,
            pointer_quality_weight=0.5,
        )
    )
    losses = criterion.compute_pointer_selection_loss(outputs, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert pointer_logits.grad is not None
    assert unary_logits.grad is not None
    assert bool(torch.isfinite(pointer_logits.grad).all())
    assert bool(torch.isfinite(unary_logits.grad).all())


def test_pointer_postprocess_obeys_stop_instead_of_filling_top4() -> None:
    pred_x = torch.stack(
        [
            torch.full((8,), 10.0),
            torch.full((8,), 30.0),
            torch.full((8,), 50.0),
        ]
    ).unsqueeze(0)
    outputs = {
        "exist_logits": torch.zeros(1, 3, 2),
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.0, 1.0]] * 3]),
        "selection_pointer_indices": torch.tensor([[2, -1, -1, -1]]),
        "selection_pointer_scores": torch.tensor([[0.9, 0.0, 0.0, 0.0]]),
    }
    lanes = predictions_to_lanes(
        outputs,
        score_thresh=0.0,
        input_w=64,
        input_h=32,
        min_pred_points=2,
        top_k=4,
        score_mode="pointer",
    )[0]
    assert len(lanes) == 1
    assert sum(x for x, _y in lanes[0]) / len(lanes[0]) == 50.0


def test_teacher_forced_pointer_loss_cannot_leak_into_geometry_features() -> None:
    torch.manual_seed(709)
    head = StructuredLaneQueryHead(
        dim=32,
        num_instances=4,
        num_rows=8,
        x_bins=16,
        input_w=64,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.0,
        evidence_x_bins=12,
        row_reference={
            "enabled": True,
            "prediction_mode": "bounded_delta",
            "detach_between_layers": True,
            "offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
            "delta_offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
            "initial_prior_sigma_px": 16.0,
        },
        lane_state={
            "enabled": True,
            "mode": "causal_set",
            "single_logit_score": True,
            "detach_score_geometry": True,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "semantic_context": {
                "enabled": True,
                "scales": ["p4", "p5"],
                "pool_size": [2, 3],
            },
        },
        set_selection={
            "enabled": True,
            "unified_score": True,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "curve_samples": 4,
            "detach_geometry_features": True,
            "use_curve_evidence": True,
            "use_semantic_decision": True,
            "candidate_interaction": "sequential_pointer",
            "pointer_max_selections": 4,
            "pointer_min_valid_rows": 2,
        },
    ).train()

    class TinyModel(nn.Module):
        def __init__(self, structured: StructuredLaneQueryHead) -> None:
            super().__init__()
            self.structured_query_head = structured

    model = TinyModel(head)
    p2 = torch.randn(1, 32, 8, 12, requires_grad=True)
    p4 = torch.randn(1, 32, 4, 6, requires_grad=True)
    p5 = torch.randn(1, 32, 2, 3, requires_grad=True)
    outputs = head(p2, multi_scale_features={"p4": p4, "p5": p5})
    targets = [
        {
            "x_rows": torch.tensor([[20.0] * 8]),
            "valid_mask": torch.ones(1, 8, dtype=torch.bool),
        }
    ]
    cfg = {
        "model": {"input_h": 32},
        "loss": {
            "input_h": 32,
            "set_selection_line_width": 12.0,
            "set_selection_min_valid_rows": 2,
        },
    }
    _apply_pointer_teacher_forcing(model, outputs, targets, cfg)
    criterion = S0Criterion(
        LossConfig(
            input_h=32,
            input_w=64,
            set_selection_line_width=12.0,
            set_selection_min_valid_rows=2,
            w_pointer_selection=1.0,
        )
    )
    criterion.compute_pointer_selection_loss(outputs, targets)["total"].backward()
    assert p2.grad is None
    assert p4.grad is None
    assert p5.grad is None
    assert all(layer.weight.grad is None for layer in head.row_delta_heads)
    assert head.set_selection_head.pointer_query.weight.grad is not None
    assert head.lane_state_layers[-1].semantic_attention[
        "p4"
    ].in_proj_weight.grad is not None


def test_v4_3_config_is_one_frozen_geometry_pointer_arm() -> None:
    cfg = load_config(CONFIG)
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    training = cfg["training"]
    assert selection["candidate_interaction"] == "sequential_pointer"
    assert selection["detach_geometry_features"] is True
    assert selection["pointer_max_selections"] == 4
    assert loss["w_pointer_selection"] == 1.0
    assert loss["w_set_selection"] == 0.0
    assert cfg["postprocess"]["score_mode"] == "pointer"
    assert training["checkpoint_interval"] == 0
    assert training["checkpoint_include_optimizer"] is False
