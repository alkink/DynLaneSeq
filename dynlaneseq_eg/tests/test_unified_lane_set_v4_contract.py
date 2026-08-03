from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.factory import build_optimizer
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0, MatcherConfig
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml"
)


def _head(*, semantic: bool = False) -> StructuredLaneQueryHead:
    return StructuredLaneQueryHead(
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
        num_groups=1,
        intermediate_supervision=True,
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
                "enabled": semantic,
                "scales": ["p4", "p5"],
                "pool_size": [2, 3],
            },
        },
    )


def test_v4_uses_separate_non_affine_bounded_delta_heads() -> None:
    torch.manual_seed(401)
    head = _head().eval()
    assert head.row_norm is None
    assert head.row_x is None
    assert len(head.row_delta_heads) == 2
    assert len({id(module) for module in head.row_delta_heads}) == 2
    assert all(not norm.elementwise_affine for norm in head.row_delta_norms)
    assert all(layer.bias is None for layer in head.row_delta_heads)

    with torch.no_grad():
        for layer in head.row_delta_heads:
            layer.weight.normal_(std=0.1)
        output = head(torch.randn(2, 32, 8, 12))

    assert output["row_x_logits"].shape == (2, 4, 8, 5)
    assert len(output["aux_outputs"]) == 1
    for stage in (*output["aux_outputs"], output):
        motion = stage["pred_x_rows"] - stage["input_reference_x_rows"]
        assert float(motion.max()) <= 16.0 + 1e-5
        assert float(motion.min()) >= -16.0 - 1e-5
        torch.testing.assert_close(
            stage["row_x_offsets_px"],
            torch.tensor([-16.0, -8.0, 0.0, 8.0, 16.0]),
        )


def test_v4_score_gradient_cannot_enter_geometry_or_feature_pyramid() -> None:
    torch.manual_seed(403)
    head = _head(semantic=True).train()
    p2 = torch.randn(2, 32, 8, 12, requires_grad=True)
    p4 = torch.randn(2, 32, 4, 6, requires_grad=True)
    p5 = torch.randn(2, 32, 2, 3, requires_grad=True)
    output = head(p2, multi_scale_features={"p4": p4, "p5": p5})
    output["score_logits"].sum().backward()

    assert p2.grad is None
    assert p4.grad is None
    assert p5.grad is None
    assert head.exist[-1].weight.grad is not None
    assert float(head.exist[-1].weight.grad.abs().sum()) > 0.0
    assert head.decision_norm is not None
    assert head.decision_norm.weight.grad is not None
    assert all(layer.weight.grad is None for layer in head.row_delta_heads)
    geometry_layer = head.lane_state_layers[-1]
    assert geometry_layer.lane_to_rows.weight.grad is None
    assert geometry_layer.rows_to_lane.in_proj_weight.grad is None
    assert (
        geometry_layer.semantic_attention["p4"].in_proj_weight.grad is not None
    )


def test_v4_geometry_still_backpropagates_through_local_evidence() -> None:
    torch.manual_seed(405)
    head = _head().train()
    with torch.no_grad():
        for layer in head.row_delta_heads:
            layer.weight.normal_(std=0.01)
    features = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = head(features)
    loss = output["pred_x_rows"].mean()
    loss = loss + output["aux_outputs"][0]["pred_x_rows"].mean()
    loss.backward()

    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0.0
    assert head.row_delta_heads[0].weight.grad is not None
    assert head.row_delta_heads[1].weight.grad is not None
    assert head.lane_state_layers[0].lane_to_rows.weight.grad is not None


def test_v4_local_distribution_focal_loss_uses_reference_relative_target() -> None:
    logits = torch.zeros((1, 1, 4, 5), requires_grad=True)
    outputs = {
        "exist_logits": torch.zeros((1, 1, 2)),
        "pred_x_rows": torch.tensor([[[20.0, 24.0, 28.0, 32.0]]]),
        "range_norm": torch.tensor([[[0.0, 1.0]]]),
        "row_x_logits": logits,
        "row_x_offsets_px": torch.tensor([-16.0, -8.0, 0.0, 8.0, 16.0]),
        "input_reference_x_rows": torch.tensor([[[20.0, 20.0, 20.0, 20.0]]]),
    }
    targets = [
        {
            "x_rows": torch.tensor([[20.0, 24.0, 28.0, 36.0]]),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 63.0]]),
        }
    ]
    matches = [
        {"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}
    ]
    criterion = S0Criterion(LossConfig(input_w=64, w_row_dfl=1.0))
    loss = criterion.compute_row_dfl_loss(outputs, targets, matches)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_v4_iou_aware_score_target_is_detached_from_geometry() -> None:
    score = torch.zeros((1, 2), requires_grad=True)
    pred_x = torch.stack(
        (torch.full((8,), 20.0), torch.full((8,), 50.0)),
        dim=0,
    ).unsqueeze(0).requires_grad_()
    ranges = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True)
    outputs = {
        "score_logits": score,
        "exist_logits": torch.stack(
            (score, torch.zeros_like(score)),
            dim=-1,
        ),
        "pred_x_rows": pred_x,
        "range_norm": ranges,
    }
    targets = [
        {
            "x_rows": torch.full((1, 8), 20.0),
            "valid_mask": torch.ones((1, 8), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 63.0]]),
        }
    ]
    matches = [
        {"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}
    ]
    criterion = S0Criterion(
        LossConfig(
            input_w=64,
            input_h=64,
            exist_target_mode="iou_aware",
            exist_quality_floor=0.5,
            exist_quality_line_width=30.0,
            exist_quality_min_valid_rows=2,
        )
    )
    quality_target = criterion.compute_exist_quality_targets(
        outputs,
        targets,
        matches,
    )
    assert float(quality_target[0, 0]) > 0.99
    assert float(quality_target[0, 1]) == 0.0
    loss = criterion.compute_exist_loss(outputs, matches, targets)
    loss.backward()
    assert score.grad is not None
    assert float(score.grad.abs().sum()) > 0.0
    assert pred_x.grad is None
    assert ranges.grad is None


def test_v4_reuses_final_geometry_assignment_for_every_auxiliary_layer() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = _head()

        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.head(features)

    matcher = HungarianMatcherS0(
        MatcherConfig(
            assignment="hungarian",
            lambda_obj=0.0,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            input_w=64,
            input_h=64,
        )
    )
    target = {
        "x_rows": torch.stack(
            (torch.linspace(12.0, 20.0, 8), torch.linspace(48.0, 40.0, 8))
        ),
        "valid_mask": torch.ones((2, 8), dtype=torch.bool),
        "range_y": torch.tensor([[0.0, 63.0], [0.0, 63.0]]),
    }
    outputs, matches = forward_with_matches(
        TinyModel().train(),
        torch.randn(2, 32, 8, 12),
        [target, target],
        matcher,
        {
            "model": {"name": "DynLaneSeqS0"},
            "matcher": {"reuse_final_assignment_for_intermediate": True},
        },
        iteration=100,
    )
    assert len(outputs["_aux_matches"]) == 1
    assert outputs["_aux_matches"][0] is matches


def test_v4_complete_training_objective_is_finite_and_backpropagates() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = _head()

        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.head(features)

    torch.manual_seed(407)
    model = TinyModel().train()
    matcher = HungarianMatcherS0(
        MatcherConfig(
            assignment="hungarian",
            lambda_obj=0.0,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            line_iou_radius=15.0,
            input_w=64,
            input_h=64,
        )
    )
    criterion = S0Criterion(
        LossConfig(
            w_exist=2.0,
            w_point=5.0,
            w_range=1.0,
            w_line_iou=2.0,
            w_row_dfl=0.5,
            w_quality=0.0,
            w_seg=0.0,
            w_centerline=0.0,
            input_w=64,
            input_h=64,
            line_iou_radius=15.0,
            lambda_intermediate=0.5,
            intermediate_layer_weights=(1.0,),
            w_intermediate_exist=0.0,
            exist_target_mode="iou_aware",
            exist_quality_floor=0.5,
            exist_quality_line_width=30.0,
            exist_quality_min_valid_rows=2,
        ),
        matcher=matcher,
    )
    x_rows = torch.stack(
        (torch.linspace(12.0, 20.0, 8), torch.linspace(48.0, 40.0, 8))
    )
    target = {
        "x_rows": x_rows,
        "valid_mask": torch.ones_like(x_rows, dtype=torch.bool),
        "range_y": torch.tensor([[0.0, 63.0], [0.0, 63.0]]),
    }
    outputs, matches = forward_with_matches(
        model,
        torch.randn(2, 32, 8, 12),
        [target, target],
        matcher,
        {
            "model": {"name": "DynLaneSeqS0"},
            "matcher": {"reuse_final_assignment_for_intermediate": True},
        },
        iteration=100,
    )
    losses = criterion(outputs, [target, target], matches)
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()
    assert model.head.exist[-1].weight.grad is not None
    assert model.head.row_delta_heads[-1].weight.grad is not None
    assert model.head.row_delta_heads[0].weight.grad is not None


def test_v4_full_config_encodes_one_278k_stable_contract() -> None:
    cfg = load_config(CONFIG)
    structured = cfg["model"]["structured_query"]
    rowref = structured["row_reference"]
    loss = cfg["loss"]
    matcher = cfg["matcher"]

    assert rowref["prediction_mode"] == "bounded_delta"
    assert len(rowref["delta_offsets_px"]) == 33
    assert structured["lane_state"]["detach_score_geometry"] is True
    assert matcher["lambda_obj"] == 0.0
    assert matcher["reuse_final_assignment_for_intermediate"] is True
    assert loss["w_intermediate_exist"] == 0.0
    assert loss["exist_target_mode"] == "iou_aware"
    assert loss["w_cardinality"] == 0.0
    assert loss["w_score_margin"] == 0.0
    assert cfg["scheduler"]["total_iters"] == 278000
    assert cfg["training"]["max_iters"] == 278000

    class HeadContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.structured_query_head = _head(semantic=True)

    optimizer = build_optimizer(cfg, HeadContainer())
    lrs = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
    assert lrs["lane_state_decay"] == 5e-5
    assert lrs["coordinate_decay"] == 5e-5
    assert lrs["lane_identity_decay"] == 5e-5
    assert lrs["evidence_decay"] == 1e-4
