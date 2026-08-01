from __future__ import annotations

from copy import deepcopy

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0, MatcherConfig
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tests.test_train_many_infer_one_gate import _report
from dynlaneseq_eg.tools.summarize_hybrid_primary_auxiliary_gate import summarize


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_slots32_aux3x8_b4x4_1600x640_"
    "bins800_fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml"
)
UNIFIED_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml"
)


def _head(*, with_training_auxiliary: bool = True) -> StructuredLaneQueryHead:
    return StructuredLaneQueryHead(
        dim=32,
        num_instances=8,
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
        training_auxiliary_group_sizes=[2, 2] if with_training_auxiliary else None,
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
    )


def _targets(batch: int = 2) -> list[dict[str, torch.Tensor]]:
    x_rows = torch.stack(
        (
            torch.linspace(12.0, 20.0, 8),
            torch.linspace(48.0, 40.0, 8),
        ),
        dim=0,
    )
    return [
        {
            "x_rows": x_rows.clone(),
            "valid_mask": torch.ones_like(x_rows, dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 63.0], [0.0, 63.0]]),
        }
        for _ in range(batch)
    ]


def test_hybrid_config_contract() -> None:
    cfg = load_config(CONFIG)
    structured = cfg["model"]["structured_query"]
    assert structured["num_instances"] == 32
    assert structured["num_groups"] == 1
    assert structured["training_auxiliary_group_sizes"] == [8, 8, 8]
    assert structured["row_reference"]["enabled"] is True
    assert structured["intermediate_supervision"] is True
    assert cfg["matcher"]["assignment"] == "hungarian"
    assert cfg["matcher"]["num_groups"] == 1
    assert cfg["matcher"]["lambda_obj"] == 0.5
    assert cfg["loss"]["lambda_training_auxiliary"] == 0.5
    assert cfg["scheduler"]["total_iters"] == 278000
    assert cfg["training"]["seed"] == 3407


def test_unified_selection_config_has_one_train_deploy_contract() -> None:
    cfg = load_config(UNIFIED_CONFIG)
    structured = cfg["model"]["structured_query"]
    selection = structured["set_selection"]

    assert structured["num_instances"] == 32
    assert structured["training_auxiliary_group_sizes"] == [8, 8, 8]
    assert structured["lane_pooling"] == "mean"
    assert selection["unified_score"] is True
    assert selection["use_curve_evidence"] is True
    assert cfg["matcher"]["cost_type"] == "range_aware_iou"
    assert cfg["matcher"]["reuse_final_assignment_for_intermediate"] is True
    assert cfg["loss"]["set_selection_share_matcher_assignment"] is True
    assert cfg["loss"]["w_exist"] == 0.0
    assert cfg["loss"]["w_quality"] == 0.0
    assert cfg["loss"]["set_selection_positive_floor"] == 0.5
    assert cfg["postprocess"]["score_mode"] == "selection"
    assert cfg["postprocess"]["lane_nms_distance_thresh_px"] == 0.0
    assert cfg["scheduler"]["total_iters"] == 10000


def test_hybrid_training_keeps_full_primary_set_and_hides_auxiliary_at_inference() -> None:
    torch.manual_seed(43)
    head = _head().eval()
    features = torch.randn(2, 32, 8, 12)
    with torch.inference_mode():
        training_view = head(features, inference_only=False)
        inference_view = head(features, inference_only=True)

    assert head.primary_num_instances == 8
    assert head.num_instances == 12
    assert head.interaction_group_sizes == (8, 2, 2)
    assert training_view["pred_x_rows"].shape == (2, 8, 8)
    assert training_view["_training_auxiliary_outputs"]["pred_x_rows"].shape == (
        2,
        4,
        8,
    )
    assert training_view["_training_auxiliary_group_sizes"] == (2, 2)
    assert len(training_view["aux_outputs"]) == 1
    assert len(training_view["_training_auxiliary_aux_outputs"]) == 1
    for key in ("exist_logits", "pred_x_rows", "range_norm", "quality_logits"):
        torch.testing.assert_close(inference_view[key], training_view[key])


def test_hybrid_addition_preserves_every_shared_initial_parameter() -> None:
    torch.manual_seed(53)
    control = _head(with_training_auxiliary=False)
    torch.manual_seed(53)
    candidate = _head(with_training_auxiliary=True)

    control_state = control.state_dict()
    candidate_state = candidate.state_dict()
    auxiliary_keys = {
        "training_auxiliary_instance_tokens.weight",
        "training_auxiliary_reference_anchor_logits",
    }
    assert set(candidate_state) == set(control_state) | auxiliary_keys
    for key, value in control_state.items():
        torch.testing.assert_close(candidate_state[key], value, rtol=0.0, atol=0.0)


def test_matcher_supports_explicit_unequal_group_sizes() -> None:
    matcher = HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=0.0,
            lambda_point=1.0,
            lambda_range=0.0,
        )
    )
    target = _targets(batch=1)
    predictions = torch.stack(
        (
            torch.linspace(12.0, 20.0, 8),
            torch.linspace(48.0, 40.0, 8),
            torch.linspace(12.0, 20.0, 8),
            torch.linspace(48.0, 40.0, 8),
            torch.linspace(12.0, 20.0, 8),
            torch.linspace(48.0, 40.0, 8),
        ),
        dim=0,
    ).unsqueeze(0)
    outputs = {
        "exist_logits": torch.zeros(1, 6, 2),
        "pred_x_rows": predictions,
        "range_norm": torch.tensor([[[0.0, 1.0]]]).expand(1, 6, 2).clone(),
    }
    matches = matcher.match_many(
        (outputs,),
        target,
        assignment="grouped_one_to_many",
        group_sizes=(2, 2, 2),
    )[0][0]

    assert matches["pred_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert matches["gt_indices"].tolist() == [0, 1, 0, 1, 0, 1]


class _TinyHybridModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = _head()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(features)


def test_hybrid_auxiliary_loss_is_separate_and_reaches_auxiliary_tokens() -> None:
    torch.manual_seed(47)
    model = _TinyHybridModel().train()
    features = torch.randn(2, 32, 8, 12)
    targets = _targets(batch=2)
    matcher = HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=0.5,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            line_iou_radius=15.0,
            input_w=64,
            input_h=64,
            object_cost_type="neg_probability",
        )
    )
    criterion = S0Criterion(
        LossConfig(
            w_exist=2.0,
            w_point=5.0,
            w_range=1.0,
            w_smooth=0.0,
            w_line_iou=1.0,
            w_quality=1.0,
            input_w=64,
            input_h=64,
            line_iou_radius=15.0,
            lambda_intermediate=0.5,
            intermediate_layer_weights=(1.0,),
            lambda_training_auxiliary=0.5,
        ),
        matcher=matcher,
    )
    cfg = {"model": {"name": "DynLaneSeqS0"}}

    outputs, matches = forward_with_matches(
        model,
        features,
        targets,
        matcher,
        cfg,
        iteration=0,
    )
    losses = criterion(outputs, targets, matches)
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert losses["weight_training_auxiliary"].item() == 0.5
    assert losses["loss_training_auxiliary_total"].item() > 0.0
    assert len(outputs["_training_auxiliary_matches"]) == 2
    assert len(outputs["_training_auxiliary_aux_matches"]) == 1
    primary_token_grad = model.head.instance_tokens.weight.grad
    auxiliary_tokens = model.head.training_auxiliary_instance_tokens
    assert primary_token_grad is not None
    assert primary_token_grad.abs().sum().item() > 0.0
    assert auxiliary_tokens is not None
    assert auxiliary_tokens.weight.grad is not None
    assert auxiliary_tokens.weight.grad.abs().sum().item() > 0.0


def test_deep_supervision_reuses_final_primary_and_auxiliary_ownership() -> None:
    torch.manual_seed(59)
    model = _TinyHybridModel().train()
    features = torch.randn(2, 32, 8, 12)
    targets = _targets(batch=2)
    matcher = HungarianMatcherS0(
        MatcherConfig(
            input_w=64,
            input_h=64,
            cost_type="range_aware_iou",
            range_aware_line_width=30.0,
        )
    )
    cfg = {
        "model": {"name": "DynLaneSeqS0"},
        "matcher": {"reuse_final_assignment_for_intermediate": True},
    }

    outputs, matches = forward_with_matches(
        model,
        features,
        targets,
        matcher,
        cfg,
        iteration=0,
    )

    assert all(layer_matches is matches for layer_matches in outputs["_aux_matches"])
    auxiliary_matches = outputs["_training_auxiliary_matches"]
    assert all(
        layer_matches is auxiliary_matches
        for layer_matches in outputs["_training_auxiliary_aux_matches"]
    )


def test_unified_contract_runs_one_complete_training_step() -> None:
    torch.manual_seed(61)
    head = StructuredLaneQueryHead(
        dim=32,
        num_instances=8,
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
        training_auxiliary_group_sizes=[2, 2],
        lane_pooling="mean",
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
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
        },
    )

    class TinyUnified(torch.nn.Module):
        def __init__(self, structured_head: StructuredLaneQueryHead) -> None:
            super().__init__()
            self.head = structured_head

        def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.head(value)

    model = TinyUnified(head).train()
    features = torch.randn(2, 32, 8, 12, requires_grad=True)
    targets = _targets(batch=2)
    matcher = HungarianMatcherS0(
        MatcherConfig(
            input_w=64,
            input_h=64,
            cost_type="range_aware_iou",
            range_aware_line_width=30.0,
        )
    )
    criterion = S0Criterion(
        LossConfig(
            w_exist=0.0,
            w_quality=0.0,
            w_point=5.0,
            w_range=1.0,
            w_line_iou=1.0,
            w_set_selection=1.0,
            input_w=64,
            input_h=64,
            line_iou_radius=15.0,
            set_selection_share_matcher_assignment=True,
            set_selection_positive_floor=0.5,
            set_selection_negative_weight=0.25,
            lambda_intermediate=0.5,
            intermediate_layer_weights=(1.0,),
            lambda_training_auxiliary=0.5,
        ),
        matcher=matcher,
    )
    cfg = {
        "model": {"name": "DynLaneSeqS0"},
        "matcher": {"reuse_final_assignment_for_intermediate": True},
    }
    outputs, matches = forward_with_matches(
        model,
        features,
        targets,
        matcher,
        cfg,
        iteration=0,
    )
    losses = criterion(outputs, targets, matches)
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert float(losses["loss_set_selection"]) > 0.0
    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0.0
    assert head.set_selection_head is not None
    assert head.set_selection_head.output.weight.grad is not None
    auxiliary_tokens = head.training_auxiliary_instance_tokens
    assert auxiliary_tokens is not None
    assert auxiliary_tokens.weight.grad is not None
    assert float(auxiliary_tokens.weight.grad.abs().sum()) > 0.0


def test_hybrid_gate_requires_primary_geometry_and_strict_f1_signal() -> None:
    control = _report(32, "base")
    candidate = deepcopy(control)
    for row in candidate["rows"]:
        if row["strategy"] in {"all_raw", "oracle_topk", "quality_topk"}:
            row["recall"] = min(1.0, float(row["recall"]) + 0.005)
        if row["strategy"] == "model_topk_nms":
            key = f"official_iou_{float(row['iou_threshold']):g}"
            metric = row[key]
            gain = 0.002 if float(row["iou_threshold"]) == 0.5 else 0.01
            metric["f1"] = float(metric["f1"]) + gain

    payload = summarize(control, candidate)
    assert payload["gate"]["geometry_preserved"] is True
    assert payload["gate"]["selection_positive"] is True
    assert payload["gate"]["verdict"] == "positive_continue_same_checkpoint"
