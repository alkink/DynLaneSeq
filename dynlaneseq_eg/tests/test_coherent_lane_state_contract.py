from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.train_one_epoch import forward_with_matches
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0, MatcherConfig
from dynlaneseq_eg.modeling.structured_queries import (
    PersistentLaneStateLayer,
    StructuredLaneQueryHead,
)
from dynlaneseq_eg.tools.analyze_oracle_topk import _resolve_operating_points
from dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability import (
    _deployment_scores,
)
from dynlaneseq_eg.tools.summarize_coherent_lane_state_gate import summarize


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_coherent_lane_state_25k.yaml"
)


def _head() -> StructuredLaneQueryHead:
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
            "detach_between_layers": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
        lane_state={
            "enabled": True,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
        },
    )


def test_persistent_lane_state_updates_each_query_from_only_its_rows() -> None:
    torch.manual_seed(201)
    layer = PersistentLaneStateLayer(
        dim=16,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    lane_state = torch.randn(2, 3, 16, requires_grad=True)
    row_states = torch.randn(2, 3, 5, 16, requires_grad=True)
    output = layer(lane_state, row_states)

    assert output.shape == lane_state.shape
    output.square().mean().backward()
    assert lane_state.grad is not None
    assert row_states.grad is not None
    assert float(row_states.grad.abs().sum()) > 0.0


def test_coherent_head_uses_persistent_state_and_detaches_next_reference() -> None:
    torch.manual_seed(203)
    head = _head().train()
    features = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = head(features)

    assert output["queries"].shape == (2, 4, 32)
    assert len(output["aux_outputs"]) == 1
    # The first block receives an image-conditioned differentiable reference.
    assert output["aux_outputs"][0]["input_reference_x_rows"].requires_grad
    # The second block receives the previous prediction as a detached sampling
    # reference, while the previous block retains its own auxiliary loss.
    assert not output["input_reference_x_rows"].requires_grad

    output["exist_logits"].square().mean().backward()
    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0.0
    assert head.lane_state_layers[-1].cross_attn.in_proj_weight.grad is not None
    assert float(
        head.lane_state_layers[-1].cross_attn.in_proj_weight.grad.abs().sum()
    ) > 0.0
    # Direct score supervision reads the same row states but does not take a
    # hidden shortcut through the coordinate-distribution prediction head.
    assert head.row_x.weight.grad is None


def test_coherent_contract_runs_independent_deep_supervision_backward() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = _head()

        def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.head(features)

    torch.manual_seed(205)
    model = TinyModel().train()
    matcher = HungarianMatcherS0(
        MatcherConfig(
            assignment="hungarian",
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
            w_line_iou=2.0,
            w_quality=0.0,
            w_row_dfl=0.5,
            input_w=64,
            input_h=64,
            line_iou_radius=15.0,
            lambda_intermediate=0.5,
            intermediate_layer_weights=(1.0,),
        ),
        matcher=matcher,
    )
    x_rows = torch.stack(
        (torch.linspace(12.0, 20.0, 8), torch.linspace(48.0, 40.0, 8)),
        dim=0,
    )
    targets = [
        {
            "x_rows": x_rows.clone(),
            "valid_mask": torch.ones_like(x_rows, dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 63.0], [0.0, 63.0]]),
        }
        for _ in range(2)
    ]
    outputs, matches = forward_with_matches(
        model,
        torch.randn(2, 32, 8, 12),
        targets,
        matcher,
        {
            "model": {"name": "DynLaneSeqS0"},
            "matcher": {"reuse_final_assignment_for_intermediate": False},
        },
        iteration=100,
    )
    assert len(matches) == 2
    assert len(outputs["_aux_matches"]) == 1
    assert outputs["_aux_matches"][0] is not matches
    criterion.set_iteration(100)
    losses = criterion(outputs, targets, matches)
    losses["loss_total"].backward()
    assert torch.isfinite(losses["loss_total"])
    assert model.head.instance_tokens.weight.grad is not None
    assert float(model.head.instance_tokens.weight.grad.abs().sum()) > 0.0


def test_coherent_lane_state_config_has_one_train_deploy_contract() -> None:
    cfg = load_config(CONFIG)
    structured = cfg["model"]["structured_query"]
    row_reference = structured["row_reference"]

    assert structured["num_instances"] == 32
    assert structured["num_groups"] == 1
    assert not structured.get("training_auxiliary_group_sizes")
    assert structured["lane_state"]["enabled"] is True
    assert row_reference["enabled"] is True
    assert row_reference["detach_between_layers"] is True
    assert structured["set_selection"]["enabled"] is False
    assert cfg["matcher"]["assignment"] == "hungarian"
    assert cfg["matcher"]["num_groups"] == 1
    assert cfg["matcher"]["lambda_obj"] == 0.5
    assert cfg["matcher"]["object_cost_type"] == "neg_probability"
    assert cfg["matcher"]["reuse_final_assignment_for_intermediate"] is False
    assert cfg["loss"]["w_exist"] == 2.0
    assert cfg["loss"]["w_quality"] == 0.0
    assert cfg["loss"]["w_set_selection"] == 0.0
    assert cfg["postprocess"]["score_mode"] == "exist"
    assert cfg["postprocess"]["lane_nms_distance_thresh_px"] == 0.0
    assert cfg["postprocess"]["top_k"] == 4
    assert cfg["scheduler"]["total_iters"] == 278000
    assert cfg["training"]["max_iters"] == 25000


def test_direct_existence_diagnostic_score_ignores_quality() -> None:
    stage = {
        "exist_logits": torch.tensor([[-1.0, 1.0], [2.0, -2.0]]),
        "quality_logits": torch.tensor([100.0, -100.0]),
    }
    score = _deployment_scores(
        stage,
        score_mode="exist",
        quality_power=0.5,
    )
    torch.testing.assert_close(
        score,
        torch.softmax(stage["exist_logits"], dim=-1)[:, 0],
    )


def test_threshold_free_diagnostic_operating_point_accepts_minus_one() -> None:
    assert _resolve_operating_points([], [0.0], [-1.0]) == [(0.0, -1.0)]


def _ranking_row(
    strategy: str,
    threshold: float,
    *,
    base: float,
    candidate: float,
    top_k: int = 4,
    quality_power: float | None = None,
    score_threshold: float | None = None,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "top_k": top_k,
        "iou_threshold": threshold,
        "quality_power": quality_power,
        "score_threshold": score_threshold,
        "base_recall": base,
        "candidate_recall": candidate,
        "delta_recall_points": 100.0 * (candidate - base),
    }


def test_coherent_gate_promotes_capacity_safe_direct_score_gain() -> None:
    rows: list[dict[str, object]] = []
    for threshold in (0.5, 0.75):
        rows.extend(
            (
                _ranking_row(
                    "all_raw",
                    threshold,
                    base=0.80,
                    candidate=0.80,
                    top_k=0,
                ),
                _ranking_row(
                    "oracle_topk",
                    threshold,
                    base=0.76,
                    candidate=0.77,
                ),
                _ranking_row(
                    "model_topk",
                    threshold,
                    base=0.65,
                    candidate=0.67,
                    quality_power=0.0,
                ),
                _ranking_row(
                    "model_topk_nms",
                    threshold,
                    base=0.68,
                    candidate=0.675,
                    quality_power=0.0,
                    score_threshold=-1.0,
                ),
            )
        )
    ranking = {
        "comparability": {"all_checks_pass": True},
        "rows": rows,
    }
    ownership = {
        "score_mode": "exist",
        "gate": {
            "weighted_consecutive_training_owner_retention_recoverable": 0.80,
        },
    }
    calibration_metadata = {
        "split": "val",
        "list_sha256": "fixed",
        "max_batches": 64,
        "num_records": 256,
        "iou_space": "official_raster",
        "sample_strategy": "uniform",
        "sampled_dataset_indices": list(range(256)),
        "nms_distance_thresh_px": 0.0,
    }

    def calibration(f1_050: float, f1_075: float) -> dict[str, object]:
        rows = []
        for threshold, f1 in ((0.5, f1_050), (0.75, f1_075)):
            rows.append(
                {
                    "stage": "main",
                    "strategy": "model_topk_nms",
                    "top_k": 4,
                    "iou_threshold": threshold,
                    "quality_power": 0.0,
                    "score_threshold": 0.20,
                    f"official_iou_{threshold:g}": {
                        "f1": f1,
                        "precision": f1 + 0.05,
                        "recall": f1 - 0.05,
                        "tp": 100,
                        "fp": 20,
                        "fn": 30,
                    },
                }
            )
        return {"metadata": dict(calibration_metadata), "rows": rows}

    result = summarize(
        ranking,
        ownership,
        calibration(0.70, 0.55),
        calibration(0.72, 0.56),
    )
    assert result["verdict"] == "positive_continue_training"
    assert (
        result["aggregate"]["mean_nms_dependency_reduction_points"] > 0.0
    )
