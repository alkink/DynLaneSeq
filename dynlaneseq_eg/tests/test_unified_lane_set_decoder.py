from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.modeling.unified_lane_set import UnifiedLaneSetLayer


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml"
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
        exist_prior_prob=0.01,
        row_reference={
            "enabled": True,
            "detach_between_layers": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
        lane_state={
            "enabled": True,
            "mode": "causal_set",
            "single_logit_score": True,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "semantic_context": {
                "enabled": True,
                "scales": ["p4", "p5"],
                "pool_size": [2, 3],
            },
        },
    )


def _features(batch: int = 2) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p2 = torch.randn(batch, 32, 8, 12, requires_grad=True)
    pyramid = {
        "p4": torch.randn(batch, 32, 4, 6, requires_grad=True),
        "p5": torch.randn(batch, 32, 2, 3, requires_grad=True),
    }
    return p2, pyramid


def test_lane_set_attention_makes_candidate_decisions_non_independent() -> None:
    torch.manual_seed(401)
    layer = UnifiedLaneSetLayer(
        dim=16,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    state = torch.randn(1, 4, 16)
    changed = state.clone()
    changed[:, 0, 0] += 3.0
    base = layer.prepare(state)
    perturbed = layer.prepare(changed)
    assert not torch.allclose(base[:, 1:], perturbed[:, 1:])


def test_geometry_loss_reaches_the_lane_set_state_through_row_injection() -> None:
    torch.manual_seed(403)
    head = _head().train()
    p2, pyramid = _features()
    output = head(p2, multi_scale_features=pyramid)
    output["pred_x_rows"].mean().backward()

    layer = head.lane_state_layers[-1]
    assert isinstance(layer, UnifiedLaneSetLayer)
    assert layer.lane_to_rows.weight.grad is not None
    assert float(layer.lane_to_rows.weight.grad.abs().sum()) > 0.0
    assert layer.set_attention.in_proj_weight.grad is not None
    assert float(layer.set_attention.in_proj_weight.grad.abs().sum()) > 0.0
    # Coarse semantics is a score-only view and cannot blur P2 geometry.
    assert pyramid["p4"].grad is None
    assert layer.semantic_attention["p4"].in_proj_weight.grad is None


def test_foreground_loss_reaches_rows_set_competition_and_coarse_context() -> None:
    torch.manual_seed(405)
    head = _head().train()
    p2, pyramid = _features()
    output = head(p2, multi_scale_features=pyramid)
    lane_logit = output["exist_logits"][..., 0] - output["exist_logits"][..., 1]
    lane_logit.square().mean().backward()

    layer = head.lane_state_layers[-1]
    assert isinstance(layer, UnifiedLaneSetLayer)
    assert layer.rows_to_lane.in_proj_weight.grad is not None
    assert float(layer.rows_to_lane.in_proj_weight.grad.abs().sum()) > 0.0
    assert layer.semantic_attention["p4"].in_proj_weight.grad is not None
    assert float(layer.semantic_attention["p4"].in_proj_weight.grad.abs().sum()) > 0.0
    assert p2.grad is not None and float(p2.grad.abs().sum()) > 0.0
    assert pyramid["p4"].grad is not None
    assert float(pyramid["p4"].grad.abs().sum()) > 0.0


def test_single_score_logit_has_one_exact_probability_everywhere() -> None:
    torch.manual_seed(407)
    head = _head().eval()
    p2, pyramid = _features(batch=1)
    output = head(p2, multi_scale_features=pyramid)
    score = output["score_logits"]
    probability = torch.softmax(output["exist_logits"], dim=-1)[..., 0]
    torch.testing.assert_close(probability, torch.sigmoid(score))
    torch.testing.assert_close(
        output["exist_logits"][..., 1],
        torch.zeros_like(score),
    )


def test_count_and_margin_losses_penalize_duplicate_foreground_scores() -> None:
    criterion = S0Criterion(
        LossConfig(score_margin=0.5, score_margin_topk_negatives=2)
    )
    good_score = torch.tensor([[4.0, -3.0, -4.0, -5.0]])
    duplicate_score = torch.tensor([[4.0, 3.5, 3.0, -5.0]])

    def outputs(score: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "exist_logits": torch.stack((score, torch.zeros_like(score)), dim=-1)
        }

    targets = [{"x_rows": torch.zeros(1, 8)}]
    matches = [
        {
            "pred_indices": torch.tensor([0]),
            "gt_indices": torch.tensor([0]),
        }
    ]
    assert criterion.compute_cardinality_loss(
        outputs(good_score), targets
    ) < criterion.compute_cardinality_loss(outputs(duplicate_score), targets)
    assert criterion.compute_score_margin_loss(
        outputs(good_score), matches
    ) < criterion.compute_score_margin_loss(outputs(duplicate_score), matches)


def test_v3_config_has_one_causal_train_deploy_contract() -> None:
    cfg = load_config(CONFIG)
    structured = cfg["model"]["structured_query"]
    state = structured["lane_state"]
    assert structured["num_instances"] == 32
    assert structured["num_groups"] == 1
    assert not structured.get("training_auxiliary_group_sizes")
    assert state["mode"] == "causal_set"
    assert state["single_logit_score"] is True
    assert state["semantic_context"]["scales"] == ["p4", "p5"]
    assert cfg["loss"]["w_cardinality"] == 0.10
    assert cfg["loss"]["w_score_margin"] == 0.25
    assert cfg["loss"]["w_quality"] == 0.0
    assert cfg["loss"]["w_set_selection"] == 0.0
    assert cfg["postprocess"]["score_mode"] == "exist"
    assert cfg["postprocess"]["lane_nms_distance_thresh_px"] == 0.0
