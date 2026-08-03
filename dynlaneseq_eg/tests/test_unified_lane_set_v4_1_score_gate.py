from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.frozen_training import (
    freeze_except_parameter_prefixes,
    set_frozen_detector_eval,
)
from dynlaneseq_eg.evaluation.candidate_diagnostics import stage_scores
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tools.summarize_v4_1_score_gate import summarize


CONFIG_PREFIX = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_"
)


def _head(*, interaction: str = "transformer") -> StructuredLaneQueryHead:
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
            "candidate_interaction": interaction,
        },
    )


def test_v4_1_selector_observes_geometry_without_leaking_gradient() -> None:
    torch.manual_seed(503)
    head = _head().train()
    assert head.set_selection_head is not None
    with torch.no_grad():
        head.set_selection_head.output.weight.fill_(0.02)
    p2 = torch.randn(2, 32, 8, 12, requires_grad=True)
    p4 = torch.randn(2, 32, 4, 6, requires_grad=True)
    p5 = torch.randn(2, 32, 2, 3, requires_grad=True)
    output = head(p2, multi_scale_features={"p4": p4, "p5": p5})
    output["selection_logits"].sum().backward()

    assert p2.grad is None
    assert p4.grad is None
    assert p5.grad is None
    assert all(layer.weight.grad is None for layer in head.row_delta_heads)
    geometry = head.lane_state_layers[-1]
    assert geometry.lane_to_rows.weight.grad is None
    assert geometry.rows_to_lane.in_proj_weight.grad is None
    # Semantic score adapters receive only detached lane/FPN inputs and remain
    # safe to fine-tune in a later score-only phase.
    assert geometry.semantic_attention["p4"].in_proj_weight.grad is not None
    assert head.set_selection_head.output.weight.grad is not None


def test_independent_arm_has_no_candidate_axis_communication() -> None:
    torch.manual_seed(505)
    selector = _head(interaction="independent").set_selection_head
    assert selector is not None
    with torch.no_grad():
        selector.output.weight.normal_(std=0.05)
    feature_dim = int(selector.input_norm.normalized_shape[0])
    features = torch.randn(2, 4, feature_dim)
    changed = features.clone()
    changed[:, 1:] = changed[:, 1:] + 100.0
    original_logits = selector.score_selection_features(features)
    changed_logits = selector.score_selection_features(changed)
    torch.testing.assert_close(original_logits[:, 0], changed_logits[:, 0])


def test_frozen_training_keeps_only_selector_live_and_detector_in_eval() -> None:
    class TinyDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
            self.structured_query_head = nn.Module()
            self.structured_query_head.set_selection_head = nn.Sequential(
                nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 1)
            )

    model = TinyDetector()
    stats = freeze_except_parameter_prefixes(
        model,
        ("structured_query_head.set_selection_head",),
    )
    assert stats["trainable_tensor_count"] == 4
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.structured_query_head.set_selection_head.parameters()
    )

    model.train()
    set_frozen_detector_eval(
        model,
        ("structured_query_head.set_selection_head",),
    )
    assert model.backbone.training is False
    assert model.structured_query_head.set_selection_head.training is True


def test_v4_1_configs_encode_the_intended_two_by_two_gate() -> None:
    configs = {
        "a": load_config(CONFIG_PREFIX + "a_mlp_shared.yaml"),
        "b": load_config(CONFIG_PREFIX + "b_mlp_unique.yaml"),
        "c": load_config(CONFIG_PREFIX + "c_set_shared.yaml"),
        "d": load_config(CONFIG_PREFIX + "d_set_unique.yaml"),
    }
    expected = {
        "a": ("independent", True, 1.0, 0.0),
        "b": ("independent", False, 2.0, 0.25),
        "c": ("transformer", True, 1.0, 0.0),
        "d": ("transformer", False, 2.0, 0.25),
    }
    for name, cfg in configs.items():
        selection = cfg["model"]["structured_query"]["set_selection"]
        loss = cfg["loss"]
        assert (
            selection["candidate_interaction"],
            loss["set_selection_share_matcher_assignment"],
            loss["set_selection_negative_weight"],
            loss["set_selection_rank_weight"],
        ) == expected[name]
        assert selection["detach_geometry_features"] is True
        assert cfg["postprocess"]["score_mode"] == "selection"
        assert cfg["training"]["frozen_detector_eval"] is True
        assert cfg["training"]["trainable_parameter_prefixes"] == [
            "structured_query_head.set_selection_head"
        ]
        assert cfg["training"]["checkpoint_model_prefixes"] == [
            "structured_query_head.set_selection_head"
        ]
        assert cfg["training"]["checkpoint_include_optimizer"] is False
        assert cfg["training"]["save_last_alias"] is False
        assert cfg["training"]["checkpoint_interval"] == 0
        assert cfg["loss"]["w_set_selection"] == 1.0
        assert cfg["loss"]["w_exist"] == 0.0


def test_candidate_diagnostics_can_read_selection_score() -> None:
    stage = {
        "pred_x_rows": torch.zeros(2, 8),
        "exist_logits": torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
        "selection_logits": torch.tensor([-2.0, 2.0]),
    }
    score = stage_scores(stage, score_mode="selection")
    torch.testing.assert_close(score, torch.sigmoid(stage["selection_logits"]))
    assert float(score[1]) > float(score[0])


def _report(
    *,
    recall_050: float,
    recall_075: float,
    f1_050: float,
    ap_050: float,
    mass: float,
) -> dict[str, object]:
    def row(recall: float, f1: float) -> dict[str, object]:
        return {
            "precision": f1,
            "recall": recall,
            "f1": f1,
            "false_positive_breakdown": {
                "duplicate_fp": {"fraction_of_fp": 0.25}
            },
        }

    return {
        "methods": {
            "score_top4": {
                "0.50": row(recall_050, f1_050),
                "0.75": row(recall_075, recall_075),
            }
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.94}},
            "0.75": {"all_candidate_oracle": {"recall": 0.83}},
        },
        "score_diagnostics": {
            "score_mode": "selection",
            "unique_candidate_ap": {"0.50": ap_050, "0.75": ap_050 - 0.1},
            "mean_foreground_probability_mass": mass,
            "mean_target_lane_count": 3.45,
        },
    }


def test_v4_1_summary_requires_geometry_invariance_and_selection_recovery() -> None:
    source = _report(
        recall_050=0.50,
        recall_075=0.40,
        f1_050=0.48,
        ap_050=0.24,
        mass=7.3,
    )
    reports = {
        "source_v4": source,
        "a_mlp_shared": _report(
            recall_050=0.56, recall_075=0.45, f1_050=0.54, ap_050=0.30, mass=6.0
        ),
        "b_mlp_unique": _report(
            recall_050=0.63, recall_075=0.52, f1_050=0.61, ap_050=0.42, mass=4.5
        ),
        "c_set_shared": _report(
            recall_050=0.66, recall_075=0.56, f1_050=0.64, ap_050=0.46, mass=4.8
        ),
        "d_set_unique": _report(
            recall_050=0.75, recall_075=0.65, f1_050=0.70, ap_050=0.56, mass=3.7
        ),
    }
    payload = summarize(reports)
    assert payload["geometry_invariance"]["all_preserved"] is True
    assert payload["combined_gate"]["passed"] is True
    assert payload["best_diagnostic_arm"] == "d_set_unique"
    assert payload["factor_effects"]["set_interaction_with_unique_loss"][
        "recall_050"
    ] > 0.0
