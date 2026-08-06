from __future__ import annotations

import json
import sys

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import build_pointer_cluster_soft_targets
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead
from dynlaneseq_eg.tools.summarize_v4_6_remaining_cluster_mixture_gate import (
    main as summarize_gate,
)


CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_"
    "v4_6_remaining_cluster_mixture_pointer.yaml"
)


def _geometry() -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    rows = 8
    outputs = {
        "pred_x_rows": torch.tensor(
            [
                [
                    [10.0] * rows,
                    [10.5] * rows,
                    [30.0] * rows,
                    [50.0] * rows,
                    [50.4] * rows,
                ]
            ]
        ),
        "range_norm": torch.tensor([[[0.0, 1.0]] * 5]),
    }
    targets = [
        {
            "x_rows": torch.tensor(
                [[10.0] * rows, [50.0] * rows, [90.0] * rows]
            ),
            "valid_mask": torch.ones(3, rows, dtype=torch.bool),
        }
    ]
    return outputs, targets


def _teacher(target_mode: str) -> dict[str, torch.Tensor]:
    outputs, targets = _geometry()
    return build_pointer_cluster_soft_targets(
        outputs,
        targets,
        max_selections=4,
        input_h=32,
        line_width=12.0,
        min_valid_rows=2,
        representable_min=0.20,
        support_quality_delta=0.10,
        temperature=0.03,
        base_seed=3407,
        iteration=105000,
        visit=0,
        target_mode=target_mode,
    )


def _selector() -> SetAwareLaneSelectionHead:
    return SetAwareLaneSelectionHead(
        8,
        input_w=100,
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
        pointer_teacher_mode="cluster_soft_remaining_mixture",
        row_grid_mode="fixed_rows",
    )


def test_v4_6_config_changes_only_the_teacher_probability_contract() -> None:
    baseline = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_unified_lane_set_"
        "v4_5_cluster_soft_pointer.yaml"
    )
    cfg = load_config(CONFIG)
    baseline_selection = baseline["model"]["structured_query"]["set_selection"]
    selection = cfg["model"]["structured_query"]["set_selection"]

    assert selection["pointer_teacher_mode"] == (
        "cluster_soft_remaining_mixture"
    )
    for name in (
        "pointer_cluster_representable_min",
        "pointer_cluster_quality_delta",
        "pointer_cluster_temperature",
        "row_grid_mode",
        "candidate_interaction",
        "detach_geometry_features",
    ):
        assert selection[name] == baseline_selection[name]
    assert cfg["loss"]["pointer_quality_weight"] == 0.10
    assert cfg["loss"]["pointer_stop_weight"] == 1.0
    assert cfg["training"]["max_iters"] == 5000
    assert cfg["training"]["checkpoint_interval"] == 2500


def test_remaining_cluster_mixture_preserves_teacher_prefix() -> None:
    sampled = _teacher("sampled_cluster")
    mixture = _teacher("remaining_cluster_mixture")

    # V4.6 changes the loss target, not the reproducibly sampled GRU prefix.
    torch.testing.assert_close(mixture["indices"], sampled["indices"])
    torch.testing.assert_close(mixture["active"], sampled["active"])
    torch.testing.assert_close(
        mixture["remaining_cluster_count"],
        torch.tensor([[2.0, 1.0, 0.0, 0.0]]),
    )

    first_target = mixture["probabilities"][0, 0]
    second_target = mixture["probabilities"][0, 1]
    # The two remaining GT clusters receive equal mass at step zero.
    torch.testing.assert_close(first_target[[0, 1]].sum(), torch.tensor(0.5))
    torch.testing.assert_close(first_target[[3, 4]].sum(), torch.tensor(0.5))
    assert float(first_target[2]) == 0.0
    assert float(first_target[5]) == 0.0

    first_choice = int(mixture["indices"][0, 0])
    if first_choice in {0, 1}:
        remaining_group = [3, 4]
        removed_group = [0, 1]
    else:
        remaining_group = [0, 1]
        removed_group = [3, 4]
    torch.testing.assert_close(
        second_target[remaining_group].sum(),
        torch.tensor(1.0),
    )
    assert float(second_target[removed_group].sum()) == 0.0


def test_mixture_removes_between_cluster_false_negative_gradient() -> None:
    target = _teacher("remaining_cluster_mixture")["probabilities"][0, 0]
    logits = torch.zeros_like(target, requires_grad=True)
    loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
    loss.backward()
    assert logits.grad is not None

    # At uniform logits each two-candidate cluster owns only 1/3 of the model
    # mass, while the target assigns it 1/2.  Both clusters therefore receive
    # an aggregate upward gradient instead of one being a random false negative.
    assert float(logits.grad[[0, 1]].sum()) < 0.0
    assert float(logits.grad[[3, 4]].sum()) < 0.0
    assert float(logits.grad[[2, 5]].sum()) > 0.0


def test_remaining_cluster_teacher_reroll_is_differentiable() -> None:
    torch.manual_seed(46)
    selector = _selector().train()
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    unary = torch.randn(1, 5, requires_grad=True)
    outputs = {
        "_selection_pointer_hidden": hidden,
        "_selection_pointer_relations": torch.zeros(1, 5, 5, 6),
        "_selection_pointer_unary_logits": unary,
        "_selection_pointer_candidate_valid": torch.ones(
            1, 5, dtype=torch.bool
        ),
        "selection_logits": unary,
    }
    teacher = _teacher("remaining_cluster_mixture")
    selector.reroll_pointer_with_cluster_teacher(outputs, teacher)
    logits = outputs["selection_pointer_logits"]
    target = outputs["selection_pointer_teacher_probabilities"]
    active = outputs["selection_pointer_teacher_active"]
    loss = -(
        target * torch.log_softmax(logits, dim=-1)
    ).sum(dim=-1)[active].mean()
    loss.backward()

    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    assert unary.grad is not None and bool(torch.isfinite(unary.grad).all())
    torch.testing.assert_close(
        outputs["selection_pointer_teacher_remaining_cluster_count"],
        teacher["remaining_cluster_count"],
    )


def test_v4_6_summary_requires_metric_and_teacher_policy_gain(
    tmp_path, monkeypatch
) -> None:
    def coverage(f1_050: float, f1_075: float) -> dict:
        def metric(f1: float) -> dict:
            return {
                "f1": f1,
                "precision": f1,
                "recall": f1,
                "mean_selected_per_image": 3.2,
                "false_positive_breakdown": {
                    "duplicate_fp": {"fraction_of_fp": 0.1}
                },
            }

        return {
            "methods": {
                "pointer_greedy": {
                    "0.50": metric(f1_050),
                    "0.75": metric(f1_075),
                }
            },
            "capacity": {
                "0.50": {"all_candidate_oracle": {"recall": 0.95}},
                "0.75": {"all_candidate_oracle": {"recall": 0.85}},
            },
        }

    def policy(step4_hit: float) -> dict:
        return {
            "teacher_contract": {
                "target_mode": "remaining_cluster_mixture"
            },
            "audit": {
                "teacher_prefix_policy": {
                    str(step): {
                        "candidate_support_hit_rate": (
                            step4_hit if step == 4 else 0.5
                        ),
                        "mean_probability_mass_on_candidate_support": 0.4,
                        "mean_soft_target_cross_entropy": 2.0,
                    }
                    for step in range(1, 5)
                },
                "teacher_support": {
                    "support_size": {"mean": 4.0},
                    "target_entropy": {"mean": 1.0},
                },
            },
        }

    paths = {
        "source": tmp_path / "source.json",
        "source_policy": tmp_path / "source_policy.json",
        "gradient": tmp_path / "gradient.json",
        "trajectory": tmp_path / "trajectory.json",
        "policy": tmp_path / "policy.json",
        "output": tmp_path / "summary.json",
    }
    payloads = {
        "source": coverage(0.820, 0.610),
        "source_policy": policy(0.40),
        "gradient": {"passed": True},
        "trajectory": coverage(0.824, 0.609),
        "policy": policy(0.52),
    }
    for name, payload in payloads.items():
        paths[name].write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--source",
            str(paths["source"]),
            "--source-policy",
            str(paths["source_policy"]),
            "--gradient-audit",
            str(paths["gradient"]),
            "--trajectory",
            f"107500={paths['trajectory']}",
            "--policy-trajectory",
            f"107500={paths['policy']}",
            "--output-json",
            str(paths["output"]),
        ],
    )
    summarize_gate()
    summary = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["passing_iterations"] == [107500]
