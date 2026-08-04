from __future__ import annotations

import torch
import torch.nn.functional as F

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.structured_queries import SetAwareLaneSelectionHead


Q1_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_"
    "v4_5_1_q1_quality_policy_split.yaml"
)
Q2_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_"
    "v4_5_1_q2_quality_policy_listwise.yaml"
)


def _selector(mode: str) -> SetAwareLaneSelectionHead:
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
        pointer_teacher_mode="cluster_soft_randomized",
        row_grid_mode="fixed_rows",
        pointer_quality_policy_mode=mode,
        pointer_quality_prior_max_scale=2.0,
    )


def _copy_common_state(
    source: SetAwareLaneSelectionHead,
    target: SetAwareLaneSelectionHead,
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    compatible = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    target.load_state_dict(compatible, strict=False)


def test_v4_5_1_configs_change_only_split_then_listwise() -> None:
    q1 = load_config(Q1_CONFIG)
    q2 = load_config(Q2_CONFIG)
    q1_selection = q1["model"]["structured_query"]["set_selection"]
    q2_selection = q2["model"]["structured_query"]["set_selection"]
    assert q1_selection["pointer_quality_policy_mode"] == "decoupled"
    assert q2_selection["pointer_quality_policy_mode"] == "decoupled"
    assert q1_selection["pointer_quality_prior_max_scale"] == 2.0
    assert q1["loss"]["pointer_cluster_listwise_weight"] == 0.0
    assert q2["loss"]["pointer_cluster_listwise_weight"] == 0.25
    assert q1["loss"]["pointer_unary_target_mode"] == "max_quality"
    assert q2["loss"]["pointer_quality_weight"] == 0.10


def test_decoupled_pointer_is_behavior_preserving_at_initialization() -> None:
    torch.manual_seed(4511)
    shared = _selector("shared").eval()
    decoupled = _selector("decoupled").eval()
    _copy_common_state(shared, decoupled)

    hidden = torch.randn(2, 5, 16)
    relations = torch.randn(2, 5, 5, 6)
    unary = torch.randn(2, 5)
    valid = torch.ones(2, 5, dtype=torch.bool)
    policy = decoupled.pointer_policy_output(
        decoupled.pointer_policy_output_norm(hidden)
    ).squeeze(-1)
    shared_result = shared.decode_pointer(hidden, relations, unary, valid)
    split_result = decoupled.decode_pointer(
        hidden,
        relations,
        unary,
        valid,
        policy_logits=policy,
    )

    torch.testing.assert_close(
        decoupled.pointer_quality_scale(),
        torch.ones(4),
    )
    torch.testing.assert_close(
        shared_result["selection_pointer_logits"],
        split_result["selection_pointer_logits"],
    )
    torch.testing.assert_close(
        shared_result["selection_pointer_indices"],
        split_result["selection_pointer_indices"],
    )


def test_decoupled_pointer_routes_sequence_and_quality_gradients_apart() -> None:
    torch.manual_seed(4512)
    selector = _selector("decoupled").train()
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    quality_input = hidden.detach()
    quality_hidden = quality_input + selector.pointer_quality_adapter(
        quality_input
    )
    unary = selector.output(selector.output_norm(quality_hidden)).squeeze(-1)
    policy = selector.pointer_policy_output(
        selector.pointer_policy_output_norm(hidden)
    ).squeeze(-1)
    relations = torch.randn(2, 5, 5, 6)
    valid = torch.ones(2, 5, dtype=torch.bool)
    teacher = torch.tensor([[0, 1, 5, -100], [2, 3, 4, 5]])
    rollout = selector.decode_pointer(
        hidden,
        relations,
        unary,
        valid,
        policy_logits=policy,
        teacher_indices=teacher,
    )
    sequence = F.cross_entropy(
        rollout["selection_pointer_logits"].reshape(-1, 6),
        teacher.reshape(-1),
        ignore_index=-100,
    )
    quality = F.binary_cross_entropy_with_logits(
        unary,
        torch.full_like(unary, 0.75),
    )

    quality_parameters = tuple(selector.pointer_quality_adapter.parameters()) + (
        *tuple(selector.output_norm.parameters()),
        *tuple(selector.output.parameters()),
    )
    policy_parameters = (
        *tuple(selector.pointer_policy_output_norm.parameters()),
        *tuple(selector.pointer_policy_output.parameters()),
    )
    sequence_quality_grads = torch.autograd.grad(
        sequence,
        quality_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    sequence_policy_grads = torch.autograd.grad(
        sequence,
        policy_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    quality_policy_grads = torch.autograd.grad(
        quality,
        policy_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    quality_quality_grads = torch.autograd.grad(
        quality,
        quality_parameters,
        allow_unused=True,
    )

    assert all(grad is None for grad in sequence_quality_grads)
    assert any(grad is not None and float(grad.norm()) > 0.0 for grad in sequence_policy_grads)
    assert all(grad is None for grad in quality_policy_grads)
    assert any(grad is not None and float(grad.norm()) > 0.0 for grad in quality_quality_grads)
    assert hidden.grad is None


def test_pointer_components_sum_and_quality_scale_is_bounded() -> None:
    torch.manual_seed(4513)
    selector = _selector("decoupled").eval()
    hidden = torch.randn(1, 4, 16)
    relations = torch.randn(1, 4, 4, 6)
    unary = torch.randn(1, 4)
    policy = selector.pointer_policy_output(
        selector.pointer_policy_output_norm(hidden)
    ).squeeze(-1)
    rollout = selector.decode_pointer(
        hidden,
        relations,
        unary,
        torch.ones(1, 4, dtype=torch.bool),
        policy_logits=policy,
    )
    reconstructed = (
        rollout["selection_pointer_unary_component"][:, 0]
        + rollout["selection_pointer_policy_component"][:, 0]
        + rollout["selection_pointer_content_component"][:, 0]
        + rollout["selection_pointer_relation_bias"][:, 0]
    )
    torch.testing.assert_close(
        reconstructed,
        rollout["selection_pointer_logits"][:, 0, :4],
    )
    scale = rollout["selection_pointer_quality_scale"]
    assert bool((scale > 0.0).all())
    assert bool((scale < 2.0).all())


def test_cluster_listwise_loss_uses_exact_soft_teacher_support() -> None:
    unary = torch.zeros(1, 3, requires_grad=True)
    pointer_logits = torch.zeros(1, 2, 4, requires_grad=True)
    teacher_probabilities = torch.tensor(
        [[[0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    outputs = {
        "selection_logits": unary,
        "selection_pointer_logits": pointer_logits,
        "selection_pointer_teacher_indices": torch.tensor([[0, 3]]),
        "selection_pointer_teacher_probabilities": teacher_probabilities,
        "selection_pointer_teacher_active": torch.ones(1, 2, dtype=torch.bool),
        "selection_pointer_teacher_candidate_steps": torch.tensor([[True, False]]),
        "pred_x_rows": torch.tensor([[[10.0] * 8, [10.5] * 8, [80.0] * 8]]),
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
            input_w=100,
            set_selection_line_width=12.0,
            set_selection_min_valid_rows=2,
            set_selection_focal_beta=0.0,
            w_pointer_selection=1.0,
            pointer_quality_weight=0.0,
            pointer_cluster_listwise_weight=1.0,
            pointer_cluster_listwise_logit_temperature=1.0,
        )
    )
    losses = criterion.compute_pointer_selection_loss(outputs, targets)
    losses["listwise"].backward()
    assert float(unary.grad[0, 0]) < 0.0
    assert float(unary.grad[0, 1]) > 0.0
    assert abs(float(unary.grad[0, 2])) < 1e-6
