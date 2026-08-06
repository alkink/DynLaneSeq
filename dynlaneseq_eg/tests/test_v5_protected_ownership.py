from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_matcher
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0, MatcherConfig
from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead


def _tiny_head(*, ownership_enabled: bool = True) -> StructuredLaneQueryHead:
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
        evidence_x_bins=16,
        intermediate_supervision=True,
        row_reference={
            "enabled": True,
            "prediction_mode": "bounded_delta",
            "detach_between_layers": True,
            "offsets_px": [-8.0, 0.0, 8.0],
            "delta_offsets_px": [-8.0, 0.0, 8.0],
        },
        lane_state={
            "enabled": True,
            "mode": "causal_set",
            "single_logit_score": True,
            "detach_score_geometry": True,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "semantic_context": {"enabled": False},
        },
        ownership=(
            {
                "enabled": True,
                "detach_geometry_inputs": True,
                "retain_diagnostic_tensors": True,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout": 0.0,
                "semantic_context": {
                    "enabled": True,
                    "scales": ["p4", "p5"],
                    "pool_size": [2, 2],
                },
            }
            if ownership_enabled
            else None
        ),
        set_selection={"enabled": False},
    )


def test_enabling_ownership_preserves_same_seed_v4_geometry_initialization() -> None:
    torch.manual_seed(17)
    control = _tiny_head(ownership_enabled=False)
    torch.manual_seed(17)
    ownership = _tiny_head(ownership_enabled=True)

    control_state = control.state_dict()
    ownership_state = ownership.state_dict()
    common_names = set(control_state).intersection(ownership_state)
    assert common_names
    for name in sorted(common_names):
        assert torch.equal(control_state[name], ownership_state[name]), name


def test_ownership_loss_cannot_backpropagate_to_geometry_or_shared_features() -> None:
    torch.manual_seed(3)
    head = _tiny_head()
    p2 = torch.randn(2, 32, 8, 16, requires_grad=True)
    p4 = torch.randn(2, 32, 4, 8, requires_grad=True)
    p5 = torch.randn(2, 32, 2, 4, requires_grad=True)
    outputs = head(p2, {"p4": p4, "p5": p5})

    loss = outputs["ownership_logits"].float().square().mean()
    for auxiliary in outputs["aux_outputs"]:
        loss = loss + auxiliary["ownership_logits"].float().square().mean()
    loss.backward()

    assert p2.grad is None
    assert p4.grad is None
    assert p5.grad is None
    assert head.instance_tokens.weight.grad is None
    assert head.row_tokens.weight.grad is None
    assert head.reference_anchor_logits.grad is None
    assert all(layer.weight.grad is None for layer in head.row_delta_heads)
    assert all(
        parameter.grad is None
        for layer in head.lane_state_layers
        for parameter in layer.parameters()
    )
    assert head.ownership_tokens is not None
    assert head.ownership_tokens.weight.grad is not None
    assert float(head.ownership_tokens.weight.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for layer in head.ownership_layers
        for parameter in layer.parameters()
    )
    assert head.exist[-1].weight.grad is not None


def test_ownership_forward_exposes_direct_and_intermediate_lane_logits() -> None:
    head = _tiny_head().eval()
    with torch.no_grad():
        outputs = head(
            torch.randn(1, 32, 8, 16),
            {
                "p4": torch.randn(1, 32, 4, 8),
                "p5": torch.randn(1, 32, 2, 4),
            },
        )
    assert outputs["ownership_logits"].shape == (1, 4, 2)
    assert outputs["ownership_state"].shape == (1, 4, 32)
    assert torch.equal(outputs["ownership_logits"], outputs["exist_logits"])
    assert len(outputs["aux_outputs"]) == 1
    assert outputs["aux_outputs"][0]["ownership_logits"].shape == (1, 4, 2)
    assert head.set_selection_head is None


def test_matcher_ownership_cost_warmup_and_linear_ramp() -> None:
    matcher = HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=0.25,
            lambda_obj_start=0.0,
            lambda_obj_end=0.25,
            lambda_obj_ramp_start_iter=10000,
            lambda_obj_ramp_end_iter=25000,
        )
    )
    expected = {
        0: 0.0,
        9999: 0.0,
        10000: 0.0,
        17500: 0.125,
        25000: 0.25,
        30000: 0.25,
    }
    for iteration, value in expected.items():
        assert matcher.effective_lambda_obj(iteration) == value


def test_v5_configs_differ_only_in_assignment_coupling() -> None:
    control = load_config(
        "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_a_protected_ownership_sidecar_25k.yaml"
    )
    assignment = load_config(
        "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_b_protected_ownership_assignment_25k.yaml"
    )
    assert control["model"] == assignment["model"]
    assert control["loss"] == assignment["loss"]
    assert control["optimizer"] == assignment["optimizer"]
    assert control["scheduler"] == assignment["scheduler"]
    assert control["training"] == assignment["training"]
    assert build_matcher(control).effective_lambda_obj(25000) == 0.0
    assert build_matcher(assignment).effective_lambda_obj(10000) == 0.0
    assert build_matcher(assignment).effective_lambda_obj(25000) == 0.25
    assert control["matcher"]["reuse_final_assignment_for_intermediate"] is False
    assert control["loss"]["exist_target_mode"] == "binary"
    assert control["loss"]["w_intermediate_exist"] == 2.0
