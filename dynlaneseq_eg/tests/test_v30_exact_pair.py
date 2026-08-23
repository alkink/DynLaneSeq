from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import restore_checkpoint_rng_state
from dynlaneseq_eg.modeling.v30_joint_slot_field import (
    FourSlotJointBeliefField,
)
from dynlaneseq_eg.tools.audit_v30_exact_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCES,
    _config_differences,
)


CONTROL_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml"
)
TREATMENT_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_v30_joint_slot_field_35k_route_residual_off.yaml"
)


def test_exact_pair_configs_differ_only_by_field_intervention() -> None:
    control = load_config(CONTROL_CONFIG)
    treatment = load_config(TREATMENT_CONFIG)
    differences = _config_differences(control, treatment)
    assert not (set(differences) - ALLOWED_CONFIG_DIFFERENCES)
    assert treatment["dataloader"]["resume_safe"] is True
    assert control["dataloader"]["resume_safe"] is True
    assert (
        treatment["model"]["structured_query"]["set_selection"]
        ["four_slot_joint_field_route_residual_scale"]
        == 0.0
    )


def test_optimizer_remap_resume_restores_checkpoint_rng() -> None:
    torch.manual_seed(3407)
    checkpoint_state = torch.random.get_rng_state().clone()
    expected = torch.rand(8)
    torch.manual_seed(9999)
    assert restore_checkpoint_rng_state(
        {"rng_state": {"torch_cpu": checkpoint_state}}
    )
    actual = torch.rand(8)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_field_forward_does_not_advance_model_rng() -> None:
    torch.manual_seed(3407)
    field = FourSlotJointBeliefField(
        feature_dim=16,
        slot_dim=12,
        num_rows=8,
        input_w=64,
        hidden_dim=10,
        route_residual_scale=0.0,
    )
    slot_states = torch.randn(2, 4, 12)
    row_features = torch.randn(2, 8, 16, 16)
    proposal_x = torch.rand(2, 6, 8) * 63.0
    proposal_range = torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(2, 6, 2)
    candidate_valid = torch.ones(2, 6, dtype=torch.bool)
    rng_before = torch.random.get_rng_state().clone()
    output = field(
        slot_states=slot_states,
        row_value_features=row_features,
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        candidate_valid=candidate_valid,
    )
    rng_after = torch.random.get_rng_state()
    torch.testing.assert_close(rng_after, rng_before, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(output["route_residual"]).item() == 0
