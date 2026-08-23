from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.four_slot_selection import (
    FourSlotLaneSelectionHead,
)


PULSE_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_v32_auxiliary_pulse_consolidation_35k_to50k.yaml"
)


def _head(*, field_forward: bool) -> FourSlotLaneSelectionHead:
    return FourSlotLaneSelectionHead(
        8,
        input_w=80,
        hidden_dim=16,
        num_slots=4,
        proposal_layers=1,
        slot_layers=1,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
        curve_samples=6,
        min_valid_rows=3,
        joint_slot_field_enabled=True,
        joint_slot_field_forward_enabled=field_forward,
        joint_slot_field_num_rows=6,
        joint_slot_field_hidden_dim=8,
        joint_slot_field_route_residual_scale=0.0,
    )


def _outputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3407)
    batch, candidates, rows, dim = 2, 5, 6, 8
    return {
        "queries": torch.randn(batch, candidates, dim, generator=generator),
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, generator=generator
        ),
        "ownership_state": torch.randn(
            batch, candidates, dim, generator=generator
        ),
        "range_norm": torch.tensor(
            [[[0.0, 1.0]] * candidates] * batch, dtype=torch.float32
        ),
        "pred_x_rows": torch.rand(
            batch, candidates, rows, generator=generator
        )
        * 79.0,
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, generator=generator
        ),
        "exist_logits": torch.randn(
            batch, candidates, 2, generator=generator
        ),
        "input_reference_x_rows": torch.rand(
            batch, candidates, rows, generator=generator
        )
        * 79.0,
    }


def test_pulse_config_disables_every_auxiliary_effect() -> None:
    cfg = load_config(PULSE_CONFIG)
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_joint_field_enabled"] is True
    assert selection["four_slot_joint_field_forward_enabled"] is False
    assert selection["four_slot_joint_field_route_residual_scale"] == 0.0
    assert selection["four_slot_selection_row_token_gradient_scale"] == 0.0
    assert cfg["loss"]["w_four_slot_joint_field"] == 0.0
    assert cfg["dataloader"]["resume_safe"] is True


def test_dormant_field_keeps_checkpoint_topology_without_forward_effect() -> None:
    torch.manual_seed(3407)
    active = _head(field_forward=True).eval()
    dormant = _head(field_forward=False).eval()
    dormant.load_state_dict(active.state_dict(), strict=True)
    assert dormant.joint_slot_field is not None
    assert dormant.joint_slot_field_forward_enabled is False
    assert dormant.requires_live_row_value_features is False

    outputs = _outputs()
    row_features = torch.randn(2, 6, 10, 8)
    with torch.no_grad():
        active_result = active(outputs, row_value_features=row_features)
        dormant_result = dormant(outputs)

    for key in (
        "selection_slot_logits",
        "selection_slot_indices",
        "selection_slot_scores",
    ):
        torch.testing.assert_close(
            dormant_result[key], active_result[key], rtol=0.0, atol=0.0
        )
    assert "selection_slot_joint_field_logits" in active_result
    assert "selection_slot_joint_field_logits" not in dormant_result
