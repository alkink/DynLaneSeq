from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.modeling.v20_slot_owned_replacement import (
    SlotOwnedSafeReplacementHead,
    complete_curve_relations,
    slot_candidate_action_valid,
)
from dynlaneseq_eg.tools.v20_replacement_targets import (
    build_one_edit_action_targets,
)


def _inputs(
    *, batch: int = 2, slots: int = 4, candidates: int = 6, rows: int = 12, dim: int = 16
) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    base = torch.linspace(20.0, 80.0, rows)
    x = base.view(1, 1, 1, rows).expand(batch, slots, candidates, rows).clone()
    slot_offset = torch.arange(slots).view(1, slots, 1, 1) * 120.0
    candidate_offset = torch.arange(candidates).view(1, 1, candidates, 1) * 4.0
    x = x + slot_offset + candidate_offset
    ranges = torch.tensor((0.05, 0.95)).view(1, 1, 1, 2).expand(
        batch, slots, candidates, 2
    ).clone()
    source_route = torch.tensor((0, 1, 2, 3)).view(1, slots).expand(batch, slots)
    return {
        "candidate_state": torch.randn(batch, slots, candidates, dim),
        "p50": torch.rand(batch, slots, candidates),
        "p75": torch.rand(batch, slots, candidates),
        "expected_iou": torch.rand(batch, slots, candidates),
        "legacy_route_logits": torch.randn(batch, slots, candidates),
        "counterfactual_x": x,
        "counterfactual_range": ranges,
        "counterfactual_valid": torch.ones(
            batch, slots, candidates, dtype=torch.bool
        ),
        "source_route": source_route,
        "source_active": torch.ones(batch, slots, dtype=torch.bool),
    }


def test_action_mask_preserves_id_uniqueness_and_excludes_keep() -> None:
    values = _inputs(batch=1)
    valid = slot_candidate_action_valid(
        values["counterfactual_valid"],
        values["source_route"],
        values["source_active"],
    )
    for slot in range(4):
        assert not bool(valid[0, slot, slot])
        for other in range(4):
            if other != slot:
                assert not bool(valid[0, slot, other])
        assert bool(valid[0, slot, 4])


def test_curve_relation_is_zero_for_identical_geometry() -> None:
    values = _inputs(batch=1)
    source_x = values["counterfactual_x"][:, :, 0]
    source_range = values["counterfactual_range"][:, :, 0]
    relation = complete_curve_relations(
        source_x.unsqueeze(2),
        source_range.unsqueeze(2),
        source_x,
        source_range,
        input_w=800,
    )
    diagonal = relation[0, torch.arange(4), 0, torch.arange(4)]
    torch.testing.assert_close(diagonal[:, :8], torch.zeros_like(diagonal[:, :8]))
    torch.testing.assert_close(diagonal[:, 8:10], torch.ones_like(diagonal[:, 8:10]))
    torch.testing.assert_close(diagonal[:, 10], torch.zeros_like(diagonal[:, 10]))


def test_zero_initialized_v20_is_exact_keep() -> None:
    values = _inputs()
    head = SlotOwnedSafeReplacementHead(
        16, hidden_dim=16, ff_dim=32, input_w=800
    )
    result = head(**values)
    torch.testing.assert_close(result["selected_route"], values["source_route"])
    assert torch.count_nonzero(result["edit_count"]) == 0
    assert torch.count_nonzero(result["policy_logits"]) == 0
    assert torch.count_nonzero(result["delta50_logits"]) == 0
    assert torch.count_nonzero(result["delta75_logits"]) == 0


def test_treatment_context_edge_isolated_from_masked_control() -> None:
    values = _inputs(batch=1)
    head = SlotOwnedSafeReplacementHead(
        16, hidden_dim=16, ff_dim=32, input_w=800
    )
    treatment = head(**values, force_context_mode="treatment")
    control = head(**values, force_context_mode="masked")
    assert not torch.equal(
        treatment["action_hidden"], control["action_hidden"]
    )
    # Zero output projections preserve the same deployment at Gate 0.
    torch.testing.assert_close(
        treatment["selected_route"], control["selected_route"]
    )


def test_deployment_never_makes_more_than_one_edit() -> None:
    values = _inputs()
    head = SlotOwnedSafeReplacementHead(
        16, hidden_dim=16, ff_dim=32, input_w=800
    )
    with torch.no_grad():
        head.delta50_output.bias[:] = torch.tensor((-2.0, -2.0, 2.0))
        head.duplicate_output.bias.fill_(-2.0)
        head.abandon_output.bias.fill_(-2.0)
        head.policy_output.bias.fill_(1.0)
    result = head(**values)
    assert bool((result["edit_count"] <= 1).all())
    for item in range(values["source_route"].shape[0]):
        route = result["selected_route"][item]
        active_route = route[values["source_active"][item]]
        assert active_route.unique().numel() == active_route.numel()


def test_exact_action_target_prefers_one_owned_lane_replacement() -> None:
    # Four active slots initially cover A,A,C,D.  Replacing slot 1 candidate
    # 1 with candidate 4 adds B and is the only TP@.50 gain.
    quality = torch.zeros(4, 4, 6)
    source_route = torch.tensor((0, 1, 2, 3))
    active = torch.ones(4, dtype=torch.bool)
    valid = torch.ones(4, 6, dtype=torch.bool)
    quality[0, 0, 0] = 0.92
    quality[0, 1, 1] = 0.90
    quality[2, 2, 2] = 0.88
    quality[3, 3, 3] = 0.86
    quality[1, 1, 4] = 0.82
    target = build_one_edit_action_targets(
        quality, valid, source_route, active
    )
    action_id = 1 + 1 * 6 + 4
    assert int(target["delta50"][action_id]) == 1
    assert int(target["delta75"][action_id]) == 1
    assert float(target["policy_target"][action_id]) == 1.0
    assert int(target["source_tp50"]) == 3
    assert int(target["best_one_edit_tp50"]) == 4


def test_exact_action_target_defaults_to_keep_without_threshold_gain() -> None:
    quality = torch.zeros(2, 2, 4)
    source_route = torch.tensor((0, 1))
    active = torch.ones(2, dtype=torch.bool)
    valid = torch.ones(2, 4, dtype=torch.bool)
    quality[0, 0, 0] = 0.80
    quality[1, 1, 1] = 0.82
    quality[0, 0, 2] = 0.90  # Better IoU but threshold-neutral.
    target = build_one_edit_action_targets(
        quality, valid, source_route, active
    )
    assert float(target["policy_target"][0]) == 1.0
    assert torch.count_nonzero(target["policy_target"][1:]) == 0


def test_v20_config_is_frozen_one_edit_and_zero_augmentation() -> None:
    config = (
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml"
    )
    cfg = load_config(config)
    selection = cfg["model"]["structured_query"]["set_selection"]
    assert selection["four_slot_counterfactual_fidelity_enabled"] is True
    assert selection["four_slot_slot_owned_safe_replacement_enabled"] is True
    assert selection["four_slot_slot_owned_safe_replacement_context_mode"] == "treatment"
    assert cfg["v20"]["max_active_edits"] == 1
    assert cfg["training"]["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.slot_owned_safe_replacement"
    ]
    augmentation = cfg["augmentation"]
    assert augmentation["horizontal_flip_prob"] == 0.0
    assert augmentation["color_jitter"] is False
    assert augmentation["affine_prob"] == 0.0
    assert augmentation["random_shadow_prob"] == 0.0


def test_v20_model_builds_frozen_v19_and_new_head() -> None:
    config = (
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml"
    )
    cfg = load_config(config)
    cfg["model"]["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model = build_model(cfg)
    selector = model.structured_query_head.set_selection_head
    assert selector.counterfactual_fidelity is not None
    assert selector.slot_owned_safe_replacement is not None
    assert selector.slot_owned_safe_replacement.context_mode == "treatment"
