from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.v19_counterfactual_fidelity import (
    FourSlotCounterfactualProposalFidelity,
    frozen_v7_counterfactual_anchors,
)


class _IdentityCounterfactualRefiner(nn.Module):
    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        proposal_row_tokens: torch.Tensor,
        proposal_x_rows: torch.Tensor,
        proposal_range_norm: torch.Tensor,
        route_indices: torch.Tensor,
        route_logits: torch.Tensor | None,
        candidate_valid: torch.Tensor,
        slot_active: torch.Tensor,
        row_value_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del slot_states, proposal_row_tokens, route_logits, row_value_features
        safe = route_indices.clamp(min=0)
        rows = int(proposal_x_rows.shape[-1])
        x = proposal_x_rows.gather(
            1, safe.unsqueeze(-1).expand(-1, -1, rows)
        )
        ranges = proposal_range_norm.gather(
            1, safe.unsqueeze(-1).expand(-1, -1, 2)
        )
        valid = candidate_valid.gather(1, safe) & slot_active
        return {
            "selection_slot_pred_x_rows": x,
            "selection_slot_range_norm": ranges,
            "selection_slot_geometry_valid": valid,
        }


def test_counterfactual_helper_expands_every_slot_candidate_pair() -> None:
    batch, slots, candidates, rows, hidden = 2, 4, 6, 5, 8
    proposal_x = torch.arange(
        batch * candidates * rows, dtype=torch.float32
    ).reshape(batch, candidates, rows)
    proposal_range = torch.tensor((0.1, 0.9)).view(1, 1, 2).expand(
        batch, candidates, 2
    )
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    valid[1, -1] = False
    result = frozen_v7_counterfactual_anchors(
        _IdentityCounterfactualRefiner(),
        slot_states=torch.randn(batch, slots, hidden),
        proposal_row_tokens=torch.randn(
            batch, candidates, rows, hidden
        ),
        proposal_x_rows=proposal_x,
        proposal_range_norm=proposal_range,
        candidate_valid=valid,
        row_value_features=torch.randn(batch, rows, 20, hidden),
    )
    assert result["x_rows"].shape == (batch, slots, candidates, rows)
    for slot in range(slots):
        torch.testing.assert_close(result["x_rows"][:, slot], proposal_x)
        assert torch.equal(result["valid"][:, slot], valid)


def test_fidelity_head_has_exact_neutral_route_and_frozen_inputs() -> None:
    torch.manual_seed(190)
    batch, slots, candidates, rows, hidden = 1, 4, 7, 8, 16
    module = FourSlotCounterfactualProposalFidelity(
        hidden,
        feature_dim=hidden,
        slot_dim=hidden,
        hidden_dim=hidden,
        input_w=160,
        num_slots=slots,
        num_heads=4,
        ff_dim=32,
        vertical_layers=1,
        dropout=0.0,
        scale_names=("p2", "p4"),
        evidence_offsets_px=(-16.0, 0.0, 16.0),
    )
    legacy = torch.randn(batch, slots, candidates)
    proposal_rows = torch.randn(
        batch, candidates, rows, hidden, requires_grad=True
    )
    counterfactual_x = torch.rand(batch, slots, candidates, rows) * 159.0
    counterfactual_range = torch.tensor((0.1, 0.9)).view(
        1, 1, 1, 2
    ).expand(batch, slots, candidates, 2)
    features = {
        "p2": torch.randn(
            batch, rows, 40, hidden, requires_grad=True
        ),
        "p4": torch.randn(
            batch, hidden, 2, 10, requires_grad=True
        ),
    }
    result = module(
        slot_states=torch.randn(batch, slots, hidden),
        legacy_route_logits=legacy,
        proposal_rows=proposal_rows,
        proposal_x=counterfactual_x[:, 0],
        proposal_range=counterfactual_range[:, 0],
        candidate_valid=torch.ones(batch, candidates, dtype=torch.bool),
        counterfactual_x=counterfactual_x,
        counterfactual_range=counterfactual_range,
        counterfactual_valid=torch.ones(
            batch, slots, candidates, dtype=torch.bool
        ),
        image_features=features,
    )
    assert torch.equal(result["calibrated_route_logits"], legacy)
    assert torch.count_nonzero(result["fidelity_delta"]) == 0
    result["quality_logits"].sum().backward()
    assert module.quality_output.weight.grad is not None
    assert proposal_rows.grad is None
    assert features["p2"].grad is None
    assert features["p4"].grad is None


def _synthetic_v19_outputs(
    logits: torch.Tensor,
    score: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    batch, slots, candidates, _ = logits.shape
    rows = 20
    gt = torch.full((1, rows), 50.0)
    gt_valid = torch.ones_like(gt, dtype=torch.bool)
    # Candidate quality decreases monotonically with lateral displacement.
    offsets = torch.tensor((0.0, 4.0, 12.0, 25.0))[:candidates]
    x = gt.view(1, 1, 1, rows) + offsets.view(1, 1, candidates, 1)
    x = x.expand(batch, slots, candidates, rows).contiguous()
    ranges = torch.tensor((0.0, 1.0)).view(1, 1, 1, 2).expand(
        batch, slots, candidates, 2
    )
    outputs = {
        "selection_slot_v19_quality_logits": logits,
        "selection_slot_v19_fidelity_delta": score,
        "selection_slot_v19_counterfactual_x_rows": x,
        "selection_slot_v19_counterfactual_range_norm": ranges,
        "selection_slot_v19_counterfactual_valid": torch.ones(
            batch, slots, candidates, dtype=torch.bool
        ),
        "selection_slot_candidate_valid": torch.ones(
            batch, candidates, dtype=torch.bool
        ),
        "selection_slot_indices": torch.zeros(
            batch, slots, dtype=torch.long
        ),
        "selection_slot_v19_v7_indices": torch.full(
            (batch, slots), candidates - 1, dtype=torch.long
        ),
    }
    return outputs, [{"x_rows": gt, "valid_mask": gt_valid}]


def test_v19_loss_directly_rewards_threshold_quality_and_pair_order() -> None:
    logits = torch.zeros(1, 2, 4, 3, requires_grad=True)
    correct_score = torch.tensor(
        [[[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    wrong_score = -correct_score.detach().clone().requires_grad_(True)
    criterion = S0Criterion(
        LossConfig(
            input_h=640,
            four_slot_line_width=30.0,
            four_slot_min_valid_rows=5,
            w_four_slot_v19=1.0,
        )
    )
    correct_outputs, targets = _synthetic_v19_outputs(logits, correct_score)
    wrong_outputs, _ = _synthetic_v19_outputs(logits, wrong_score)
    correct = criterion.compute_four_slot_v19_loss(correct_outputs, targets)
    wrong = criterion.compute_four_slot_v19_loss(wrong_outputs, targets)
    assert correct["rank"] < wrong["rank"]
    assert correct["pair_accuracy"] == 1.0
    assert correct["threshold_pair_accuracy"] == 1.0
    assert correct["selected_quality"] > correct["v7_selected_quality"]
    assert torch.isfinite(correct["total"])
    correct["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert correct_score.grad is not None


def test_v19_config_freezes_exact_v7_and_opens_only_fidelity() -> None:
    cfg = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_v19_"
        "frozen_counterfactual_fidelity_225k_to233k.yaml"
    )
    selection = cfg["model"]["structured_query"]["set_selection"]
    training = cfg["training"]
    loss = cfg["loss"]
    assert selection["four_slot_counterfactual_fidelity_enabled"] is True
    assert selection["four_slot_joint_exact_set_energy_enabled"] is False
    assert selection["four_slot_counterfactual_fidelity_scale_names"] == [
        "p2",
        "p4",
        "p5",
    ]
    assert training["frozen_detector_eval"] is True
    assert training["trainable_parameter_prefixes"] == [
        "structured_query_head.set_selection_head.counterfactual_fidelity"
    ]
    assert training["trainable_module_prefixes"] == [
        "structured_query_head.set_selection_head.counterfactual_fidelity"
    ]
    assert loss["w_four_slot_v19"] == 1.0
    assert loss["w_exist"] == 0.0
    assert loss["w_four_slot_selection"] == 0.0
    assert loss["w_four_slot_geometry"] == 0.0
