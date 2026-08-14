from __future__ import annotations

import itertools

import torch
from torch import nn

from dynlaneseq_eg.engine.train_one_epoch import _v18_conflict_safe_backward
from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.modeling.four_slot_selection import FourSlotLaneSelectionHead
from dynlaneseq_eg.modeling.v18_joint_exact_set_energy import (
    FourSlotJointExactSetEnergy,
    build_exact_four_set_tables,
    decode_exact_ordered_set,
    exact_ordered_set_energies,
    exact_unordered_set_rewards,
    masked_unordered_set_listwise_loss,
    unordered_set_log_scores,
)


def test_invalid_candidates_are_finite_and_never_decoded() -> None:
    torch.manual_seed(18)
    batch, candidates, slots = 2, 6, 4
    combinations, _permutations, ordered = build_exact_four_set_tables(
        candidates
    )
    slot_pairs = torch.tensor(
        tuple(itertools.combinations(range(slots), 2)), dtype=torch.long
    )
    unary = torch.randn(batch, slots, candidates, requires_grad=True)
    pair = torch.randn(batch, 6, candidates, candidates, requires_grad=True)
    active = torch.randn(batch, slots)
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    valid[0, -1] = False
    ordered_energy, valid_set = exact_ordered_set_energies(
        unary, pair, active, valid, ordered, slot_pairs
    )
    scores = unordered_set_log_scores(ordered_energy, valid_set)
    assert torch.isfinite(scores).all()
    decoded = decode_exact_ordered_set(ordered_energy, valid_set, ordered)
    assert 5 not in decoded["indices"][0].tolist()

    quality = torch.rand(batch, candidates, 3)
    gt_valid = torch.tensor([[True, True, False], [True, True, True]])
    reward, target_valid = exact_unordered_set_rewards(
        quality, gt_valid, valid, combinations
    )
    loss, diagnostics = masked_unordered_set_listwise_loss(
        scores,
        reward,
        valid_set & target_valid,
        support_delta=0.10,
        target_temperature=0.10,
        model_temperature=1.0,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    loss.backward()
    assert unary.grad is not None and torch.isfinite(unary.grad).all()
    assert pair.grad is not None and torch.isfinite(pair.grad).all()


def test_unordered_reward_marginalizes_unused_members_for_two_gt() -> None:
    candidates = 6
    combinations, _permutations, _ordered = build_exact_four_set_tables(
        candidates
    )
    quality = torch.zeros(1, candidates, 2)
    quality[0, 0, 0] = 1.0
    quality[0, 1, 1] = 1.0
    reward, valid = exact_unordered_set_rewards(
        quality,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, candidates, dtype=torch.bool),
        combinations,
    )
    best = reward[0, valid[0]].max()
    best_sets = combinations[valid[0]][reward[0, valid[0]] == best]
    # Any two of the four remaining proposals may fill the unused positions.
    assert int(best_sets.shape[0]) == 6
    assert all(0 in row.tolist() and 1 in row.tolist() for row in best_sets)


def test_vectorized_reward_matches_permutation_reference() -> None:
    torch.manual_seed(184)
    batch, candidates, max_gt = 4, 7, 4
    combinations, _permutations, _ordered = build_exact_four_set_tables(
        candidates
    )
    quality = torch.rand(batch, candidates, max_gt)
    gt_valid = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, False, True, False],
            [True, True, True, True],
        ]
    )
    candidate_valid = torch.ones(batch, candidates, dtype=torch.bool)
    candidate_valid[2, -1] = False
    actual, actual_valid = exact_unordered_set_rewards(
        quality,
        gt_valid,
        candidate_valid,
        combinations,
    )

    reference = torch.zeros_like(actual)
    for batch_index in range(batch):
        gt_ids = torch.nonzero(gt_valid[batch_index], as_tuple=False).flatten()[:4]
        gt_count = int(gt_ids.numel())
        if gt_count == 0:
            continue
        set_quality = quality[batch_index, combinations].index_select(-1, gt_ids)
        assignment_reward = []
        gt_axis = torch.arange(gt_count)
        for positions in itertools.permutations(range(4), gt_count):
            selected = set_quality[:, torch.tensor(positions), gt_axis]
            lane_reward = (
                torch.sigmoid((selected - 0.50) / 0.03)
                + 0.5 * torch.sigmoid((selected - 0.75) / 0.03)
                + 0.1 * selected
            )
            assignment_reward.append(lane_reward.sum(dim=-1))
        reference[batch_index] = torch.stack(
            assignment_reward, dim=-1
        ).amax(dim=-1)
    reference_valid = candidate_valid[:, combinations].all(dim=-1)
    assert torch.equal(actual_valid, reference_valid)
    torch.testing.assert_close(actual, reference, rtol=1.0e-6, atol=1.0e-7)


def test_vectorized_listwise_matches_filtered_reference_and_gradient() -> None:
    torch.manual_seed(185)
    score = torch.randn(3, 17, requires_grad=True)
    reference_score = score.detach().clone().requires_grad_(True)
    reward = torch.rand(3, 17)
    valid = torch.rand(3, 17) > 0.25
    valid[2] = False

    actual, actual_diagnostics = masked_unordered_set_listwise_loss(
        score,
        reward,
        valid,
        support_delta=0.10,
        target_temperature=0.10,
        model_temperature=1.0,
    )

    losses = []
    entropies = []
    support_sizes = []
    regrets = []
    for batch_index in range(3):
        batch_valid = valid[batch_index]
        if not bool(batch_valid.any()):
            continue
        batch_score = reference_score[batch_index, batch_valid]
        batch_reward = reward[batch_index, batch_valid]
        best_reward = batch_reward.max()
        support = batch_reward >= best_reward - 0.10
        target_probability = torch.softmax(batch_reward[support] / 0.10, dim=-1)
        model_log_probability = torch.log_softmax(batch_score, dim=-1)
        support_indices = torch.nonzero(support, as_tuple=False).flatten()
        losses.append(
            -(target_probability * model_log_probability[support_indices]).sum()
        )
        entropies.append(
            -(
                target_probability
                * target_probability.clamp_min(1.0e-12).log()
            ).sum()
        )
        support_sizes.append(support.float().sum())
        regrets.append(best_reward - batch_reward[batch_score.argmax()])
    reference_loss = torch.stack(losses).mean()
    reference_diagnostics = {
        "target_entropy": torch.stack(entropies).mean(),
        "support_size": torch.stack(support_sizes).mean(),
        "chosen_regret": torch.stack(regrets).mean(),
    }
    torch.testing.assert_close(actual, reference_loss, rtol=1.0e-6, atol=1.0e-7)
    for key, value in actual_diagnostics.items():
        torch.testing.assert_close(
            value,
            reference_diagnostics[key],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    actual.backward()
    reference_loss.backward()
    torch.testing.assert_close(
        score.grad,
        reference_score.grad,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_zero_residual_exact_parity_and_two_phase_gradient_contract() -> None:
    torch.manual_seed(180)
    batch, candidates, slots, rows, hidden = 1, 32, 4, 8, 32
    module = FourSlotJointExactSetEnergy(
        hidden,
        feature_dim=hidden,
        slot_dim=hidden,
        hidden_dim=hidden,
        input_w=1600,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
    )
    slot_state = torch.randn(batch, slots, hidden)
    legacy = torch.randn(batch, slots, candidates)
    active = torch.randn(batch, slots)
    proposal_rows = torch.randn(
        batch, candidates, rows, hidden, requires_grad=True
    )
    proposal_x = torch.rand(batch, candidates, rows) * 1599.0
    proposal_range = torch.tensor([0.1, 0.9]).view(1, 1, 2).expand(
        batch, candidates, 2
    )
    candidate_valid = torch.ones(batch, candidates, dtype=torch.bool)
    features = {
        "p2": torch.randn(batch, rows, 40, hidden, requires_grad=True),
        "p3": torch.randn(batch, hidden, 4, 10, requires_grad=True),
        "p4": torch.randn(batch, hidden, 2, 5, requires_grad=True),
    }
    result = module.route(
        slot_states=slot_state,
        legacy_route_logits=legacy,
        legacy_active_logits=active,
        proposal_rows=proposal_rows,
        proposal_x=proposal_x,
        proposal_range=proposal_range,
        candidate_valid=candidate_valid,
        image_features=features,
    )
    assert torch.equal(result["unary"], legacy)
    assert torch.count_nonzero(result["pair_energy"]) == 0
    zero_pair = torch.zeros_like(result["pair_energy"])
    expected_energy, expected_valid = exact_ordered_set_energies(
        legacy,
        zero_pair,
        active,
        candidate_valid,
        module.ordered_assignments,
        module.slot_pairs,
    )
    expected = decode_exact_ordered_set(
        expected_energy, expected_valid, module.ordered_assignments
    )["indices"]
    assert torch.equal(result["indices"], expected)

    result["set_scores"].mean().backward()
    assert module.unary_interaction[-1].weight.grad.norm() > 0
    assert module.pair_output.weight.grad.norm() > 0
    assert proposal_rows.grad is not None
    assert proposal_rows.grad.norm() == 0
    assert features["p2"].grad is not None
    assert features["p2"].grad.norm() == 0

    module.zero_grad(set_to_none=True)
    proposal_rows.grad = None
    for feature in features.values():
        feature.grad = None
    with torch.no_grad():
        module.unary_interaction[-1].weight.normal_(std=1.0e-4)
        module.pair_output.weight.normal_(std=1.0e-4)
    second = module.route(
        slot_states=slot_state,
        legacy_route_logits=legacy,
        legacy_active_logits=active,
        proposal_rows=proposal_rows,
        proposal_x=proposal_x,
        proposal_range=proposal_range,
        candidate_valid=candidate_valid,
        image_features=features,
    )
    second["set_scores"].mean().backward()
    assert proposal_rows.grad is not None and proposal_rows.grad.norm() > 0
    assert features["p2"].grad is not None and features["p2"].grad.norm() > 0


def test_detach_control_has_same_forward_and_blocks_association_gradient() -> None:
    torch.manual_seed(181)
    kwargs = dict(
        proposal_dim=32,
        feature_dim=32,
        slot_dim=32,
        hidden_dim=32,
        input_w=1600,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
    )
    treatment = FourSlotJointExactSetEnergy(
        **kwargs, detach_association_for_set_loss=False
    )
    control = FourSlotJointExactSetEnergy(
        **kwargs, detach_association_for_set_loss=True
    )
    control.load_state_dict(treatment.state_dict())
    with torch.no_grad():
        for module in (treatment, control):
            module.unary_interaction[-1].weight.normal_(std=1.0e-4)
            module.pair_output.weight.normal_(std=1.0e-4)
        control.unary_interaction[-1].weight.copy_(
            treatment.unary_interaction[-1].weight
        )
        control.pair_output.weight.copy_(treatment.pair_output.weight)

    common = {
        "slot_states": torch.randn(1, 4, 32),
        "legacy_route_logits": torch.randn(1, 4, 32),
        "legacy_active_logits": torch.randn(1, 4),
        "proposal_x": torch.rand(1, 32, 8) * 1599.0,
        "proposal_range": torch.tensor([0.1, 0.9]).view(1, 1, 2).expand(1, 32, 2),
        "candidate_valid": torch.ones(1, 32, dtype=torch.bool),
    }

    def inputs() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rows = torch.randn(1, 32, 8, 32, requires_grad=True)
        features = {
            "p2": torch.randn(1, 8, 40, 32, requires_grad=True),
            "p3": torch.randn(1, 32, 4, 10, requires_grad=True),
            "p4": torch.randn(1, 32, 2, 5, requires_grad=True),
        }
        return rows, features

    treatment_rows, treatment_features = inputs()
    control_rows = treatment_rows.detach().clone().requires_grad_(True)
    control_features = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in treatment_features.items()
    }
    treatment_result = treatment.route(
        **common,
        proposal_rows=treatment_rows,
        image_features=treatment_features,
    )
    control_result = control.route(
        **common,
        proposal_rows=control_rows,
        image_features=control_features,
    )
    assert torch.equal(treatment_result["indices"], control_result["indices"])
    assert torch.equal(treatment_result["set_scores"], control_result["set_scores"])
    treatment_result["set_scores"].mean().backward()
    control_result["set_scores"].mean().backward()
    assert treatment_rows.grad is not None and treatment_rows.grad.norm() > 0
    assert control_rows.grad is None
    assert treatment_features["p2"].grad is not None
    assert treatment_features["p2"].grad.norm() > 0
    assert control_features["p2"].grad is None


def test_geometry_cannot_rewrite_v7_anchor_or_proposal_coordinates() -> None:
    torch.manual_seed(182)
    module = FourSlotJointExactSetEnergy(
        32,
        feature_dim=32,
        slot_dim=32,
        hidden_dim=32,
        input_w=1600,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
    )
    with torch.no_grad():
        module.refiner.delta_head.weight.normal_(std=1.0e-3)
        module.refiner.range_head.weight.normal_(std=1.0e-3)
        module.refiner.activity_head.weight.normal_(std=1.0e-3)
        module.refiner.policy_head.weight.normal_(std=1.0e-3)

    slot = torch.randn(1, 4, 32, requires_grad=True)
    legacy_route = torch.randn(1, 4, 32, requires_grad=True)
    legacy_active = torch.randn(1, 4, requires_grad=True)
    proposal_rows = torch.randn(1, 32, 8, 32, requires_grad=True)
    proposal_x = (torch.rand(1, 32, 8) * 1599.0).requires_grad_(True)
    proposal_range = (
        torch.tensor([0.1, 0.9]).view(1, 1, 2).expand(1, 32, 2).clone()
    ).requires_grad_(True)
    features = {
        "p2": torch.randn(1, 8, 40, 32, requires_grad=True),
        "p3": torch.randn(1, 32, 4, 10, requires_grad=True),
        "p4": torch.randn(1, 32, 2, 5, requires_grad=True),
    }
    valid = torch.ones(1, 32, dtype=torch.bool)
    route = module.route(
        slot_states=slot,
        legacy_route_logits=legacy_route,
        legacy_active_logits=legacy_active,
        proposal_rows=proposal_rows,
        proposal_x=proposal_x,
        proposal_range=proposal_range,
        candidate_valid=valid,
        image_features=features,
    )
    anchor_x = (torch.rand(1, 4, 8) * 1599.0).requires_grad_(True)
    anchor_range = (
        torch.tensor([0.15, 0.85]).view(1, 1, 2).expand(1, 4, 2).clone()
    ).requires_grad_(True)
    refined = module.refine(
        route_result=route,
        slot_states=slot,
        anchor_x=anchor_x,
        anchor_range=anchor_range,
        geometry_valid=torch.ones(1, 4, dtype=torch.bool),
        proposal_rows=proposal_rows,
        proposal_x=proposal_x,
        proposal_range=proposal_range,
        candidate_valid=valid,
        image_features=features,
        legacy_active_logits=legacy_active,
    )
    loss = (
        refined["refined_x"].mean()
        + refined["refined_range"].mean()
        + refined["active_logits"].mean()
        + refined["policy_logits"].mean()
    )
    loss.backward()
    assert anchor_x.grad is None and anchor_range.grad is None
    assert proposal_x.grad is None and proposal_range.grad is None
    assert slot.grad is None and legacy_route.grad is None
    assert legacy_active.grad is None
    assert proposal_rows.grad is not None and proposal_rows.grad.norm() > 0
    assert features["p2"].grad is not None and features["p2"].grad.norm() > 0


def test_conflict_projection_replaces_opposing_shared_v18_component() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.proj = nn.Linear(2, 1, bias=False)

    model = TinyModel()
    parameter = model.encoder.proj.weight
    with torch.no_grad():
        parameter.fill_(1.0)
    v18 = parameter.sum()
    proposal = -parameter.sum()
    total = v18 + proposal
    stats = _v18_conflict_safe_backward(
        model,
        {
            "loss_v18_set_backward": v18,
            "loss_v18_proposal_protection": proposal,
        },
        total,
        {
            "training": {
                "v18_gradient_conflict_projection": {
                    "enabled": True,
                    "shared_prefixes": ["encoder.proj."],
                    "exclude_prefixes": [],
                }
            }
        },
        accumulation_steps=1,
        scaler=None,
        amp=False,
    )
    assert stats is not None
    assert stats["v18_shared_projection_active"] == 1
    # The opposing V18 component is projected away; the mature proposal
    # objective remains exactly [-1, -1].
    assert torch.equal(parameter.grad, -torch.ones_like(parameter))


def test_partitioned_two_pass_matches_reference_gradient_with_accumulation() -> None:
    class TinyPartitionedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.proj = nn.Linear(3, 2, bias=True)
            self.v18_private = nn.Linear(2, 1, bias=False)
            self.proposal_private = nn.Parameter(torch.randn(2))

    torch.manual_seed(183)
    reference = TinyPartitionedModel()
    optimized = TinyPartitionedModel()
    optimized.load_state_dict(reference.state_dict())
    for left, right in zip(reference.parameters(), optimized.parameters()):
        initial_gradient = torch.randn_like(left)
        left.grad = initial_gradient.clone()
        right.grad = initial_gradient.clone()

    inputs = torch.randn(5, 3)

    def losses(model: TinyPartitionedModel) -> tuple[torch.Tensor, torch.Tensor]:
        shared = model.encoder.proj(inputs)
        set_loss = model.v18_private(shared).square().mean()
        protection_loss = (
            (shared - 0.75).square().mean()
            + 0.2 * model.proposal_private.square().sum()
        )
        return set_loss, protection_loss

    def run(model: TinyPartitionedModel, backward_mode: str):
        set_loss, protection_loss = losses(model)
        stats = _v18_conflict_safe_backward(
            model,
            {
                "loss_v18_set_backward": set_loss,
                "loss_v18_proposal_protection": protection_loss,
            },
            (set_loss + protection_loss) / 4.0,
            {
                "training": {
                    "v18_gradient_conflict_projection": {
                        "enabled": True,
                        "backward_mode": backward_mode,
                        "complete_loss_partition": True,
                        "shared_prefixes": ["encoder.proj."],
                        "exclude_prefixes": [],
                    }
                }
            },
            accumulation_steps=4,
            scaler=None,
            amp=False,
        )
        assert stats is not None
        return stats

    reference_stats = run(reference, "reference_three_pass")
    optimized_stats = run(optimized, "partitioned_two_pass")
    for reference_parameter, optimized_parameter in zip(
        reference.parameters(), optimized.parameters()
    ):
        assert reference_parameter.grad is not None
        assert optimized_parameter.grad is not None
        torch.testing.assert_close(
            optimized_parameter.grad,
            reference_parameter.grad,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
    for key in reference_stats:
        torch.testing.assert_close(
            optimized_stats[key],
            reference_stats[key],
            rtol=1.0e-6,
            atol=1.0e-7,
        )


def _integrated_head() -> FourSlotLaneSelectionHead:
    return FourSlotLaneSelectionHead(
        16,
        input_w=100,
        hidden_dim=32,
        num_slots=4,
        proposal_layers=1,
        slot_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        curve_samples=6,
        min_valid_rows=3,
        factorized_routing=True,
        active_prior_prob=0.99,
        refinement_enabled=True,
        refinement_hidden_dim=32,
        refinement_delta_offsets_px=(-12.0, -6.0, 0.0, 6.0, 12.0),
        refinement_straight_through_routing=True,
        refinement_detach_slot_states=True,
        refinement_structured_unique_routing=True,
        refinement_route_gradient_scale=0.0,
        range_refinement_enabled=True,
        range_delta_offsets_norm=(-0.1, 0.0, 0.1),
        joint_exact_set_energy_enabled=True,
        joint_exact_set_energy_hidden_dim=32,
        joint_exact_set_energy_num_heads=4,
        joint_exact_set_energy_ff_dim=64,
        joint_exact_set_energy_dropout=0.0,
        joint_exact_set_energy_scale_names=("p2", "p3", "p4"),
        joint_exact_set_energy_association_offsets_px=(-8.0, 0.0, 8.0),
        joint_exact_set_energy_visual_offsets_px=(-8.0, 0.0, 8.0),
        joint_exact_set_energy_delta_offsets_px=(-4.0, 0.0, 4.0),
        joint_exact_set_energy_range_offsets_norm=(-0.01, 0.0, 0.01),
    )


def _integrated_inputs() -> tuple[
    dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]
]:
    batch, candidates, rows, dim = 1, 32, 8, 16
    centers = torch.linspace(5.0, 94.0, candidates).view(1, candidates, 1)
    curve = centers + torch.linspace(-2.0, 2.0, rows).view(1, 1, rows)
    proposals = {
        "structured_row_tokens": torch.randn(
            batch, candidates, rows, dim, requires_grad=True
        ),
        "queries": torch.randn(batch, candidates, dim, requires_grad=True),
        "ownership_state": torch.randn(
            batch, candidates, dim, requires_grad=True
        ),
        "range_norm": torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(
            batch, candidates, 2
        ).clone().requires_grad_(),
        "pred_x_rows": curve.clone().requires_grad_(),
        "row_x_logits": torch.randn(
            batch, candidates, rows, 7, requires_grad=True
        ),
        "exist_logits": torch.randn(
            batch, candidates, 2, requires_grad=True
        ),
        "input_reference_x_rows": curve.clone().requires_grad_(),
    }
    p2 = torch.randn(batch, rows, 25, dim, requires_grad=True)
    multi = {
        "p2": p2.permute(0, 3, 1, 2),
        "p3": torch.randn(batch, dim, 4, 13, requires_grad=True),
        "p4": torch.randn(batch, dim, 2, 7, requires_grad=True),
    }
    return proposals, p2, multi


def test_integrated_v18_step_zero_is_exact_v7_and_loss_is_finite() -> None:
    torch.manual_seed(182)
    head = _integrated_head()
    proposals, p2, multi = _integrated_inputs()
    module = head.joint_exact_set_energy
    assert module is not None
    head.joint_exact_set_energy = None
    with torch.no_grad():
        source = head(
            proposals, row_value_features=p2, multi_scale_features=multi
        )
    head.joint_exact_set_energy = module
    treatment = head(
        proposals, row_value_features=p2, multi_scale_features=multi
    )
    for name in (
        "selection_slot_geometry_route_indices",
        "selection_slot_indices",
        "selection_slot_scores",
        "selection_slot_active_logits",
        "selection_slot_active",
        "selection_slot_pred_x_rows",
        "selection_slot_range_norm",
    ):
        assert torch.equal(source[name], treatment[name]), name
    assert torch.count_nonzero(
        treatment["selection_slot_v18_unary_residual"]
    ) == 0
    assert torch.count_nonzero(treatment["selection_slot_v18_pair_energy"]) == 0
    assert torch.count_nonzero(treatment["selection_slot_v18_policy"]) == 0

    gt_x = torch.stack(
        (proposals["pred_x_rows"][0, 3], proposals["pred_x_rows"][0, 14], proposals["pred_x_rows"][0, 26])
    ).detach()
    targets = [
        {
            "x_rows": gt_x,
            "valid_mask": torch.ones_like(gt_x, dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 49.0]] * 3),
        }
    ]
    criterion = S0Criterion(
        LossConfig(
            input_w=100,
            input_h=50,
            four_slot_line_width=15.0,
            four_slot_min_valid_rows=3,
            w_four_slot_v18=1.0,
        )
    )
    merged = {**proposals, **treatment}
    loss = criterion.compute_four_slot_v18_loss(merged, targets)
    assert torch.isfinite(loss["total"])
    assert torch.isfinite(loss["set"])
    loss["total"].backward()
    assert module.unary_interaction[-1].weight.grad is not None
    assert module.unary_interaction[-1].weight.grad.norm() > 0
    assert module.pair_output.weight.grad is not None
    assert module.pair_output.weight.grad.norm() > 0
    assert module.refiner.delta_head.weight.grad is not None
    assert module.refiner.delta_head.weight.grad.norm() > 0
