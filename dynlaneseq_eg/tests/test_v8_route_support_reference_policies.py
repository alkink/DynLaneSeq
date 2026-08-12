from __future__ import annotations

from argparse import Namespace

import torch

from dynlaneseq_eg.losses.loss_s0 import (
    _slot_assignment_paths,
    _vectorized_slot_path_costs,
)
from dynlaneseq_eg.tools.audit_v8_route_support_reference_policies import (
    _hard_min_slots,
    _limited_indices,
    _sampling_limit,
    _target_hard_unique_routes,
    _target_soft_weights,
)


def test_sampling_limit_supports_complete_fixed_lists() -> None:
    args = Namespace(max_images=0, eval_batch_size=8)
    assert _sampling_limit(args) == (0, None)
    assert _limited_indices([1, 2, 3], None) == [1, 2, 3]
    args.max_images = 17
    assert _sampling_limit(args) == (3, 17)
    assert _limited_indices(list(range(20)), 17) == list(range(17))


def test_hard_min_slots_respects_route_and_activity_cost() -> None:
    active = torch.tensor([4.0, 4.0, -4.0, -4.0])
    logits = torch.tensor(
        [
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, 8.0],
        ]
    )
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert _hard_min_slots(active, logits, target) == [0, 1]


def test_hard_min_slots_matches_production_vectorized_contract() -> None:
    generator = torch.Generator().manual_seed(3407)
    for gt_count in (1, 2, 3, 4):
        for _ in range(20):
            active = torch.randn(4, generator=generator)
            logits = torch.randn(4, 32, generator=generator)
            target = torch.softmax(
                torch.randn(gt_count, 32, generator=generator), dim=-1
            )
            candidate_cost = -torch.einsum(
                "sn,gn->sg", torch.log_softmax(logits, dim=-1), target
            )
            candidate_cost += torch.nn.functional.softplus(-active).unsqueeze(-1)
            inactive_cost = torch.nn.functional.softplus(active)
            paths, inactive_mask = _slot_assignment_paths(4, gt_count, logits.device)
            costs = _vectorized_slot_path_costs(
                candidate_cost.unsqueeze(0),
                inactive_cost.unsqueeze(0),
                paths,
                inactive_mask,
            )[0]
            expected = paths[int(costs.argmin())].tolist()
            assert _hard_min_slots(active, logits, target) == expected


def test_target_hard_routes_are_unique_even_when_argmax_collides() -> None:
    target = torch.tensor(
        [[0.6, 0.4, 0.0, 0.0], [0.7, 0.0, 0.3, 0.0]]
    )
    current = torch.tensor([0, 1, 2, 3])
    logits = torch.zeros((4, 4))
    routes, argmax, collisions = _target_hard_unique_routes(
        target,
        [0, 1],
        current,
        logits,
        torch.ones(4, dtype=torch.bool),
    )
    assert argmax == [0, 0]
    assert collisions == 1
    assert len(set(routes.tolist())) == 4
    assert routes[0].item() in {0, 1}
    assert routes[1].item() in {0, 2}


def test_target_soft_weights_preserve_assigned_distributions() -> None:
    target = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.2, 0.8]])
    weight = _target_soft_weights(
        target,
        [1, 3],
        torch.tensor([0, 1, 2, 0]),
        3,
    )
    assert torch.equal(weight[1], target[0])
    assert torch.equal(weight[3], target[1])
    assert torch.equal(weight[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(weight[2], torch.tensor([0.0, 0.0, 1.0]))
