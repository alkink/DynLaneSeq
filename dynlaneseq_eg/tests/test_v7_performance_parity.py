from __future__ import annotations

from itertools import permutations
import math

import pytest
import torch
from torch.nn import functional as F

from dynlaneseq_eg.losses.loss_s0 import (
    build_four_slot_cluster_targets,
    four_slot_factorized_permutation_loss,
)
from dynlaneseq_eg.losses.matcher_s0 import HungarianMatcherS0, MatcherConfig
from dynlaneseq_eg.losses.range_aware_iou import (
    batched_pairwise_range_aware_row_strip_iou,
    pairwise_range_aware_row_strip_iou,
)
from dynlaneseq_eg.modeling.four_slot_selection import (
    decode_unique_real_slot_routes,
    structured_unique_route_marginals,
)
from dynlaneseq_eg.modeling.common import (
    fixed_linspace,
    fixed_row_fractions,
    fixed_sample_indices,
    fixed_y_rows,
    soft_expected_x,
)
from dynlaneseq_eg.modeling.position_encoding import SinePositionEncoding2D
from dynlaneseq_eg.modeling.structured_queries import (
    prepare_shared_grid_sample_feature_map,
    reuse_shared_grid_sample_feature_map,
)


def _reference_structured_marginals(
    logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    batch, slots, candidates = logits.shape
    outputs: list[torch.Tensor] = []
    for batch_index in range(batch):
        valid_ids = torch.nonzero(
            candidate_valid[batch_index], as_tuple=False
        ).flatten()
        scores = logits[batch_index, :, valid_ids].float() / temperature
        dummy = scores.new_zeros((int(valid_ids.numel()) - slots, int(valid_ids.numel())))
        log_transport = torch.cat((scores, dummy), dim=0)
        for _ in range(20):
            log_transport = log_transport - torch.logsumexp(
                log_transport, dim=1, keepdim=True
            )
            log_transport = log_transport - torch.logsumexp(
                log_transport, dim=0, keepdim=True
            )
        marginal = logits.new_zeros((slots, candidates), dtype=torch.float32)
        marginal[:, valid_ids] = log_transport[:slots].exp()
        outputs.append(marginal)
    return torch.stack(outputs)


def _reference_factorized_loss(
    active_logits: torch.Tensor,
    real_route_logits: torch.Tensor,
    target_rows: list[torch.Tensor],
    *,
    temperature: float,
    mode: str,
) -> torch.Tensor:
    _batch, slots, candidates = real_route_logits.shape
    real_log_probability = F.log_softmax(real_route_logits.float(), dim=-1)
    active_cost = F.softplus(-active_logits.float())
    inactive_cost = F.softplus(active_logits.float())
    losses: list[torch.Tensor] = []
    for batch_index, target_value in enumerate(target_rows):
        target = target_value.to(
            device=real_route_logits.device,
            dtype=real_log_probability.dtype,
        )
        gt_count = int(target.shape[0])
        if gt_count == 0:
            losses.append(inactive_cost[batch_index].sum())
            continue
        candidate_cost = -torch.einsum(
            "sn,gn->sg",
            real_log_probability[batch_index],
            target[:, :candidates],
        )
        candidate_cost = candidate_cost + active_cost[batch_index].unsqueeze(-1)
        path_costs: list[torch.Tensor] = []
        for assigned_slots in permutations(range(int(slots)), gt_count):
            assigned = set(int(value) for value in assigned_slots)
            cost = candidate_cost.new_zeros(())
            for gt_index, slot_index in enumerate(assigned_slots):
                cost = cost + candidate_cost[int(slot_index), gt_index]
            for slot_index in range(int(slots)):
                if slot_index not in assigned:
                    cost = cost + inactive_cost[batch_index, slot_index]
            path_costs.append(cost)
        stacked = torch.stack(path_costs)
        if mode == "hard_min":
            losses.append(stacked[stacked.detach().argmin()])
        else:
            losses.append(
                -temperature * torch.logsumexp(-stacked / temperature, dim=0)
                + temperature * math.log(float(len(path_costs)))
            )
    return torch.stack(losses).mean() / float(slots)


def _soft_target_rows(
    *, candidates: int, counts: tuple[int, ...]
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3407)
    rows: list[torch.Tensor] = []
    for count in counts:
        value = torch.rand(count, candidates + 1, generator=generator)
        if count:
            value[:, -1] = 0.0
            value[:, :-1] /= value[:, :-1].sum(dim=-1, keepdim=True)
        rows.append(value)
    return rows


def test_cached_row_constants_preserve_exact_values_and_expected_x_gradient():
    first = fixed_y_rows(17, 123, dtype=torch.float32)
    second = fixed_y_rows(17, 123, dtype=torch.float32)
    reference_rows = torch.arange(17, dtype=torch.float32) * (123.0 / 17.0)
    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(first, reference_rows)
    assert torch.equal(
        fixed_linspace(-1.0, 1.0, 17),
        torch.linspace(-1.0, 1.0, 17),
    )
    assert torch.equal(
        fixed_row_fractions(17),
        torch.arange(17, dtype=torch.float32) / 17.0,
    )
    assert torch.equal(
        fixed_sample_indices(17, 7),
        torch.linspace(0, 16, 7).round().long(),
    )

    generator = torch.Generator().manual_seed(1984)
    fast_logits = torch.randn(2, 3, 17, generator=generator, requires_grad=True)
    reference_logits = fast_logits.detach().clone().requires_grad_(True)
    fast = soft_expected_x(fast_logits, input_w=321, x_bins=17)
    probability = torch.softmax(reference_logits, dim=-1)
    centers = torch.arange(17, dtype=reference_logits.dtype)
    reference = (probability * centers).sum(dim=-1) * (321.0 / 17.0)
    assert torch.equal(fast, reference)
    fast.sum().backward()
    reference.sum().backward()
    assert torch.equal(fast_logits.grad, reference_logits.grad)


def test_cached_sine_position_encoding_is_bit_identical_to_original_formula():
    module = SinePositionEncoding2D(dim=32, temperature=10000)
    feature = torch.zeros(2, 7, 5, 9, dtype=torch.float32)
    first = module(feature)
    second = module(feature)
    y = torch.linspace(0, 1, 5)
    xcoord = torch.linspace(0, 1, 9)
    yy, xx = torch.meshgrid(y, xcoord, indexing="ij")
    omega = torch.arange(8, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / 7.0))
    out_x = xx[..., None] * omega * 2 * math.pi
    out_y = yy[..., None] * omega * 2 * math.pi
    reference = torch.cat(
        (out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()),
        dim=-1,
    ).permute(2, 0, 1).unsqueeze(0)
    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(first, reference)


def test_shared_fp32_feature_map_preserves_per_layer_values_and_bf16_gradient():
    generator = torch.Generator().manual_seed(9173)
    fast_source = torch.randn(
        2,
        5,
        7,
        8,
        generator=generator,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_source = fast_source.detach().clone().requires_grad_(True)
    cached = prepare_shared_grid_sample_feature_map(fast_source)
    fast_values = [
        reuse_shared_grid_sample_feature_map(fast_source, cached)
        for _ in range(4)
    ]
    reference_values = [
        reference_source.float().permute(0, 3, 1, 2).contiguous()
        for _ in range(4)
    ]
    assert all(torch.equal(value, reference_values[0]) for value in fast_values)
    assert all(value.data_ptr() == cached.data_ptr() for value in fast_values)

    weights = [
        torch.randn(cached.shape, generator=generator)
        for _ in range(4)
    ]
    fast_loss = sum(
        (value * weight).square().mean()
        for value, weight in zip(fast_values, weights)
    )
    reference_loss = sum(
        (value * weight).square().mean()
        for value, weight in zip(reference_values, weights)
    )
    assert torch.equal(fast_loss, reference_loss)
    fast_loss.backward()
    reference_loss.backward()
    assert torch.equal(fast_source.grad, reference_source.grad)


def test_batched_structured_sinkhorn_matches_image_loop_and_gradient():
    generator = torch.Generator().manual_seed(3407)
    fast_logits = torch.randn(3, 4, 9, generator=generator, requires_grad=True)
    reference_logits = fast_logits.detach().clone().requires_grad_(True)
    valid = torch.ones((3, 9), dtype=torch.bool)
    fast = structured_unique_route_marginals(
        fast_logits, valid, temperature=0.7
    )
    reference = _reference_structured_marginals(
        reference_logits, valid, temperature=0.7
    )
    assert torch.allclose(fast, reference, atol=1.0e-7, rtol=1.0e-6)
    weight = torch.randn(fast.shape, generator=generator)
    (fast * weight).sum().backward()
    (reference * weight).sum().backward()
    assert torch.allclose(
        fast_logits.grad,
        reference_logits.grad,
        atol=2.0e-7,
        rtol=2.0e-6,
    )


def test_masked_batched_sinkhorn_matches_variable_valid_image_loops():
    generator = torch.Generator().manual_seed(1181)
    fast_logits = torch.randn(3, 4, 9, generator=generator, requires_grad=True)
    reference_logits = fast_logits.detach().clone().requires_grad_(True)
    valid = torch.ones((3, 9), dtype=torch.bool)
    valid[1, -2:] = False
    valid[2, 1::2] = False
    fast = structured_unique_route_marginals(
        fast_logits,
        valid,
        temperature=0.7,
    )
    reference = _reference_structured_marginals(
        reference_logits,
        valid,
        temperature=0.7,
    )
    assert torch.allclose(fast, reference, atol=2.0e-7, rtol=2.0e-6)
    weight = torch.randn(fast.shape, generator=generator)
    (fast * weight).sum().backward()
    (reference * weight).sum().backward()
    assert torch.allclose(
        fast_logits.grad,
        reference_logits.grad,
        atol=3.0e-7,
        rtol=3.0e-6,
    )

    insufficient = torch.zeros((2, 9), dtype=torch.bool)
    insufficient[0, :3] = True
    fallback = structured_unique_route_marginals(
        torch.randn(2, 4, 9, generator=generator),
        insufficient,
    )
    assert torch.count_nonzero(fallback) == 0


def test_batched_row_strip_iou_matches_per_image_values_and_gradients():
    generator = torch.Generator().manual_seed(8128)
    pred_batched = (torch.rand(3, 6, 12, generator=generator) * 100.0).requires_grad_(
        True
    )
    pred_reference = pred_batched.detach().clone().requires_grad_(True)
    ranges = torch.rand(3, 6, 2, generator=generator)
    ranges = torch.sort(ranges, dim=-1).values
    gt = torch.rand(3, 4, 12, generator=generator) * 100.0
    valid = torch.rand(3, 4, 12, generator=generator) > 0.15
    valid[1, 3] = False

    batched, batched_candidate, batched_gt = (
        batched_pairwise_range_aware_row_strip_iou(
            pred_batched,
            ranges,
            gt,
            valid,
            input_h=120,
            line_width=30.0,
            min_valid_rows=5,
        )
    )
    reference_values = []
    reference_candidates = []
    reference_gt = []
    for batch_index in range(3):
        value, candidate_ok, gt_ok = pairwise_range_aware_row_strip_iou(
            pred_reference[batch_index],
            ranges[batch_index],
            gt[batch_index],
            valid[batch_index],
            input_h=120,
            line_width=30.0,
            min_valid_rows=5,
        )
        reference_values.append(value)
        reference_candidates.append(candidate_ok)
        reference_gt.append(gt_ok)
    reference = torch.stack(reference_values)
    assert torch.equal(batched_candidate, torch.stack(reference_candidates))
    assert torch.equal(batched_gt, torch.stack(reference_gt))
    assert torch.allclose(batched, reference, atol=1.0e-7, rtol=1.0e-6)

    weight = torch.randn(batched.shape, generator=generator)
    (batched * weight).sum().backward()
    (reference * weight).sum().backward()
    assert torch.allclose(
        pred_batched.grad,
        pred_reference.grad,
        atol=2.0e-7,
        rtol=2.0e-6,
    )


@pytest.mark.parametrize("mode", ["hard_min", "marginal"])
def test_vectorized_factorized_loss_matches_scalar_paths_and_gradient(mode: str):
    generator = torch.Generator().manual_seed(1701)
    active_fast = torch.randn(4, 4, generator=generator, requires_grad=True)
    route_fast = torch.randn(4, 4, 7, generator=generator, requires_grad=True)
    active_reference = active_fast.detach().clone().requires_grad_(True)
    route_reference = route_fast.detach().clone().requires_grad_(True)
    rows = _soft_target_rows(candidates=7, counts=(0, 2, 3, 4))
    fast = four_slot_factorized_permutation_loss(
        active_fast,
        route_fast,
        rows,
        permutation_temperature=0.8,
        assignment_mode=mode,
    )
    reference = _reference_factorized_loss(
        active_reference,
        route_reference,
        rows,
        temperature=0.8,
        mode=mode,
    )
    assert torch.allclose(fast, reference, atol=5.0e-7, rtol=2.0e-6)
    fast.backward()
    reference.backward()
    assert torch.allclose(
        active_fast.grad,
        active_reference.grad,
        atol=5.0e-7,
        rtol=2.0e-6,
    )
    assert torch.allclose(
        route_fast.grad,
        route_reference.grad,
        atol=5.0e-7,
        rtol=2.0e-6,
    )


def test_precomputed_real_route_combinations_preserve_exact_decode():
    generator = torch.Generator().manual_seed(90210)
    logits = torch.randn(3, 4, 11, generator=generator)
    valid = torch.ones((3, 11), dtype=torch.bool)
    valid[1, -2:] = False
    combinations = torch.tensor(
        tuple(__import__("itertools").product(range(4), repeat=4)),
        dtype=torch.long,
    )
    baseline = decode_unique_real_slot_routes(logits, valid)
    cached = decode_unique_real_slot_routes(logits, valid, combinations)
    for name in baseline:
        assert torch.equal(baseline[name], cached[name])


def test_all_gt_with_at_most_four_lanes_skips_joint_hungarian(monkeypatch):
    rows = 10
    outputs = {
        "pred_x_rows": torch.stack(
            (
                torch.full((rows,), 10.0),
                torch.full((rows,), 12.0),
                torch.full((rows,), 70.0),
            )
        ).unsqueeze(0),
        "range_norm": torch.tensor([[[0.0, 0.9]] * 3]),
    }
    targets = [
        {
            "x_rows": torch.stack(
                (torch.full((rows,), 10.0), torch.full((rows,), 70.0))
            ),
            "valid_mask": torch.ones((2, rows), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 90.0], [0.0, 90.0]]),
        }
    ]

    def fail_if_called(_cost):
        raise AssertionError("all_gt <= slots must not invoke Hungarian")

    monkeypatch.setattr(
        HungarianMatcherS0,
        "_linear_sum_assignment",
        staticmethod(fail_if_called),
    )
    result = build_four_slot_cluster_targets(
        outputs,
        targets,
        num_slots=4,
        input_h=100,
        line_width=30.0,
        min_valid_rows=5,
        representable_min=0.0,
        cluster_min=0.0,
        cluster_delta=0.1,
        temperature=0.03,
        target_mode="all_gt",
    )
    target = result["rows"][0]
    assert target.shape == (2, 4)
    assert torch.allclose(target.sum(dim=-1), torch.ones(2))


def test_batched_matcher_costs_preserve_per_image_costs_stats_and_assignment():
    generator = torch.Generator().manual_seed(6007)
    layers, batch, candidates, rows = 3, 3, 7, 11
    output_sequence = []
    for _ in range(layers):
        output_sequence.append(
            {
                "exist_logits": torch.randn(
                    batch, candidates, 2, generator=generator
                ),
                "pred_x_rows": torch.rand(
                    batch, candidates, rows, generator=generator
                )
                * 160.0,
                "range_norm": torch.rand(
                    batch, candidates, 2, generator=generator
                ),
            }
        )
    targets = []
    for gt_count in (0, 2, 4):
        valid = torch.rand(gt_count, rows, generator=generator) > 0.2
        if gt_count:
            valid[:, :5] = True
        targets.append(
            {
                "x_rows": torch.rand(
                    gt_count, rows, generator=generator
                )
                * 160.0,
                "valid_mask": valid,
                "range_y": torch.rand(
                    gt_count, 2, generator=generator
                )
                * 64.0,
            }
        )
    matcher = HungarianMatcherS0(
        MatcherConfig(
            input_w=160,
            input_h=64,
            lambda_obj=0.25,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            line_iou_radius=15.0,
            object_cost_type="neg_probability",
        )
    )
    batched_cost, batched_stats, counts = matcher.compute_cost_many(
        output_sequence,
        targets,
    )
    assert counts == (0, 2, 4)
    batched_matches = matcher.match_many(output_sequence, targets)
    for layer_index, output in enumerate(output_sequence):
        for batch_index, target in enumerate(targets):
            reference_cost, reference_stats = matcher.compute_cost_for_image(
                output["exist_logits"][batch_index],
                output["pred_x_rows"][batch_index],
                output["range_norm"][batch_index],
                target,
            )
            count = counts[batch_index]
            value = batched_cost[layer_index, batch_index, :, :count]
            assert torch.allclose(value, reference_cost, atol=2.0e-6, rtol=2.0e-6)
            for name, reference_stat in reference_stats.items():
                assert torch.allclose(
                    batched_stats[name][layer_index, batch_index],
                    reference_stat,
                    atol=2.0e-6,
                    rtol=2.0e-6,
                )
            reference_pred, reference_gt = matcher._linear_sum_assignment(
                reference_cost
            )
            assert torch.equal(
                batched_matches[layer_index][batch_index]["pred_indices"].cpu(),
                reference_pred,
            )
            assert torch.equal(
                batched_matches[layer_index][batch_index]["gt_indices"].cpu(),
                reference_gt,
            )
