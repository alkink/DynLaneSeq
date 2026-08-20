from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v28_refined_belief_router import (
    V28RefinedBeliefRouter,
    build_v28_refined_route_targets,
    decode_v28_unique_routes,
    gather_canonical_slots,
    gather_slot_candidates,
    restore_public_slots,
    score_slot_candidate_paths,
    v28_refined_belief_loss,
)


def _ranges(*shape: int) -> torch.Tensor:
    return torch.tensor([0.0, 1.0]).view(*([1] * len(shape)), 2).expand(
        *shape, 2
    ).clone()


def test_canonical_slot_round_trip_and_candidate_gather() -> None:
    value = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    order = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    canonical = gather_canonical_slots(value, order)
    assert torch.equal(restore_public_slots(canonical, order), value)

    bank = torch.arange(2 * 4 * 5 * 3).reshape(2, 4, 5, 3)
    indices = torch.tensor([[0, 2, 4, 1], [4, 3, 2, 1]])
    selected = gather_slot_candidates(bank, indices)
    for batch in range(2):
        for slot in range(4):
            assert torch.equal(
                selected[batch, slot],
                bank[batch, slot, indices[batch, slot]],
            )


def test_path_score_reads_slot_specific_refined_curves() -> None:
    rows, bins, width = 8, 32, 128
    logits = torch.full((1, 4, rows, bins), -8.0)
    peak_bins = (4, 10, 18, 26)
    for slot, peak in enumerate(peak_bins):
        logits[0, slot, :, peak] = 8.0
    log_probability = logits.log_softmax(dim=-1)
    candidate_x = torch.zeros(1, 4, 3, rows)
    for slot, peak in enumerate(peak_bins):
        # Bin centre mapping in score_slot_candidate_paths.
        candidate_x[0, slot, 0] = (peak + 0.5) * (width / bins)
        candidate_x[0, slot, 1] = ((peak + 5) % bins + 0.5) * (width / bins)
        candidate_x[0, slot, 2] = ((peak + 9) % bins + 0.5) * (width / bins)
    scores, valid = score_slot_candidate_paths(
        log_probability,
        candidate_x=candidate_x,
        candidate_range=_ranges(1, 4, 3),
        candidate_valid=torch.ones(1, 4, 3, dtype=torch.bool),
        input_w=width,
        minimum_valid_rows=5,
    )
    assert valid.all()
    assert scores.argmax(dim=-1).tolist() == [[0, 0, 0, 0]]


def test_exact_unique_decoder_respects_slot_specific_masks() -> None:
    scores = torch.tensor(
        [[[9.0, 8.0, 0.0], [9.0, 7.0, 6.5], [1.0, 8.0, 7.0]]]
    )
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[0, 1, 0] = False
    routes = decode_v28_unique_routes(scores, valid)
    assert routes.tolist() == [[0, 2, 1]]
    assert len(set(routes[0].tolist())) == 3


def test_refined_route_target_prefers_owned_candidate() -> None:
    rows = 12
    owned = torch.tensor([20.0, 45.0, 70.0, 95.0]).view(1, 4, 1)
    owned = owned.expand(1, 4, rows).clone()
    proposal_positions = torch.tensor([21.0, 46.0, 71.0, 96.0, 115.0])
    candidate_x = proposal_positions.view(1, 1, 5, 1).expand(
        1, 4, 5, rows
    ).clone()
    result = build_v28_refined_route_targets(
        candidate_x=candidate_x,
        candidate_range=_ranges(1, 4, 5),
        candidate_valid=torch.ones(1, 4, 5, dtype=torch.bool),
        owned_x=owned,
        owned_valid=torch.ones_like(owned, dtype=torch.bool),
        owned_matched=torch.ones(1, 4, dtype=torch.bool),
        input_h=64,
        line_width=30.0,
        minimum_valid_rows=5,
        temperature=0.05,
        support_delta=0.01,
        support_floor=0.0,
    )
    assert result["quality"].argmax(dim=-1).tolist() == [[0, 1, 2, 3]]
    assert result["matched"].all()
    torch.testing.assert_close(
        result["probability"].sum(dim=-1),
        torch.ones(1, 4),
    )


def test_arm_b_route_gradient_reaches_image_encoder_and_arm_c_adds_field() -> None:
    torch.manual_seed(19)
    rows, bins, width = 16, 32, 128
    model = V28RefinedBeliefRouter(
        input_h=64,
        input_w=width,
        num_rows=rows,
        x_bins=bins,
        fpn_channels=32,
        hidden_dim=32,
        query_dim=32,
        vertical_layers=1,
        num_heads=4,
        ff_dim=64,
    ).train()
    source_x = torch.tensor([20.0, 45.0, 70.0, 95.0]).view(1, 4, 1)
    source_x = source_x.expand(1, 4, rows).clone()
    source_range = _ranges(1, 4)
    source_active = torch.ones(1, 4, dtype=torch.bool)
    source_route = torch.tensor([[0, 1, 2, 3]])
    positions = torch.tensor([22.0, 47.0, 72.0, 97.0, 116.0])
    candidate_x = positions.view(1, 1, 5, 1).expand(
        1, 4, 5, rows
    ).clone()
    candidate_range = _ranges(1, 4, 5)
    candidate_valid = torch.ones(1, 4, 5, dtype=torch.bool)
    target_x = positions[:4].view(4, 1).expand(4, rows).clone()
    targets = [
        {
            "x_rows": target_x,
            "valid_mask": torch.ones_like(target_x, dtype=torch.bool),
        }
    ]
    output = model(
        torch.randn(1, 3, 64, width),
        source_x=source_x,
        source_range=source_range,
        source_active=source_active,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=candidate_valid,
    )
    loss_b, diagnostics_b = v28_refined_belief_loss(
        output,
        targets,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=candidate_valid,
        source_x=source_x,
        source_range=source_range,
        source_active=source_active,
        source_route=source_route,
        input_h=64,
        input_w=width,
        arm="B",
    )
    loss_c, diagnostics_c = v28_refined_belief_loss(
        output,
        targets,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=candidate_valid,
        source_x=source_x,
        source_range=source_range,
        source_active=source_active,
        source_route=source_route,
        input_h=64,
        input_w=width,
        arm="C",
    )
    torch.testing.assert_close(
        loss_c, loss_b + diagnostics_c["loss_field"], rtol=1.0e-6, atol=1.0e-6
    )
    assert diagnostics_b["loss_route"] == diagnostics_c["loss_route"]
    loss_b.backward()
    groups = (
        model.backbone,
        model.fpn,
        model.key_projection,
        model.vertical_encoder,
    )
    for group in groups:
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in group.parameters()
        )
