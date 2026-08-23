from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v25_dual_energy_multi_path import (
    V25DualEnergyMultiPath,
    diverse_viterbi_paths,
    dual_energy_log_mixture,
    exact_small_path_set_decode,
)
from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    V25ImageMediatedLaneObjects,
    V25LossWeights,
    v25_lane_object_loss,
)


def _target(rows: int) -> list[dict[str, torch.Tensor]]:
    lane0 = torch.linspace(12.0, 20.0, rows)
    lane1 = torch.linspace(42.0, 52.0, rows)
    x = torch.stack((lane0, lane1))
    return [{"x_rows": x, "valid_mask": torch.ones_like(x, dtype=torch.bool)}]


def test_diverse_viterbi_keeps_two_separate_spatial_modes() -> None:
    rows, bins = 7, 24
    first = torch.tensor([4, 5, 6, 7, 8, 9, 10])
    second = torch.tensor([15, 15, 16, 16, 17, 17, 18])
    logits = torch.full((1, 1, rows, bins), -12.0)
    logits[0, 0, torch.arange(rows), first] = 12.0
    logits[0, 0, torch.arange(rows), second] = 10.0
    result = diverse_viterbi_paths(
        logits,
        num_hypotheses=2,
        transition_radius_bins=2,
        transition_penalty=0.1,
        suppression_radius_bins=2,
        suppression_penalty=30.0,
    )
    assert torch.equal(result.indices[0, 0, 0], first)
    assert torch.equal(result.indices[0, 0, 1], second)
    assert result.scores[0, 0, 0] > result.scores[0, 0, 1]


def test_dual_energy_mixture_preserves_two_modes() -> None:
    image = torch.full((1, 1, 1, 20), -20.0)
    proposal = torch.full_like(image, -20.0)
    image[..., 4] = 20.0
    proposal[..., 15] = 20.0
    fused, weights = dual_energy_log_mixture(
        image, proposal, torch.zeros((1, 1, 2))
    )
    top = fused[0, 0, 0].topk(2).indices.sort().values
    assert torch.equal(top, torch.tensor([4, 15]))
    assert torch.allclose(weights, torch.full_like(weights, 0.5))


def test_exact_set_decode_uses_alternative_to_avoid_crossing() -> None:
    rows = 5
    # Slot 0's locally strongest path crosses slot 1. Its second path is valid.
    hypotheses = torch.tensor(
        [[
            [[60.0] * rows, [20.0] * rows],
            [[40.0] * rows, [80.0] * rows],
            [[100.0] * rows, [110.0] * rows],
            [[130.0] * rows, [140.0] * rows],
        ]]
    )
    scores = torch.tensor([[[0.0, -0.2], [0.0, -2.0], [0.0, -2.0], [0.0, -2.0]]])
    exist = torch.tensor([[[8.0, -8.0]] * 4])
    decoded = exact_small_path_set_decode(
        hypotheses,
        scores,
        exist,
        input_w=160,
        minimum_spacing_px=10.0,
        order_penalty=20.0,
        duplicate_penalty=5.0,
    )
    assert decoded["selected_path_indices"][0, 0].item() == 1
    assert torch.all(decoded["selected_paths"][0, 0] == 20.0)


def test_advanced_graph_has_dual_energy_visibility_and_trainable_proposals() -> None:
    torch.manual_seed(17)
    model = V25DualEnergyMultiPath(
        input_h=32,
        input_w=64,
        num_rows=6,
        x_bins=16,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        transition_penalty=0.1,
        enable_competition=True,
        enable_slot_interaction=True,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
        proposal_count=8,
        proposal_groups=4,
        num_path_hypotheses=2,
        exact_set_selection=True,
        proposal_dropout=0.0,
    ).train()
    output = model(torch.randn(1, 3, 32, 64))
    assert output["image_energy_logits"].shape == (1, 4, 6, 16)
    assert output["proposal_energy_logits"].shape == (1, 4, 6, 16)
    assert output["proposal_unary_logits"].shape == (1, 8, 6, 16)
    assert output["mixture_weights"].shape == (1, 4, 2)
    assert output["row_visibility_logits"].shape == (1, 4, 6)
    loss, diagnostics = v25_lane_object_loss(
        output,
        _target(6),
        input_w=64,
        weights=V25LossWeights(
            minimum_valid_rows=2,
            visibility=0.5,
            proposal_coverage=0.5,
            proposal_groups=4,
        ),
    )
    assert torch.isfinite(loss)
    assert diagnostics["loss_visibility"] > 0
    assert diagnostics["loss_proposal_coverage"] > 0
    loss.backward()
    for prefix in ("proposal_memory", "energy_mixture", "row_visibility_head", "reliability_head"):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients, prefix
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
            for gradient in gradients
        ), prefix


def test_advanced_eval_writes_one_of_the_hard_hypotheses() -> None:
    torch.manual_seed(23)
    model = V25DualEnergyMultiPath(
        input_h=32,
        input_w=64,
        num_rows=6,
        x_bins=16,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
        proposal_count=8,
        proposal_groups=4,
        num_path_hypotheses=2,
        exact_set_selection=True,
        proposal_dropout=0.0,
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 64))
    assert output["path_hypotheses"].shape == (1, 4, 2, 6)
    assert output["selected_path_indices"].shape == (1, 4)
    assert output["pred_x_rows"].shape == (1, 4, 6)
    for slot in range(4):
        choice = int(output["selected_path_indices"][0, slot])
        if choice < 2:
            assert torch.equal(
                output["pred_x_rows"][0, slot],
                output["path_hypotheses"][0, slot, choice],
            )


def test_aux_only_arm_preserves_primary_writer_at_initialization() -> None:
    """An unfused auxiliary bank must be training-only at gate zero.

    V33 Arm B is allowed to send proposal-coverage gradients into the shared
    image encoder, but the newly initialised private proposal modules must not
    change the primary prediction before any update.  This makes Arm A/B a
    causal auxiliary-supervision comparison rather than an inference rewrite.
    """

    common = dict(
        input_h=32,
        input_w=64,
        num_rows=6,
        x_bins=16,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        transition_penalty=0.1,
        enable_competition=False,
        enable_slot_interaction=False,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
    )
    torch.manual_seed(101)
    primary = V25ImageMediatedLaneObjects(**common).eval()
    torch.manual_seed(202)
    auxiliary = V25DualEnergyMultiPath(
        **common,
        proposal_count=8,
        proposal_groups=4,
        proposal_dropout=0.0,
        enable_proposal_fusion=False,
        num_path_hypotheses=1,
        exact_set_selection=False,
    ).eval()
    advanced_state = auxiliary.state_dict()
    for name, value in primary.state_dict().items():
        assert name in advanced_state
        advanced_state[name] = value.detach().clone()
    auxiliary.load_state_dict(advanced_state, strict=True)

    images = torch.randn(2, 3, 32, 64)
    with torch.no_grad():
        source = primary(images)
        treatment = auxiliary(images)
    for name in (
        "unary_logits",
        "path_logits",
        "soft_x_rows",
        "hard_path_x_rows",
        "pred_x_rows",
        "exist_logits",
        "range_norm",
    ):
        assert torch.allclose(source[name], treatment[name], atol=1.0e-6), name

    auxiliary.train()
    auxiliary.zero_grad(set_to_none=True)
    output = auxiliary(images[:1])
    loss, diagnostics = v25_lane_object_loss(
        output,
        _target(6),
        input_w=64,
        weights=V25LossWeights(
            minimum_valid_rows=2,
            quality50=0.0,
            quality75=0.0,
            visibility=0.0,
            proposal_coverage=1.0,
            proposal_groups=4,
        ),
    )
    assert diagnostics["loss_proposal_coverage"] > 0
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in auxiliary.proposal_memory.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in auxiliary.backbone.parameters()
    )
