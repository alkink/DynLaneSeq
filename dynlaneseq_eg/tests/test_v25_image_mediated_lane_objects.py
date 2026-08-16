from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.v25_image_mediated_lane_objects import (
    V25ImageMediatedLaneObjects,
    V25LossWeights,
    build_ordered_lane_targets,
    hard_viterbi_paths,
    image_mediated_other_coverage,
    v25_lane_object_loss,
    v25_model_contract,
)


def _target(rows: int, lanes: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    if not lanes:
        return {
            "x_rows": torch.zeros((0, rows)),
            "valid_mask": torch.zeros((0, rows), dtype=torch.bool),
        }
    x = torch.stack(lanes)
    return {"x_rows": x, "valid_mask": torch.ones_like(x, dtype=torch.bool)}


def test_hard_viterbi_recovers_exact_coherent_path() -> None:
    rows, bins = 9, 21
    expected = torch.tensor([4, 5, 6, 7, 8, 9, 8, 7, 6])
    logits = torch.full((1, 1, rows, bins), -20.0)
    logits[0, 0, torch.arange(rows), expected] = 20.0
    path = hard_viterbi_paths(
        logits, transition_radius_bins=2, transition_penalty=0.2
    )
    assert torch.equal(path[0, 0], expected)


def test_ordered_targets_sort_by_bottom_x_and_pad() -> None:
    rows = 8
    right = torch.linspace(100.0, 130.0, rows)
    left = torch.linspace(20.0, 40.0, rows)
    ordered = build_ordered_lane_targets(
        [_target(rows, [right, left])],
        device=torch.device("cpu"),
        slots=4,
        rows=rows,
        input_w=160,
        minimum_valid_rows=2,
    )
    assert ordered["active"].tolist() == [[True, True, False, False]]
    assert torch.equal(ordered["x_rows"][0, 0], left)
    assert torch.equal(ordered["x_rows"][0, 1], right)
    assert ordered["counts"].tolist() == [2]


def test_other_coverage_excludes_current_slot() -> None:
    probability = torch.zeros((1, 4, 1, 5))
    probability[0, 0, 0, 2] = 0.8
    probability[0, 1, 0, 2] = 0.5
    other = image_mediated_other_coverage(probability)
    assert torch.allclose(other[0, 0, 0, 2], torch.tensor(0.5))
    assert torch.allclose(other[0, 1, 0, 2], torch.tensor(0.8))
    assert torch.allclose(other[0, 2, 0, 2], torch.tensor(0.9))


def test_v25_core_owns_writer_geometry_and_all_major_blocks_get_gradient() -> None:
    torch.manual_seed(7)
    rows, bins, width, height = 8, 32, 128, 64
    model = V25ImageMediatedLaneObjects(
        input_h=height,
        input_w=width,
        num_rows=rows,
        x_bins=bins,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=2,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        transition_penalty=0.1,
        enable_competition=False,
        enable_slot_interaction=False,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
        freeze_batch_norm_stats=True,
    ).train()
    images = torch.randn(2, 3, height, width)
    output = model(images)
    assert output["pred_x_rows"].shape == (2, 4, rows)
    assert output["range_norm"].shape == (2, 4, 2)
    assert output["exist_logits"].shape == (2, 4, 2)
    assert output["unary_logits"].shape == (2, 4, rows, bins)
    targets = [
        _target(
            rows,
            [
                torch.linspace(20.0, 28.0, rows),
                torch.linspace(76.0, 82.0, rows),
            ],
        ),
        _target(rows, [torch.linspace(45.0, 52.0, rows)]),
    ]
    loss, diagnostics = v25_lane_object_loss(
        output,
        targets,
        input_w=width,
        weights=V25LossWeights(minimum_valid_rows=2),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(diagnostics["mean_soft_iou"])
    loss.backward()
    named = dict(model.named_parameters())
    for prefix in (
        "backbone.base_layer.0",
        "fpn.lateral.c2",
        "fine_stem.0",
        "decoder.0.context_projection",
        "decoder.0.vertical",
        "exist_head",
        "range_head",
        "quality_head",
    ):
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if name.startswith(prefix)
        ]
        assert gradients, prefix
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum() > 0
            for gradient in gradients
        ), prefix
    contract = v25_model_contract(model)
    assert contract["final_geometry_owner"] == "four_image_mediated_lane_objects"
    assert contract["writer_depends_on_v7"] is False
    assert contract["global_geometry_gate_present"] is False
    forbidden = ("teacher", "geometry_gate", "router")
    assert not any(
        token in name
        for name, _parameter in model.named_parameters()
        for token in forbidden
    )


def test_v25_cost_volume_depends_on_the_correct_image() -> None:
    torch.manual_seed(11)
    model = V25ImageMediatedLaneObjects(
        input_h=64,
        input_w=128,
        num_rows=8,
        x_bins=32,
        fpn_channels=32,
        hidden_dim=32,
        decoder_layers=1,
        num_heads=4,
        ff_dim=64,
        dropout=0.0,
        transition_radius_bins=2,
        pretrained_backbone=False,
        require_pretrained_backbone=False,
    ).eval()
    first = model(torch.zeros(1, 3, 64, 128))["unary_logits"]
    second = model(torch.ones(1, 3, 64, 128))["unary_logits"]
    assert not torch.equal(first, second)
    assert float((first - second).detach().abs().mean()) > 1.0e-6
