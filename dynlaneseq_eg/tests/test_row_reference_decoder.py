from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.structured_queries import (
    ReferenceGuidedRowLayer,
    StructuredLaneQueryHead,
)


def _head(*, intermediate_supervision: bool = True) -> StructuredLaneQueryHead:
    return StructuredLaneQueryHead(
        dim=32,
        num_instances=4,
        num_rows=8,
        x_bins=16,
        input_w=64,
        num_heads=4,
        num_layers=2,
        ff_dim=64,
        dropout=0.0,
        evidence_x_bins=12,
        num_groups=1,
        intermediate_supervision=intermediate_supervision,
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
    )


def test_reference_decoder_preserves_public_output_contract() -> None:
    torch.manual_seed(7)
    head = _head().eval()
    features = torch.randn(2, 32, 8, 12)
    with torch.inference_mode():
        output = head(features)
        inference = head(features, inference_only=True)

    assert output["exist_logits"].shape == (2, 4, 2)
    assert output["pred_x_rows"].shape == (2, 4, 8)
    assert output["row_x_logits"].shape == (2, 4, 8, 16)
    assert output["input_reference_x_rows"].shape == (2, 4, 8)
    assert len(output["aux_outputs"]) == 1
    assert output["aux_outputs"][0]["input_reference_x_rows"].shape == (2, 4, 8)
    assert set(inference) == {
        "exist_logits",
        "pred_x_rows",
        "range_norm",
        "quality_logits",
    }


def test_reference_decoder_backpropagates_through_image_and_reference() -> None:
    torch.manual_seed(11)
    head = _head()
    features = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = head(features)
    loss = output["pred_x_rows"].mean()
    loss = loss + output["aux_outputs"][0]["pred_x_rows"].mean()
    loss.backward()

    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert float(features.grad.abs().sum()) > 0.0
    assert head.reference_anchor_logits is not None
    assert head.reference_anchor_logits.grad is not None
    assert float(head.reference_anchor_logits.grad.abs().sum()) > 0.0
    assert head.reference_query is not None
    assert head.reference_query.weight.grad is not None
    assert float(head.reference_query.weight.grad.abs().sum()) > 0.0


def test_local_profile_sampler_tracks_explicit_x_reference() -> None:
    layer = ReferenceGuidedRowLayer(
        dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        offsets_px=(-8.0, 0.0, 8.0),
    )
    # Every channel stores the horizontal feature coordinate.
    horizontal = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1)
    evidence = horizontal.expand(1, 3, 9, 8).contiguous()
    reference = torch.full((1, 2, 3), 32.0)
    profiles = layer._sample_local_profiles(evidence, reference, input_w=64)

    assert profiles.shape == (1, 2, 3, 3, 8)
    # x=32 maps to the center of a nine-bin feature row (index ~= 4.06).
    torch.testing.assert_close(
        profiles[:, :, :, 1].mean(),
        torch.tensor(32.0 / 63.0 * 8.0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert float(profiles[:, :, :, 0].mean()) < float(profiles[:, :, :, 1].mean())
    assert float(profiles[:, :, :, 2].mean()) > float(profiles[:, :, :, 1].mean())


def test_linear_sampler_matches_grid_sampler_values_and_gradients() -> None:
    torch.manual_seed(23)
    grid = ReferenceGuidedRowLayer(
        dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        offsets_px=(-9.0, -3.0, 0.0, 5.0, 11.0),
        sampling_backend="grid_sample",
    )
    linear = ReferenceGuidedRowLayer(
        dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        offsets_px=(-9.0, -3.0, 0.0, 5.0, 11.0),
        sampling_backend="linear_gather",
    )
    evidence_grid = torch.randn(2, 5, 13, 8, requires_grad=True)
    reference_grid = (torch.rand(2, 3, 5) * 90.0 + 3.0).requires_grad_(True)
    evidence_linear = evidence_grid.detach().clone().requires_grad_(True)
    reference_linear = reference_grid.detach().clone().requires_grad_(True)

    output_grid = grid._sample_local_profiles(
        evidence_grid,
        reference_grid,
        input_w=96,
    )
    output_linear = linear._sample_local_profiles(
        evidence_linear,
        reference_linear,
        input_w=96,
    )
    torch.testing.assert_close(output_linear, output_grid, atol=5e-6, rtol=2e-5)

    weight = torch.randn_like(output_grid)
    (output_grid * weight).sum().backward()
    (output_linear * weight).sum().backward()
    torch.testing.assert_close(
        evidence_linear.grad,
        evidence_grid.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        reference_linear.grad,
        reference_grid.grad,
        atol=2e-5,
        rtol=2e-5,
    )


def test_linear_sampler_preserves_bfloat16_contract() -> None:
    torch.manual_seed(29)
    evidence = torch.randn(1, 6, 17, 8, dtype=torch.bfloat16)
    reference = (torch.rand(1, 3, 6) * 60.0 + 2.0).to(torch.bfloat16)
    common = {
        "dim": 8,
        "num_heads": 2,
        "ff_dim": 16,
        "dropout": 0.0,
        "offsets_px": (-8.0, 0.0, 8.0),
    }
    grid = ReferenceGuidedRowLayer(**common, sampling_backend="grid_sample")
    linear = ReferenceGuidedRowLayer(**common, sampling_backend="linear_gather")

    expected = grid._sample_local_profiles(evidence, reference, input_w=64)
    actual = linear._sample_local_profiles(evidence, reference, input_w=64)

    assert actual.dtype == torch.bfloat16
    # Native BF16 interpolation differs from the legacy FP32 island by no
    # more than one BF16 quantization step in this representative profile.
    torch.testing.assert_close(actual.float(), expected.float(), atol=8e-3, rtol=8e-3)


def test_fused_local_key_value_projection_matches_separate_linears() -> None:
    torch.manual_seed(31)
    layer = ReferenceGuidedRowLayer(
        dim=8,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
    )
    profiles = torch.randn(2, 3, 4, 5, 8)

    expected_key = layer.local_key(profiles)
    expected_value = layer.local_value(profiles)
    fused = torch.nn.functional.linear(
        profiles,
        torch.cat((layer.local_key.weight, layer.local_value.weight), dim=0),
    )
    actual_key, actual_value = fused.split(layer.dim, dim=-1)

    torch.testing.assert_close(actual_key, expected_key)
    torch.testing.assert_close(actual_value, expected_value)


def test_reference_gate_configs_differ_only_in_reference_switch() -> None:
    control = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_reference_gate_control_10k.yaml"
    )
    candidate = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_row_reference_gate_10k.yaml"
    )
    assert not bool(
        control["model"]["structured_query"].get("row_reference", {}).get("enabled", False)
    )
    assert candidate["model"]["structured_query"]["row_reference"]["enabled"] is True
    for key in ("matcher", "loss", "optimizer", "scheduler", "training"):
        assert control[key] == candidate[key]


def test_full_reference_config_preserves_validated_candidate_contract() -> None:
    short = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_row_reference_gate_10k.yaml"
    )
    full = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_"
        "fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml"
    )

    assert full["model"]["structured_query"] == short["model"]["structured_query"]
    assert full["matcher"] == short["matcher"]
    assert full["loss"] == short["loss"]
    assert full["optimizer"] == short["optimizer"]
    assert full["training"]["seed"] == 3407
    assert full["training"]["batch_size"] == 4
    assert full["training"]["gradient_accumulation_steps"] == 4
    assert full["training"]["max_iters"] == 278000
    assert full["scheduler"]["total_iters"] == 278000
