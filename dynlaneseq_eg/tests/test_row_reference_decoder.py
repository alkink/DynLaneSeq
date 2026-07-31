from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_criterion, build_matcher
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


def test_set_selection_starts_as_exact_existing_score_residual() -> None:
    torch.manual_seed(9)
    head = StructuredLaneQueryHead(
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
        row_reference={
            "enabled": True,
            "offsets_px": [-16.0, 0.0, 16.0],
            "initial_prior_sigma_px": 16.0,
            "output_prior_sigma_px": 8.0,
        },
        set_selection={
            "enabled": True,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "curve_samples": 4,
            "base_quality_power": 0.5,
        },
    ).eval()
    features = torch.randn(2, 32, 8, 12)
    with torch.inference_mode():
        output = head(features)
        inference = head(features, inference_only=True)

    expected = torch.softmax(output["exist_logits"].float(), dim=-1)[..., 0]
    expected = expected * torch.sigmoid(
        output["quality_logits"].float()
    ).pow(0.5)
    torch.testing.assert_close(
        torch.sigmoid(output["selection_logits"]),
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    assert float(output["selection_delta_logits"].abs().max()) == 0.0
    assert "selection_logits" in inference
    torch.testing.assert_close(
        inference["selection_logits"],
        output["selection_logits"],
    )


def test_set_selection_can_backpropagate_into_decoder_state_features() -> None:
    torch.manual_seed(10)
    head = StructuredLaneQueryHead(
        dim=32,
        num_instances=4,
        num_rows=8,
        x_bins=16,
        input_w=64,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        dropout=0.0,
        evidence_x_bins=12,
        num_groups=1,
        set_selection={
            "enabled": True,
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "ff_dim": 64,
            "dropout": 0.0,
            "curve_samples": 4,
        },
    )
    assert head.set_selection_head is not None
    with torch.no_grad():
        head.set_selection_head.output.weight.fill_(0.01)
    features = torch.randn(2, 32, 8, 12, requires_grad=True)
    output = head(features)
    output["selection_logits"].sum().backward()

    assert features.grad is not None
    assert float(features.grad.abs().sum()) > 0.0
    assert head.layers[0].ffn[0].weight.grad is not None
    assert float(head.layers[0].ffn[0].weight.grad.abs().sum()) > 0.0


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


def test_from50k_cooldown_changes_only_schedule_and_evidence_lr() -> None:
    full = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_"
        "fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml"
    )
    cooldown = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from50k_cooldown_5k.yaml"
    )

    for key in ("model", "matcher", "loss", "augmentation", "dataset", "dataloader"):
        assert cooldown[key] == full[key]
    assert cooldown["optimizer"]["base_lr"] == full["optimizer"]["base_lr"]
    assert cooldown["optimizer"]["backbone_lr"] == full["optimizer"]["backbone_lr"]
    assert cooldown["optimizer"]["weight_decay"] == full["optimizer"]["weight_decay"]
    assert cooldown["optimizer"]["betas"] == full["optimizer"]["betas"]
    assert full["optimizer"]["evidence_lr"] == 0.0002
    assert cooldown["optimizer"]["evidence_lr"] == 0.00002
    assert cooldown["training"]["seed"] == full["training"]["seed"] == 3407
    assert cooldown["training"]["batch_size"] == full["training"]["batch_size"] == 4
    assert cooldown["training"]["gradient_accumulation_steps"] == 4
    assert cooldown["training"]["max_iters"] == 5000
    assert cooldown["training"]["checkpoint_interval"] == 2500
    assert cooldown["scheduler"] == {
        "name": "cosine",
        "total_iters": 5000,
        "warmup_iters": 0,
        "min_lr_ratio": 0.1,
    }


def test_from55k_extension_preserves_model_and_uses_constant_floor_lr() -> None:
    cooldown = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from50k_cooldown_5k.yaml"
    )
    extension = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml"
    )

    for key in (
        "model",
        "matcher",
        "loss",
        "augmentation",
        "dataset",
        "dataloader",
        "optimizer",
    ):
        assert extension[key] == cooldown[key]
    assert extension["training"]["seed"] == cooldown["training"]["seed"] == 3407
    assert extension["training"]["batch_size"] == 4
    assert extension["training"]["gradient_accumulation_steps"] == 4
    assert extension["training"]["max_iters"] == 20000
    assert extension["training"]["checkpoint_interval"] == 5000
    assert extension["scheduler"] == {
        "name": "constant",
        "total_iters": 20000,
        "warmup_iters": 0,
        "min_lr_ratio": 1.0,
    }


def test_from55k_evidence_lr_probe_changes_only_optimizer_rates_and_horizon() -> None:
    floor = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml"
    )
    probe = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_probe_5k.yaml"
    )

    for key in ("model", "matcher", "loss", "augmentation", "dataset", "dataloader"):
        assert probe[key] == floor[key]
    assert probe["optimizer"]["backbone_lr"] == 1e-6
    assert probe["optimizer"]["base_lr"] == 1e-5
    assert probe["optimizer"]["evidence_lr"] == 1e-5
    assert probe["optimizer"]["weight_decay"] == floor["optimizer"]["weight_decay"]
    assert probe["optimizer"]["betas"] == floor["optimizer"]["betas"]
    assert probe["training"]["seed"] == floor["training"]["seed"] == 3407
    assert probe["training"]["batch_size"] == 4
    assert probe["training"]["gradient_accumulation_steps"] == 4
    assert probe["training"]["max_iters"] == 5000
    assert probe["training"]["checkpoint_interval"] == 2500
    assert probe["scheduler"] == {
        "name": "constant",
        "total_iters": 5000,
        "warmup_iters": 0,
        "min_lr_ratio": 1.0,
    }


def test_from55k_evidence_lr_trajectory_only_extends_probe_horizon() -> None:
    probe = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_probe_5k.yaml"
    )
    trajectory = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml"
    )

    for key in (
        "model",
        "matcher",
        "loss",
        "augmentation",
        "dataset",
        "dataloader",
        "optimizer",
    ):
        assert trajectory[key] == probe[key]
    assert trajectory["training"]["seed"] == probe["training"]["seed"] == 3407
    assert trajectory["training"]["batch_size"] == 4
    assert trajectory["training"]["gradient_accumulation_steps"] == 4
    assert trajectory["training"]["max_iters"] == 20000
    assert trajectory["training"]["checkpoint_interval"] == 5000
    assert trajectory["scheduler"] == {
        "name": "constant",
        "total_iters": 20000,
        "warmup_iters": 0,
        "min_lr_ratio": 1.0,
    }


def test_joint_set_selection_config_is_matched_65k_to70k_intervention() -> None:
    control = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml"
    )
    candidate = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from65k_joint_set_selection_5k.yaml"
    )

    for key in ("matcher", "augmentation", "dataset", "dataloader", "scheduler"):
        assert candidate[key] == control[key]
    assert candidate["training"]["seed"] == control["training"]["seed"] == 3407
    assert candidate["training"]["batch_size"] == control["training"]["batch_size"]
    assert candidate["training"]["gradient_accumulation_steps"] == 4
    assert candidate["training"]["max_iters"] == 5000

    assert candidate["model"]["structured_query"]["set_selection"]["enabled"] is True
    assert "set_selection" not in control["model"]["structured_query"]
    assert candidate["loss"]["w_set_selection"] == 1.0
    assert control["loss"].get("w_set_selection", 0.0) == 0.0
    candidate_groups = {
        group["name"]: group
        for group in candidate["optimizer"]["parameter_groups"]
    }
    control_groups = {
        group["name"]: group
        for group in control["optimizer"]["parameter_groups"]
    }
    assert set(candidate_groups) == set(control_groups) | {"set_selection"}
    for name in control_groups:
        assert candidate_groups[name] == control_groups[name]
    assert candidate_groups["set_selection"]["lr"] == 1e-4


def test_no_object_matcher_config_is_single_change_65k_to70k_intervention() -> None:
    control = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from65k_selective_cooldown_10k.yaml"
    )
    candidate = load_config(
        "dynlaneseq_eg/configs/"
        "culane_s0_structured_query_dla34_rowref_from65k_no_object_matcher_5k.yaml"
    )

    for key in (
        "model",
        "loss",
        "augmentation",
        "dataset",
        "dataloader",
        "optimizer",
        "scheduler",
    ):
        assert candidate[key] == control[key]

    expected_matcher = dict(control["matcher"])
    expected_matcher["lambda_obj"] = 0.0
    assert candidate["matcher"] == expected_matcher
    assert control["matcher"]["lambda_obj"] == 2.0
    assert candidate["matcher"]["lambda_point"] == 5.0
    assert candidate["matcher"]["lambda_range"] == 1.0
    assert candidate["matcher"]["lambda_line_iou"] == 1.0
    assert candidate["matcher"]["object_cost_type"] == "neg_probability"
    assert candidate["training"]["seed"] == control["training"]["seed"] == 3407
    assert candidate["training"]["batch_size"] == control["training"]["batch_size"]
    assert candidate["training"]["gradient_accumulation_steps"] == 4
    assert candidate["training"]["max_iters"] == 5000

    final_matcher = build_matcher(candidate)
    criterion = build_criterion(candidate)
    assert final_matcher.cfg.lambda_obj == 0.0
    assert criterion.matcher is not None
    assert criterion.matcher.cfg.lambda_obj == 0.0
