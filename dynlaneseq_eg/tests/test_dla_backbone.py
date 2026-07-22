from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.backbone_dla import DLA34Backbone
from dynlaneseq_eg.modeling.dynlaneseq_s0 import DynLaneSeqEncoder
from dynlaneseq_eg.modeling.fpn import SimpleFPN


def test_dla34_backbone_feature_contract() -> None:
    model = DLA34Backbone(pretrained=False).eval()
    with torch.inference_mode():
        features = model(torch.randn(1, 3, 64, 96))
    assert tuple(features) == ("c2", "c3", "c4", "c5")
    assert tuple(features[name].shape[1] for name in features) == (64, 128, 256, 512)
    assert tuple(features[name].shape[-2:] for name in features) == (
        (16, 24),
        (8, 12),
        (4, 6),
        (2, 3),
    )


def test_dla34_backbone_connects_to_existing_fpn() -> None:
    backbone = DLA34Backbone(pretrained=False).eval()
    fpn = SimpleFPN(in_channels=backbone.out_channels, out_channels=32).eval()
    with torch.inference_mode():
        features = backbone(torch.randn(1, 3, 64, 96))
        p2 = fpn(features)
    assert p2.shape == (1, 32, 16, 24)


def test_dla34_fpn_path_backpropagates_finite_gradients() -> None:
    backbone = DLA34Backbone(pretrained=False).train()
    fpn = SimpleFPN(in_channels=backbone.out_channels, out_channels=16).train()
    p2 = fpn(backbone(torch.randn(2, 3, 64, 96)))
    p2.square().mean().backward()
    grad = backbone.base_layer[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_dla34_encoder_selection_and_frozen_bn() -> None:
    cfg = {
        "model": {
            "backbone_name": "dla34",
            "pretrained_backbone": False,
            "freeze_backbone_bn": True,
            "input_h": 64,
            "input_w": 96,
            "fpn_channels": 32,
            "dim": 32,
            "num_slots": 4,
            "num_rows": 8,
            "x_bins": 24,
            "decoder_layers": 0,
            "num_heads": 4,
            "decoder_ff_dim": 64,
            "structured_query": {"enabled": False},
            "seg_aux": {"enabled": False},
            "centerline_aux": {"enabled": False},
        }
    }
    encoder = DynLaneSeqEncoder(cfg).train()
    assert encoder.backbone_name == "dla34"
    assert all(not module.training for module in encoder.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d))
    with torch.inference_mode():
        outputs = encoder.forward_features(torch.randn(1, 3, 64, 96))
    assert outputs["features"].shape == (1, 32, 16, 24)


def test_tusimple_dla34_config_is_res34_control_except_backbone_and_output() -> None:
    config_dir = Path("dynlaneseq_eg/configs")
    res34 = load_config(
        config_dir / "tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml"
    )
    dla34 = load_config(
        config_dir / "tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml"
    )
    assert res34["model"]["backbone_name"] == "resnet"
    assert res34["model"]["resnet_depth"] == 34
    assert dla34["model"]["backbone_name"] == "dla34"

    for cfg in (res34, dla34):
        cfg.pop("_config_path", None)
        cfg.pop("output_dir", None)
        cfg["model"].pop("backbone_name", None)
        cfg["model"].pop("resnet_depth", None)
    assert dla34 == res34


def test_culane_dla34_config_is_res34_control_except_backbone_metadata() -> None:
    config_dir = Path("dynlaneseq_eg/configs")
    res34 = load_config(
        config_dir / "culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml"
    )
    dla34 = load_config(
        config_dir / "culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml"
    )
    assert dla34["model"]["backbone_name"] == "dla34"
    assert dla34["model"]["require_pretrained_backbone"] is True
    assert dla34["training"]["seed"] == 3407
    assert dla34["training"]["vis_interval"] == 25000

    for cfg in (res34, dla34):
        cfg.pop("_config_path", None)
        cfg.pop("output_dir", None)
        cfg["model"].pop("backbone_name", None)
        cfg["model"].pop("require_pretrained_backbone", None)
        # Seed and visualization cadence are run bookkeeping rather than model,
        # optimization, or data-pipeline differences.  The historical control
        # did not record a seed, while the new DLA run deliberately does.
        cfg["training"].pop("seed", None)
        cfg["training"].pop("vis_interval", None)
    assert dla34 == res34


def test_culane_dla34_radius15_deep_supervision_config_contract() -> None:
    cfg = load_config(
        "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_"
        "1600x640_bins800_fpn256_l4_dfl_r15_deepsup_50ep.yaml"
    )
    assert cfg["model"]["backbone_name"] == "dla34"
    assert cfg["model"]["structured_query"]["num_layers"] == 4
    assert cfg["model"]["structured_query"]["intermediate_supervision"] is True
    assert cfg["matcher"]["line_iou_radius"] == 15.0
    assert cfg["loss"]["line_iou_radius"] == 15.0
    assert cfg["loss"]["lambda_intermediate"] == 0.5
    assert cfg["loss"]["intermediate_layer_weights"] == [1.0, 2.0, 3.0]
    assert cfg["training"]["seed"] == 3407
    assert cfg["training"]["amp_dtype"] == "bfloat16"
    assert cfg["training"]["batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 4
    assert cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"] == 16


def test_unknown_backbone_name_fails_clearly() -> None:
    cfg = {
        "model": {
            "backbone_name": "unknown",
            "pretrained_backbone": False,
        }
    }
    with pytest.raises(ValueError, match="Unsupported model.backbone_name"):
        DynLaneSeqEncoder(cfg)


def test_seg_aux_precision_override_is_isolated_to_segmentation_head() -> None:
    cfg = {
        "model": {
            "backbone_name": "dla34",
            "pretrained_backbone": False,
            "fpn_channels": 64,
            "dim": 64,
            "input_h": 64,
            "input_w": 128,
            "seg_aux": {"enabled": True, "amp_dtype": "bfloat16"},
            "centerline_aux": {"enabled": True},
        }
    }
    encoder = DynLaneSeqEncoder(cfg)
    assert encoder.seg_aux_head is not None
    assert encoder.seg_aux_head.amp_dtype == "bfloat16"
    assert not hasattr(encoder.centerline_aux_head, "amp_dtype")
