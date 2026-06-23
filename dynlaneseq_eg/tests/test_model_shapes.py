from __future__ import annotations

import torch
from PIL import Image

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.data.transforms import LaneTransforms, TransformConfig
from dynlaneseq_eg.factory import build_criterion, build_matcher, build_model
from dynlaneseq_eg.losses.loss_s2 import S2Criterion, S2LossConfig
from dynlaneseq_eg.modeling import DynLaneSeqS0, DynLaneSeqS1, DynLaneSeqS2, DynLaneSeqS3
from dynlaneseq_eg.modeling.dynlaneseq_s2 import PixelOnlyCorridorSearch
from dynlaneseq_eg.modeling.evidence import (
    AsymmetricContextModulationBridge,
    DynamicDepthwiseBridge,
    DynamicOffsetFusion,
    MultiScaleCurveAlignedSampler,
)


def _cfg(name: str):
    return {
        "model": {
            "name": name,
            "input_h": 288,
            "input_w": 800,
            "fpn_channels": 128,
            "dim": 256,
            "pretrained_backbone": False,
            "num_slots": 20,
            "num_rows": 72,
            "x_bins": 200,
            "decoder_layers": 1,
            "row_decoder_layers": 1,
            "num_heads": 8,
            "decoder_ff_dim": 512,
            "row_decoder_ff_dim": 512,
            "dropout": 0.0,
            "bridge": {"type": "film"},
        }
    }


def test_s0_forward_shapes():
    model = DynLaneSeqS0(_cfg("DynLaneSeqS0")).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 20, 2)
    assert out["row_x_logits"].shape == (1, 20, 72, 200)
    assert out["pred_x_rows"].shape == (1, 20, 72)
    assert out["range_norm"].shape == (1, 20, 2)


def test_centerline_aux_propagates_to_s3_outputs():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["centerline_aux"] = {"enabled": True, "dropout": 0.0}
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["centerline_logits"].shape == (1, 1, 72, 200)


def test_dynamic_evidence_starts_as_query_identity():
    cfg = _cfg("DynLaneSeqS0")
    cfg["model"]["dynamic_evidence"] = {
        "enabled": True,
        "num_points": 5,
        "hidden_dim": 64,
        "dropout": 0.0,
    }
    model = DynLaneSeqS0(cfg).eval()
    with torch.no_grad():
        enc = model.encoder.forward_features(torch.randn(1, 3, 288, 800))
    assert enc["queries"].shape == (1, 20, 256)
    assert torch.allclose(enc["queries"], enc["queries_pre_dynamic"], atol=1e-6)
    assert enc["dynamic_evidence"]["dynamic_evidence_refs"].shape == (1, 20, 5, 2)
    assert enc["dynamic_evidence"]["dynamic_evidence_delta_abs"].item() == 0.0


def test_dynamic_proposal_outputs_are_isolated_from_static_slots():
    cfg = _cfg("DynLaneSeqS0")
    cfg["model"]["dynamic_proposal"] = {
        "enabled": True,
        "top_k": 7,
        "hidden_dim": 64,
        "heatmap_bias_init": -4.6,
    }
    model = DynLaneSeqS0(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["pred_x_rows"].shape == (1, 20, 72)
    proposals = out["dynamic_proposals"]
    assert proposals["stage"]["pred_x_rows"].shape == (1, 7, 72)
    assert proposals["stage"]["range_norm"].shape == (1, 7, 2)
    assert proposals["stage"]["exist_logits"].shape == (1, 7, 2)
    assert proposals["queries"].shape == (1, 7, 256)
    assert proposals["dense"]["heatmap_logits"].shape == (1, 1, 72, 200)
    assert proposals["dense"]["x_rows"].shape == (1, 72, 72, 200)
    assert proposals["dense"]["range_norm"].shape == (1, 2, 72, 200)


def test_structured_query_s0_forward_shapes():
    cfg = _cfg("DynLaneSeqS0")
    cfg["model"]["dim"] = 64
    cfg["model"]["fpn_channels"] = 64
    cfg["model"]["num_slots"] = 16
    cfg["model"]["num_heads"] = 4
    cfg["model"]["decoder_layers"] = 0
    cfg["model"]["decoder_ff_dim"] = 128
    cfg["model"]["structured_query"] = {
        "enabled": True,
        "num_instances": 16,
        "num_groups": 4,
        "num_layers": 1,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.0,
    }
    model = DynLaneSeqS0(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 16, 2)
    assert out["row_x_logits"].shape == (1, 16, 72, 200)
    assert out["pred_x_rows"].shape == (1, 16, 72)
    assert out["range_norm"].shape == (1, 16, 2)
    assert out["queries"].shape == (1, 16, 64)
    assert out["structured_row_tokens"].shape == (1, 16, 72, 64)


def test_structured_query_debug_config_builds():
    cfg = load_config("dynlaneseq_eg/configs/debug/culane_s0_structured_query_2k.yaml")
    cfg["model"]["pretrained_backbone"] = False
    cfg["model"]["dim"] = 64
    cfg["model"]["fpn_channels"] = 64
    cfg["model"]["num_heads"] = 4
    cfg["model"]["decoder_ff_dim"] = 128
    cfg["model"]["structured_query"]["num_heads"] = 4
    cfg["model"]["structured_query"]["ff_dim"] = 128
    cfg["model"]["structured_query"]["num_layers"] = 1
    model = build_model(cfg).eval()
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 64, 2)
    assert out["pred_x_rows"].shape == (1, 64, 72)
    assert matcher.cfg.assignment == "grouped_one_to_many"
    assert matcher.cfg.num_groups == 4
    assert criterion.cfg.exist_loss_type == "focal"


def test_structured_stage_debug_configs_build():
    paths = [
        "dynlaneseq_eg/configs/debug/culane_s1_residual_structured_query_2k_init_structured.yaml",
        "dynlaneseq_eg/configs/debug/culane_s1_igt_structured_query_2k_from_s0_frozen.yaml",
        "dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_2k_from_s1.yaml",
        "dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_csr_2k_from_s1.yaml",
        "dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_structured_query_2k_from_s2.yaml",
    ]
    for path in paths:
        cfg = load_config(path)
        cfg["model"]["pretrained_backbone"] = False
        model = build_model(cfg)
        matcher = build_matcher(cfg)
        criterion = build_criterion(cfg)
        assert model.structured_query_head.num_instances == 64
        assert matcher.cfg.assignment == "grouped_one_to_many"
        assert matcher.cfg.num_groups == 4
        assert criterion.cfg.exist_loss_type == "focal"


def test_structured_full_configs_build():
    paths = [
        "dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16.yaml",
        "dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_local_speedtest.yaml",
        "dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_continue_75k.yaml",
        "dynlaneseq_eg/configs/culane_s1_residual_structured_query_res34_b16_from_s0.yaml",
        "dynlaneseq_eg/configs/culane_s1_igt_structured_query_res34_b16_from_s0_50ep_frozen.yaml",
        "dynlaneseq_eg/configs/culane_s1_igt_structured_query_res34_b8_from_s0_50ep_frozen.yaml",
        "dynlaneseq_eg/configs/culane_s2_residual_structured_query_res34_b16_from_s1.yaml",
        "dynlaneseq_eg/configs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2.yaml",
        "dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml",
    ]
    for path in paths:
        cfg = load_config(path)
        cfg["model"]["pretrained_backbone"] = False
        model = build_model(cfg)
        matcher = build_matcher(cfg)
        criterion = build_criterion(cfg)
        assert model.structured_query_head.num_instances == 64
        assert matcher.cfg.assignment == "grouped_one_to_many"
        assert matcher.cfg.num_groups == 4
        assert criterion.cfg.exist_loss_type == "focal"


def _structured_stage_cfg(name: str):
    cfg = _cfg(name)
    cfg["model"].update(
        {
            "dim": 64,
            "fpn_channels": 64,
            "num_slots": 16,
            "num_heads": 4,
            "decoder_layers": 0,
            "decoder_ff_dim": 128,
            "row_decoder_ff_dim": 128,
            "structured_query": {
                "enabled": True,
                "num_instances": 16,
                "num_groups": 4,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.0,
            },
        }
    )
    if name == "DynLaneSeqS1":
        cfg["model"]["s1_mode"] = "residual"
    else:
        cfg["model"]["s2_mode"] = "residual"
    return cfg


def test_structured_query_propagates_to_s1_outputs():
    cfg = _structured_stage_cfg("DynLaneSeqS1")
    model = DynLaneSeqS1(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 16, 2)
    assert out["pred_x_rows"].shape == (1, 16, 72)
    assert out["coarse"]["pred_x_rows"].shape == (1, 16, 72)
    assert out["structured_row_tokens"].shape == (1, 16, 72, 64)
    assert out["row_hidden"].shape == (1, 16, 72, 64)


def test_igt_s1_zero_init_preserves_structured_coarse_outputs():
    cfg = _structured_stage_cfg("DynLaneSeqS1")
    cfg["model"]["freeze_s0_frontend"] = True
    cfg["model"]["s1_refiner_type"] = "igt"
    cfg["model"]["residual_logit_scale"] = 0.5
    cfg["model"]["row_visibility"] = {"enabled": True}
    cfg["model"]["igt_refiner"] = {
        "hidden_dim": 128,
        "num_blocks": 2,
        "kernel_size": 5,
        "dilations": [1, 2],
        "dropout": 0.0,
        "zero_init": True,
        "num_groups": 4,
        "detach_quality_base": True,
        "visibility_head": True,
    }
    model = DynLaneSeqS1(cfg)
    assert not any(param.requires_grad for param in model.encoder.parameters())
    assert not any(param.requires_grad for param in model.structured_query_head.parameters())
    assert any(param.requires_grad for param in model.igt_refiner.parameters())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 16, 2)
    assert out["pred_x_rows"].shape == (1, 16, 72)
    assert out["row_hidden"].shape == (1, 16, 72, 64)
    assert out["row_delta_logits"].shape == (1, 16, 72, 200)
    assert out["row_visibility_logits"].shape == (1, 16, 72)
    assert torch.allclose(out["row_x_logits"], out["coarse"]["row_x_logits"], atol=1e-6)
    assert torch.allclose(out["pred_x_rows"], out["coarse"]["pred_x_rows"], atol=1e-5)
    assert torch.allclose(out["quality_logits"], out["coarse"]["quality_logits"], atol=1e-6)


def test_structured_query_propagates_to_s2_outputs():
    cfg = _structured_stage_cfg("DynLaneSeqS2")
    model = DynLaneSeqS2(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["coarse"]["pred_x_rows"].shape == (1, 16, 72)
    assert out["final"]["pred_x_rows"].shape == (1, 16, 72)
    assert out["structured_row_tokens"].shape == (1, 16, 72, 64)
    assert out["evidence"]["E_seq"].shape == (1, 16, 72, 64)


def test_s2_csr_conv1d_zero_init_preserves_coarse_outputs():
    cfg = _structured_stage_cfg("DynLaneSeqS2")
    cfg["model"]["s2_refiner_type"] = "csr_conv1d"
    cfg["model"]["s2_csr"] = {
        "hidden_dim": 128,
        "num_blocks": 2,
        "kernel_size": 5,
        "dilations": [1, 2],
        "dropout": 0.0,
        "zero_init": True,
        "quality_base": "coarse",
        "detach_quality_base": True,
    }
    model = DynLaneSeqS2(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert model.adapter is None
    assert model.row_decoder is None
    assert out["evidence"]["E_seq"].shape == (1, 16, 72, 64)
    assert torch.allclose(out["final"]["row_x_logits"], out["coarse"]["row_x_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["pred_x_rows"], out["coarse"]["pred_x_rows"], atol=1e-5)
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_logits"], out["coarse"]["quality_logits"], atol=1e-6)
    assert out["evidence"]["evidence_scale"].item() == 0.0
    assert out["evidence"]["csr_offset_profile_abs"].item() > 0.0
    assert out["evidence"]["csr_fused_evidence_abs"].item() == 0.0
    assert out["evidence"]["csr_delta_logits_abs"].item() == 0.0


def test_structured_query_propagates_to_s3_outputs():
    cfg = _structured_stage_cfg("DynLaneSeqS3")
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["quality_calibrator"] = {
        "enabled": True,
        "range_padding_px": 4.0,
        "detach_row_hidden": True,
        "quality_base": "none",
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["coarse"]["pred_x_rows"].shape == (1, 16, 72)
    assert out["final"]["pred_x_rows"].shape == (1, 16, 72)
    assert out["structured_row_tokens"].shape == (1, 16, 72, 64)
    assert out["evidence"]["active_offset_logits"].shape == (1, 16, 72, 5)
    assert torch.allclose(out["final"]["quality_logits"], torch.zeros_like(out["final"]["quality_logits"]), atol=1e-6)


def test_dynamic_evidence_propagates_to_s3_outputs():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["dynamic_evidence"] = {"enabled": True, "num_points": 5, "hidden_dim": 64}
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["final"]["pred_x_rows"].shape == (1, 20, 72)
    assert out["dynamic_evidence"]["dynamic_evidence_refs"].shape == (1, 20, 5, 2)
    assert out["dynamic_evidence"]["dynamic_evidence_delta_abs"].item() == 0.0


def test_s0_geometry_evidence_starts_as_shared_head_identity():
    cfg = _cfg("DynLaneSeqS0")
    cfg["model"]["s0_geometry_evidence"] = {
        "enabled": True,
        "hidden_dim": 64,
        "dropout": 0.0,
        "pooling": "mean_max",
        "local_window_enabled": True,
        "offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
        "local_reduce": "max",
        "detach_draft_x": True,
    }
    model = DynLaneSeqS0(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert "coarse" in out and "final" in out
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["row_x_logits"], out["coarse"]["row_x_logits"], atol=1e-6)
    assert torch.allclose(out["queries"], out["queries_pre_geometry"], atol=1e-6)
    assert out["geometry_evidence"]["s0_geometry_delta_abs"].item() == 0.0
    assert out["geometry_evidence"]["s0_geometry_local_window_abs"].item() > 0.0


def test_s0_geometry_evidence_propagates_to_s3_outputs():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["s0_geometry_evidence"] = {
        "enabled": True,
        "hidden_dim": 64,
        "dropout": 0.0,
        "pooling": "mean_max",
        "local_window_enabled": True,
        "offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
        "local_reduce": "max",
        "detach_draft_x": True,
    }
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["final"]["pred_x_rows"].shape == (1, 20, 72)
    assert out["s0_geometry_draft"]["pred_x_rows"].shape == (1, 20, 72)
    assert torch.allclose(out["queries"], out["queries_pre_geometry"], atol=1e-6)


def test_image_only_hard_augmentations_keep_lane_geometry():
    cfg = TransformConfig(
        input_w=64,
        input_h=32,
        gamma_jitter_prob=1.0,
        low_light_prob=1.0,
        motion_blur_prob=1.0,
        motion_blur_kernel_choices=(3,),
        random_shadow_prob=1.0,
        random_occlusion_prob=1.0,
    )
    tfm = LaneTransforms(cfg)
    image = Image.new("RGB", (64, 32), (128, 128, 128))
    lanes = [[(10.0, 5.0), (12.0, 20.0)]]
    tensor, out_lanes, seg, meta = tfm(image, lanes, training=True)
    assert tensor.shape == (3, 32, 64)
    assert out_lanes == lanes
    assert seg is None
    assert meta["flipped"] is False


def test_s0_geometry_evidence_propagates_to_s1_outputs():
    cfg = _cfg("DynLaneSeqS1")
    cfg["model"]["s1_mode"] = "residual"
    cfg["model"]["s0_geometry_evidence"] = {
        "enabled": True,
        "hidden_dim": 64,
        "dropout": 0.0,
        "pooling": "mean_max",
        "local_window_enabled": True,
        "offsets_px": [-16.0, -8.0, 0.0, 8.0, 16.0],
        "local_reduce": "max",
        "detach_draft_x": True,
    }
    model = DynLaneSeqS1(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["pred_x_rows"].shape == (1, 20, 72)
    assert out["s0_geometry_draft"]["pred_x_rows"].shape == (1, 20, 72)
    assert torch.allclose(out["queries"], out["queries_pre_geometry"], atol=1e-6)
    assert out["geometry_evidence"]["s0_geometry_delta_abs"].item() == 0.0
    assert out["geometry_evidence"]["s0_geometry_local_window_abs"].item() > 0.0


def test_s3_oracle_coarse_uses_target_geometry():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["oracle_coarse"] = {"enabled": True, "score_logit": 8.0, "background_logit": 8.0}
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    target = {
        "x_rows": torch.full((1, 72), 123.0),
        "x_bins": torch.full((1, 72), 31, dtype=torch.long),
        "valid_mask": torch.ones((1, 72), dtype=torch.bool),
        "range_y": torch.tensor([[0.0, 288.0]]),
    }
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800), targets=[target])
    assert torch.allclose(out["coarse"]["pred_x_rows"][0, 0], target["x_rows"][0], atol=1e-6)
    assert out["coarse"]["exist_logits"][0, 0, 0] > out["coarse"]["exist_logits"][0, 0, 1]
    assert out["coarse"]["exist_logits"][0, 1, 1] > out["coarse"]["exist_logits"][0, 1, 0]
    assert torch.allclose(out["evidence"]["active_center_x_rows"][0, 0], target["x_rows"][0], atol=1e-6)


def test_s1_s2_s3_forward_contracts():
    x = torch.randn(1, 3, 288, 800)
    for cls, name in [(DynLaneSeqS1, "DynLaneSeqS1"), (DynLaneSeqS2, "DynLaneSeqS2"), (DynLaneSeqS3, "DynLaneSeqS3")]:
        model = cls(_cfg(name)).eval()
        with torch.no_grad():
            out = model(x)
        if name == "DynLaneSeqS1":
            assert out["row_hidden"].shape[:3] == (1, 20, 72)
        else:
            assert "coarse" in out and "final" in out and "evidence" in out
            assert out["evidence"]["E_seq"].shape == (1, 20, 72, 256)


def test_dynamic_offset_fusion_starts_as_uniform_mean():
    fusion = DynamicOffsetFusion(dim=8, num_offsets=5, hidden_dim=16, zero_init=True).eval()
    samples = torch.randn(2, 3, 4, 5, 8)
    queries = torch.randn(2, 3, 8)
    row_embedding = torch.randn(4, 8)
    with torch.no_grad():
        fused, debug = fusion(samples, queries, row_embedding)
    assert torch.allclose(fused, samples.mean(dim=3), atol=1e-6)
    assert abs(debug["offset_weight_center"].item() - 0.2) < 1e-6


def test_dynamic_depthwise_bridge_zero_scale_preserves_evidence():
    bridge = DynamicDepthwiseBridge(dim=8, kernel_size=3, bridge_scale_init=0.0).eval()
    evidence = torch.randn(2, 3, 4, 8)
    queries = torch.randn(2, 3, 8)
    with torch.no_grad():
        out, debug = bridge(evidence, queries)
    assert torch.allclose(out, evidence, atol=1e-6)
    assert debug["delta_ratio"].item() == 0.0


def test_acm_bridge_starts_close_to_base_evidence():
    bridge = AsymmetricContextModulationBridge(dim=8, kernel_size=3, bridge_scale_init=0.0, acm_scale_init=0.0).eval()
    evidence = {"p2": torch.randn(2, 3, 4, 8), "p3": torch.randn(2, 3, 4, 8)}
    queries = torch.randn(2, 3, 8)
    row_embedding = torch.randn(4, 8)
    with torch.no_grad():
        out, debug = bridge(evidence, queries, row_embedding)
    assert torch.allclose(out, evidence["p2"], atol=1e-6)
    assert debug["acm_scale"].item() == 0.0


def test_acm_bridge_without_scale_starts_as_identity():
    bridge = AsymmetricContextModulationBridge(
        dim=8,
        kernel_size=3,
        bridge_scale_init=0.0,
        use_acm_scale=False,
        context_dropout=0.1,
    ).eval()
    evidence = {"p2": torch.randn(2, 3, 4, 8), "p3": torch.randn(2, 3, 4, 8)}
    queries = torch.randn(2, 3, 8)
    row_embedding = torch.randn(4, 8)
    with torch.no_grad():
        out, debug = bridge(evidence, queries, row_embedding)
    assert torch.allclose(out, evidence["p2"], atol=1e-6)
    assert "acm_scale" not in debug


def test_multi_scale_sampler_starts_as_uniform_fusion():
    sampler = MultiScaleCurveAlignedSampler(
        input_w=16,
        input_h=8,
        num_rows=4,
        dim=3,
        scales=["p2", "p3"],
        gate_hidden_dim=8,
        zero_init_gate=True,
    ).eval()
    sampler.scale_embeddings.data.zero_()
    features = {
        "p2": torch.ones(1, 3, 4, 8),
        "p3": torch.ones(1, 3, 2, 4) * 3.0,
    }
    sample_x = torch.full((1, 2, 4), 8.0)
    queries = torch.randn(1, 2, 3)
    row_embedding = torch.randn(4, 3)
    with torch.no_grad():
        out, debug = sampler(features, sample_x, queries, row_embedding)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-6)
    assert abs(debug["ms_gate_p2"].item() - 0.5) < 1e-6
    assert abs(debug["ms_gate_p3"].item() - 0.5) < 1e-6


def test_multi_scale_residual_starts_from_base_scale():
    sampler = MultiScaleCurveAlignedSampler(
        input_w=16,
        input_h=8,
        num_rows=4,
        dim=3,
        scales=["p2", "p3", "p4"],
        gate_hidden_dim=8,
        zero_init_gate=True,
        fusion_mode="residual",
        base_scale="p2",
        residual_scale_init=0.0,
        initial_gate_bias=[2.0, -1.0, -2.0],
    ).eval()
    sampler.scale_embeddings.data.normal_()
    features = {
        "p2": torch.ones(1, 3, 4, 8),
        "p3": torch.ones(1, 3, 2, 4) * 3.0,
        "p4": torch.ones(1, 3, 1, 2) * 7.0,
    }
    sample_x = torch.full((1, 2, 4), 8.0)
    queries = torch.randn(1, 2, 3)
    row_embedding = torch.randn(4, 3)
    with torch.no_grad():
        out, debug = sampler(features, sample_x, queries, row_embedding)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)
    assert debug["ms_gate_p2"].item() > 0.9
    assert debug["ms_residual_scale"].item() == 0.0


def test_s3_multiscale_forward_contract():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["multi_scale_evidence"] = {"enabled": True, "scales": ["p2", "p3", "p4"], "gate_hidden_dim": 64}
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800), return_features=True)
    assert out["evidence"]["E_seq"].shape == (1, 20, 72, 256)
    assert set(out["multi_scale_features"]) == {"p2", "p3", "p4"}
    assert "ms_gate_p2" in out["evidence"]


def test_s3_acm_forward_contract():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["multi_scale_evidence"] = {
        "enabled": True,
        "scales": ["p2", "p3"],
        "return_separate": True,
    }
    cfg["model"]["bridge"] = {
        "type": "asymmetric_context_modulation",
        "base_scale": "p2",
        "context_scale": "p3",
        "kernel_size": 3,
        "bridge_scale_init": 0.0,
        "acm_scale_init": 0.01,
    }
    cfg["model"]["seg_aux"] = {"enabled": True, "extra_scales": ["p3"]}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800), return_features=True)
    assert out["evidence"]["E_seq"].shape == (1, 20, 72, 256)
    assert "acm_scale" in out["evidence"]
    assert "seg_logits" in out
    assert "seg_logits_p3" in out


def test_s3_final_decision_zero_init_preserves_coarse_exist():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    cfg["model"]["final_decision"] = {"enabled": True, "pooling": "range_mean", "detach_base": True}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert out["evidence"]["final_delta_exist_abs"].item() == 0.0


def test_s3_active_corridor_forward_contract():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["evidence_gamma_init"] = 0.01
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["evidence"]["E_seq"].shape == (1, 20, 72, 256)
    assert out["evidence"]["active_offset_logits"].shape == (1, 20, 72, 5)
    assert out["evidence"]["active_pred_delta_x_rows"].abs().max() <= 16.0
    assert out["evidence"]["active_offset_center_prob"].item() > 0.3


def test_s3_active_corridor_can_be_authoritative_geometry():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["evidence"]["active_authoritative_geometry"].item() == 1.0
    assert torch.allclose(out["final"]["pred_x_rows"], out["evidence"]["active_refined_x_rows"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_pred_x_rows"], out["final"]["pred_x_rows"], atol=1e-6)


def test_s3_active_corridor_lateral_profile_contract():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "scorer_type": "lateral_profile",
        "offsets_px": [-16, -8, 0, 8, 16],
        "profile_dim": 32,
        "profile_blocks": 2,
        "profile_kernel_size": 3,
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["evidence"]["active_offset_logits"].shape == (1, 20, 72, 5)
    assert out["evidence"]["active_profile_relative_abs"].item() > 0.0
    assert out["evidence"]["active_profile_context_abs"].item() > 0.0
    assert torch.allclose(out["final"]["pred_x_rows"], out["coarse"]["pred_x_rows"], atol=1e-5)


def test_s3_active_corridor_coarse_to_fine_contract_and_identity_init():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "scorer_type": "coarse_to_fine_2d",
        "offsets_px": [-16, -8, 0, 8, 16],
        "fine_offsets_px": [-4, -2, 0, 2, 4],
        "profile_dim": 32,
        "profile_blocks": 2,
        "row_kernel_size": 3,
        "profile_kernel_size": 3,
        "use_dense_priors": True,
        "prior_channels": 2,
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["evidence"]["active_coarse_offset_logits"].shape == (1, 20, 72, 5)
    assert out["evidence"]["active_fine_offset_logits"].shape == (1, 20, 72, 5)
    assert out["evidence"]["active_offset_logits"].shape == (1, 20, 72, 25)
    assert out["evidence"]["active_dense_prior_abs"].item() >= 0.0
    assert torch.allclose(out["final"]["pred_x_rows"], out["coarse"]["pred_x_rows"], atol=1e-5)


def test_pixel_only_corridor_is_query_and_row_identity_invariant():
    scorer = PixelOnlyCorridorSearch(
        dim=16,
        num_rows=8,
        offsets_px=[-8, -4, 0, 4, 8],
        profile_dim=16,
        num_blocks=1,
        row_kernel=3,
        offset_kernel=3,
        zero_init=False,
    ).eval()
    samples = torch.randn(2, 3, 8, 5, 16)
    query_a = torch.randn(2, 3, 16)
    query_b = torch.randn(2, 3, 16) * 100.0
    row_a = torch.randn(8, 16)
    row_b = torch.randn(8, 16) * 100.0
    with torch.no_grad():
        output_a = scorer(samples, query_a, row_a)
        output_b = scorer(samples, query_b, row_b)
    assert torch.allclose(output_a[1], output_b[1], atol=1e-6)
    assert torch.allclose(output_a[2], output_b[2], atol=1e-6)


def test_pixel_only_training_center_jitter_is_disabled_at_evaluation():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "scorer_type": "pixel_only_2d",
        "offsets_px": [-16, -8, 0, 8, 16],
        "profile_dim": 16,
        "profile_blocks": 1,
        "row_kernel_size": 3,
        "profile_kernel_size": 3,
        "train_center_jitter_px": 12.0,
        "train_center_jitter_knots": 3,
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
        "skip_row_decoder": True,
    }
    cfg["model"]["seg_aux"] = {"enabled": True, "dropout": 0.0}
    cfg["model"]["centerline_aux"] = {"enabled": True, "dropout": 0.0}
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg)
    with torch.no_grad():
        train_out = model.train()(torch.randn(1, 3, 288, 800))
        eval_out = model.eval()(torch.randn(1, 3, 288, 800))
    assert train_out["evidence"]["active_center_jitter_abs"].item() > 0.0
    assert eval_out["evidence"]["active_center_jitter_abs"].item() == 0.0
    assert torch.allclose(eval_out["final"]["pred_x_rows"], eval_out["coarse"]["pred_x_rows"], atol=1e-5)


def test_lane_evidence_tower_contract_identity_and_freeze_boundary():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "scorer_type": "lane_evidence_tower",
        "offsets_px": [-16, -8, 0, 8, 16],
        "evidence_dim": 32,
        "tower_blocks": 1,
        "profile_dim": 32,
        "profile_blocks": 1,
        "row_kernel_size": 3,
        "profile_kernel_size": 3,
        "gate_hidden_dim": 16,
        "gate_bias": -1.0,
        "gate_enabled": True,
        "train_center_jitter_px": 12.0,
        "train_center_jitter_prob": 1.0,
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
        "freeze_non_active_modules": True,
        "skip_row_decoder": True,
    }
    cfg["model"]["seg_aux"] = {"enabled": True, "dropout": 0.0}
    cfg["model"]["centerline_aux"] = {"enabled": True, "dropout": 0.0}
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(name.startswith("active_corridor.") for name in trainable_names)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["seg_logits"].shape == (1, 1, 72, 200)
    assert out["centerline_logits"].shape == (1, 1, 72, 200)
    assert out["coarse_seg_logits"].shape == (1, 1, 288, 800)
    assert out["coarse_centerline_logits"].shape == (1, 1, 72, 200)
    assert out["evidence"]["active_gate_logits"].shape == (1, 20, 72)
    assert out["evidence"]["active_gate_mean"].item() < 0.5
    assert torch.allclose(out["final"]["pred_x_rows"], out["coarse"]["pred_x_rows"], atol=1e-5)
    with torch.no_grad():
        train_out = model.train()(torch.randn(1, 3, 288, 800))
    assert train_out["evidence"]["active_center_jitter_abs"].item() > 0.0


def test_coarse_to_fine_offset_losses_are_finite_and_backpropagate():
    cfg = _cfg("DynLaneSeqS2")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "scorer_type": "coarse_to_fine_2d",
        "offsets_px": [-16, -8, 0, 8, 16],
        "fine_offsets_px": [-4, -2, 0, 2, 4],
        "profile_dim": 32,
        "profile_blocks": 1,
        "row_kernel_size": 3,
        "profile_kernel_size": 3,
        "center_init_bias": 0.0,
        "authoritative_geometry": True,
    }
    model = DynLaneSeqS2(cfg)
    images = torch.randn(1, 3, 288, 800)
    targets = [
        {
            "x_rows": torch.linspace(100.0, 150.0, 72).view(1, 72),
            "valid_mask": torch.ones((1, 72), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 287.0]]),
            "x_bins": torch.linspace(25, 37, 72).round().long().view(1, 72),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    outputs = model(images, targets=targets, matches=matches)
    criterion = S2Criterion(
        S2LossConfig(
            w_active_offset_reg=1.0,
            w_active_offset_ce=0.5,
            w_active_offset_tangent=0.1,
            active_offset_max=16.0,
            active_offset_reg_beta_px=2.0,
            active_offset_soft_target_sigma_px=4.0,
            active_offset_fine_target_sigma_px=1.5,
            active_offset_magnitude_weight=1.0,
        )
    )
    loss = criterion(outputs, targets, matches)
    assert torch.isfinite(loss["loss_active_offset_reg"])
    assert torch.isfinite(loss["loss_active_offset_ce"])
    assert torch.isfinite(loss["loss_active_offset_tangent"])
    loss["loss_total"].backward()
    assert model.active_corridor.coarse_head.weight.grad is not None
    assert model.active_corridor.fine_head.weight.grad is not None


def test_s3_active_corridor_quality_calibrator_starts_neutral():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["evidence_gamma_init"] = 0.01
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["quality_calibrator"] = {
        "enabled": True,
        "range_padding_px": 4.0,
        "detach_row_hidden": True,
        "quality_base": "none",
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_logits"], torch.zeros_like(out["final"]["quality_logits"]), atol=1e-6)
    assert out["evidence"]["quality_calib_delta_exist_abs"].item() == 0.0
    assert out["evidence"]["quality_calib_quality_abs"].item() == 0.0


def test_s3_quality_calibrator_mean_max_pooling_starts_neutral():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["quality_calibrator"] = {
        "enabled": True,
        "pooling": "range_mean_max",
        "calibrate_exist": False,
        "quality_base": "none",
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_logits"], torch.zeros_like(out["final"]["quality_logits"]), atol=1e-6)


def test_s3_quality_calibrator_curvature_feature_starts_neutral():
    cfg = _cfg("DynLaneSeqS3")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    cfg["model"]["quality_calibrator"] = {
        "enabled": True,
        "pooling": "range_mean",
        "curvature_feature": True,
        "quality_base": "none",
    }
    cfg["model"]["bridge"] = {"type": "dynamic_depthwise_sequence", "kernel_size": 3, "bridge_scale_init": 0.0}
    model = DynLaneSeqS3(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_logits"], torch.zeros_like(out["final"]["quality_logits"]), atol=1e-6)


def test_active_corridor_offset_loss_is_finite_and_backprops():
    cfg = _cfg("DynLaneSeqS2")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["active_corridor"] = {
        "enabled": True,
        "offsets_px": [-16, -8, 0, 8, 16],
        "center_init_bias": 2.0,
    }
    model = DynLaneSeqS2(cfg)
    images = torch.randn(1, 3, 288, 800)
    targets = [
        {
            "x_rows": torch.full((1, 72), 120.0),
            "valid_mask": torch.ones((1, 72), dtype=torch.bool),
            "range_y": torch.tensor([[0.0, 287.0]]),
            "x_bins": torch.full((1, 72), 30, dtype=torch.long),
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    outputs = model(images, targets=targets, matches=matches)
    loss = S2Criterion(
        S2LossConfig(w_active_offset_reg=1.0, w_active_offset_ce=0.1, active_offset_max=16.0)
    )(outputs, targets, matches)
    assert torch.isfinite(loss["loss_active_offset_reg"])
    assert torch.isfinite(loss["loss_active_offset_ce"])
    loss["loss_total"].backward()
    assert model.active_corridor.offset_bias.grad is not None


def test_s2_decision_refiner_zero_init_preserves_coarse_decisions():
    cfg = _cfg("DynLaneSeqS2")
    cfg["model"]["s2_mode"] = "residual"
    cfg["model"]["dim"] = 64
    cfg["model"]["fpn_channels"] = 64
    cfg["model"]["num_heads"] = 4
    cfg["model"]["decoder_ff_dim"] = 128
    cfg["model"]["row_decoder_ff_dim"] = 128
    cfg["model"]["s2_decision_refiner"] = {
        "enabled": True,
        "hidden_dim": 64,
        "dropout": 0.0,
        "zero_init": True,
        "detach_base": True,
        "exist_delta_scale": 0.5,
        "quality_base": "coarse",
    }
    model = DynLaneSeqS2(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert torch.allclose(out["final"]["exist_logits"], out["coarse"]["exist_logits"], atol=1e-6)
    assert torch.allclose(out["final"]["quality_logits"], out["coarse"]["quality_logits"], atol=1e-6)
    assert out["evidence"]["s2_decision_delta_exist_abs"].item() == 0.0
    assert out["evidence"]["s2_decision_delta_quality_abs"].item() == 0.0
