from __future__ import annotations

import torch

from dynlaneseq_eg.losses.loss_s2 import S2Criterion, S2LossConfig
from dynlaneseq_eg.modeling import DynLaneSeqS0, DynLaneSeqS1, DynLaneSeqS2, DynLaneSeqS3
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
