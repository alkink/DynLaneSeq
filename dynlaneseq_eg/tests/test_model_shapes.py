from __future__ import annotations

import torch

from dynlaneseq_eg.modeling import DynLaneSeqS0, DynLaneSeqS1, DynLaneSeqS2, DynLaneSeqS3
from dynlaneseq_eg.modeling.evidence import DynamicDepthwiseBridge, DynamicOffsetFusion


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
