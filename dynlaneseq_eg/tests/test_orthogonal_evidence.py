from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.common import fixed_y_rows
from dynlaneseq_eg.modeling.dynlaneseq_s0 import DynLaneSeqS0
from dynlaneseq_eg.modeling.evidence import OrthogonalEvidenceSampler


def test_orthogonal_sampler_preserves_rows_and_samples_offsets() -> None:
    sampler = OrthogonalEvidenceSampler(input_w=800, input_h=288, num_rows=72)
    features = torch.randn(1, 8, 72, 200)
    x_rows = torch.full((1, 3, 72), 400.0)
    samples, valid, debug = sampler(features, x_rows)
    assert samples.shape == (1, 3, 72, 7, 8)
    assert valid.shape == (1, 3, 72, 7)
    assert valid.all()
    assert debug["orthogonal_valid_frac"].item() == 1.0


def test_orthogonal_sampler_tangent_uses_pixel_space_and_keeps_endpoints() -> None:
    sampler = OrthogonalEvidenceSampler(input_w=800, input_h=288, num_rows=72)
    y = fixed_y_rows(72, input_h=288).view(1, 1, 72)
    x_rows = 100.0 + 0.5 * y
    tx, ty, nx, ny = sampler.tangent_normal(x_rows)
    expected_tx = torch.full_like(tx, 0.5 / ((0.5**2 + 1.0) ** 0.5))
    expected_ty = torch.full_like(ty, 1.0 / ((0.5**2 + 1.0) ** 0.5))
    assert tx.shape == (1, 1, 72)
    assert ty.shape == (1, 1, 72)
    assert nx.shape == (1, 1, 72)
    assert ny.shape == (1, 1, 72)
    assert torch.allclose(tx, expected_tx, atol=1e-5)
    assert torch.allclose(ty, expected_ty, atol=1e-5)
    assert torch.allclose(nx, -expected_ty, atol=1e-5)
    assert torch.allclose(ny, expected_tx, atol=1e-5)


def test_orthogonal_sampler_detaches_sample_coordinates() -> None:
    sampler = OrthogonalEvidenceSampler(input_w=800, input_h=288, num_rows=72, detach_sample_x=True)
    features = torch.randn(1, 4, 72, 200, requires_grad=True)
    x_rows = torch.full((1, 2, 72), 400.0, requires_grad=True)
    samples, _, _ = sampler(features, x_rows)
    samples.sum().backward()
    assert features.grad is not None
    assert features.grad.abs().sum().item() > 0.0
    assert x_rows.grad is None


def test_structured_s0_orthogonal_verifier_quality_only_zero_init_and_freeze() -> None:
    cfg = {
        "model": {
            "name": "DynLaneSeqS0",
            "input_h": 288,
            "input_w": 800,
            "fpn_channels": 32,
            "dim": 32,
            "pretrained_backbone": False,
            "freeze_backbone_bn": True,
            "freeze_all_bn": True,
            "num_slots": 8,
            "num_rows": 72,
            "x_bins": 200,
            "decoder_layers": 0,
            "num_heads": 4,
            "decoder_ff_dim": 128,
            "dropout": 0.0,
            "structured_query": {
                "enabled": True,
                "num_instances": 8,
                "num_groups": 4,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.0,
                "orthogonal_verifier": {
                    "enabled": True,
                    "freeze_base": True,
                    "evidence_dim": 16,
                    "hidden_dim": 32,
                    "offsets_px": [-8.0, 0.0, 8.0],
                    "detach_sample_x": True,
                    "zero_init": True,
                },
            },
        }
    }
    model = DynLaneSeqS0(cfg).eval()
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("structured_query_head.orthogonal_verifier.") for name in trainable)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 8, 2)
    assert out["row_x_logits"].shape == (1, 8, 72, 200)
    assert out["pred_x_rows"].shape == (1, 8, 72)
    assert out["quality_logits"].shape == (1, 8)
    assert out["structured_debug"]["orthogonal_quality_delta_abs"].item() == 0.0


def test_structured_s0_orthogonal_grounder_zero_init_and_freeze() -> None:
    cfg = {
        "model": {
            "name": "DynLaneSeqS0",
            "input_h": 288,
            "input_w": 800,
            "fpn_channels": 32,
            "dim": 32,
            "pretrained_backbone": False,
            "freeze_backbone_bn": True,
            "freeze_all_bn": True,
            "num_slots": 8,
            "num_rows": 72,
            "x_bins": 200,
            "decoder_layers": 0,
            "num_heads": 4,
            "decoder_ff_dim": 128,
            "dropout": 0.0,
            "structured_query": {
                "enabled": True,
                "num_instances": 8,
                "num_groups": 4,
                "num_layers": 1,
                "num_heads": 4,
                "ff_dim": 128,
                "dropout": 0.0,
                "orthogonal_grounder": {
                    "enabled": True,
                    "freeze_base": True,
                    "evidence_dim": 16,
                    "hidden_dim": 32,
                    "offsets_px": [-8.0, 0.0, 8.0],
                    "detach_sample_x": True,
                    "zero_init": True,
                },
            },
        }
    }
    model = DynLaneSeqS0(cfg).eval()
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(name.startswith("structured_query_head.orthogonal_grounder.") for name in trainable)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 288, 800))
    assert out["exist_logits"].shape == (1, 8, 2)
    assert out["row_x_logits"].shape == (1, 8, 72, 200)
    assert out["pred_x_rows"].shape == (1, 8, 72)
    assert out["quality_logits"].shape == (1, 8)
    assert out["structured_debug"]["orthogonal_grounder_delta_abs"].item() == 0.0
