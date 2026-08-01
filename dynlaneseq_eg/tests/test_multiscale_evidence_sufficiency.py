from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dynlaneseq_eg.tools.probe_multiscale_evidence_sufficiency import (
    ARM_NAMES,
    CandidateSequenceDescriptor,
    EvidenceSelectionArm,
    _validate_offsets,
    evidence_masks,
    extract_frozen_sources,
)


def _descriptor(mask: torch.Tensor) -> CandidateSequenceDescriptor:
    offsets = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-2.0, 0.0, 2.0],
        ]
    )
    return CandidateSequenceDescriptor(
        base_dim=10,
        row_state_dim=8,
        feature_dim=8,
        rows=5,
        offsets_px=offsets,
        evidence_mask=mask,
        sequence_dim=8,
        output_dim=16,
        dropout=0.0,
    ).eval()


def test_evidence_masks_partition_modalities() -> None:
    masks = evidence_masks(7)
    assert tuple(masks) == ARM_NAMES
    assert not bool(masks["state"].any())
    assert int(masks["p2_center"].sum()) == 1
    assert bool(masks["p2_center"][0, 3])
    assert int(masks["p2_strip"].sum()) == 7
    assert int(masks["p34_context"].sum()) == 14
    assert int(masks["joint"].sum()) == 21


def test_state_arm_is_invariant_to_visual_profiles() -> None:
    torch.manual_seed(3)
    descriptor = _descriptor(evidence_masks(3)["state"])
    base = torch.randn(2, 4, 10)
    states = torch.randn(2, 4, 5, 8)
    profiles_a = torch.randn(2, 4, 5, 3, 3, 8)
    profiles_b = torch.randn_like(profiles_a) * 100.0
    weights = torch.ones(2, 4, 5)
    with torch.no_grad():
        output_a = descriptor(base, states, profiles_a, weights)
        output_b = descriptor(base, states, profiles_b, weights)
    torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)


def test_joint_arm_responds_to_visual_profiles() -> None:
    torch.manual_seed(4)
    descriptor = _descriptor(evidence_masks(3)["joint"])
    base = torch.randn(2, 4, 10)
    states = torch.randn(2, 4, 5, 8)
    profiles = torch.randn(2, 4, 5, 3, 3, 8)
    weights = torch.ones(2, 4, 5)
    with torch.no_grad():
        output = descriptor(base, states, profiles, weights)
        zero = descriptor(base, states, torch.zeros_like(profiles), weights)
    assert float((output - zero).abs().max()) > 1e-5


def test_all_selection_arms_have_identical_parameter_counts() -> None:
    masks = evidence_masks(3)
    offsets = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-2.0, 0.0, 2.0],
        ]
    )
    counts = []
    for name in ARM_NAMES:
        arm = EvidenceSelectionArm(
            base_dim=10,
            row_state_dim=8,
            feature_dim=8,
            rows=5,
            offsets_px=offsets,
            evidence_mask=masks[name],
            sequence_dim=8,
            hidden_dim=16,
            num_heads=4,
            ff_dim=32,
            dropout=0.0,
            top_k=4,
        )
        counts.append(sum(parameter.numel() for parameter in arm.parameters()))
    assert len(set(counts)) == 1


def test_offset_validation_rejects_asymmetric_contract() -> None:
    assert _validate_offsets([-2.0, 0.0, 2.0], "test") == [-2.0, 0.0, 2.0]
    try:
        _validate_offsets([-2.0, 0.0, 3.0], "test")
    except ValueError as error:
        assert "symmetric" in str(error)
    else:
        raise AssertionError("asymmetric offsets must be rejected")


class _FakeBackbone(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        base = images.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return {
            "c2": F.avg_pool2d(base, 4),
            "c3": F.avg_pool2d(base, 8),
            "c4": F.avg_pool2d(base, 16),
            "c5": F.avg_pool2d(base, 32),
        }


class _FakeFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lateral = nn.ModuleDict(
            {name: nn.Identity() for name in ("c2", "c3", "c4", "c5")}
        )
        self.output = nn.Identity()


class _FakeRowLayer(nn.Module):
    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return rows


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeRowLayer()])

    def forward(
        self,
        features: torch.Tensor,
        inference_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, channels, _height, _width = features.shape
        candidates, rows = 3, 5
        state = features.mean(dim=(-2, -1)).view(batch, 1, 1, channels)
        state = state.expand(batch, candidates, rows, channels).contiguous()
        state = self.layers[-1](state)
        x = torch.linspace(8.0, 56.0, rows, device=features.device)
        x = x.view(1, 1, rows).expand(batch, candidates, rows)
        ranges = torch.tensor([0.0, 1.0], device=features.device)
        return {
            "pred_x_rows": x,
            "range_norm": ranges.view(1, 1, 2).expand(batch, candidates, 2),
        }


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.fpn = _FakeFPN()
        self.proj = nn.Identity()


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.structured_query_head = _FakeHead()


def test_frozen_source_extraction_preserves_scale_row_offset_contract() -> None:
    model = _FakeModel().eval()
    images = torch.stack(
        (
            torch.ones(3, 32, 64),
            torch.full((3, 32, 64), 2.0),
        )
    )
    curves = torch.linspace(8.0, 56.0, 5).view(1, 1, 5).expand(2, 3, 5)
    result = extract_frozen_sources(
        model,
        images,
        cached_curves=curves,
        row_indices=torch.arange(5),
        p2_offsets=torch.tensor([-4.0, 0.0, 4.0]),
        context_offsets=torch.tensor([-8.0, 0.0, 8.0]),
        input_h=32,
        input_w=64,
        range_temperature=0.02,
        amp_dtype=None,
        include_wrong_image=True,
    )
    assert result["profiles"].shape == (2, 3, 5, 3, 3, 4)
    assert result["wrong_profiles"].shape == result["profiles"].shape
    assert result["row_states"].shape == (2, 3, 5, 4)
    assert float(result["prediction_parity_max_abs_px"]) == 0.0
    assert float((result["profiles"] - result["wrong_profiles"]).abs().max()) > 0.0
