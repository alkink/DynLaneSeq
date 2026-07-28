from __future__ import annotations

import torch

from dynlaneseq_eg.modeling.structured_queries import StructuredLaneQueryHead
from dynlaneseq_eg.tools.analyze_attention_acquisition import (
    AttentionStats,
    LaneRecord,
    _build_corridor_indicator,
    _structured_forward_with_attention,
)
from dynlaneseq_eg.tools.analyze_position_transport import (
    _attention_position_fusion,
    _classifier_coordinate_codebook,
    _position_variant,
)


def test_corridor_indicator_targets_only_group_zero_assignment() -> None:
    targets = [
        {
            "x_rows": torch.tensor([[10.0, 30.0, 50.0]]),
            "valid_mask": torch.tensor([[True, True, False]]),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([1, 3]),
            "gt_indices": torch.tensor([0, 0]),
        }
    ]
    indicator = _build_corridor_indicator(
        targets=targets,
        matches=matches,
        batch=1,
        instances=4,
        rows=3,
        x_bins=10,
        group_size=2,
        input_w=100.0,
        radius_px=6.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert indicator.shape == (1, 3, 4, 10)
    assert bool(indicator[0, 0, 1].any())
    assert bool(indicator[0, 1, 1].any())
    assert not bool(indicator[0, 2, 1].any())
    assert not bool(indicator[:, :, 0].any())
    assert not bool(indicator[:, :, 2:].any())


def test_manual_attention_forward_matches_production_head() -> None:
    torch.manual_seed(7)
    head = StructuredLaneQueryHead(
        dim=16,
        num_instances=4,
        num_rows=3,
        x_bins=8,
        input_w=80,
        num_heads=4,
        num_layers=2,
        ff_dim=32,
        dropout=0.0,
        evidence_x_bins=8,
        num_groups=2,
    ).eval()
    features = torch.randn(2, 16, 3, 8)
    expected = head(features)
    actual, stages, attention = _structured_forward_with_attention(head, features)
    assert len(stages) == 2
    assert len(attention) == 2
    assert attention[0].shape == (2, 3, 4, 4, 8)
    assert torch.allclose(
        actual["pred_x_rows"],
        expected["pred_x_rows"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        actual["exist_logits"],
        expected["exist_logits"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_coordinate_feedback_is_zero_preserving_and_causally_active() -> None:
    torch.manual_seed(11)
    head = StructuredLaneQueryHead(
        dim=16,
        num_instances=4,
        num_rows=3,
        x_bins=8,
        input_w=80,
        num_heads=4,
        num_layers=2,
        ff_dim=32,
        dropout=0.0,
        evidence_x_bins=8,
        num_groups=2,
    ).eval()
    features = torch.randn(1, 16, 3, 8)
    baseline, _stages, _attention = _structured_forward_with_attention(
        head,
        features,
    )
    codebook = _classifier_coordinate_codebook(
        row_classifier_weight=head.row_x.weight,
        evidence_bins=8,
    )
    zero, _stages, _attention = _structured_forward_with_attention(
        head,
        features,
        coordinate_feedback_by_layer={0: (codebook, 0.0)},
    )
    changed, _stages, _attention = _structured_forward_with_attention(
        head,
        features,
        coordinate_feedback_by_layer={0: (codebook, 1.0)},
    )
    torch.testing.assert_close(
        zero["pred_x_rows"],
        baseline["pred_x_rows"],
    )
    assert not torch.allclose(
        changed["pred_x_rows"],
        baseline["pred_x_rows"],
    )


def test_classifier_coordinate_codebook_has_unit_rows() -> None:
    torch.manual_seed(13)
    weight = torch.randn(8, 16)
    codebook = _classifier_coordinate_codebook(
        row_classifier_weight=weight,
        evidence_bins=4,
    )
    assert codebook.shape == (4, 16)
    torch.testing.assert_close(
        codebook.norm(dim=-1),
        torch.ones(4),
        atol=1e-6,
        rtol=1e-6,
    )


def test_position_variants_preserve_shape_and_shuffle_deterministically() -> None:
    original = torch.arange(24, dtype=torch.float32).view(6, 4)
    reverse = _position_variant(
        original,
        "reverse",
        roll_columns=2,
        seed=7,
    )
    shuffled_a = _position_variant(
        original,
        "shuffle",
        roll_columns=2,
        seed=7,
    )
    shuffled_b = _position_variant(
        original,
        "shuffle",
        roll_columns=2,
        seed=7,
    )
    torch.testing.assert_close(reverse, original.flip(0))
    torch.testing.assert_close(shuffled_a, shuffled_b)
    assert reverse.shape == original.shape
    assert not torch.equal(shuffled_a, original)


def test_attention_position_fusion_has_exact_endpoints() -> None:
    row_logits = torch.tensor([[[[8.0, 0.0, 0.0, 0.0]]]])
    # [B=1, R=1, H=1, N=1, E=2], all attention on the right half.
    attention = torch.tensor([[[[[0.0, 1.0]]]]])
    model_only = _attention_position_fusion(
        row_logits=row_logits,
        attention=attention,
        input_w=80.0,
        alpha=0.0,
    )
    attention_only = _attention_position_fusion(
        row_logits=row_logits,
        attention=attention,
        input_w=80.0,
        alpha=1.0,
    )
    reversed_only = _attention_position_fusion(
        row_logits=row_logits,
        attention=attention,
        input_w=80.0,
        alpha=1.0,
        reverse_attention=True,
    )
    assert float(model_only) < 11.0
    assert float(attention_only) > 55.0
    assert float(reversed_only) < 25.0


def test_attention_stats_detects_corridor_concentration() -> None:
    # [B=1, R=2, H=1, N=1, X=4]
    attention = torch.tensor(
        [
            [
                [[[0.90, 0.05, 0.03, 0.02]]],
                [[[0.02, 0.03, 0.05, 0.90]]],
            ]
        ],
        dtype=torch.float32,
    )
    target = {
        "x_rows": torch.tensor([[10.0, 70.0]]),
        "valid_mask": torch.tensor([[True, True]]),
    }
    record = LaneRecord(
        image_index=0,
        gt_index=0,
        pred_index=0,
        stage_ious=(0.0, 0.0, 0.0, 0.0),
    )
    stats = AttentionStats()
    stats.update(
        attention,
        target,
        record,
        input_w=80.0,
        radius_px=6.0,
        line_width=30.0,
    )
    summary = stats.summary()
    assert summary["valid_lane_rows"] == 2
    assert summary["corridor_attention_mass"] > 0.89
    assert summary["corridor_enrichment"] > 3.5
    assert summary["top1_inside_fraction"] == 1.0
    assert summary["attention_peak_mean_iou"] > 0.99
    assert summary["attention_peak_recall_050"] == 1.0
