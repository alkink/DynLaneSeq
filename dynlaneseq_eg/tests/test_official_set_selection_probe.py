from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.probe_official_set_selection import (
    SetAwareQualityProbe,
    assert_disjoint_image_paths,
    official_unique_quality_targets,
    selection_features,
    selection_verdict,
    training_index_schedule,
)


def test_official_unique_target_keeps_one_candidate_per_gt() -> None:
    official_iou = torch.tensor(
        [
            [0.95, 0.90, 0.05, 0.00],
            [0.10, 0.15, 0.92, 0.20],
        ]
    )
    targets = official_unique_quality_targets(
        official_iou,
        torch.tensor([True, True, True, False]),
    )
    assert targets.shape == (4,)
    assert targets[0] == pytest.approx(0.95)
    assert targets[1] == 0.0
    assert targets[2] == pytest.approx(0.92)
    assert targets[3] == 0.0


def test_set_aware_probe_is_permutation_equivariant() -> None:
    torch.manual_seed(7)
    probe = SetAwareQualityProbe(
        12,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    ).eval()
    features = torch.randn(2, 5, 12)
    permutation = torch.tensor([2, 4, 0, 3, 1])
    expected = probe(features)[:, permutation]
    actual = probe(features[:, permutation])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_selection_features_append_curve_and_confidence_samples() -> None:
    batch, candidates, rows, channels, bins = 2, 3, 8, 6, 12
    outputs = {
        "structured_row_tokens": torch.randn(
            batch,
            candidates,
            rows,
            channels,
        ),
        "queries": torch.randn(batch, candidates, channels),
        "range_norm": torch.tensor(
            [[[0.1, 0.9]] * candidates, [[0.2, 0.8]] * candidates]
        ),
        "pred_x_rows": torch.rand(batch, candidates, rows) * 100.0,
        "row_x_logits": torch.randn(batch, candidates, rows, bins),
        "input_reference_x_rows": torch.rand(batch, candidates, rows) * 100.0,
    }
    features = selection_features(
        outputs,
        input_w=100,
        curve_samples=5,
    )
    expected_dim = 2 * channels + 10 + 2 * 5
    assert features.shape == (batch, candidates, expected_dim)
    assert torch.isfinite(features).all()


def test_cache_splits_must_be_disjoint() -> None:
    assert_disjoint_image_paths(
        ("train/a.jpg", "train/b.jpg"),
        ("val/a.jpg", "val/b.jpg"),
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_image_paths(
            ("shared.jpg", "train/b.jpg"),
            ("shared.jpg", "val/b.jpg"),
        )


def test_training_schedule_is_seed_deterministic() -> None:
    first = training_index_schedule(
        num_examples=41,
        batch_size=7,
        steps=9,
        seed=3407,
    )
    second = training_index_schedule(
        num_examples=41,
        batch_size=7,
        steps=9,
        seed=3407,
    )
    different = training_index_schedule(
        num_examples=41,
        batch_size=7,
        steps=9,
        seed=3408,
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_verdict_separates_target_and_set_context() -> None:
    def row(f1_050: float, f1_070: float) -> dict[str, float]:
        return {"f1_050": f1_050, "f1_070": f1_070}

    evaluation = {
        "modes": {
            "nms_top4": {
                "current_exist_quality": row(0.70, 0.50),
                "official_scalar_quality": row(0.705, 0.502),
                "official_set_quality": row(0.72, 0.51),
            }
        }
    }
    verdict = selection_verdict(
        evaluation,
        min_gain_050_points=1.0,
        min_gain_070_points=0.5,
    )
    assert verdict["arms"]["official_scalar_quality"]["positive"] is False
    assert verdict["arms"]["official_set_quality"]["positive"] is True
    assert verdict["recommendation"] == "add_set_aware_selection_head"
