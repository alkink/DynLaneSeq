from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v4_pointer_row_sequence_observability import (
    CandidateResidualProbe,
    RowSequenceResidualProbe,
    build_row_sequence_features,
    fixed_rademacher_projection,
    full_cluster_residual_loss,
    prepare_full_cluster_targets,
)


def _quality() -> torch.Tensor:
    return torch.tensor(
        [
            [0.90, 0.05],
            [0.80, 0.04],
            [0.03, 0.88],
            [0.02, 0.84],
        ],
        dtype=torch.float32,
    )


def _cache() -> dict:
    quality = _quality()
    cache = {
        "quality": [quality],
        "candidate_valid": torch.ones(1, 4, dtype=torch.bool),
    }
    prepare_full_cluster_targets(
        cache,
        representable_min=0.50,
        cluster_quality_min=1e-4,
        target_temperature=0.05,
        pair_min_quality_gap=0.02,
    )
    return cache


def test_full_cluster_targets_cover_each_natural_gt_cluster() -> None:
    cache = _cache()
    active = cache["_full_cluster_active"][0]
    target = cache["_full_cluster_target"][0, active].float()

    assert int(active.sum()) == 2
    assert torch.allclose(target.sum(dim=-1), torch.ones(2), atol=1e-3)
    assert cache["_full_cluster_pairs"][0].tolist() == [[0, 1], [2, 3]]


def test_full_cluster_residual_loss_rewards_correct_global_order() -> None:
    cache = _cache()
    correct = torch.tensor([[4.0, 1.0, 3.0, 0.0]])
    reversed_order = -correct
    residual = torch.zeros_like(correct)

    correct_loss, _ = full_cluster_residual_loss(
        correct,
        residual,
        [0],
        cache,
        representable_min=0.50,
        cluster_quality_min=1e-4,
        target_temperature=0.05,
        pair_min_quality_gap=0.02,
        pairwise_weight=0.50,
        residual_l2_weight=0.0,
    )
    reversed_loss, _ = full_cluster_residual_loss(
        reversed_order,
        residual,
        [0],
        cache,
        representable_min=0.50,
        cluster_quality_min=1e-4,
        target_temperature=0.05,
        pair_min_quality_gap=0.02,
        pairwise_weight=0.50,
        residual_l2_weight=0.0,
    )

    assert correct_loss < reversed_loss


def test_candidate_residual_starts_at_checkpoint_unary() -> None:
    probe = CandidateResidualProbe(
        torch.zeros(5),
        torch.ones(5),
        nonlinear=False,
        hidden=8,
        dropout=0.0,
    )
    baseline = torch.randn(2, 4)
    score, residual = probe(torch.randn(2, 4, 5), baseline)

    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(score, baseline)


def test_fixed_projection_is_reproducible_and_seed_sensitive() -> None:
    first = fixed_rademacher_projection(8, 4, 7)
    second = fixed_rademacher_projection(8, 4, 7)
    other = fixed_rademacher_projection(8, 4, 8)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (8, 4)


class _Selector:
    input_w = 100

    @staticmethod
    def _lane_row_grid(
        rows: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.arange(rows, device=device, dtype=dtype) / float(rows)


def test_row_feature_builder_preserves_candidate_and_row_axes() -> None:
    outputs = {
        "structured_row_tokens": torch.randn(1, 2, 4, 3),
        "selection_curve_evidence": torch.randn(1, 2, 4, 3),
        "row_x_logits": torch.randn(1, 2, 4, 5),
        "pred_x_rows": torch.rand(1, 2, 4) * 99.0,
        "range_norm": torch.tensor([[[0.0, 1.0], [0.25, 0.75]]]),
        "input_reference_x_rows": torch.rand(1, 2, 4) * 99.0,
    }
    projection = fixed_rademacher_projection(3, 2, 3)
    features, visible = build_row_sequence_features(
        outputs,
        _Selector(),
        projection,
        projection,
        torch.tensor([0, 2]),
    )

    assert features.shape == (1, 2, 2, 10)
    assert visible.shape == (1, 2, 2)
    assert torch.isfinite(features).all()


def test_row_probe_handles_fully_masked_invalid_candidate() -> None:
    probe = RowSequenceResidualProbe(
        torch.zeros(10),
        torch.ones(10),
        row_count=3,
        hidden=16,
        layers=1,
        heads=4,
        dropout=0.0,
    )
    features = torch.randn(2, 4, 3, 10)
    visible = torch.ones(2, 4, 3, dtype=torch.bool)
    visible[0, 0] = False
    baseline = torch.randn(2, 4)
    score, residual = probe(features, visible, baseline)

    assert score.shape == (2, 4)
    assert residual.shape == (2, 4)
    assert torch.isfinite(score).all()
