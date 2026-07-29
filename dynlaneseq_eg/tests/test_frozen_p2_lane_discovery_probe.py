from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_frozen_p2_lane_discovery import (
    FrozenP2LaneDiscoveryProbe,
    TeacherSeedConditionMetrics,
    _curve_loss,
    _gaussian_heatmap_targets,
    _topk_seeds,
)


def _target() -> dict[str, torch.Tensor]:
    x_rows = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    return {
        "x_rows": x_rows,
        "valid_mask": torch.ones_like(x_rows, dtype=torch.bool),
    }


def test_gaussian_target_places_lane_endpoint_peak() -> None:
    heatmap, seeds = _gaussian_heatmap_targets(
        [_target()],
        height=4,
        width=10,
        num_rows=4,
        input_w=100.0,
        sigma=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert seeds == [[(0, 3, 4)]]
    assert heatmap.shape == (1, 4, 10)
    assert torch.isclose(heatmap[0, 3, 4], torch.tensor(1.0))


def test_seeded_curve_loss_is_finite_and_backpropagates() -> None:
    target = _target()
    seeds = [[(0, 3, 4)]]
    probe = FrozenP2LaneDiscoveryProbe(
        in_dim=8,
        hidden_dim=8,
        num_rows=4,
        decoder_rows=4,
        probe_height=4,
        probe_width=10,
        input_w=100,
    )
    hidden = torch.randn((1, 8, 4, 10), requires_grad=True)
    loss, ce, point = _curve_loss(
        probe,
        hidden,
        [target],
        seeds,
        input_w=100.0,
        beta=0.01,
    )
    assert torch.isfinite(loss)
    assert float(ce) > 0.0
    assert float(point) >= 0.0
    loss.backward()
    assert hidden.grad is not None


def test_topk_seeds_returns_heatmap_peak() -> None:
    heatmap_logits = torch.full((1, 4, 5), -10.0)
    heatmap_logits[0, 2, 3] = 10.0
    scores, peaks = _topk_seeds(
        {"heatmap_logits": heatmap_logits},
        top_k=1,
        nms_radius=1,
    )
    assert peaks.tolist() == [[[2, 3]]]
    assert scores[0, 0] > 0.99


def test_teacher_seed_metrics_partition_base_hits_and_misses() -> None:
    metrics = TeacherSeedConditionMetrics()
    metrics.update(
        torch.tensor([0.8, 0.2, 0.6, 0.4]),
        torch.tensor([0.9, 0.7, 0.4, 0.1]),
    )
    summary = metrics.summary()
    assert summary["lanes"] == 4
    assert summary["paired_recall_050"] == 0.5
    assert summary["base_hit_lanes"] == 2
    assert summary["base_hit_paired_recall_050"] == 0.5
    assert summary["base_miss_lanes"] == 2
    assert summary["base_miss_paired_recall_050"] == 0.5
