from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    cardinality_oracle_assignment,
    evaluator_hungarian_assignment,
)
from dynlaneseq_eg.tools.analyze_v4_5_cluster_support import (
    analyze_cluster_support_image,
    cluster_soft_distribution,
    maximum_cardinality_support_assignment,
    summarize_cluster_support,
)


def _analyze(
    quality: torch.Tensor,
    *,
    support_mins: tuple[float, ...] = (0.50,),
    quality_deltas: tuple[float, ...] = (0.05,),
    temperatures: tuple[float, ...] = (0.05,),
) -> dict:
    return analyze_cluster_support_image(
        quality,
        torch.ones(quality.shape[1], dtype=torch.bool),
        representable_thresholds=(0.50,),
        support_mins=support_mins,
        quality_deltas=quality_deltas,
        temperatures=temperatures,
        top_k=4,
    )


def test_cardinality_first_matching_avoids_post_hungarian_hit_loss() -> None:
    # Maximum summed IoU chooses (GT0,C0)=1.00 and (GT1,C1)=.49, which
    # produces one hit after thresholding.  The crossed assignment produces
    # two valid .74 hits and is therefore the correct representability oracle.
    quality = torch.tensor([[1.00, 0.74], [0.74, 0.49]])
    vanilla = evaluator_hungarian_assignment(quality, (0, 1), threshold=0.50)
    cardinality = cardinality_oracle_assignment(
        quality,
        threshold=0.50,
        top_k=4,
        candidate_valid=torch.ones(2, dtype=torch.bool),
    )
    assert vanilla.hit_count == 1
    assert cardinality.hit_count == 2


def test_support_assignment_prioritizes_joint_coverage() -> None:
    quality = torch.tensor([[1.00, 0.74], [0.74, 0.49]])
    support = quality > 0.50
    pairs = maximum_cardinality_support_assignment(quality, support, top_k=4)
    assert set(pairs) == {(0, 1), (1, 0)}


def test_cluster_soft_distribution_is_normalized_and_quality_weighted() -> None:
    support, probability, cutoff = cluster_soft_distribution(
        torch.tensor([0.90, 0.87, 0.70]),
        torch.ones(3, dtype=torch.bool),
        support_min=0.50,
        quality_delta=0.05,
        temperature=0.05,
    )
    assert cutoff == pytest.approx(0.85)
    assert support.tolist() == [True, True, False]
    assert float(probability.sum()) == pytest.approx(1.0)
    assert float(probability[0]) > float(probability[1]) > 0.0
    assert float(probability[2]) == 0.0


def test_audit_detects_cross_gt_support_overlap_and_sampling_collision() -> None:
    row = _analyze(
        torch.tensor(
            [
                [0.90, 0.86, 0.10],
                [0.10, 0.84, 0.80],
            ]
        ),
        quality_deltas=(0.10,),
    )
    grid = next(iter(row["thresholds"]["0.500"]["grids"].values()))
    assert grid["full_joint_support_coverage"] is True
    assert grid["multi_cluster_candidate_count"] == 1
    assert grid["intersecting_gt_pair_count"] == 1
    assert grid["expected_pair_collision_sum"] > 0.0


def test_support_min_045_can_leak_target_mass_below_metric_boundary() -> None:
    rows = [
        _analyze(
            torch.tensor([[0.55, 0.49, 0.10]]),
            support_mins=(0.45, 0.50),
            quality_deltas=(0.10,),
            temperatures=(0.05,),
        )
    ]
    report = summarize_cluster_support(rows, representable_thresholds=(0.50,))
    grids = report["thresholds"]["0.500"]["grids"]
    loose = next(
        value
        for value in grids.values()
        if value["parameters"]["support_min"] == 0.45
    )
    strict = next(
        value
        for value in grids.values()
        if value["parameters"]["support_min"] == 0.50
    )
    assert loose["target_mass_at_or_below_representable_threshold"]["mean"] > 0.0
    assert loose["preflight_checks"]["no_subthreshold_target_mass"] is False
    assert strict["target_mass_at_or_below_representable_threshold"]["mean"] == 0.0


def test_disjoint_clusters_need_no_collision_safe_teacher_sampler() -> None:
    rows = [
        _analyze(
            torch.tensor(
                [
                    [0.90, 0.87, 0.10, 0.10],
                    [0.10, 0.10, 0.91, 0.88],
                ]
            )
        )
    ]
    report = summarize_cluster_support(rows, representable_thresholds=(0.50,))
    preferred = report["teacher_only_gate_screening"]["provisional_preferred"]
    assert preferred is not None
    assert preferred["requires_collision_safe_teacher_sampling"] is False
