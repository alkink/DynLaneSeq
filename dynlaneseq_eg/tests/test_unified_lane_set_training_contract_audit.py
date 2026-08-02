from __future__ import annotations

import torch

from dynlaneseq_eg.losses.loss_s0 import LossConfig, S0Criterion
from dynlaneseq_eg.tools.analyze_unified_lane_set_training_contract import (
    GradientAlignmentStats,
    _average_precision,
    _build_verdict,
    _loss_groups,
)


def test_gradient_alignment_reports_opposing_score_update() -> None:
    stats = GradientAlignmentStats()
    stats.update(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-2.0, 0.0]),),
    )
    summary = stats.summary()
    assert summary["cosine_geometry_vs_score"]["mean"] == -1.0
    assert summary["score_to_geometry_norm_ratio"]["mean"] == 2.0
    assert summary["score_projection_on_geometry"]["mean"] == -2.0
    assert summary["negative_cosine_fraction"] == 1.0


def test_loss_partition_reconstructs_v3_objective() -> None:
    cfg = LossConfig(
        w_exist=2.0,
        w_point=5.0,
        w_range=1.0,
        w_smooth=0.0,
        w_line_iou=2.0,
        w_seg=1.0,
        w_quality=0.0,
        w_centerline=0.25,
        w_row_dfl=0.5,
        row_dfl_warmup_iters=0,
        lambda_intermediate=0.5,
        w_cardinality=0.1,
        w_score_margin=0.25,
        w_set_selection=0.0,
    )
    criterion = S0Criterion(cfg)
    one = torch.tensor(1.0, requires_grad=True)
    losses = {
        "loss_exist": one,
        "loss_point": one,
        "loss_range": one,
        "loss_smooth": one,
        "loss_line_iou": one,
        "loss_seg": one,
        "loss_quality": one,
        "loss_cardinality": one,
        "loss_score_margin": one,
        "loss_set_selection": one,
        "loss_centerline": one,
        "loss_row_dfl": one,
        "loss_intermediate_exist": one,
        "loss_intermediate_point": one,
        "loss_intermediate_range": one,
        "loss_intermediate_line_iou": one,
        "loss_intermediate_row_dfl": one,
    }
    groups = _loss_groups(criterion, losses)
    assert torch.isclose(groups["score"], torch.tensor(3.35))
    assert torch.isclose(groups["geometry"], torch.tensor(12.75))
    assert torch.isclose(groups["dense"], torch.tensor(1.25))


def test_candidate_average_precision_is_threshold_free() -> None:
    assert _average_precision([0.9, 0.2, 0.8], [1, 0, 1]) == 1.0
    assert _average_precision([0.1, 0.2], [0, 0]) is None


def test_verdict_can_report_mixed_geometry_and_scoring_failure() -> None:
    verdict = _build_verdict(
        gradient={},
        alignment={
            "pearson_score_vs_best_official_iou": 0.1,
            "unique_candidate_ap_050": 0.3,
        },
        capacity={
            "0.50": {
                "direct_topk_recall": 0.60,
                "oracle_topk_recall": 0.72,
                "all_candidates_recall": 0.70,
            }
        },
        count={"probability_mass_vs_training_count_mae": 0.2},
    )
    assert verdict["primary_signal"] == "mixed_geometry_and_scoring_bottleneck"
    assert "weak_score_localization_alignment" in verdict["evidence"]
    assert "candidate_geometry_capacity_is_still_limited" in verdict["evidence"]
