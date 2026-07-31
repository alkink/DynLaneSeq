from __future__ import annotations

import torch

from dynlaneseq_eg.losses.matcher_s0 import (
    HungarianMatcherS0,
    MatcherConfig,
)
from dynlaneseq_eg.tools.analyze_matcher_cost_counterfactual import (
    PairStats,
    build_counterfactual_configs,
    matcher_cost_components,
    weighted_matcher_cost,
)


def _target() -> dict[str, torch.Tensor]:
    return {
        "x_rows": torch.tensor([[10.0, 20.0, 30.0]]),
        "valid_mask": torch.tensor([[True, True, True]]),
        "range_y": torch.tensor([[0.0, 20.0]]),
    }


def test_decomposed_cost_reconstructs_matcher_cost() -> None:
    matcher = HungarianMatcherS0(
        MatcherConfig(
            lambda_obj=2.0,
            lambda_point=5.0,
            lambda_range=1.0,
            lambda_line_iou=1.0,
            line_iou_radius=5.0,
            input_w=100,
            input_h=30,
            object_cost_type="neg_probability",
        )
    )
    exist_logits = torch.tensor([[3.0, -1.0], [0.0, 1.0]])
    pred_x = torch.tensor(
        [[11.0, 21.0, 31.0], [40.0, 40.0, 40.0]]
    )
    ranges = torch.tensor([[0.0, 0.8], [0.2, 1.0]])
    expected, _ = matcher.compute_cost_for_image(
        exist_logits,
        pred_x,
        ranges,
        _target(),
    )
    components = matcher_cost_components(
        matcher,
        exist_logits,
        pred_x,
        ranges,
        _target(),
    )

    assert torch.allclose(
        weighted_matcher_cost(components, matcher.cfg),
        expected,
    )


def test_counterfactuals_change_only_declared_weights() -> None:
    configured = MatcherConfig(
        lambda_obj=2.0,
        lambda_point=5.0,
        lambda_range=1.0,
        lambda_line_iou=1.0,
    )
    variants = build_counterfactual_configs(configured)

    assert variants["no_object"].lambda_obj == 0.0
    assert variants["no_object"].lambda_point == 5.0
    assert variants["no_range"].lambda_range == 0.0
    assert variants["no_line_iou"].lambda_line_iou == 0.0
    assert variants["point_only"].lambda_obj == 0.0
    assert variants["point_only"].lambda_range == 0.0
    assert variants["point_only"].lambda_line_iou == 0.0


def test_pair_stats_reports_net_strict_iou_rescue() -> None:
    stats = PairStats()
    stats.update(
        reference_candidate=0,
        variant_candidate=1,
        reference_iou=0.65,
        variant_iou=0.78,
        win_margin=0.02,
    )
    summary = stats.summary()

    assert summary["variant_wins"] == 1
    assert summary["reference_wins"] == 0
    assert summary["threshold_transitions"]["0.70"] == {
        "reference_miss_variant_hit": 1,
        "reference_hit_variant_miss": 0,
    }
