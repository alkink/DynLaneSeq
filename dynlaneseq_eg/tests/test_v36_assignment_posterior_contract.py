from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v36_assignment_posterior_contract import (
    classify_verdict,
    route_target_distribution,
    target_with_range,
)


def _overall(
    *,
    route_preference: float,
    route_top1: float,
    matcher_preference: float,
    good_positive: float,
    good_intended: float,
    counterfactual: float,
) -> dict:
    return {
        "metrics": {
            "route_target_good_preference": {"mean": route_preference},
            "route_target_good_top1": {"mean": route_top1},
            "configured_cost_good_preference": {"mean": matcher_preference},
            "configured_good_any_positive": {"mean": good_positive},
            "configured_good_intended": {"mean": good_intended},
        },
        "assignments": {
            "configured": {"good_intended": {"mean": good_intended}},
            "no_object": {"good_intended": {"mean": counterfactual}},
            "point_only": {"good_intended": {"mean": counterfactual}},
            "target_row_iou": {"good_intended": {"mean": counterfactual}},
        },
    }


def test_target_with_range_uses_fixed_training_rows() -> None:
    target = {
        "x_rows": torch.zeros((2, 4)),
        "valid_mask": torch.tensor(
            [[False, True, True, False], [False, False, False, False]]
        ),
    }
    result = target_with_range(target, input_h=8)
    assert torch.equal(result["range_y"][0], torch.tensor([2.0, 4.0]))
    assert torch.equal(result["range_y"][1], torch.tensor([0.0, 0.0]))


def test_route_target_distribution_is_near_best_and_normalized() -> None:
    quality = torch.tensor(
        [[0.80], [0.75], [0.68], [0.99]], dtype=torch.float32
    )
    valid = torch.tensor([True, True, True, False])
    target = route_target_distribution(
        quality,
        valid,
        0,
        cluster_delta=0.10,
        temperature=0.03,
    )
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert target[0] > target[1] > 0.0
    assert target[2] == 0.0
    assert target[3] == 0.0


def test_verdict_detects_target_misalignment() -> None:
    verdict = classify_verdict(
        _overall(
            route_preference=0.69,
            route_top1=0.90,
            matcher_preference=0.90,
            good_positive=0.90,
            good_intended=0.90,
            counterfactual=0.95,
        )
    )
    assert verdict["decision"] == "ROUTE_TARGET_MISALIGNED"


def test_verdict_detects_assignment_starvation() -> None:
    verdict = classify_verdict(
        _overall(
            route_preference=0.85,
            route_top1=0.80,
            matcher_preference=0.60,
            good_positive=0.60,
            good_intended=0.50,
            counterfactual=0.70,
        )
    )
    assert verdict["decision"] == "HARD_ASSIGNMENT_STARVATION"


def test_verdict_detects_post_assignment_failure() -> None:
    verdict = classify_verdict(
        _overall(
            route_preference=0.90,
            route_top1=0.85,
            matcher_preference=0.80,
            good_positive=0.90,
            good_intended=0.78,
            counterfactual=0.82,
        )
    )
    assert verdict["decision"] == "POST_ASSIGNMENT_BELIEF_FAILURE"
