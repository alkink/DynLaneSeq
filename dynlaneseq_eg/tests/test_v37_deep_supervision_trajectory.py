from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.audit_v37_deep_supervision_trajectory import (
    classify_verdict,
    exact_candidate_descent_update,
    layer_loss_coefficients,
)


def _metric(mean: float) -> dict[str, float]:
    return {"mean": float(mean)}


def _summary(
    *,
    good_up: float = 0.95,
    wrong_down: float = 0.95,
    good_down: float = 0.0,
    all_layer_intended: float = 0.9,
) -> dict[str, object]:
    return {
        "metrics": {
            "good_total_update_up": _metric(good_up),
            "wrong_total_update_down": _metric(wrong_down),
            "good_total_update_down": _metric(good_down),
            "good_all_layers_intended": _metric(all_layer_intended),
        }
    }


def _gradients(*, negative: float, ratio: float) -> dict[str, object]:
    return {
        "negative_fraction": float(negative),
        "right_to_left_norm_ratio": {"median": float(ratio)},
    }


def test_v7_layer_loss_coefficients_are_exact() -> None:
    assert layer_loss_coefficients([1.0, 2.0, 3.0], 0.5) == pytest.approx(
        [1.0 / 12.0, 1.0 / 6.0, 1.0 / 4.0, 1.0]
    )


def test_exact_candidate_descent_update_has_correct_sign_and_weight() -> None:
    logits = torch.zeros(1, 2, 2)
    labels = torch.tensor([[1.0, 0.0]])
    update = exact_candidate_descent_update(
        logits,
        labels,
        no_lane_weight=0.1,
        coefficient=0.5,
        exist_loss_weight=2.0,
    )

    # Weighted CE uses denominator 1.0 + 0.1.  Gradient descent raises the
    # foreground logit for the positive and lowers it for the negative.
    assert float(update[0, 0]) == pytest.approx(0.5 / 1.1)
    assert float(update[0, 1]) == pytest.approx(-0.05 / 1.1)


@pytest.mark.parametrize(
    ("overall", "final_positive", "gradients", "expected"),
    [
        (
            _summary(),
            _summary(good_down=0.30),
            _gradients(negative=0.0, ratio=0.1),
            "DEEP_SUPERVISION_SCORE_CONFLICT",
        ),
        (
            _summary(good_up=0.80, wrong_down=0.80),
            _summary(good_down=0.02, all_layer_intended=0.50),
            _gradients(negative=0.10, ratio=0.5),
            "ASSIGNMENT_CHURN_WITH_ALIGNED_SCORE_GRADIENT",
        ),
        (
            _summary(good_up=0.95, wrong_down=0.96),
            _summary(good_down=0.02, all_layer_intended=0.90),
            _gradients(negative=0.10, ratio=0.5),
            "DEEP_SUPERVISION_NOT_PRIMARY",
        ),
    ],
)
def test_predeclared_verdicts(
    overall: dict[str, object],
    final_positive: dict[str, object],
    gradients: dict[str, object],
    expected: str,
) -> None:
    verdict = classify_verdict(overall, final_positive, gradients)
    assert verdict["decision"] == expected
    assert verdict["test_split_used"] is False
