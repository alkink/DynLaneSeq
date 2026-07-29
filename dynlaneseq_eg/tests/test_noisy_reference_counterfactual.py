import torch

from dynlaneseq_eg.tools.analyze_noisy_reference_counterfactual import (
    _bootstrap_mean_ci,
    _condition_gate,
    _counterfactual_p2,
    _paired_lane_visual_summary,
)
from dynlaneseq_eg.tools.probe_gt_curve_aligned_features import (
    _evaluation_residuals,
)


def test_counterfactual_p2_replaces_only_image_content() -> None:
    p2 = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    controls = _counterfactual_p2(p2)

    assert torch.equal(controls["correct_p2"], p2)
    assert torch.equal(controls["wrong_image_p2"], torch.roll(p2, 1, 0))
    assert int(torch.count_nonzero(controls["zero_p2"])) == 0
    horizontal = controls["horizontal_mean_p2"]
    assert torch.allclose(horizontal, horizontal[..., :1].expand_as(horizontal))


def test_counterfactual_p2_rejects_single_image_wrong_control() -> None:
    try:
        _counterfactual_p2(torch.zeros(1, 4, 2, 2))
    except ValueError as error:
        assert "eval_batch_size >= 2" in str(error)
    else:
        raise AssertionError("single-image wrong control must be rejected")


def test_condition_gate_requires_visual_margin_over_every_control() -> None:
    def summary(recall: float, iou_gain: float, direction: float):
        return {
            "base_miss": {
                "corrected_recall_050": recall,
                "mean_iou_gain": iou_gain,
                "direction_accuracy": direction,
            }
        }

    summaries = {
        "correct_p2": summary(0.40, 0.20, 0.80),
        "reference_only": summary(0.05, 0.00, 0.50),
        "wrong_image_p2": summary(0.10, 0.03, 0.55),
        "zero_p2": summary(0.06, 0.01, 0.51),
        "horizontal_mean_p2": summary(0.09, 0.04, 0.58),
    }
    assert _condition_gate(summaries)["positive_gate"] is True

    summaries["wrong_image_p2"] = summary(0.35, 0.18, 0.76)
    assert _condition_gate(summaries)["positive_gate"] is False


def test_evaluation_residuals_respect_requested_shift() -> None:
    residuals = _evaluation_residuals(
        5,
        device=torch.device("cpu"),
        shift_px=24.0,
    )
    assert torch.equal(residuals[0], torch.full((5,), -24.0))
    assert torch.equal(residuals[1], torch.full((5,), 24.0))
    assert float(residuals[2, 0]) == -24.0
    assert float(residuals[2, -1]) == 24.0


def test_paired_lane_summary_uses_lane_level_patterns() -> None:
    values = {
        (0, 0, 0): {
            "reference_only": [0.1, 0.2],
            "correct_p2": [0.6, 0.7],
            "wrong_image_p2": [0.2, 0.3],
            "zero_p2": [0.1, 0.2],
            "horizontal_mean_p2": [0.1, 0.2],
        },
        (0, 0, 1): {
            "reference_only": [0.2, 0.2],
            "correct_p2": [0.4, 0.6],
            "wrong_image_p2": [0.1, 0.4],
            "zero_p2": [0.2, 0.2],
            "horizontal_mean_p2": [0.2, 0.2],
        },
    }
    summary = _paired_lane_visual_summary(values, seed=7)
    wrong = summary["controls"]["wrong_image_p2"]
    assert summary["base_miss_lanes"] == 2
    assert wrong["correct_p2_best_of_four_recall_050"] == 1.0
    assert wrong["control_best_of_four_recall_050"] == 0.0
    assert wrong["correct_only_pattern_hits_050"] == 3


def test_bootstrap_mean_ci_is_deterministic() -> None:
    values = torch.tensor([0.1, 0.2, 0.3])
    assert _bootstrap_mean_ci(values, seed=11) == _bootstrap_mean_ci(
        values,
        seed=11,
    )
