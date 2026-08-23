from __future__ import annotations

from dynlaneseq_eg.tools.summarize_v33_primary_aux_gate import build


def _values(f1: float, tp: int, fp: int, fn: int) -> dict[str, float | int]:
    return {
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": tp / (tp + fp),
        "Recall": tp / (tp + fn),
    }


def test_v33_summary_keeps_the_two_causal_deltas_separate() -> None:
    arm_a = {
        "0.5": _values(0.720, 720, 280, 280),
        "0.75": _values(0.400, 400, 600, 600),
    }
    arm_b = {
        "0.5": _values(0.725, 725, 275, 275),
        "0.75": _values(0.401, 401, 599, 599),
    }
    arm_c = {
        "0.5": _values(0.729, 729, 271, 271),
        "0.75": _values(0.403, 403, 597, 597),
    }
    wrong = {
        "0.5": _values(0.200, 200, 800, 800),
        "0.75": _values(0.100, 100, 900, 900),
    }
    first = {"metrics": {"control": arm_a, "image_ownership": arm_b}}
    second = {
        "metrics": {
            "control": arm_b,
            "image_ownership": arm_c,
            "image_ownership_wrong_image": wrong,
        }
    }
    result = build(first, second)
    assert abs(result["deltas_F1_points"]["B_minus_A"]["0.5"] - 0.5) < 1.0e-9
    assert abs(result["deltas_F1_points"]["C_minus_B"]["0.5"] - 0.4) < 1.0e-9
    assert result["gates"]["auxiliary_training_only"]["passed"]
    assert result["gates"]["auxiliary_memory"]["passed"]
    assert result["gates"]["overall_primary_auxiliary_mechanism"]["passed"]

