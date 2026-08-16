from __future__ import annotations

import torch

from dynlaneseq_eg.tools.audit_v23_geometry_gate import (
    _interpret,
    raw_student_public_output,
)


def test_raw_student_is_restored_to_public_v7_slot_order() -> None:
    raw = torch.tensor(
        [[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]]
    )
    # Canonical slots correspond to public slots [2, 0, 3, 1].
    source_slot_indices = torch.tensor([[2, 0, 3, 1]])
    output = {
        "student_x_rows": raw,
        "source_slot_indices": source_slot_indices,
        "exist_logits": torch.zeros(1, 4, 2),
        "range_norm": torch.zeros(1, 4, 2),
        "quality_logits": torch.zeros(1, 4),
    }
    public = raw_student_public_output(output)
    assert public["pred_x_rows"].tolist() == [
        [[20.0, 21.0], [40.0, 41.0], [10.0, 11.0], [30.0, 31.0]]
    ]


def _metric(f1, tp):
    return {"F1": f1, "TP": tp}


def test_interpretation_calls_gate_a_bottleneck_only_when_raw_wins_both() -> None:
    summary = {
        "metrics": {
            "source_v7": {
                "0.50": _metric(0.80, 100),
                "0.75": _metric(0.60, 70),
            },
            "learned_gate_v23": {
                "0.50": _metric(0.801, 101),
                "0.75": _metric(0.601, 71),
            },
            "raw_student_correct_image": {
                "0.50": _metric(0.82, 110),
                "0.75": _metric(0.63, 80),
            },
            "raw_student_wrong_image": {
                "0.50": _metric(0.78, 90),
                "0.75": _metric(0.55, 60),
            },
        }
    }
    result = _interpret(summary)
    assert result["verdict"] == "geometry_gate_bottleneck_supported"
    assert all(result["checks"].values())


def test_interpretation_detects_gate_protection() -> None:
    summary = {
        "metrics": {
            "source_v7": {
                "0.50": _metric(0.80, 100),
                "0.75": _metric(0.60, 70),
            },
            "learned_gate_v23": {
                "0.50": _metric(0.801, 101),
                "0.75": _metric(0.601, 71),
            },
            "raw_student_correct_image": {
                "0.50": _metric(0.70, 80),
                "0.75": _metric(0.50, 50),
            },
            "raw_student_wrong_image": {
                "0.50": _metric(0.69, 70),
                "0.75": _metric(0.49, 40),
            },
        }
    }
    result = _interpret(summary)
    assert result["verdict"] == (
        "geometry_gate_protected_source_from_worse_raw_student"
    )
