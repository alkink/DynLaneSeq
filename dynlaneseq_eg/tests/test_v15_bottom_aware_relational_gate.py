from __future__ import annotations

from copy import deepcopy

from dynlaneseq_eg.tools.summarize_v15_bottom_aware_relational_gate import (
    _domain,
)


def _row(tp: int, predictions: int = 100, gt: int = 100) -> dict[str, float | int]:
    fp = predictions - tp
    fn = gt - tp
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predictions": predictions,
        "gt": gt,
        "f1": 0.0 if denominator == 0 else 2.0 * tp / denominator,
    }


def _delta(left: dict[str, float | int], right: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "delta_tp": int(left["tp"]) - int(right["tp"]),
        "delta_fp": int(left["fp"]) - int(right["fp"]),
        "delta_fn": int(left["fn"]) - int(right["fn"]),
        "delta_predictions": int(left["predictions"])
        - int(right["predictions"]),
        "delta_f1": float(left["f1"]) - float(right["f1"]),
        "delta_f1_points": 100.0
        * (float(left["f1"]) - float(right["f1"])),
    }


def _report() -> dict:
    rows = {
        "v7_anchor": {"0.50": _row(80), "0.75": _row(60)},
        "v15_final": {"0.50": _row(84), "0.75": _row(61)},
        "cross_clip_wrong_p2_final": {
            "0.50": _row(81),
            "0.75": _row(59),
        },
        "identity_graph_final": {"0.50": _row(82), "0.75": _row(60)},
        "geometry_shuffled_graph_final": {
            "0.50": _row(81),
            "0.75": _row(59),
        },
        "no_proposal_context_final": {
            "0.50": _row(83),
            "0.75": _row(60),
        },
    }
    metrics = {
        policy: {
            mode: {"thresholds": deepcopy(thresholds)}
            for mode in ("neural_active", "writer_valid")
        }
        for policy, thresholds in rows.items()
    }
    comparisons = {
        "v15_minus_v7": ("v15_final", "v7_anchor"),
        "correct_minus_cross_clip_wrong_p2": (
            "v15_final",
            "cross_clip_wrong_p2_final",
        ),
        "correct_minus_identity_graph": (
            "v15_final",
            "identity_graph_final",
        ),
        "correct_minus_geometry_shuffled_graph": (
            "v15_final",
            "geometry_shuffled_graph_final",
        ),
        "correct_minus_no_proposal_context": (
            "v15_final",
            "no_proposal_context_final",
        ),
    }
    deltas = {
        name: {
            mode: {
                threshold: _delta(rows[left][threshold], rows[right][threshold])
                for threshold in ("0.50", "0.75")
            }
            for mode in ("neural_active", "writer_valid")
        }
        for name, (left, right) in comparisons.items()
    }
    return {
        "metrics": metrics,
        "deltas": deltas,
        "cross_clip_runtime_same_image": 0,
        "cross_clip_runtime_same_clip": 0,
        "hard_cluster_or_prototype_used": False,
        "proposal_id_supervision_used": False,
        "activity_score_route_source": "exact_v7",
        "test_set_used": False,
    }


def _coverage(duplicate: float = 0.0) -> dict:
    return {
        "four_slot_diagnostics": {
            "semantic_duplicate_cluster_fraction": duplicate,
        },
        "capacity": {
            threshold: {"all_candidate_oracle": {"hits": hits}}
            for threshold, hits in (("0.50", 97), ("0.75", 88))
        },
    }


def test_v15_gate_requires_transfer_and_both_causal_advantages() -> None:
    domain = _domain(_report(), _coverage(), _coverage())
    assert domain["passed"] is True
    assert domain["checks"]["correct_p2_at_least_wrong_plus_2_tp_050"]
    assert domain["checks"]["correct_graph_at_least_identity_plus_2_tp_050"]


def test_v15_gate_rejects_nominal_gain_without_graph_advantage() -> None:
    report = _report()
    report["deltas"]["correct_minus_identity_graph"]["writer_valid"][
        "0.50"
    ]["delta_tp"] = 1
    domain = _domain(report, _coverage(), _coverage())
    assert domain["passed"] is False
    assert not domain["checks"][
        "correct_graph_at_least_identity_plus_2_tp_050"
    ]
