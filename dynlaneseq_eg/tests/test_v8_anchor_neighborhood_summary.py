from __future__ import annotations

from dynlaneseq_eg.tools.summarize_v8_anchor_neighborhood_gate import summarize


def _report(f1_050: float, f1_075: float, mix: float):
    def method(f1: float):
        return {
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "tp": 100,
            "fp": 10,
            "fn": 10,
            "mean_selected_per_image": 3.2,
        }

    return {
        "methods": {
            "four_slot_refined": {"0.50": method(f1_050), "0.75": method(f1_075)},
            "four_slot_global_unique": {"0.50": method(0.75), "0.75": method(0.55)},
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.94}},
            "0.75": {"all_candidate_oracle": {"recall": 0.80}},
        },
        "four_slot_diagnostics": {
            "cardinality": {"mean_absolute_error": 0.2},
            "neighborhood": {"mean_mix": mix},
            "refinement": {"mean_abs_delta_px": 2.0},
        },
    }


def test_v8_summary_requires_strict_gain_without_loose_regression():
    source = _report(0.79, 0.56, 0.0)
    final = _report(0.793, 0.571, 0.2)
    result = summarize(source, [(227000, final)], {"passed": True})
    assert result["verdict"] == "strong_pass"
    assert result["full_validation_authorized"] is True
    assert result["long_training_authorized"] is False


def test_v8_summary_blocks_changed_frozen_anchor():
    source = _report(0.79, 0.56, 0.0)
    final = _report(0.80, 0.58, 0.2)
    final["capacity"]["0.50"]["all_candidate_oracle"]["recall"] = 0.90
    result = summarize(source, [(227000, final)], {"passed": True})
    assert result["verdict"] == "fail"
    assert result["full_validation_authorized"] is False


def _full_validation(f1_050: float, f1_075: float):
    def metric(f1: float):
        return {
            "TP": 100,
            "FP": 10,
            "FN": 10,
            "Precision": f1,
            "Recall": f1,
            "F1": f1,
        }

    return {
        "split": "val",
        "score_thresh": 0.0,
        "lane_nms_distance_thresh_px": 0.0,
        "top_k": 4,
        "row_visibility_thresh": 0.0,
        "quality_score_power": 0.0,
        "score_mode": "four_slot",
        "eval_batch_size": 8,
        "channels_last": True,
        "inference_only": True,
        "amp_dtype": "none",
        "compile_model": False,
        "no_pretrained_init": True,
        "results": {
            "0.5": metric(f1_050),
            "0.75": metric(f1_075),
        },
    }


def test_v8_summary_closes_long_training_when_uniform_gain_does_not_replicate():
    source = _report(0.79, 0.56, 0.0)
    final = _report(0.793, 0.571, 0.2)
    source_full = _full_validation(0.81, 0.60)
    candidate_full = _full_validation(0.809, 0.602)
    result = summarize(
        source,
        [(227000, final)],
        {"passed": True},
        source_full,
        candidate_full,
    )
    assert result["full_validation"]["protocol_equal"] is True
    assert result["full_validation"]["uniform_primary_signal_replicated"] is False
    assert result["full_validation"]["verdict"] == (
        "strict_only_improved_primary_regressed"
    )
    assert result["long_training_authorized"] is False
    assert result["next_action"] == (
        "stop_long_training_and_audit_why_uniform_gain_did_not_generalize"
    )
