from __future__ import annotations

from dynlaneseq_eg.tools.summarize_v4_4_set_oracle_gate import summarize_gate


def _metric_summary() -> dict:
    return {
        "pointer": {
            "f1_050": 0.81,
            "recall_050": 0.80,
            "f1_075": 0.66,
            "recall_075": 0.65,
            "mean_selected_per_image": 3.4,
        },
        "checks": {
            "gradient_isolation_passed": True,
            "geometry_oracle_050_preserved": True,
            "geometry_oracle_075_preserved": True,
            "duplicate_fp_fraction_below_25pct": True,
            "variable_cardinality_active": True,
        },
    }


def _alignment() -> dict:
    return {
        "analysis": {
            "pointer_target_set_jaccard": 0.50,
            "exact_pointer_target_set_rate": 0.30,
            "pointer_rollout_dynamics": {
                "fraction_of_unordered_failures_starting_at_step1": 0.60,
                "lane_images_first_step": {
                    "any_target_representative_rate": 0.55,
                },
            },
        }
    }


def test_v4_4_gate_passes_only_the_complete_contract() -> None:
    report = summarize_gate(_metric_summary(), _alignment())
    assert report["passed"] is True
    assert report["decision"] == "set_oracle_pointer_passed_run_full_validation"


def test_v4_4_gate_rejects_weak_first_step_acquisition() -> None:
    alignment = _alignment()
    alignment["analysis"]["pointer_rollout_dynamics"][
        "lane_images_first_step"
    ]["any_target_representative_rate"] = 0.21
    report = summarize_gate(_metric_summary(), alignment)
    assert report["passed"] is False
    assert report["checks"]["first_step_any_target_at_least_40pct"] is False
