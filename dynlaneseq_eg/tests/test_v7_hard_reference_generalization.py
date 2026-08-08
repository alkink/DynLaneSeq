from __future__ import annotations

import json
from pathlib import Path

from dynlaneseq_eg.tools.summarize_v7_hard_reference_generalization import (
    summarize,
)


def _report(path: Path, *, f1_050: float, f1_075: float) -> Path:
    def method(f1: float):
        return {
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "tp": 100,
            "fp": 10,
            "fn": 10,
            "mean_selected_per_image": 3.1,
            "selected_curve_diversity": {
                "close_pair_fraction_below_20px": 0.005,
            },
        }

    payload = {
        "methods": {
            "four_slot_refined": {
                "0.50": method(f1_050),
                "0.75": method(f1_075),
            },
            "four_slot_global_unique": {
                "0.50": method(f1_050 - 0.01),
                "0.75": method(f1_075 - 0.01),
            },
        },
        "four_slot_diagnostics": {
            "cardinality": {
                "exact_fraction": 0.75,
                "under_fraction": 0.15,
                "over_fraction": 0.10,
                "mean_absolute_error": 0.25,
            },
            "semantic_duplicate_cluster_fraction": 0.005,
            "global_assignment_repair_fraction": 0.005,
            "mean_route_entropy": 1.2,
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.94}},
            "0.75": {"all_candidate_oracle": {"recall": 0.80}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_strong_generalization_signal_authorizes_only_full_validation(
    tmp_path: Path,
):
    reports = [
        (25500, _report(tmp_path / "a.json", f1_050=0.77, f1_075=0.56)),
        (28000, _report(tmp_path / "b.json", f1_050=0.81, f1_075=0.60)),
    ]
    result = summarize(reports, {"passed": True})
    assert result["verdict"] == "pass"
    assert result["full_validation_authorized"] is True
    assert result["long_run_authorized"] is False
    assert result["test_split_closed"] is True


def test_conditional_signal_requests_only_short_extension(tmp_path: Path):
    reports = [
        (25500, _report(tmp_path / "a.json", f1_050=0.73, f1_075=0.54)),
        (28000, _report(tmp_path / "b.json", f1_050=0.77, f1_075=0.56)),
    ]
    result = summarize(reports, {"passed": True})
    assert result["verdict"] == "conditional"
    assert result["next_action"] == "extend_same_frozen_head_by_2k_without_test"
    assert result["full_validation_authorized"] is False


def test_completed_continuation_stops_frozen_arm_but_not_joint_hypothesis(
    tmp_path: Path,
):
    reports = [
        (28000, _report(tmp_path / "a.json", f1_050=0.79, f1_075=0.57)),
        (30000, _report(tmp_path / "b.json", f1_050=0.79, f1_075=0.57)),
    ]
    result = summarize(
        reports,
        {"passed": True},
        continuation_complete=True,
    )
    assert result["verdict"] == "frozen_head_plateau"
    assert (
        result["next_action"]
        == "stop_frozen_head_and_test_joint_training_hypothesis"
    )
    assert result["full_validation_authorized"] is False
    assert result["long_run_authorized"] is False
    assert result["joint_training_ruled_out"] is False


def test_geometry_or_gradient_failure_blocks_gate(tmp_path: Path):
    reports = [
        (25500, _report(tmp_path / "a.json", f1_050=0.81, f1_075=0.60)),
        (28000, _report(tmp_path / "b.json", f1_050=0.81, f1_075=0.60)),
    ]
    result = summarize(reports, {"passed": False})
    assert result["verdict"] == "fail"
    assert result["safety_checks"]["gradient_contract"] is False
    assert result["full_validation_authorized"] is False
