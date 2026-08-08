from __future__ import annotations

import json
from pathlib import Path

from dynlaneseq_eg.tools.summarize_v7_joint_four_slot_10k_gate import (
    summarize,
)


def _report(
    path: Path,
    *,
    f1_050: float,
    f1_075: float,
    oracle_050: float = 0.94,
    oracle_075: float = 0.79,
) -> Path:
    def method(f1: float) -> dict:
        return {
            "f1": f1,
            "precision": f1,
            "recall": f1,
            "tp": 100,
            "fp": 10,
            "fn": 10,
            "mean_selected_per_image": 3.3,
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
                "exact_fraction": 0.70,
                "mean_absolute_error": 0.30,
            },
            "semantic_duplicate_cluster_fraction": 0.005,
            "global_assignment_repair_fraction": 0.005,
            "mean_route_entropy": 1.2,
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": oracle_050}},
            "0.75": {"all_candidate_oracle": {"recall": oracle_075}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _contract(
    *,
    iteration: int,
    passed: bool = True,
    cosine: float = 0.1,
    ratio: float = 0.5,
):
    return {
        "passed": passed,
        "iteration": iteration,
        "checks": {"hard_slot_assignment": True},
        "route_geometry_selection_cosine": cosine,
        "route_geometry_to_selection_norm_ratio": ratio,
    }


def test_safe_learning_authorizes_only_exact_25k_resume(tmp_path: Path):
    reports = [
        (5000, _report(tmp_path / "5k.json", f1_050=0.50, f1_075=0.30)),
        (10000, _report(tmp_path / "10k.json", f1_050=0.58, f1_075=0.40)),
    ]
    result = summarize(
        reports,
        _contract(iteration=0),
        _contract(iteration=10000),
    )
    assert result["verdict"] == "pass"
    assert result["joint_25k_continuation_authorized"] is True
    assert result["full_validation_authorized"] is False
    assert result["long_run_authorized"] is False
    assert result["test_split_closed"] is True


def test_gradient_dominance_blocks_joint_continuation(tmp_path: Path):
    reports = [
        (5000, _report(tmp_path / "5k.json", f1_050=0.56, f1_075=0.38)),
        (10000, _report(tmp_path / "10k.json", f1_050=0.58, f1_075=0.40)),
    ]
    result = summarize(
        reports,
        _contract(iteration=0),
        _contract(iteration=10000, ratio=9.0),
    )
    assert result["verdict"] == "fail"
    assert result["safety_checks"]["route_geometry_norm_bounded"] is False
    assert result["joint_25k_continuation_authorized"] is False


def test_oracle_collapse_blocks_joint_continuation(tmp_path: Path):
    reports = [
        (5000, _report(tmp_path / "5k.json", f1_050=0.56, f1_075=0.38)),
        (
            10000,
            _report(
                tmp_path / "10k.json",
                f1_050=0.58,
                f1_075=0.40,
                oracle_050=0.85,
                oracle_075=0.65,
            ),
        ),
    ]
    result = summarize(
        reports,
        _contract(iteration=0),
        _contract(iteration=10000),
    )
    assert result["verdict"] == "fail"
    assert result["safety_checks"]["oracle_050_floor"] is False
    assert result["safety_checks"]["oracle_075_floor"] is False
