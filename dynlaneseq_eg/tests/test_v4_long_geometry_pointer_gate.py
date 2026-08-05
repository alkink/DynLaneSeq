from __future__ import annotations

import pytest

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.tools.summarize_v4_long_geometry_pointer_gate import summarize


LONG_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_long_geometry.yaml"
)
BASE_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml"
)


def _pointer_report(f1_050: float, f1_075: float) -> dict:
    def row(f1: float) -> dict:
        return {
            "f1": f1,
            "precision": f1 + 0.02,
            "recall": f1 - 0.02,
            "mean_selected_per_image": 3.2,
            "false_positive_breakdown": {
                "duplicate_fp": {"fraction_of_fp": 0.0}
            },
        }

    return {
        "methods": {
            "pointer_greedy": {"0.50": row(f1_050), "0.75": row(f1_075)}
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"recall": 0.96}},
            "0.75": {"all_candidate_oracle": {"recall": 0.84}},
        },
    }


def test_long_geometry_config_changes_only_checkpoint_policy() -> None:
    base = load_config(BASE_CONFIG)
    long = load_config(LONG_CONFIG)
    for key in ("model", "matcher", "loss", "optimizer", "scheduler"):
        assert long[key] == base[key]
    assert long["training"]["max_iters"] == 278000
    assert long["training"]["checkpoint_interval"] == 25000
    assert long["training"]["checkpoint_include_optimizer"] is True
    assert long["training"]["save_last_alias"] is False


def test_long_geometry_summary_requires_joint_pointer_improvement() -> None:
    baseline = _pointer_report(0.82, 0.61)
    geometry = {
        "rows": [
            {
                "iteration": 50000,
                "all_candidates_recall_050": 0.95,
                "all_candidates_recall_075": 0.82,
            },
            {
                "iteration": 125000,
                "all_candidates_recall_050": 0.96,
                "all_candidates_recall_075": 0.84,
            },
        ]
    }
    mature = {
        "gradient_isolation_passed": True,
        "trajectory": [
            {
                "iteration": 135000,
                "f1_050": 0.83,
                "precision_050": 0.85,
                "recall_050": 0.81,
                "f1_075": 0.62,
                "precision_075": 0.64,
                "recall_075": 0.60,
                "mean_selected_per_image": 3.2,
                "duplicate_fp_fraction_050": 0.0,
                "oracle_recall_050": 0.96,
                "oracle_recall_075": 0.84,
            }
        ],
    }
    result = summarize(baseline, geometry, mature)
    assert result["passed"] is True
    assert result["winner"]["iteration"] == 135000
    assert result["geometry"]["delta_oracle_recall_075"] == pytest.approx(0.02)

    mature["trajectory"][0]["f1_075"] = 0.59
    failed = summarize(baseline, geometry, mature)
    assert failed["passed"] is False
    assert failed["winner"] is None
