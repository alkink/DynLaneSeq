from __future__ import annotations

from copy import deepcopy

import pytest

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.tools.summarize_fromscratch_matcher_gate import summarize


CONTROL_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_"
    "fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml"
)
CANDIDATE_CONFIG = (
    "dynlaneseq_eg/configs/"
    "culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_"
    "fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml"
)


def _normalized(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    cfg["matcher"]["lambda_obj"] = 2.0
    cfg["training"]["checkpoint_interval"] = 25000
    return cfg


def test_fromscratch_config_is_a_single_optimization_intervention() -> None:
    control = load_config(CONTROL_CONFIG)
    candidate = load_config(CANDIDATE_CONFIG)
    assert control["matcher"]["lambda_obj"] == 2.0
    assert candidate["matcher"]["lambda_obj"] == 0.5
    assert candidate["scheduler"]["total_iters"] == 278000
    assert candidate["scheduler"]["warmup_iters"] == 1000
    assert candidate["training"]["seed"] == 3407
    assert _normalized(control) == _normalized(candidate)


def _paired_row(strategy, threshold, base, candidate, quality=None, score=None, top_k=4):
    return {
        "strategy": strategy,
        "top_k": top_k,
        "iou_threshold": threshold,
        "quality_power": quality,
        "score_threshold": score,
        "base_recall": base,
        "candidate_recall": candidate,
        "delta_recall_points": 100.0 * (candidate - base),
    }


def test_summary_uses_threshold_free_official_set_metrics() -> None:
    rows = []
    for threshold in (0.5, 0.75):
        rows.extend(
            [
                _paired_row("all_raw", threshold, 0.60, 0.62, top_k=0),
                _paired_row("oracle_topk", threshold, 0.58, 0.60),
                _paired_row(
                    "model_topk", threshold, 0.50, 0.52, quality=0.25
                ),
                _paired_row(
                    "model_topk_nms",
                    threshold,
                    0.55,
                    0.57,
                    quality=0.25,
                    score=-1.0,
                ),
            ]
        )
    ranking = {
        "comparability": {"all_checks_pass": True},
        "rows": rows,
    }
    arm = {
        "mean_active_queries_iou030_per_image": 3.0,
        "mean_active_queries_iou050_per_image": 2.0,
        "mean_query_supporters_iou050_per_lane": 0.7,
        "per_query": [
            {
                "assignment_rate_per_image": 0.6,
                "useful_iou050_image_fraction": 0.5,
            },
            {
                "assignment_rate_per_image": 0.0,
                "useful_iou050_image_fraction": 0.0,
            },
        ],
    }
    assignment = {"query_specialization": {"r34": arm, "dla34": arm}}
    result = summarize(ranking, assignment)
    assert result["verdict"] == "positive_continue_training"
    assert result["primary_setting"]["score_threshold"] is None
    assert (
        result["metrics"]["0.50"]["nms_model_top4_q0p25_no_threshold"]
        ["delta_recall_points"]
        == pytest.approx(2.0)
    )
