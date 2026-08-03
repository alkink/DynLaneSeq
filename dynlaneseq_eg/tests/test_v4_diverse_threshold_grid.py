from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.tools.summarize_v4_diverse_threshold_grid import summarize
from dynlaneseq_eg.tools.sweep_v4_diverse_thresholds import (
    SelectionSpec,
    _select_ids,
)


def test_thresholded_mmr_prioritizes_diversity_and_removes_low_score() -> None:
    scores = torch.tensor([0.90, 0.85, 0.80, 0.05])
    distance = torch.tensor(
        [
            [0.0, 4.0, 200.0, 300.0],
            [4.0, 0.0, 196.0, 296.0],
            [200.0, 196.0, 0.0, 100.0],
            [300.0, 296.0, 100.0, 0.0],
        ]
    )
    spec = SelectionSpec(
        "mmr",
        0.10,
        sigma_px=10.0,
        penalty=0.50,
    )
    selected = _select_ids(
        spec,
        scores,
        distance,
        torch.ones(4, dtype=torch.bool),
        top_k=2,
    )
    assert selected == [0, 2]
    assert 3 not in selected


def _metric(f1: float, *, tp: int = 8, fp: int = 2) -> dict:
    recall = 0.8
    precision = float(tp) / float(tp + fp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_breakdown": {
            "duplicate_fp": {"count": 1, "fraction_of_fp": 0.5},
            "near_miss_fp": {"count": 0, "fraction_of_fp": 0.0},
            "background_fp": {"count": 0, "fraction_of_fp": 0.0},
            "empty_scene_fp": {"count": 1, "fraction_of_fp": 0.5},
        },
    }


def _row(
    key: str,
    family: str,
    f1_050: float,
    f1_075: float,
    *,
    score_threshold: float = 0.0,
) -> dict:
    return {
        "key": key,
        "family": family,
        "score_threshold": score_threshold,
        "distance_px": 20.0 if family == "hard_diversity" else None,
        "sigma_px": 20.0 if family == "mmr" else None,
        "penalty": 0.5 if family == "mmr" else None,
        "metrics": {
            "0.50": _metric(f1_050),
            "0.75": _metric(f1_075),
        },
        "objectives": {
            "f1_050": f1_050,
            "mean_f1": 0.5 * (f1_050 + f1_075),
        },
    }


def _report(arm: str, best_f1: float) -> dict:
    raw = _row("score_topk_thr0", "score_topk", 0.50, 0.40)
    thresholded = _row(
        "score_topk_thr0p1",
        "score_topk",
        0.60,
        0.45,
        score_threshold=0.1,
    )
    hard = _row("hard_d20_thr0p1", "hard_diversity", 0.70, 0.50)
    mmr = _row("mmr_s20_p0p5_thr0p1", "mmr", best_f1, 0.60)
    return {
        "metadata": {
            "score_mode": "exist" if arm == "source_v4" else "selection",
            "sampled_dataset_indices": [1, 2, 3],
            "list_sha256": "same-list",
        },
        "all_candidate_oracle": {
            "0.50": {"gt": 10, "hits": 9, "recall": 0.9},
            "0.75": {"gt": 10, "hits": 8, "recall": 0.8},
        },
        "best_by_family": {
            "score_topk": {"f1_050": thresholded, "mean_f1": thresholded},
            "hard_diversity": {"f1_050": hard, "mean_f1": hard},
            "mmr": {"f1_050": mmr, "mean_f1": mmr},
        },
        "best_overall_diversity": {"f1_050": mmr, "mean_f1": mmr},
        "rows": [raw, thresholded, hard, mmr],
    }


def test_summary_selects_the_best_paired_teacher() -> None:
    payload = summarize(
        {
            "source_v4": _report("source_v4", 0.78),
            "c_set_shared": _report("c_set_shared", 0.79),
            "d_set_unique": _report("d_set_unique", 0.82),
        }
    )
    assert payload["paired_contract"]["passed"] is True
    assert payload["best_primary_arm"] == "d_set_unique"
    assert payload["primary_subset_gate_passed"] is True
    effects = payload["arms"]["d_set_unique"]["effects"]
    assert effects["threshold_only_f1_050_gain"] == pytest.approx(0.10)
    assert effects["diversity_beyond_best_threshold_f1_050_gain"] == pytest.approx(
        0.22
    )
