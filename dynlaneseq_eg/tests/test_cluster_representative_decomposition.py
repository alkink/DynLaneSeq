from __future__ import annotations

import torch

from dynlaneseq_eg.tools.decompose_cluster_representative_selection import (
    _dominant_diagnosis,
    _finish,
    decompose_one_image,
    nms_clusters_from_trace,
    verify_reference_report,
)


def test_nms_trace_is_converted_to_complete_partition() -> None:
    trace = {
        "nms_kept_ids": [0, 2],
        "suppressed_by": {1: 0, 3: 2},
        "eligible_ids": [0, 1, 2, 3],
    }
    assert nms_clusters_from_trace(trace) == {0: [0, 1], 2: [2, 3]}


def test_decomposition_detects_within_cluster_representative_loss() -> None:
    iou = torch.tensor(
        [
            [0.20, 0.90, 0.00, 0.00],
            [0.00, 0.00, 0.80, 0.10],
        ]
    )
    row = decompose_one_image(
        iou,
        candidate_valid=torch.ones(4, dtype=torch.bool),
        clusters={0: [0, 1], 2: [2, 3]},
        selected_keepers=[0, 2],
        raw_topk_ids=[0, 2],
        threshold=0.5,
        top_k=2,
    )
    assert row["actual_nms"] == 1
    assert row["selected_cluster_oracle_representative"] == 2
    assert row["oracle_cluster_source_representative"] == 1
    assert row["oracle_cluster_and_representative"] == 2
    assert row["candidate_oracle"] == 2


def test_decomposition_detects_cluster_ranking_loss() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.80, 0.00],
            [0.00, 0.00, 0.90],
        ]
    )
    row = decompose_one_image(
        iou,
        candidate_valid=torch.ones(3, dtype=torch.bool),
        clusters={0: [0], 1: [1], 2: [2]},
        selected_keepers=[0, 1],
        raw_topk_ids=[0, 1],
        threshold=0.5,
        top_k=2,
    )
    assert row["actual_nms"] == 1
    assert row["selected_cluster_oracle_representative"] == 1
    assert row["oracle_cluster_source_representative"] == 2
    assert row["oracle_cluster_and_representative"] == 2
    assert row["candidate_oracle"] == 2


def test_decomposition_detects_nms_partition_loss() -> None:
    iou = torch.tensor(
        [
            [0.90, 0.00],
            [0.00, 0.90],
        ]
    )
    row = decompose_one_image(
        iou,
        candidate_valid=torch.ones(2, dtype=torch.bool),
        clusters={0: [0, 1]},
        selected_keepers=[0],
        raw_topk_ids=[0, 1],
        threshold=0.5,
        top_k=2,
    )
    assert row["actual_nms"] == 1
    assert row["oracle_cluster_and_representative"] == 1
    assert row["candidate_oracle"] == 2


def test_reference_report_must_match_recomputed_raw_and_nms() -> None:
    results = {
        "0.50": {
            "modes": {
                "raw_topk": {"recall": 0.50},
                "actual_nms": {"recall": 0.70},
            }
        },
        "0.70": {
            "modes": {
                "raw_topk": {"recall": 0.40},
                "actual_nms": {"recall": 0.60},
            }
        },
    }
    reference = {
        "source": {
            "raw_top4": {"recall_050": 0.50, "recall_070": 0.40},
            "nms_top4": {"recall_050": 0.70, "recall_070": 0.60},
        }
    }
    assert verify_reference_report(results, reference)["matched"] is True


def test_joint_interaction_is_reported_separately() -> None:
    result = _finish(
        {
            "gt_lanes": 2,
            "raw_topk": 1,
            "actual_nms": 1,
            "selected_cluster_oracle_representative": 1,
            "oracle_cluster_source_representative": 1,
            "oracle_cluster_and_representative": 2,
            "candidate_oracle": 2,
        }
    )
    headroom = result["headroom_points"]
    assert headroom["within_selected_cluster_representative"] == 0.0
    assert headroom["cluster_ranking_with_source_representatives"] == 0.0
    assert headroom["joint_interaction_beyond_best_single"] == 50.0


def test_dominant_diagnosis_returns_cluster_ranking() -> None:
    row = {
        "headroom_points": {
            "within_selected_cluster_representative": 5.0,
            "cluster_ranking_with_source_representatives": 15.0,
            "joint_interaction_beyond_best_single": 3.0,
            "nms_partition_loss": 0.0,
        }
    }
    decision = _dominant_diagnosis({"0.50": row, "0.70": row})
    assert decision["diagnosis"] == "cluster_ranking_is_primary"
    assert decision["confidence"] == "strong"
