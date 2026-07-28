from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_group_duplication_pair import (
    _analyze_arm,
    _block_ids,
    _pairwise_close_counts,
)


def _stage() -> dict[str, torch.Tensor]:
    # Blocks are [0, 1] and [2, 3]. Candidates 0/2 are cross-block
    # duplicates; candidates 0/1 are far apart within the first block.
    return {
        "pred_x_rows": torch.tensor(
            [
                [10.0, 10.0, 10.0, 10.0],
                [60.0, 60.0, 60.0, 60.0],
                [11.0, 11.0, 11.0, 11.0],
                [90.0, 90.0, 90.0, 90.0],
            ]
        ),
        "range_norm": torch.tensor([[0.0, 1.0]]).repeat(4, 1),
    }


def test_block_ids_are_contiguous() -> None:
    assert _block_ids(8, 4) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_pairwise_close_counts_separate_cross_block_duplicates() -> None:
    result = _pairwise_close_counts(
        _stage(),
        input_h=4,
        input_w=100,
        block_ids=_block_ids(4, 2),
        min_valid_rows=1,
        row_visibility_thresh=0.0,
        distance_threshold=20.0,
        min_overlap_points=1,
    )
    assert result["same_block_close_pairs"] == 0
    assert result["cross_block_close_pairs"] == 1
    assert result["close_pair_matrix"] == [[0, 1], [1, 0]]


def test_arm_analysis_exposes_duplicate_rank_then_nms_recovery(tmp_path) -> None:
    stage = _stage()
    stage["exist_logits"] = torch.tensor(
        [
            [8.0, -8.0],
            [1.0, -1.0],
            [7.0, -7.0],
            [-2.0, 2.0],
        ]
    )
    stage["quality_logits"] = torch.zeros(4)
    cache_path = tmp_path / "cache.pt"
    cache = {
        "metadata": {
            "input_h": 4,
            "input_w": 100,
        },
        "records": [
            {
                "image_id": "synthetic",
                "target": {
                    "x_rows": torch.tensor(
                        [
                            [10.0, 10.0, 10.0, 10.0],
                            [60.0, 60.0, 60.0, 60.0],
                        ]
                    ),
                    "valid_mask": torch.ones((2, 4), dtype=torch.bool),
                },
                "stages": {"main": stage},
            }
        ],
    }
    report = {
        "metadata": {
            "config": str(tmp_path / "missing.yaml"),
            "checkpoint": "synthetic.pt",
        }
    }
    result = _analyze_arm(
        report,
        cache,
        report_path=str(tmp_path / "report.json"),
        cache_path=cache_path,
        stage_name="main",
        num_query_blocks=2,
        top_k=2,
        quality_power=0.0,
        iou_thresholds=[0.5],
        line_width=30.0,
        min_valid_rows=1,
        row_visibility_thresh=0.0,
        nms_distance_thresh_px=20.0,
        nms_min_overlap_points=1,
    )
    metrics = result["thresholds"]["0.50"]
    assert metrics["all_candidates_capacity"]["hits"] == 2
    assert metrics["raw_model_topk"]["hits"] == 1
    assert metrics["nms_model_topk"]["hits"] == 2
    assert result["selection_and_nms"]["cross_block_nms_suppressions"] >= 1
