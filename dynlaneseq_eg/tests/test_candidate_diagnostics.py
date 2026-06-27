from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dynlaneseq_eg.evaluation.candidate_diagnostics import (
    CACHE_VERSION,
    _cache_path,
    cardinality_oracle_assignment,
    candidate_row_masks,
    exact_official_counts,
    evaluator_hungarian_assignment,
    load_or_collect_cache,
    official_proposal_gt_iou_matrix,
    proposal_gt_iou_matrix,
    recall_from_ids,
    stage_lane_points,
    trace_postprocess,
    unique_candidate_labels,
)
from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes
from dynlaneseq_eg.tools.analyze_stage_transitions import summarize_pair


def _stage(xs: list[float], scores: list[float] | None = None) -> dict[str, torch.Tensor]:
    pred_x = torch.stack([torch.full((72,), x) for x in xs])
    lane_scores = scores or [0.9] * len(xs)
    logits = torch.tensor([[[score, 1.0 - score] for score in lane_scores]], dtype=torch.float32)
    return {
        "pred_x_rows": pred_x,
        "exist_logits": logits[0],
        "quality_logits": torch.zeros(len(xs)),
        "range_norm": torch.tensor([[0.0, 1.0]] * len(xs)),
    }


def test_oracle_top2_avoids_duplicate_proposals_for_same_gt():
    iou = torch.tensor(
        [
            [0.95, 0.94, 0.93, 0.92, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.80],
        ]
    )
    oracle = cardinality_oracle_assignment(iou, threshold=0.5, top_k=2)
    assert oracle.hit_count == 2
    assert 4 in oracle.proposal_ids
    assert len(set(oracle.proposal_ids) & {0, 1, 2, 3}) == 1


def test_invalid_range_candidate_is_not_selected_by_oracle():
    stage = _stage([100.0, 100.0])
    stage["range_norm"][0] = torch.tensor([0.0, 0.01])
    target = {
        "x_rows": torch.full((1, 72), 100.0),
        "valid_mask": torch.ones((1, 72), dtype=torch.bool),
    }
    iou, _valid_gt, candidate_valid = proposal_gt_iou_matrix(stage, target, min_valid_rows=5)
    oracle = cardinality_oracle_assignment(iou, threshold=0.5, top_k=1, candidate_valid=candidate_valid)
    assert candidate_valid.tolist() == [False, True]
    assert oracle.proposal_ids == (1,)


def test_oracle_handles_all_invalid_candidates():
    oracle = cardinality_oracle_assignment(
        torch.tensor([[0.9, 0.8]]),
        threshold=0.5,
        top_k=1,
        candidate_valid=torch.zeros(2, dtype=torch.bool),
    )
    assert oracle.hit_count == 0
    assert oracle.proposal_ids == ()


def test_unique_candidate_labels_separate_duplicates_from_primary_tp():
    iou = torch.tensor([[0.9, 0.8, 0.1]])
    labels, assignment = unique_candidate_labels(iou, threshold=0.5, candidate_valid=torch.ones(3, dtype=torch.bool))
    assert assignment.hit_count == 1
    assert labels.count("unique_tp") == 1
    assert labels.count("duplicate") == 1
    assert labels.count("background") == 1


def test_trace_reports_nms_keeper_identity():
    stage = _stage([100.0, 101.0, 250.0], scores=[0.99, 0.90, 0.80])
    trace = trace_postprocess(stage, score_thresh=0.0, top_k=4, nms_distance_thresh_px=20.0)
    assert trace["selected_ids"] == [0, 2]
    assert trace["status"][1] == "nms_removed"
    assert trace["suppressed_by"][1] == 0


def test_trace_selected_lanes_match_production_postprocess():
    stage = _stage([100.0, 101.0, 250.0], scores=[0.99, 0.90, 0.80])
    batched = {key: value.unsqueeze(0) for key, value in stage.items()}
    production = predictions_to_lanes(
        batched,
        score_thresh=0.0,
        nms_distance_thresh_px=20.0,
        nms_min_overlap_points=5,
        top_k=2,
    )[0]
    trace = trace_postprocess(stage, score_thresh=0.0, top_k=2, nms_distance_thresh_px=20.0)
    lanes, _valid = stage_lane_points(stage, input_h=288, input_w=800, min_valid_rows=5, row_visibility_thresh=0.0)
    traced = [lanes[index] for index in trace["selected_ids"]]
    assert traced == production


def test_cross_checkpoint_summary_omits_slot_transitions():
    before = [
        {
            "states": ["selected_tp"],
            "raw_matchable": [True],
            "selected_tp": 1,
            "selected_fp": 2,
            "slot_status": ["selected"],
            "slot_best_iou": torch.tensor([0.8]),
        }
    ]
    after = [
        {
            "states": ["below_threshold"],
            "raw_matchable": [True],
            "selected_tp": 0,
            "selected_fp": 1,
            "slot_status": ["below_threshold"],
            "slot_best_iou": torch.tensor([0.8]),
        }
    ]
    summary = summarize_pair(before, after, same_model_slots=False)
    assert summary["slot_transitions"] is None
    assert summary["summary"]["killed_by_score"] == 1
    assert summary["summary"]["fp_removed_net"] == 1


def test_candidate_masks_require_minimum_predicted_range_rows():
    stage = _stage([100.0])
    stage["range_norm"][0] = torch.tensor([0.0, 0.01])
    _pred_x, masks, valid = candidate_row_masks(stage, input_h=288, input_w=800, min_valid_rows=5)
    assert int(masks.sum()) < 5
    assert valid.tolist() == [False]


def test_reuse_cache_returns_bit_identical_cpu_tensors(tmp_path: Path):
    list_path = tmp_path / "eval.txt"
    list_path.write_text("/driver/a.jpg\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset:\n  root: .\n  lists:\n    val: eval.txt\nmodel:\n  name: DynLaneSeqS0\n",
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"cache-key-only")
    cache_dir = tmp_path / "cache"
    cache_path = _cache_path(cache_dir, config_path, checkpoint_path, list_path, "val", 0)
    cache_path.parent.mkdir(parents=True)
    expected = {
        "cache_version": CACHE_VERSION,
        "metadata": {"list_path": str(list_path), "input_h": 288, "input_w": 800},
        "records": [{"image_id": "a", "tensor": torch.tensor([1.25, 2.5])}],
    }
    torch.save(expected, cache_path)
    loaded = load_or_collect_cache(
        config_path,
        checkpoint_path,
        split="val",
        list_path=list_path,
        device="cpu",
        cache_dir=cache_dir,
        reuse_cache=True,
    )
    assert loaded["cache_version"] == CACHE_VERSION
    assert torch.equal(loaded["records"][0]["tensor"], expected["records"][0]["tensor"])


def test_official_iou_matrix_matches_exact_polyline(tmp_path: Path):
    anno = tmp_path / "lane.lines.txt"
    anno.write_text("100 0 100 100 100 200 100 284\n", encoding="utf-8")
    stage = _stage([100.0])
    record = {
        "image_id": "synthetic",
        "meta": {
            "anno_path": str(anno),
            "orig_h": 288,
            "orig_w": 800,
            "input_h": 288,
            "input_w": 800,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "crop_x": 0.0,
            "crop_y": 0.0,
        },
        "stages": {"main": stage},
    }
    matrix, valid = official_proposal_gt_iou_matrix(record, "main", line_width=30.0)
    assert valid.tolist() == [True]
    assert matrix.shape == (1, 1)
    assert matrix[0, 0].item() > 0.95


def test_exact_official_counts_use_unique_assignment_from_cached_matrix():
    stage = _stage([100.0, 102.0, 300.0])
    stage["official_iou"] = torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.0, 0.7]])
    record = {
        "image_id": "cached",
        "meta": {},
        "stages": {"main": stage},
    }
    counts = exact_official_counts([record], "main", {"cached": [0, 1, 2]}, iou_threshold=0.5)
    assert counts["tp"] == 2
    assert counts["fp"] == 1
    assert counts["fn"] == 0


def test_oracle_and_deployed_hungarian_are_intentionally_distinct():
    iou = torch.tensor([[0.51, 0.49], [0.99, 0.51]])
    oracle = cardinality_oracle_assignment(iou, threshold=0.5, top_k=2)
    deployed = evaluator_hungarian_assignment(iou, proposal_ids=[0, 1], threshold=0.5)
    assert oracle.hit_count == 2
    assert deployed.hit_count == 1


def test_model_recall_count_is_duplicate_safe():
    iou = torch.tensor([[0.90], [0.80]])
    hits, gt_count, best = recall_from_ids(iou, proposal_ids=[0], threshold=0.5)
    assert hits == 1
    assert gt_count == 2
    assert best.tolist() == pytest.approx([0.90, 0.80])
