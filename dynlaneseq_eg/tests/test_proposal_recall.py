from __future__ import annotations

import torch

from dynlaneseq_eg.evaluation.proposal_recall import (
    ProposalRecallStats,
    collect_prediction_stages,
    line_iou_against_gt,
    select_candidates,
    update_stage_recall,
)


def test_line_iou_against_gt_exact_match_is_one():
    pred = torch.full((2, 72), 100.0)
    pred[1] = 200.0
    gt = torch.full((72,), 100.0)
    valid = torch.ones((72,), dtype=torch.bool)
    ious = line_iou_against_gt(pred, gt, valid, line_width=30.0)
    assert torch.allclose(ious[0], torch.tensor(1.0), atol=1e-6)
    assert ious[1].item() == 0.0


def test_line_iou_against_gt_ignores_invalid_rows():
    pred = torch.full((1, 72), 100.0)
    pred[0, 20:] = 400.0
    gt = torch.full((72,), 100.0)
    valid = torch.zeros((72,), dtype=torch.bool)
    valid[:20] = True
    ious = line_iou_against_gt(pred, gt, valid, line_width=30.0)
    assert torch.allclose(ious[0], torch.tensor(1.0), atol=1e-6)


def test_select_candidates_can_rank_by_score_quality():
    outputs = {
        "pred_x_rows": torch.arange(4 * 72, dtype=torch.float32).view(1, 4, 72),
        "exist_logits": torch.tensor([[[0.0, 1.0], [3.0, -3.0], [2.0, -2.0], [1.0, -1.0]]]),
        "quality_logits": torch.tensor([[5.0, -5.0, 3.0, 1.0]]),
    }
    selected = select_candidates(outputs, 0, top_k=2, rank_by="score_quality")
    assert torch.equal(selected[0], outputs["pred_x_rows"][0, 2])
    assert torch.equal(selected[1], outputs["pred_x_rows"][0, 3])


def test_collect_prediction_stages_prefers_named_stages():
    outputs = {
        "pred_x_rows": torch.zeros((1, 2, 72)),
        "coarse": {"pred_x_rows": torch.ones((1, 2, 72))},
        "final": {"pred_x_rows": torch.full((1, 2, 72), 2.0)},
    }
    stages = collect_prediction_stages(outputs)
    assert set(stages) == {"coarse", "final"}


def test_collect_prediction_stages_includes_dynamic_proposals():
    outputs = {
        "dynamic_proposals": {
            "stage": {
                "pred_x_rows": torch.full((1, 3, 72), 5.0),
                "exist_logits": torch.zeros((1, 3, 2)),
            }
        },
        "pred_x_rows": torch.zeros((1, 2, 72)),
    }
    stages = collect_prediction_stages(outputs)
    assert set(stages) == {"dynamic_proposals", "main"}
    assert torch.equal(stages["dynamic_proposals"]["pred_x_rows"], outputs["dynamic_proposals"]["stage"]["pred_x_rows"])


def test_update_stage_recall_counts_per_gt_best_candidate():
    stage = {
        "pred_x_rows": torch.stack(
            [
                torch.full((72,), 100.0),
                torch.full((72,), 220.0),
                torch.full((72,), 500.0),
            ]
        ).unsqueeze(0)
    }
    targets = [
        {
            "x_rows": torch.stack([torch.full((72,), 100.0), torch.full((72,), 220.0)]),
            "valid_mask": torch.ones((2, 72), dtype=torch.bool),
        }
    ]
    stats = ProposalRecallStats(thresholds=(0.5, 0.9))
    update_stage_recall(stats, stage, targets, line_width=30.0)
    summary = stats.summary()
    assert summary["gt"] == 2.0
    assert summary["recall@0.5"] == 1.0
    assert summary["recall@0.9"] == 1.0
