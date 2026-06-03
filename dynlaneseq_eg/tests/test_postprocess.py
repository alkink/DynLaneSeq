from __future__ import annotations

import torch

from dynlaneseq_eg.evaluation.postprocess import predictions_to_lanes


def test_lane_nms_suppresses_overlapping_lower_score_lane():
    pred_x = torch.full((1, 3, 72), 100.0)
    pred_x[:, 1, :] = 108.0
    pred_x[:, 2, :] = 260.0
    outputs = {
        "exist_logits": torch.tensor([[[5.0, -5.0], [4.0, -4.0], [3.0, -3.0]]]),
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]]),
    }
    lanes = predictions_to_lanes(
        outputs,
        score_thresh=0.5,
        nms_distance_thresh_px=20.0,
        nms_min_overlap_points=5,
    )[0]
    assert len(lanes) == 2


def test_top_k_limits_lanes_without_nms():
    pred_x = torch.stack(
        [torch.full((72,), 80.0), torch.full((72,), 160.0), torch.full((72,), 240.0)],
        dim=0,
    ).unsqueeze(0)
    outputs = {
        "exist_logits": torch.tensor([[[5.0, -5.0], [4.0, -4.0], [3.0, -3.0]]]),
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]]),
    }
    lanes = predictions_to_lanes(outputs, score_thresh=0.5, top_k=2)[0]
    assert len(lanes) == 2


def test_quality_score_can_rerank_lanes():
    pred_x = torch.stack(
        [torch.full((72,), 80.0), torch.full((72,), 220.0)],
        dim=0,
    ).unsqueeze(0)
    outputs = {
        "exist_logits": torch.tensor([[[2.0, -2.0], [1.0, -1.0]]]),
        "quality_logits": torch.tensor([[-5.0, 5.0]]),
        "pred_x_rows": pred_x,
        "range_norm": torch.tensor([[[0.1, 0.9], [0.1, 0.9]]]),
    }
    lanes = predictions_to_lanes(
        outputs,
        score_thresh=0.0,
        top_k=1,
        quality_score_power=1.0,
    )[0]
    assert len(lanes) == 1
    mean_x = sum(x for x, _ in lanes[0]) / len(lanes[0])
    assert mean_x > 150.0
