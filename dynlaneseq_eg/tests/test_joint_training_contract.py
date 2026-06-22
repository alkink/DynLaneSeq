from __future__ import annotations

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.losses.loss_s2 import S2Criterion, S2LossConfig


def _stage(value: float) -> dict[str, torch.Tensor]:
    return {
        "exist_logits": torch.zeros((1, 2, 2), requires_grad=True),
        "row_x_logits": torch.zeros((1, 2, 4, 8), requires_grad=True),
        "pred_x_rows": torch.full((1, 2, 4), value, requires_grad=True),
        "range_raw": torch.zeros((1, 2, 2), requires_grad=True),
        "range_norm": torch.tensor([[[0.1, 0.9], [0.2, 0.8]]], requires_grad=True),
        "quality_logits": torch.zeros((1, 2), requires_grad=True),
    }


def test_full_coarse_anchor_supervises_decisions_geometry_quality_and_dense_heads() -> None:
    coarse = _stage(120.0)
    final = _stage(124.0)
    tower_seg = torch.zeros((1, 1, 4, 8), requires_grad=True)
    tower_centerline = torch.zeros((1, 1, 4, 8), requires_grad=True)
    coarse_seg = torch.zeros((1, 1, 4, 8), requires_grad=True)
    coarse_centerline = torch.zeros((1, 1, 4, 8), requires_grad=True)
    outputs = {
        "coarse": coarse,
        "final": final,
        "seg_logits": tower_seg,
        "centerline_logits": tower_centerline,
        "coarse_seg_logits": coarse_seg,
        "coarse_centerline_logits": coarse_centerline,
        "evidence": {},
    }
    targets = [
        {
            "x_rows": torch.tensor([[110.0, 115.0, 120.0, 125.0]]),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
            "range_y": torch.tensor([[28.8, 259.2]]),
            "x_bins": torch.tensor([[1, 1, 1, 1]]),
            "seg_mask": torch.ones((1, 4, 8)),
            "seg_valid": True,
        }
    ]
    matches = [{"pred_indices": torch.tensor([0]), "gt_indices": torch.tensor([0])}]
    criterion = S2Criterion(
        S2LossConfig(
            coarse_anchor_mode="full",
            coarse_dense_anchor=True,
            lambda_coarse=1.0,
            w_exist=2.0,
            w_point=5.0,
            w_range=1.0,
            w_smooth=0.05,
            w_line_iou=2.0,
            w_seg=1.0,
            w_centerline=0.25,
            w_quality=0.5,
            w_token=0.0,
            input_w=800,
            input_h=288,
        )
    )

    losses = criterion(outputs, targets, matches)
    for name in (
        "loss_exist_coarse",
        "loss_point_coarse",
        "loss_range_coarse",
        "loss_smooth_coarse",
        "loss_line_iou_coarse",
        "loss_quality_coarse",
        "loss_seg_coarse",
        "loss_centerline_coarse",
        "loss_coarse_total",
    ):
        assert name in losses
        assert torch.isfinite(losses[name])

    losses["loss_total"].backward()
    assert coarse["exist_logits"].grad is not None
    assert coarse["pred_x_rows"].grad is not None
    assert coarse["range_norm"].grad is not None
    assert coarse["quality_logits"].grad is not None
    assert coarse_seg.grad is not None
    assert coarse_centerline.grad is not None
    assert tower_seg.grad is not None
    assert tower_centerline.grad is not None


def test_controlled_joint_config_keeps_the_gate_isolated() -> None:
    cfg = load_config("dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml")
    model = cfg["model"]
    active = model["active_corridor"]
    loss = cfg["loss"]
    optimizer = cfg["optimizer"]
    scheduler = cfg["scheduler"]

    assert model["freeze_s0_frontend"] is False
    assert model["freeze_backbone_bn"] is True
    assert active["freeze_non_active_modules"] is False
    assert active["detach_center"] is False
    assert active["gate_enabled"] is False
    assert active["train_center_jitter_px"] == 0.0
    assert model["quality_calibrator"]["enabled"] is False
    assert loss["coarse_anchor_mode"] == "full"
    assert loss["coarse_dense_anchor"] is True
    assert loss["lambda_coarse"] == 1.0
    assert optimizer["evidence_lr"] > optimizer["structured_lr"] > optimizer["backbone_lr"]
    assert scheduler["total_iters"] == 278000
    assert cfg["training"]["max_iters"] == 25000
