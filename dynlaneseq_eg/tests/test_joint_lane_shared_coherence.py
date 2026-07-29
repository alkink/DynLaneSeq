from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.tools.finetune_joint_lane_shared_coherence import (
    build_joint_optimizer,
    select_joint_trainable_modules,
)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.fpn = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)
        self.seg_aux_head = nn.Linear(4, 1)
        self.centerline_aux_head = nn.Linear(4, 1)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.structured_query_head = nn.Linear(4, 4)
        self.unused = nn.Linear(4, 4)


def test_joint_diagnostic_freezes_backbone_and_unrelated_modules() -> None:
    model = _Model()
    counts = select_joint_trainable_modules(model)
    assert counts["fpn"] > 0
    assert not any(p.requires_grad for p in model.encoder.backbone.parameters())
    assert not any(p.requires_grad for p in model.unused.parameters())
    assert all(p.requires_grad for p in model.encoder.fpn.parameters())
    assert all(p.requires_grad for p in model.structured_query_head.parameters())


def test_joint_optimizer_has_separate_causal_learning_rates() -> None:
    model = _Model()
    select_joint_trainable_modules(model)
    probe = nn.Linear(4, 4)
    optimizer = build_joint_optimizer(
        model,
        probe,
        model_lr=2e-5,
        fpn_lr=1e-5,
        probe_lr=2e-4,
        weight_decay=1e-4,
    )
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {
        "fpn": 1e-5,
        "structured_model": 2e-5,
        "lane_shared_probe": 2e-4,
    }
