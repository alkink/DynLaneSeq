from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.factory import build_optimizer


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.fpn = nn.Linear(2, 2)
        self.proj = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.structured_query_head = nn.Linear(2, 2)
        self.active_corridor = nn.Linear(2, 2)
        self.row_decoder = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)


def test_optimizer_uses_separate_structured_learning_rate() -> None:
    model = _Model()
    optimizer = build_optimizer(
        {
            "optimizer": {
                "base_lr": 1e-5,
                "backbone_lr": 2e-6,
                "row_decoder_lr": 3e-5,
                "evidence_lr": 4e-5,
                "structured_lr": 5e-6,
            }
        },
        model,
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["structured_decay"]["lr"] == 5e-6
    assert groups["structured_no_decay"]["lr"] == 5e-6
    assert groups["evidence_decay"]["lr"] == 4e-5
    assert groups["row_decoder_decay"]["lr"] == 3e-5
    assert groups["backbone_decay"]["lr"] == 2e-6

    grouped = [param for group in optimizer.param_groups for param in group["params"]]
    assert len(grouped) == len({id(param) for param in grouped})
    assert {id(param) for param in grouped} == {id(param) for param in model.parameters()}
