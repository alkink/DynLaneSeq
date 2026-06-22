from __future__ import annotations

import torch
from torch import nn

from dynlaneseq_eg.engine.train_one_epoch import clip_optimizer_gradients, train_one_epoch


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return {"value": images * self.weight}


class _ToyCriterion:
    def __call__(self, outputs, targets, matches):
        del targets, matches
        return {"loss_total": outputs["value"].square().mean()}


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_gradient_accumulation_counts_optimizer_steps_not_micro_batches():
    model = _ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _CountingScheduler()
    batch = (torch.ones(1, 1), [], [])
    dataloader = [batch, batch]
    cfg = {
        "model": {"name": "Toy"},
        "training": {
            "amp": False,
            "gradient_accumulation_steps": 2,
            "clip_grad_norm": 10.0,
        },
    }
    end_iter = train_one_epoch(
        model,
        dataloader,
        lambda outputs, targets: [],
        _ToyCriterion(),
        optimizer,
        torch.device("cpu"),
        cfg,
        max_iters=2,
        scheduler=scheduler,
    )
    assert end_iter == 2
    assert model.forward_calls == 4
    assert scheduler.steps == 2


def test_optimizer_group_clipping_does_not_shrink_small_groups_with_large_backbone_gradient():
    large = nn.Parameter(torch.tensor(0.0))
    small = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [large], "lr": 1e-6},
            {"params": [small], "lr": 1e-4},
        ]
    )
    model = nn.ParameterList([large, small])
    large.grad = torch.tensor(100.0)
    small.grad = torch.tensor(1.0)

    norm = clip_optimizer_gradients(model, optimizer, max_norm=1.0, mode="optimizer_groups")

    assert norm > 100.0
    assert torch.allclose(large.grad, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(small.grad, torch.tensor(1.0), atol=1e-5)
