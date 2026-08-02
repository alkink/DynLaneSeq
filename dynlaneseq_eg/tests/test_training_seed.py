from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dynlaneseq_eg.tools.train import (
    align_scheduler_to_iteration,
    apply_optimizer_group_lr_overrides,
    model_init_start_iteration,
    parse_optimizer_group_lr_overrides,
    seed_everything,
)


def _sample_rngs() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_seed_everything_repeats_python_numpy_and_torch_rngs() -> None:
    seed_everything(3407)
    first = _sample_rngs()
    seed_everything(3407)
    second = _sample_rngs()

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_model_only_init_can_preserve_logical_training_iteration() -> None:
    assert model_init_start_iteration("/tmp/iter_0050000.pt", 50000) == 50000
    assert model_init_start_iteration("/tmp/weights.pt", -1) == 0
    with pytest.raises(ValueError, match="requires --init-from"):
        model_init_start_iteration("", 50000)


def test_resume_lr_override_changes_only_named_group() -> None:
    evidence = torch.nn.Parameter(torch.tensor(1.0))
    model = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [evidence], "lr": 2e-6, "name": "evidence_decay"},
            {"params": [model], "lr": 1e-5, "name": "model_decay"},
        ]
    )

    parsed = parse_optimizer_group_lr_overrides(["evidence_decay=1e-5"])
    changes = apply_optimizer_group_lr_overrides(optimizer, parsed)

    assert changes == {
        "evidence_decay": {"before": 2e-6, "after": 1e-5}
    }
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-5


@pytest.mark.parametrize(
    "spec",
    ["evidence_decay", "=1e-5", "evidence_decay=nan", "evidence_decay=-1"],
)
def test_resume_lr_override_rejects_invalid_spec(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_optimizer_group_lr_overrides([spec])


def test_resume_lr_override_rejects_unknown_group() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "lr": 2e-6, "name": "evidence_decay"}]
    )
    with pytest.raises(ValueError, match="missing groups"):
        apply_optimizer_group_lr_overrides(
            optimizer,
            {"evidence_no_decay": 1e-5},
        )


def test_scheduler_alignment_keeps_new_group_bases_at_resume_phase() -> None:
    slow = torch.nn.Parameter(torch.tensor(1.0))
    fast = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [slow], "lr": 5e-5, "name": "slow"},
            {"params": [fast], "lr": 2e-4, "name": "fast"},
        ]
    )

    def schedule(step: int) -> float:
        return 0.01 + 0.99 * 0.5 * (
            1.0 + np.cos(np.pi * float(step) / 278000.0)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    report = align_scheduler_to_iteration(scheduler, optimizer, 25000)
    expected_factor = schedule(25000)

    assert report["last_epoch"] == 25000
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        5e-5 * expected_factor
    )
    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        2e-4 * expected_factor
    )
    scheduler.step()
    assert scheduler.last_epoch == 25001
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        5e-5 * schedule(25001)
    )
