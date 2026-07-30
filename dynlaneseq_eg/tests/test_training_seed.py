from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dynlaneseq_eg.tools.train import (
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
