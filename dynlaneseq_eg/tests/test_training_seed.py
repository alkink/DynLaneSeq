from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dynlaneseq_eg.tools.train import model_init_start_iteration, seed_everything


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
