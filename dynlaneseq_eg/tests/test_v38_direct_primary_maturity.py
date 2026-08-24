from __future__ import annotations

from pathlib import Path

import pytest

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.tools.evaluate_v38_direct_primary_maturity import decide_v38
from dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects import (
    _set_learning_rate,
)
from dynlaneseq_eg.tools.train_v38_direct_primary_maturity import (
    FIXED_EFFECTIVE_BATCH,
    FIXED_ENDPOINT,
    FIXED_SCHEDULE,
    validate_v38_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "dynlaneseq_eg/configs/culane_v38_direct_primary_maturity_50k.yaml"


def test_v38_config_contract_is_exact() -> None:
    cfg = load_config(CONFIG)
    contract = validate_v38_contract(cfg)
    assert contract["passed"], contract
    assert contract["effective_batch"] == FIXED_EFFECTIVE_BATCH == 16
    assert contract["endpoint_iteration"] == FIXED_ENDPOINT == 50_000
    assert contract["scheduler_total_iters"] == FIXED_SCHEDULE == 278_000


def test_v38_lr_does_not_decay_to_minimum_at_50k() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "lr": 2e-4, "initial_lr": 2e-4}]
    )
    ratio = _set_learning_rate(
        optimizer,
        step=FIXED_ENDPOINT,
        total_steps=FIXED_SCHEDULE,
        warmup_steps=1000,
        minimum_ratio=0.01,
    )
    assert 0.90 < ratio < 1.0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * ratio)


@pytest.mark.parametrize(
    ("f50", "f75", "decision"),
    [
        (0.8040, 0.5780, "DIRECT_PRIMARY_50K_WIN"),
        (0.7970, 0.5740, "AUTHORIZE_DIRECT_PRIMARY_MATURE_EXTENSION"),
        (0.7900, 0.5700, "DIRECT_PRIMARY_50K_FAIL"),
    ],
)
def test_v38_predeclared_decision(f50: float, f75: float, decision: str) -> None:
    metrics = {"0.5": {"F1": f50}, "0.75": {"F1": f75}}
    reference = {
        "0.5": {"F1": 0.8006774963430595},
        "0.75": {"F1": 0.5776888136115174},
    }
    assert decide_v38(metrics, reference)["decision"] == decision
