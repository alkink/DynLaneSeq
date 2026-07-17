from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from dynlaneseq_eg.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "dynlaneseq_eg" / "configs"
STEM = "slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml"


def _load(name: str) -> dict:
    return load_config(CONFIG_DIR / f"curvelanes_s0_structured_query_{name}_{STEM}")


def _remove_backbone_identity(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    cfg["model"].pop("backbone_name", None)
    cfg["model"].pop("resnet_depth", None)
    return cfg


@pytest.mark.parametrize(("name", "depth"), [("res18", 18), ("res34", 34), ("res101", 101)])
def test_curvelanes_resnet_scaling_configs_are_matched(name: str, depth: int) -> None:
    control = _load("res34")
    candidate = _load(name)

    assert candidate["model"]["backbone_name"] == "resnet"
    assert candidate["model"]["resnet_depth"] == depth
    assert candidate["training"]["max_iters"] == 312500
    assert candidate["training"]["seed"] == 3407
    assert _remove_backbone_identity(candidate) == _remove_backbone_identity(control)


def test_curvelanes_dla34_config_is_matched_to_res34_control() -> None:
    control = _load("res34")
    candidate = _load("dla34")

    assert candidate["model"]["backbone_name"] == "dla34"
    assert candidate["model"]["require_pretrained_backbone"] is True
    assert candidate["training"]["max_iters"] == 312500
    assert candidate["training"]["seed"] == 3407
    assert _remove_backbone_identity(candidate) == _remove_backbone_identity(control)
