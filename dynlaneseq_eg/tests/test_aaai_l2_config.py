from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "dynlaneseq_eg" / "configs"
FULL_CONFIG = CONFIG_DIR / (
    "culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_"
    "fpn256_l4_dfl_full_seed3407_278k.yaml"
)
L2_CONFIG = CONFIG_DIR / (
    "culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_"
    "fpn256_l2_dfl_seed3407_278k.yaml"
)


def test_l2_config_changes_only_depth_and_output_directory() -> None:
    full = load_config(FULL_CONFIG)
    l2 = load_config(L2_CONFIG)

    assert l2["seed"] == 3407
    assert l2["training"]["max_iters"] == 278000
    assert l2["training"]["checkpoint_interval"] == 12500
    assert l2["training"]["vis_interval"] == 25000
    assert full["model"]["structured_query"]["num_layers"] == 4
    assert l2["model"]["structured_query"]["num_layers"] == 2
    assert l2["model"]["structured_query"]["use_intra_attention"] is True

    comparable_full = deepcopy(full)
    comparable_l2 = deepcopy(l2)
    comparable_full["output_dir"] = "<ignored>"
    comparable_l2["output_dir"] = "<ignored>"
    comparable_full["_config_path"] = "<ignored>"
    comparable_l2["_config_path"] = "<ignored>"
    comparable_full["model"]["structured_query"]["num_layers"] = 2
    assert comparable_l2 == comparable_full


def test_l2_config_builds_exactly_two_structured_layers() -> None:
    cfg = load_config(L2_CONFIG)
    cfg["model"]["pretrained_backbone"] = False
    model = build_model(cfg)
    assert model.structured_query_head is not None
    assert len(model.structured_query_head.layers) == 2
