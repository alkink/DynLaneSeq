from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model


ROOT = Path(__file__).resolve().parents[2]
STRUCTURED_CONFIG = ROOT / "dynlaneseq_eg/configs/tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml"
UNSTRUCTURED_CONFIG = ROOT / "dynlaneseq_eg/configs/tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml"


def _without_intended_differences(cfg: dict) -> dict:
    cfg = deepcopy(cfg)
    cfg.pop("_config_path", None)
    cfg.pop("output_dir", None)
    model_cfg = cfg["model"]
    model_cfg.pop("decoder_layers", None)
    model_cfg["structured_query"].pop("enabled", None)
    return cfg


def test_tusimple_unstructured_is_a_matched_representation_ablation() -> None:
    structured = load_config(STRUCTURED_CONFIG)
    unstructured = load_config(UNSTRUCTURED_CONFIG)

    assert structured["model"]["structured_query"]["enabled"] is True
    assert structured["model"]["decoder_layers"] == 0
    assert unstructured["model"]["structured_query"]["enabled"] is False
    assert unstructured["model"]["decoder_layers"] == 4
    assert _without_intended_differences(structured) == _without_intended_differences(unstructured)


def test_tusimple_unstructured_builds_the_holistic_decoder() -> None:
    cfg = load_config(UNSTRUCTURED_CONFIG)
    cfg["model"]["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model = build_model(cfg)

    assert model.structured_query_head is None
    assert len(model.encoder.decoder.layers) == 4
    assert cfg["training"]["seed"] == 3407
    assert cfg["training"]["max_iters"] == 15890
