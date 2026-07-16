from __future__ import annotations

import pytest
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.modeling.backbone_resnet import ResNetBackbone


@pytest.mark.parametrize(
    ("depth", "channels"),
    [
        (18, (64, 128, 256, 512)),
        (34, (64, 128, 256, 512)),
        (101, (256, 512, 1024, 2048)),
    ],
)
def test_resnet_backbone_feature_contract(depth: int, channels: tuple[int, ...]) -> None:
    model = ResNetBackbone(depth=depth, pretrained=False).eval()
    with torch.inference_mode():
        features = model(torch.randn(1, 3, 64, 64))
    assert tuple(features) == ("c2", "c3", "c4", "c5")
    assert tuple(features[name].shape[1] for name in features) == channels
    assert tuple(features[name].shape[-1] for name in features) == (16, 8, 4, 2)


def test_tusimple_backbone_configs_only_change_depth_and_output() -> None:
    configs = []
    for depth in (18, 34, 101):
        cfg = load_config(
            "dynlaneseq_eg/configs/"
            f"tusimple_s0_structured_query_res{depth}_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.yaml"
        )
        assert cfg["model"]["resnet_depth"] == depth
        cfg.pop("_config_path", None)
        cfg.pop("output_dir", None)
        cfg["model"].pop("resnet_depth")
        configs.append(cfg)
    assert configs[0] == configs[1] == configs[2]
