from __future__ import annotations

import numpy as np
from PIL import Image

from dynlaneseq_eg.data.transforms import LaneTransforms, TransformConfig


def test_random_shadow_darkens_image_without_changing_lanes():
    np.random.seed(0)
    image = Image.new("RGB", (64, 32), (255, 255, 255))
    lanes = [[(10.0, 20.0), (20.0, 24.0)]]
    transforms = LaneTransforms(
        TransformConfig(
            input_w=64,
            input_h=32,
            random_shadow_prob=1.0,
            random_shadow_min_opacity=0.8,
            random_shadow_max_opacity=0.8,
            random_shadow_roi_start_y=0.0,
        )
    )
    tensor, out_lanes, _, _ = transforms(image, lanes, training=True)
    assert out_lanes == lanes
    assert tensor.min().item() < 0.0
