from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_oracle_attention_counterfactual import (
    _counterfactual_features,
)


def test_counterfactual_features_remove_correct_spatial_content() -> None:
    features = torch.arange(16.0).view(2, 1, 2, 4)
    controls = _counterfactual_features(features)

    assert torch.equal(
        controls["wrong_image"][0],
        features[1],
    )
    assert int(torch.count_nonzero(controls["zero_image"])) == 0
    assert torch.equal(
        controls["horizontal_mean"][..., 0],
        controls["horizontal_mean"][..., -1],
    )
    assert torch.equal(controls["correct_image"], features)
