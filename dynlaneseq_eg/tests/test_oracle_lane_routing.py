from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_oracle_lane_routing import _build_corridor_mask


def test_endpoint_corridor_masks_only_assigned_endpoint_row() -> None:
    target = {
        "x_rows": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "valid_mask": torch.tensor([[True, True, True, True]]),
    }
    match = {
        "pred_indices": torch.tensor([1]),
        "gt_indices": torch.tensor([0]),
    }
    mask = _build_corridor_mask(
        targets=[target],
        matches=[match],
        batch=1,
        instances=2,
        rows=4,
        x_bins=10,
        group_size=2,
        input_w=100.0,
        radius_px=10.0,
        endpoint_only=True,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 4, 2, 10)
    assert not bool(mask[0, :3].any())
    assert not bool(mask[0, 3, 0].any())
    assert bool(mask[0, 3, 1].any())
    assert bool((~mask[0, 3, 1]).any())
