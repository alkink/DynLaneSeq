from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_attention_coordinate_adapter import (
    CoordinateResidualAdapter,
    _attention_coordinate_features,
    _matched_examples,
)


def test_attention_coordinate_features_carry_expected_and_peak_x() -> None:
    # [B=1, R=2, H=1, N=1, X=4]
    attention = torch.tensor(
        [
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0, 1.0]]],
            ]
        ],
        dtype=torch.float32,
    )
    anchor_x = torch.tensor([[[12.5, 87.5]]])
    features = _attention_coordinate_features(
        [attention],
        selected_layers=(0,),
        anchor_x=anchor_x,
        input_w=100.0,
    )
    assert features.shape == (1, 1, 2, 4)
    # Expected and peak positions exactly match the anchors at both rows.
    assert torch.allclose(features[..., 0], torch.zeros(1, 1, 2))
    assert torch.allclose(features[..., 1], torch.zeros(1, 1, 2))
    assert torch.allclose(
        features[..., 2],
        torch.zeros(1, 1, 2),
        atol=1e-6,
    )
    assert torch.allclose(features[..., 3], torch.ones(1, 1, 2))


def test_coordinate_adapters_are_initially_noop_with_equal_capacity() -> None:
    modes = {
        "anchor": (False, False),
        "state": (True, False),
        "attention": (False, True),
        "combined": (True, True),
    }
    adapters = {
        name: CoordinateResidualAdapter(
            state_dim=8,
            coordinate_dim=4,
            hidden_dim=16,
            num_rows=3,
            max_update_px=48.0,
            use_state=mode[0],
            use_attention=mode[1],
        )
        for name, mode in modes.items()
    }
    reference = adapters["anchor"].state_dict()
    for name in ("state", "attention", "combined"):
        adapters[name].load_state_dict(reference, strict=True)
    counts = {
        name: sum(parameter.numel() for parameter in adapter.parameters())
        for name, adapter in adapters.items()
    }
    assert len(set(counts.values())) == 1

    row_state = torch.randn(2, 3, 8)
    coordinates = torch.randn(2, 3, 4)
    anchor_x = torch.rand(2, 3) * 99.0
    for adapter in adapters.values():
        delta = adapter(
            row_state,
            coordinates,
            anchor_x,
            input_w=100.0,
        )
        assert torch.equal(delta, torch.zeros_like(delta))


def test_matched_examples_can_isolate_group_zero() -> None:
    row_state = torch.randn(1, 4, 5, 8)
    coordinates = torch.randn(1, 4, 5, 4)
    anchor_x = torch.randn(1, 4, 5)
    targets = [
        {
            "x_rows": torch.tensor(
                [[10.0, 20.0, 30.0, 40.0, 50.0]]
            ),
            "valid_mask": torch.ones(1, 5, dtype=torch.bool),
        }
    ]
    matches = [
        {
            "pred_indices": torch.tensor([1, 3]),
            "gt_indices": torch.tensor([0, 0]),
        }
    ]
    all_examples = _matched_examples(
        row_state=row_state,
        coordinate_features=coordinates,
        anchor_x=anchor_x,
        targets=targets,
        matches=matches,
        group_size=2,
        group_mode="all",
    )
    group0_examples = _matched_examples(
        row_state=row_state,
        coordinate_features=coordinates,
        anchor_x=anchor_x,
        targets=targets,
        matches=matches,
        group_size=2,
        group_mode="group0",
    )
    assert all_examples is not None and all_examples.count == 2
    assert group0_examples is not None and group0_examples.count == 1
