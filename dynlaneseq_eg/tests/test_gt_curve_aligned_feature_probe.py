from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_gt_curve_aligned_features import (
    CurveAlignedSequenceProbe,
    _build_lane_examples,
    _evaluation_residuals,
    _paired_line_iou,
    _profiles_for_source,
    _selected_row_indices,
)
from dynlaneseq_eg.evaluation.proposal_recall import line_iou_against_gt


def _targets(rows: int = 16) -> list[dict[str, torch.Tensor]]:
    x = torch.stack(
        (
            torch.linspace(300.0, 500.0, rows),
            torch.linspace(900.0, 700.0, rows),
        )
    )
    valid = torch.ones_like(x, dtype=torch.bool)
    return [{"x_rows": x, "valid_mask": valid}]


def test_eval_examples_have_four_controlled_perturbations() -> None:
    row_indices = _selected_row_indices(16, 8, torch.device("cpu"))
    examples = _build_lane_examples(
        _targets(),
        row_indices=row_indices,
        input_w=1600,
        max_offset=48.0,
        training=False,
        max_train_shift=0.0,
    )
    assert examples is not None
    assert examples.count == 8
    assert sorted(examples.pattern_indices.unique().tolist()) == [0, 1, 2, 3]
    expected = _evaluation_residuals(8, device=torch.device("cpu"))
    for pattern in range(4):
        selected = examples.pattern_indices == pattern
        assert torch.allclose(
            examples.target_residual[selected][0],
            expected[pattern],
        )


def test_equal_capacity_probe_shapes_and_gradients() -> None:
    probe = CurveAlignedSequenceProbe(
        common_channels=16,
        hidden_dim=32,
        num_rows=8,
        offsets_px=[-8.0, 0.0, 8.0],
        max_scales=3,
        num_layers=1,
    )
    profiles = torch.randn(2, 8, 3, 3, 16)
    valid = torch.ones(2, 8, dtype=torch.bool)
    one_scale = torch.tensor([[True, False, False], [True, False, False]])
    three_scales = torch.ones(2, 3, dtype=torch.bool)
    output_one = probe(profiles, one_scale, valid)
    output_three = probe(profiles, three_scales, valid)
    assert output_one["logits"].shape == (2, 8, 3)
    assert output_one["residual"].shape == (2, 8)
    assert output_three["scale_weights"].shape == (2, 8, 3, 3)
    output_one["residual"].sum().backward()
    assert any(parameter.grad is not None for parameter in probe.parameters())


def test_source_profiles_are_channel_padded_and_scale_masked() -> None:
    rows = 8
    examples = _build_lane_examples(
        _targets(rows=16),
        row_indices=_selected_row_indices(16, rows, torch.device("cpu")),
        input_w=1600,
        max_offset=8.0,
        training=True,
        max_train_shift=4.0,
    )
    assert examples is not None
    offsets = torch.tensor([-8.0, 0.0, 8.0])
    feature = torch.randn(1, 4, 4, 8)
    profiles, mask = _profiles_for_source(
        [feature],
        examples,
        offsets,
        common_channels=16,
        max_scales=3,
        input_w=1600,
        input_h=640,
    )
    assert profiles.shape == (2, rows, 3, 3, 16)
    assert torch.equal(mask[0], torch.tensor([True, False, False]))
    assert torch.count_nonzero(profiles[..., 1:, :]) == 0


def test_vectorized_paired_iou_matches_reference() -> None:
    predictions = torch.tensor(
        [[100.0, 110.0, 120.0], [300.0, 315.0, 330.0]]
    )
    targets = torch.tensor(
        [[102.0, 108.0, 119.0], [330.0, 315.0, 300.0]]
    )
    valid = torch.tensor([[True, True, False], [True, True, True]])
    paired = _paired_line_iou(
        predictions,
        targets,
        valid,
        line_width=30.0,
    )
    reference = torch.stack(
        [
            line_iou_against_gt(
                predictions[index : index + 1],
                targets[index],
                valid[index],
                line_width=30.0,
            )[0]
            for index in range(2)
        ]
    )
    assert torch.allclose(paired, reference)
