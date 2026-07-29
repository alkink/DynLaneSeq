import torch

from dynlaneseq_eg.tools.analyze_learned_reference_refinement import (
    _expand_selected_residual,
    _predicted_row_mask,
)


def test_predicted_row_mask_respects_sorted_range() -> None:
    ranges = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
    mask = _predicted_row_mask(ranges, num_rows=20, input_h=100)
    assert mask.shape == (1, 1, 20)
    assert bool(mask[0, 0, 0]) is False
    assert bool(mask[0, 0, 10]) is True
    assert bool(mask[0, 0, -1]) is False


def test_predicted_row_mask_falls_back_for_degenerate_range() -> None:
    ranges = torch.tensor([[[0.49, 0.51]]], dtype=torch.float32)
    mask = _predicted_row_mask(ranges, num_rows=8, input_h=80)
    assert bool(mask.all())


def test_expand_selected_residual_preserves_endpoints() -> None:
    residual = torch.tensor([[[1.0, 3.0, 5.0]]])
    expanded = _expand_selected_residual(residual, output_rows=7)
    assert expanded.shape == (1, 1, 7)
    assert float(expanded[0, 0, 0]) == 1.0
    assert float(expanded[0, 0, -1]) == 5.0
