from __future__ import annotations

import torch

from dynlaneseq_eg.tools.analyze_centerline_path_linking import (
    extract_paths,
    link_best_path,
)


def test_link_best_path_follows_a_smooth_diagonal() -> None:
    score = torch.full((1, 6, 12), -8.0)
    expected = torch.tensor([2, 3, 4, 5, 6, 7])
    score[0, torch.arange(6), expected] = 4.0
    path = link_best_path(
        score,
        max_step_bins=1,
        transition_penalty=0.1,
    )
    assert torch.equal(path[0], expected)


def test_extract_paths_suppresses_the_first_lane() -> None:
    logits = torch.full((1, 1, 5, 14), -8.0)
    logits[0, 0, :, 3] = 6.0
    logits[0, 0, :, 10] = 5.0
    paths = extract_paths(
        logits,
        num_paths=2,
        max_step_bins=1,
        transition_penalty=0.0,
        suppression_radius_bins=1,
        input_w=140,
    )
    assert paths.shape == (1, 2, 5)
    assert torch.allclose(paths[0, 0], torch.full((5,), 35.0))
    assert torch.allclose(paths[0, 1], torch.full((5,), 105.0))
