from __future__ import annotations

import torch

from dynlaneseq_eg.tools.probe_seed_conditioned_p2_identity import (
    SeedConditionedP2IdentityProbe,
    _lane_seed_coordinates,
    _seed_curve_loss,
)


def _target() -> dict[str, torch.Tensor]:
    return {
        "x_rows": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "valid_mask": torch.ones((1, 4), dtype=torch.bool),
    }


def test_seed_identity_probe_shapes_and_gradients() -> None:
    probe = SeedConditionedP2IdentityProbe(
        in_dim=8,
        hidden_dim=8,
        num_rows=4,
        x_bins=10,
        input_w=100,
    )
    p2 = torch.randn((1, 8, 4, 10))
    seed_features, keys = probe.encode(p2)
    seed_yx, lane_indices = _lane_seed_coordinates(
        _target(),
        num_rows=4,
        x_bins=10,
        input_w=100.0,
        device=torch.device("cpu"),
    )
    assert lane_indices == [0]
    assert seed_yx.tolist() == [[3, 4]]
    queries = probe.seed_queries(seed_features, seed_yx)
    logits = probe.score(queries, keys)
    assert logits.shape == (1, 4, 10)
    assert probe.decode(logits).shape == (1, 4)
    loss, lanes, rows = _seed_curve_loss(
        probe,
        seed_features,
        keys,
        [_target()],
        input_w=100.0,
        point_loss_weight=5.0,
        point_beta=0.01,
    )
    assert torch.isfinite(loss)
    assert lanes == 1
    assert rows == 4
    loss.backward()
    assert probe.tower[0].weight.grad is not None
