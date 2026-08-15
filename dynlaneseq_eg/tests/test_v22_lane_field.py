from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dynlaneseq_eg.modeling.v22_lane_field import (
    V22LaneFieldStageA,
    build_lane_field_targets,
    lane_field_loss,
    score_candidates_from_lane_field,
)
from dynlaneseq_eg.tools.v22_official_protocol import (
    OFFICIAL_CULANE_LISTS,
    official_culane_list_contract,
)


def test_official_protocol_preserves_all_rows_and_rejects_alternate_list(
    tmp_path: Path,
) -> None:
    root = tmp_path / "CULane"
    official = root / "list" / "val.txt"
    official.parent.mkdir(parents=True)
    _relative, rows = OFFICIAL_CULANE_LISTS["val"]
    official.write_text("/driver/clip/frame.jpg\n" * rows, encoding="utf-8")
    contract = official_culane_list_contract(root, split="val")
    assert contract["observed_nonempty_rows"] == rows
    assert contract["repeated_row_count_preserved"] == rows - 1
    assert contract["rows_removed"] == 0
    assert contract["deduplication_performed"] is False
    alternate = root / "list" / "val_subset.txt"
    alternate.write_text(official.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="official CULane val list"):
        official_culane_list_contract(
            root, split="val", supplied_path=alternate
        )


def test_lane_field_target_signed_distance_points_to_lane_center() -> None:
    target = {
        "x_rows": torch.tensor([[20.0, 20.0]]),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    built = build_lane_field_targets(
        [target],
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_rows=2,
        x_bins=32,
        input_w=64,
        centerline_sigma_px=2.0,
        distance_limit_px=16.0,
    )
    # Bin 15 is centred at x=31. The lane lies 11 pixels to its left.
    torch.testing.assert_close(
        built["distance_norm"][0, 0, :, 15],
        torch.full((2,), -11.0 / 16.0),
    )
    assert bool((built["support"][0, 0, :, 15] == 1.0).all())


def test_stage_a_field_forward_loss_and_encoder_gradients() -> None:
    torch.manual_seed(7)
    model = V22LaneFieldStageA(
        input_h=64,
        input_w=128,
        num_rows=16,
        x_bins=64,
        fpn_channels=32,
        hidden_dim=32,
        distance_limit_px=24.0,
        freeze_batch_norm_stats=True,
    ).train()
    images = torch.randn(1, 3, 64, 128)
    x = torch.linspace(38.0, 54.0, 16).view(1, 16)
    targets = [{"x_rows": x, "valid_mask": torch.ones_like(x).bool()}]
    output = model(images)
    assert output["centerline_logits"].shape == (1, 1, 16, 64)
    loss, diagnostics = lane_field_loss(
        output,
        targets,
        input_w=128,
        centerline_sigma_px=2.0,
        distance_limit_px=24.0,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    loss.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.backbone.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.fpn.parameters()
    )


def test_signed_field_correction_ranks_true_candidate_over_source_geometry() -> None:
    batch, slots, choices, rows, bins = 1, 1, 2, 8, 50
    distance_limit = 40.0
    # tanh(raw) * 40 = -20 pixels everywhere.
    distance_raw = torch.full(
        (batch, 1, rows, bins),
        torch.atanh(torch.tensor(-0.5)),
    )
    outputs = {
        "distance_raw": distance_raw,
        "support_logits": torch.full((batch, 1, rows, bins), 20.0),
    }
    source_x = torch.full((batch, slots, rows), 80.0)
    candidate_x = torch.stack(
        (
            torch.full((batch, slots, rows), 80.0),
            torch.full((batch, slots, rows), 60.0),
        ),
        dim=2,
    )
    source_range = torch.tensor([[[0.0, 1.0]]])
    candidate_range = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    scored = score_candidates_from_lane_field(
        outputs,
        source_x=source_x,
        source_range=source_range,
        candidate_x=candidate_x,
        candidate_range=candidate_range,
        candidate_valid=torch.ones(batch, slots, choices, dtype=torch.bool),
        input_w=100,
        distance_limit_px=distance_limit,
    )
    assert int(scored["field_score"].argmax(dim=2)) == 1
    assert int(scored["geometry_score"].argmax(dim=2)) == 0
