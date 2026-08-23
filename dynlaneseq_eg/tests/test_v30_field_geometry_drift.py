from pathlib import Path

import torch

from dynlaneseq_eg.tools.audit_v30_field_geometry_drift import (
    _official_target_rows,
)


def test_official_target_rows_preserves_short_gt_lane_and_order(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.lines.txt"
    annotation.write_text(
        "100 0 100 636\n"
        "200 0 212 12\n",
        encoding="utf-8",
    )
    record = {
        "meta": {
            "anno_path": str(annotation),
            "input_h": 640,
            "input_w": 1600,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "crop_x": 0.0,
            "crop_y": 0.0,
        },
        # The cache target deliberately represents the training contract: the
        # short second official lane was filtered by min_valid_rows.
        "target": {
            "x_rows": torch.zeros((1, 160)),
            "valid_mask": torch.ones((1, 160), dtype=torch.bool),
        },
    }

    x_rows, valid = _official_target_rows(record)

    assert x_rows.shape == (2, 160)
    assert valid.shape == (2, 160)
    assert int(valid[0].sum()) == 160
    assert int(valid[1].sum()) == 4
    assert torch.allclose(x_rows[0, valid[0]], torch.full((160,), 100.0))
    assert torch.allclose(
        x_rows[1, :4], torch.tensor([200.0, 204.0, 208.0, 212.0])
    )
