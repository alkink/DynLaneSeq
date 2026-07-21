from __future__ import annotations

import torch

from dynlaneseq_eg.engine.train_one_epoch import format_loss_diagnostics


def test_format_loss_diagnostics_marks_nonfinite_terms_and_samples() -> None:
    text = format_loss_diagnostics(
        {
            "loss_total": torch.tensor(float("nan")),
            "loss_seg": torch.tensor(float("inf")),
            "loss_point": torch.tensor(0.125),
            "not_scalar": torch.zeros(2),
        },
        [{"image_path": "/driver/frame.jpg"}],
    )

    assert "loss_total=nan*" in text
    assert "loss_seg=inf*" in text
    assert "loss_point=0.125" in text
    assert "not_scalar" not in text
    assert "/driver/frame.jpg" in text
