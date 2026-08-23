from __future__ import annotations

from pathlib import Path

import torch

from dynlaneseq_eg.engine import visualizer


class _SavedImage:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"visualizer-smoke")


def test_visualizer_ignores_scalar_diagnostic_tensors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[dict[str, torch.Tensor]] = []

    monkeypatch.setattr(visualizer, "tensor_to_pil", lambda _: object())
    monkeypatch.setattr(visualizer, "draw_lanes", lambda *_args, **_kwargs: _SavedImage())

    def fake_predictions_to_lanes(pred, **_kwargs):
        captured.append(pred)
        return [[]]

    monkeypatch.setattr(
        visualizer,
        "predictions_to_lanes",
        fake_predictions_to_lanes,
    )
    visualizer.save_prediction_visuals(
        images=torch.zeros(2, 3, 8, 8),
        targets=[{}, {}],
        metas=[{}, {}],
        outputs={
            "batched": torch.zeros(2, 4),
            "scalar_diagnostic": torch.tensor(0.1),
        },
        out_dir=tmp_path,
        step=50000,
    )

    assert len(captured) == 2
    assert all("batched" in pred for pred in captured)
    assert all("scalar_diagnostic" not in pred for pred in captured)
