from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from dynlaneseq_eg.engine.checkpoint import load_checkpoint, save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.structured_query_head = nn.Module()
        self.structured_query_head.set_selection_head = nn.Linear(4, 1)


def test_compact_checkpoint_overlays_selector_on_exact_base(tmp_path: Path) -> None:
    torch.manual_seed(601)
    source = _TinyModel()
    base_backbone = source.backbone.weight.detach().clone()
    base_path = tmp_path / "base.pt"
    save_checkpoint(base_path, source, iteration=50000)

    with torch.no_grad():
        source.backbone.weight.add_(100.0)
        source.structured_query_head.set_selection_head.weight.add_(3.0)
    selector_weight = (
        source.structured_query_head.set_selection_head.weight.detach().clone()
    )
    delta_path = tmp_path / "selector.pt"
    save_checkpoint(
        delta_path,
        source,
        iteration=53000,
        model_state_prefixes=("structured_query_head.set_selection_head",),
        base_checkpoint=base_path,
    )

    payload = torch.load(delta_path, map_location="cpu", weights_only=False)
    assert payload["model_state_mode"] == "delta"
    assert payload["iteration"] == 53000
    assert set(payload["model"]) == {
        "structured_query_head.set_selection_head.weight",
        "structured_query_head.set_selection_head.bias",
    }
    assert delta_path.stat().st_size < base_path.stat().st_size

    restored = _TinyModel()
    iteration = load_checkpoint(delta_path, restored, strict=True)
    assert iteration == 53000
    torch.testing.assert_close(restored.backbone.weight, base_backbone)
    torch.testing.assert_close(
        restored.structured_query_head.set_selection_head.weight,
        selector_weight,
    )


def test_atomic_checkpoint_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, iteration=10)

    def fail_save(_payload, temporary) -> None:
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated full filesystem")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        save_checkpoint(path, model, iteration=20)

    assert load_checkpoint(path, _TinyModel()) == 10
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_checkpoint_can_restore_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    model = _TinyModel()
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    path = tmp_path / "rng.pt"
    save_checkpoint(path, model, iteration=10000, include_rng_state=True)

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(4)

    random.seed(701)
    np.random.seed(702)
    torch.manual_seed(703)
    restored = _TinyModel()
    iteration = load_checkpoint(
        path,
        restored,
        restore_rng_state=True,
    )

    assert iteration == 10000
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)
