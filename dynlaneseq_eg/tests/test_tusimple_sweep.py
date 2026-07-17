from __future__ import annotations

from pathlib import Path

import pytest

from dynlaneseq_eg.tools.sweep_tusimple_thresholds import (
    _checkpoint_iteration,
    selection_key,
    validate_held_out_split,
)


def _cfg(tmp_path: Path, train: list[str], val: list[str]) -> dict:
    return {
        "dataset": {
            "root": str(tmp_path),
            "splits": {
                "train": {"annotations": train},
                "val": {"annotations": val},
            },
        }
    }


def test_validation_sweep_rejects_training_overlap(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, ["0313.json", "0531.json"], ["0531.json"])
    with pytest.raises(ValueError, match="overlap"):
        validate_held_out_split(cfg, "val")


def test_validation_sweep_accepts_disjoint_annotations(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, ["0313.json", "0601.json"], ["0531.json"])
    validate_held_out_split(cfg, "val")


def test_selection_key_prioritizes_official_accuracy() -> None:
    higher_accuracy = {
        "Accuracy": 0.96,
        "F1_score": 0.95,
        "FP": 0.03,
        "FN": 0.04,
        "iteration": 100,
    }
    higher_f1 = {
        "Accuracy": 0.95,
        "F1_score": 0.99,
        "FP": 0.01,
        "FN": 0.01,
        "iteration": 200,
    }
    assert selection_key(higher_accuracy) > selection_key(higher_f1)


def test_checkpoint_iteration_is_read_from_filename() -> None:
    assert _checkpoint_iteration(Path("iter_0014315.pt")) == 14315
