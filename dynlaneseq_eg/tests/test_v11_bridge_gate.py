from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pytest

from dynlaneseq_eg.tools.build_v11_bridge_lists import build
from dynlaneseq_eg.tools import summarize_v11_bridge_gate


def _paths(lines: list[str]) -> set[str]:
    return {line.split()[0] for line in lines}


def _clips(paths: set[str]) -> set[str]:
    return {"/".join(path.split("/")[:-1]) for path in paths}


def test_bridge_list_builder_is_deterministic_and_clip_disjoint(tmp_path: Path):
    train = tmp_path / "train_gt.txt"
    val = tmp_path / "val.txt"
    train_rows = []
    for clip_index in range(6):
        for frame_index in range(5):
            image = f"/driver_train/clip_{clip_index:02d}/{frame_index:04d}.jpg"
            train_rows.append(f"{image} /seg/{clip_index:02d}_{frame_index:04d}.png 1 1 0 0")
    val_rows = [
        f"/driver_val/clip_{clip_index:02d}/{frame_index:04d}.jpg"
        for clip_index in range(2)
        for frame_index in range(4)
    ]
    train.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    val.write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    def run(destination: Path) -> dict[str, object]:
        return build(
            argparse.Namespace(
                train_list=str(train),
                val_list=str(val),
                output_dir=str(destination),
                seed=3407,
                train_clips=4,
                train_images=8,
                seen_images=2,
                same_clip_unseen_images=2,
                heldout_clips=2,
                heldout_images=4,
                val_images=4,
                output_json=str(destination / "protocol.json"),
            )
        )

    first = run(tmp_path / "first")
    second = run(tmp_path / "second")
    assert first["passed"] is True
    assert first["test_set_used"] is False
    assert {
        name: value["sha256"] for name, value in first["lists"].items()
    } == {
        name: value["sha256"] for name, value in second["lists"].items()
    }

    lists = {
        name: Path(value["path"]).read_text(encoding="utf-8").splitlines()
        for name, value in first["lists"].items()
    }
    train_paths = _paths(lists["train"])
    seen_paths = _paths(lists["seen_train"])
    same_paths = _paths(lists["same_clip_unseen"])
    heldout_paths = _paths(lists["heldout_clip"])
    val_paths = _paths(lists["val"])
    assert seen_paths <= train_paths
    assert not (same_paths & train_paths)
    assert _clips(same_paths) <= _clips(train_paths)
    assert not (_clips(heldout_paths) & _clips(train_paths))
    assert not (_clips(val_paths) & _clips(train_paths))


def test_bridge_list_builder_balances_a_nondivisible_train_budget(
    tmp_path: Path,
):
    train = tmp_path / "train_gt.txt"
    val = tmp_path / "val.txt"
    train.write_text(
        "\n".join(
            f"/driver_train/clip_{clip:02d}/{frame:04d}.jpg "
            f"/seg/{clip:02d}_{frame:04d}.png 1 1 0 0"
            for clip in range(8)
            for frame in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    val.write_text(
        "\n".join(
            f"/driver_val/clip_{clip:02d}/{frame:04d}.jpg"
            for clip in range(2)
            for frame in range(4)
        )
        + "\n",
        encoding="utf-8",
    )

    arguments = dict(
        train_list=str(train),
        val_list=str(val),
        output_dir=str(tmp_path / "balanced"),
        seed=3407,
        train_clips=5,
        train_images=13,
        seen_images=2,
        same_clip_unseen_images=2,
        heldout_clips=2,
        heldout_images=4,
        val_images=4,
        output_json=str(tmp_path / "balanced" / "protocol.json"),
    )
    with pytest.raises(ValueError, match="divide evenly"):
        build(argparse.Namespace(**arguments))

    report = build(
        argparse.Namespace(**arguments, balanced_train_remainder=True)
    )
    assert report["passed"] is True
    assert report["parameters"]["train_images_per_clip_base"] == 2
    assert report["parameters"]["train_clips_with_one_extra_image"] == 3
    train_lines = Path(report["lists"]["train"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    per_clip = Counter(
        "/".join(line.split()[0].split("/")[:-1]) for line in train_lines
    )
    assert sorted(per_clip.values()) == [2, 2, 3, 3, 3]


def _metric(tp: int, fp: int, fn: int, mean_selected: float = 3.0):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_selected_per_image": mean_selected,
    }


def _coverage(
    tp050: int,
    tp075: int,
    *,
    split: str,
    mean_selected: float = 3.0,
):
    return {
        "split": split,
        "test_set_used": False,
        "methods": {
            "four_slot_refined": {
                "0.50": _metric(tp050, 120 - tp050, 120 - tp050, mean_selected),
                "0.75": _metric(tp075, 120 - tp075, 120 - tp075, mean_selected),
            }
        },
        "capacity": {
            "0.50": {"all_candidate_oracle": {"hits": 118}},
            "0.75": {"all_candidate_oracle": {"hits": 110}},
        },
        "four_slot_diagnostics": {
            "cardinality": {"exact_fraction": 0.96, "mean_absolute_error": 0.04},
            "semantic_duplicate_cluster_fraction": 0.0,
        },
    }


def _state(value: float, split: str):
    return {
        "split": split,
        "test_set_used": False,
        "statistics": {"target_support_mass": {"mean": value}},
    }


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_bridge_summary_requires_independent_clip_and_val_transfer(
    tmp_path: Path,
    monkeypatch,
):
    protocol = _write(
        tmp_path / "protocol.json",
        {"passed": True, "test_set_used": False},
    )
    contract = _write(tmp_path / "contract.json", {"passed": True})
    domains = {}
    endpoints = {
        "seen_train": (110, 90, 0.10),
        "same_clip_unseen": (108, 88, 0.08),
        "heldout_clip": (101, 72, 0.021),
        "val": (103, 76, 0.051),
    }
    for name, (tp050, tp075, state_gain) in endpoints.items():
        split = "val" if name == "val" else "train"
        prefix = tmp_path / name
        domains[name] = {
            "split": split,
            "list_path": str(tmp_path / f"{name}.txt"),
            "source_coverage": _write(
                Path(f"{prefix}_source.json"), _coverage(100, 70, split=split)
            ),
            "init_coverage": _write(
                Path(f"{prefix}_init.json"), _coverage(100, 70, split=split)
            ),
            "end_coverage": _write(
                Path(f"{prefix}_end.json"), _coverage(tp050, tp075, split=split)
            ),
            "init_state": _write(
                Path(f"{prefix}_init_state.json"), _state(0.40, split)
            ),
            "end_state": _write(
                Path(f"{prefix}_end_state.json"), _state(0.40 + state_gain, split)
            ),
        }
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "protocol": protocol,
            "contract": contract,
            "iteration": 228000,
            "test_set_used": False,
            "domains": domains,
        },
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_v11_bridge_gate",
            "--manifest",
            manifest,
            "--output-json",
            str(output),
        ],
    )
    summarize_v11_bridge_gate.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["diagnosis"] == "clip_disjoint_and_validation_transfer"
    assert report["long_training_authorized"] is False

    # Preserve strong seen/same-clip results while removing independent val
    # transfer.  The bridge must diagnose leakage-like memorization, not pass.
    _write(Path(domains["val"]["end_coverage"]), _coverage(100, 70, split="val"))
    second_output = tmp_path / "summary_fail.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_v11_bridge_gate",
            "--manifest",
            manifest,
            "--output-json",
            str(second_output),
        ],
    )
    summarize_v11_bridge_gate.main()
    failed = json.loads(second_output.read_text(encoding="utf-8"))
    assert failed["passed"] is False
    assert failed["diagnosis"] == (
        "scene_or_clip_memorization_without_independent_transfer"
    )
