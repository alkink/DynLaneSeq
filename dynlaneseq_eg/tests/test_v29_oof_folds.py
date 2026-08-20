from __future__ import annotations

from pathlib import Path

import pytest

from dynlaneseq_eg.tools.build_v29_oof_folds import (
    build,
    validate_fold_contract,
    validate_support_fold_contract,
)


def _rows() -> tuple[list[str], list[str]]:
    images: list[str] = []
    gt: list[str] = []
    for clip, count in (("clip_a.MP4", 5), ("clip_b.MP4", 4), ("clip_c.MP4", 3)):
        for index in range(count):
            image = f"/driver/{clip}/{index:05d}.jpg"
            images.append(image)
            gt.append(f"{image} /labels/{clip}/{index:05d}.png 1 1 0 0")
    return images, gt


def test_v29_folds_are_deterministic_complete_and_clip_disjoint(
    tmp_path: Path,
) -> None:
    images, gt = _rows()
    train = tmp_path / "train.txt"
    train_gt = tmp_path / "train_gt.txt"
    train.write_text("\n".join(images) + "\n", encoding="utf-8")
    train_gt.write_text("\n".join(gt) + "\n", encoding="utf-8")

    first = build(
        train_list=train,
        train_gt_list=train_gt,
        output_dir=tmp_path / "first",
        seed=3407,
    )
    second = build(
        train_list=train,
        train_gt_list=train_gt,
        output_dir=tmp_path / "second",
        seed=3407,
    )
    assert first["passed"] is True
    assert second["passed"] is True
    first_members = {
        fold: {
            row.split()[0]
            for row in Path(first["folds"][fold]["image_list"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        for fold in ("a", "b")
    }
    second_members = {
        fold: {
            row.split()[0]
            for row in Path(second["folds"][fold]["image_list"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        for fold in ("a", "b")
    }
    assert first_members == second_members
    assert not (first_members["a"] & first_members["b"])
    assert first_members["a"] | first_members["b"] == set(images)
    clips = {
        fold: {str(Path(image).parent) for image in members}
        for fold, members in first_members.items()
    }
    assert not (clips["a"] & clips["b"])


def test_v29_fold_contract_rejects_wrong_fold_list(tmp_path: Path) -> None:
    images, gt = _rows()
    train = tmp_path / "train.txt"
    train_gt = tmp_path / "train_gt.txt"
    train.write_text("\n".join(images) + "\n", encoding="utf-8")
    train_gt.write_text("\n".join(gt) + "\n", encoding="utf-8")
    report = build(
        train_list=train,
        train_gt_list=train_gt,
        output_dir=tmp_path / "folds",
        seed=3407,
    )
    report_path = tmp_path / "folds" / "fold_contract.json"
    valid = validate_fold_contract(
        report_path,
        fold="a",
        list_path=report["folds"]["a"]["image_list"],
    )
    assert valid["passed"] is True
    assert valid["expected_nonempty_rows"] > 0
    with pytest.raises(ValueError, match="list mismatch"):
        validate_fold_contract(
            report_path,
            fold="a",
            list_path=report["folds"]["b"]["image_list"],
        )

    support = validate_support_fold_contract(
        report_path,
        fold="a",
        gt_list_path=report["folds"]["a"]["gt_list"],
    )
    assert support["passed"] is True
    assert support["row_count"] == valid["expected_nonempty_rows"]
    with pytest.raises(ValueError, match="GT-list mismatch"):
        validate_support_fold_contract(
            report_path,
            fold="a",
            gt_list_path=report["folds"]["b"]["gt_list"],
        )
