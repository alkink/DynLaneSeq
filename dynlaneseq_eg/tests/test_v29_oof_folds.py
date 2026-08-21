from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file
from dynlaneseq_eg.tools.evaluate_v28_refined_belief_gate import (
    _validate_endpoint,
    _validate_v29_support,
)
from dynlaneseq_eg.tools.summarize_v29_oof_belief_gate import _direction
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


def test_v29_eval_contract_binds_support_and_belief_fold(tmp_path: Path) -> None:
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
    fold_contract = tmp_path / "folds" / "fold_contract.json"

    support_checkpoint = tmp_path / "support.pt"
    support_checkpoint.write_bytes(b"support-v7")
    support_report = tmp_path / "support_report.json"
    support_report.write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(support_checkpoint),
                "iteration": 112_500,
                "support_train_fold": "a",
                "belief_train_fold": "b",
                "fold_contract_sha256": sha256_file(fold_contract),
                "training_list_sha256": report["folds"]["a"][
                    "gt_list_sha256"
                ],
                "validation_used_for_checkpoint_selection": False,
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    contract = _validate_v29_support(
        checkpoint=support_checkpoint,
        report_path=support_report,
        fold_contract_path=fold_contract,
        belief_fold="b",
    )
    assert contract["support_fold"] == "a"
    assert all(contract["checks"].values())

    arm_checkpoint = tmp_path / "arm.pt"
    arm_checkpoint.write_bytes(b"arm-b")
    arm_report = tmp_path / "arm_report.json"
    arm_report.write_text(
        json.dumps(
            {
                "arm": "B",
                "iteration": 6_000,
                "checkpoint_sha256": sha256_file(arm_checkpoint),
                "gate_zero": {"passed": True},
                "teacher_state_still_exact": True,
                "checkpoint_selection_performed": False,
                "threshold_selection_performed": False,
                "test_set_used": False,
                "training_population_is_oof": True,
                "oof_fold": "b",
                "v7_checkpoint_sha256": sha256_file(support_checkpoint),
                "v7_iteration": 112_500,
                "official_train_population_contract": {
                    "fold": "b",
                    "fold_contract_sha256": sha256_file(fold_contract),
                },
            }
        ),
        encoding="utf-8",
    )
    endpoint = _validate_endpoint(
        arm_checkpoint,
        arm_report,
        arm="B",
        router_only=False,
        oof_fold="b",
        fold_contract_sha256=sha256_file(fold_contract),
        expected_v7_sha256=sha256_file(support_checkpoint),
    )
    assert all(endpoint["checks"].values())


def test_v29_eval_contract_accepts_predeclared_early_support_iteration(
    tmp_path: Path,
) -> None:
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
    fold_contract = tmp_path / "folds" / "fold_contract.json"
    support_checkpoint = tmp_path / "support_30k.pt"
    support_checkpoint.write_bytes(b"support-v7-30k")
    support_report = tmp_path / "support_report.json"
    support_report.write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(support_checkpoint),
                "iteration": 30_000,
                "support_train_fold": "a",
                "belief_train_fold": "b",
                "fold_contract_sha256": sha256_file(fold_contract),
                "training_list_sha256": report["folds"]["a"][
                    "gt_list_sha256"
                ],
                "validation_used_for_checkpoint_selection": False,
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    contract = _validate_v29_support(
        checkpoint=support_checkpoint,
        report_path=support_report,
        fold_contract_path=fold_contract,
        belief_fold="b",
        expected_iteration=30_000,
    )
    assert all(contract["checks"].values())

    arm_checkpoint = tmp_path / "arm.pt"
    arm_checkpoint.write_bytes(b"arm-c")
    arm_report = tmp_path / "arm_report.json"
    arm_report.write_text(
        json.dumps(
            {
                "arm": "C",
                "iteration": 6_000,
                "router_only_checkpoint_sha256": sha256_file(arm_checkpoint),
                "gate_zero": {"passed": True},
                "teacher_state_still_exact": True,
                "checkpoint_selection_performed": False,
                "threshold_selection_performed": False,
                "test_set_used": False,
                "training_population_is_oof": True,
                "oof_fold": "b",
                "v7_checkpoint_sha256": sha256_file(support_checkpoint),
                "v7_iteration": 30_000,
                "official_train_population_contract": {
                    "fold": "b",
                    "fold_contract_sha256": sha256_file(fold_contract),
                },
            }
        ),
        encoding="utf-8",
    )
    endpoint = _validate_endpoint(
        arm_checkpoint,
        arm_report,
        arm="C",
        router_only=True,
        oof_fold="b",
        fold_contract_sha256=sha256_file(fold_contract),
        expected_v7_sha256=sha256_file(support_checkpoint),
        expected_v7_iteration=30_000,
    )
    assert all(endpoint["checks"].values())


def test_v29_summary_requires_a_passing_oof_direction(tmp_path: Path) -> None:
    root = tmp_path / "support_a_to_fold_b"
    paired_dir = root / "paired"
    confidence_dir = root / "switch_confidence"
    paired_dir.mkdir(parents=True)
    confidence_dir.mkdir(parents=True)

    def metric(f1: float) -> dict[str, dict[str, float | int]]:
        value = {
            "TP": 10,
            "FP": 1,
            "FN": 1,
            "Precision": f1,
            "Recall": f1,
            "F1": f1,
        }
        return {"0.50": dict(value), "0.75": dict(value)}

    paired = {
        "oof_support_contract": {
            "belief_fold": "b",
            "checks": {"all": True},
        },
        "domains": {
            "full_validation": {
                "metrics": {
                    "source_v7": metric(0.80),
                    "arm_b": metric(0.80),
                    "arm_c": metric(0.81),
                    "arm_c_wrong_image": metric(0.20),
                }
            }
        },
        "gate": {
            "passed": True,
            "deltas": {"full_validation": {"c_minus_b_f1_50_points": 1.0}},
        },
        "contract": {"test_set_used": False},
    }
    confidence = {
        "oof_support_contract": {"belief_fold": "b"},
        "margin_auc": {
            "quality_delta_0p01": {"auc": 0.70},
            "threshold_0p50": {"auc": 0.65},
            "threshold_0p75": {"auc": 0.65},
        },
        "risk_curve": {
            key: {
                "best_validation_diagnostic_at_harmful_rate_le_1pct": {
                    "net": 1
                }
            }
            for key in ("0.50", "0.75")
        },
        "contract": {"test_set_used": False},
    }
    (paired_dir / "official_val_report.json").write_text(
        json.dumps(paired), encoding="utf-8"
    )
    (confidence_dir / "switch_confidence_report.json").write_text(
        json.dumps(confidence), encoding="utf-8"
    )
    result = _direction(
        root,
        expected_name="support_a_to_fold_b",
        expected_belief_fold="b",
    )
    assert result["passed"] is True
