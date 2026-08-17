from pathlib import Path

from dynlaneseq_eg.tools.build_v25_s1_oof_folds import (
    build_oof_folds,
    validate_fold_training_population,
)


def test_two_fold_partition_preserves_every_official_row_and_clip(tmp_path: Path) -> None:
    root = tmp_path / "CULane"
    list_dir = root / "list"
    list_dir.mkdir(parents=True)
    rows = [
        f"/driver_{index % 5}/clip_{index // 40:05d}/{index:05d}.jpg"
        for index in range(88_880)
    ]
    (list_dir / "train.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = build_oof_folds(root, tmp_path / "folds")

    assert manifest["passed"] is True
    assert sum(row["holdout_rows"] for row in manifest["folds"]) == 88_880
    for fold in manifest["folds"]:
        train_rows = Path(fold["train_list"]).read_text().splitlines()
        holdout_rows = Path(fold["holdout_list"]).read_text().splitlines()
        train_clips = {str(Path(row).parent) for row in train_rows}
        holdout_clips = {str(Path(row).parent) for row in holdout_rows}
        assert not train_clips & holdout_clips

        contract = validate_fold_training_population(
            tmp_path / "folds/oof_fold_manifest.json",
            fold=int(fold["fold"]),
            supplied_train_list=fold["train_list"],
        )
        assert contract["passed"] is True
        assert contract["expected_nonempty_rows"] == len(train_rows)
