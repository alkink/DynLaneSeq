from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from dynlaneseq_eg.engine.checkpoint import _torch_load
from dynlaneseq_eg.tools.build_v29_oof_folds import (
    FOLD_NAMES,
    validate_support_fold_contract,
)


ENDPOINT_ITERATION = 112_500
ORIGINAL_V7_ENDPOINT = 225_000
ORIGINAL_V7_SCHEDULE = 278_000
FOLD_V7_SCHEDULE = 139_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train the two fold-specific V7 support models used "
            "by the V29 OOF belief experiment."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--fold-contract", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"invalid training checkpoint name: {path}")
    return int(match.group(1))


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("iter_*.pt"),
        key=_checkpoint_iteration,
    )
    valid = [
        path
        for path in checkpoints
        if 0 < _checkpoint_iteration(path) <= ENDPOINT_ITERATION
    ]
    return valid[-1] if valid else None


def _run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": command}) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(
            f"V29 support training failed with return code {return_code}"
        )


def _verify_endpoint(
    endpoint: Path,
    *,
    fold_contract: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    payload = _torch_load(endpoint)
    iteration = int(payload.get("iteration", -1))
    cfg = payload.get("cfg")
    if iteration != ENDPOINT_ITERATION:
        raise ValueError(
            f"V29 support endpoint iteration {iteration} != {ENDPOINT_ITERATION}"
        )
    if not isinstance(cfg, dict):
        raise ValueError("V29 support endpoint is missing its expanded config")
    configured_list = Path(cfg["dataset"]["lists"]["train"]).expanduser().resolve()
    expected_list = Path(fold_contract["gt_list_path"]).expanduser().resolve()
    if configured_list != expected_list:
        raise ValueError("V29 support endpoint trained on the wrong fold list")
    configured_root = Path(cfg["dataset"]["root"]).expanduser().resolve()
    if configured_root != dataset_root:
        raise ValueError("V29 support endpoint used the wrong dataset root")
    if int(cfg["scheduler"]["total_iters"]) != FOLD_V7_SCHEDULE:
        raise ValueError("V29 support endpoint used the wrong cosine horizon")
    return {
        "checkpoint": str(endpoint),
        "checkpoint_sha256": _sha256(endpoint),
        "iteration": iteration,
        "model_name": str(cfg["model"]["name"]),
        "training_list": str(configured_list),
        "training_list_sha256": str(fold_contract["gt_list_sha256"]),
        "training_rows": int(fold_contract["row_count"]),
        "training_clips": int(fold_contract["clip_count"]),
        "original_v7_endpoint_iteration": ORIGINAL_V7_ENDPOINT,
        "original_v7_schedule_iterations": ORIGINAL_V7_SCHEDULE,
        "fold_v7_endpoint_iteration": ENDPOINT_ITERATION,
        "fold_v7_schedule_iterations": FOLD_V7_SCHEDULE,
        "cosine_phase_ratio": ENDPOINT_ITERATION / FOLD_V7_SCHEDULE,
        "test_set_used": False,
        "validation_used_for_checkpoint_selection": False,
    }


def main() -> None:
    args = parse_args()
    config = Path(args.config).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    fold_report_path = Path(args.fold_contract).expanduser().resolve()
    fold_report = json.loads(fold_report_path.read_text(encoding="utf-8"))
    if fold_report.get("passed") is not True:
        raise ValueError("V29 support training requires a passing fold contract")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    started = time.perf_counter()

    for fold in FOLD_NAMES:
        fold_output = output_root / f"support_fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        gt_list = Path(fold_report["folds"][fold]["gt_list"])
        fold_contract = validate_support_fold_contract(
            fold_report_path,
            fold=fold,
            gt_list_path=gt_list,
        )
        endpoint = fold_output / f"iter_{ENDPOINT_ITERATION:07d}.pt"
        if not endpoint.is_file():
            latest = _latest_checkpoint(fold_output)
            command = [
                sys.executable,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.train",
                "--config",
                str(config),
                "--device",
                str(args.device),
                "--dataset-root",
                str(dataset_root),
                "--train-list",
                str(gt_list),
                "--output-dir",
                str(fold_output),
                "--max-iters",
                str(ENDPOINT_ITERATION),
                "--checkpoint-interval",
                "10000",
                "--num-workers",
                str(args.num_workers),
                "--resume-safe-data",
                "true",
            ]
            if latest is not None:
                command.extend(("--resume", str(latest)))
            _run_and_log(command, fold_output / "training.log")
        report = _verify_endpoint(
            endpoint,
            fold_contract=fold_contract,
            dataset_root=dataset_root,
        )
        report.update(
            {
                "experiment": "V29 fold-specific V7 OOF support",
                "support_train_fold": fold,
                "belief_train_fold": "b" if fold == "a" else "a",
                "fold_contract": str(fold_report_path),
                "fold_contract_sha256": _sha256(fold_report_path),
                "config": str(config),
                "config_sha256": _sha256(config),
            }
        )
        (fold_output / "support_training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports[fold] = report

    summary = {
        "experiment": "V29 two-fold OOF support training",
        "passed": len(reports) == len(FOLD_NAMES),
        "elapsed_seconds": time.perf_counter() - started,
        "fold_contract": str(fold_report_path),
        "fold_contract_sha256": _sha256(fold_report_path),
        "supports": reports,
        "checkpoint_selection_performed": False,
        "test_set_used": False,
    }
    (output_root / "support_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
