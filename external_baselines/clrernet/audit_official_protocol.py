#!/usr/bin/env python3
"""Fail closed unless the independent CLRerNet run uses untouched splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import mmcv
from mmengine.config import Config
import torch


EXPECTED = {
    "upstream_commit": "e77f10ddffb16e3986f815d31b83fc56675e8851",
    "train_rows": 88_880,
    "val_rows": 9_675,
    "train_sha256": "b8512d9f1954a290d156cd522b6c11220e5bcd7bbd965016251167f426334908",
    "val_sha256": "01f04df210879e3c52bc3994b834eba71135250c2a06c56bc8fef0450dd7610d",
    "threshold": 0.41,
    "epochs": 15,
    "nms_patch_sha256": "0635c02f18d53a08007902f9369a57b2fb8b6c98229c534e7aa606c7f768ec61",
    "mmcv_shim_sha256": "f5e22597f563b837b4ab0e7bfc05c6ea43b961285b328d098946fda3ad2a1b8e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clrernet-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--nms-patch", type=Path, required=True)
    parser.add_argument("--mmcv-shim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config.resolve()))
    train_list = args.data_root / "list" / "train_gt.txt"
    val_list = args.data_root / "list" / "val.txt"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.clrernet_root, text=True
    ).strip()
    modified_sources = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=args.clrernet_root, text=True
    ).splitlines()
    reverse_patch = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(args.nms_patch.resolve())],
        cwd=args.clrernet_root,
        text=True,
        capture_output=True,
        check=False,
    )

    train_ds = cfg.train_dataloader.dataset
    val_ds = cfg.val_dataloader.dataset
    installed_mmcv_shim = Path(mmcv.__file__).resolve().parent / "_ext.py"
    facts = {
        "upstream_commit": commit,
        "config": str(args.config.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_list": str(train_list.resolve()),
        "val_list": str(val_list.resolve()),
        "train_rows": row_count(train_list),
        "val_rows": row_count(val_list),
        "train_sha256": sha256(train_list),
        "val_sha256": sha256(val_list),
        "configured_train_list": str(Path(train_ds.data_list).resolve()),
        "configured_val_list": str(Path(val_ds.data_list).resolve()),
        "diff_file": train_ds.get("diff_file"),
        "diff_thr": train_ds.get("diff_thr"),
        "threshold": float(cfg.model.test_cfg.conf_threshold),
        "epochs": int(cfg.train_cfg.max_epochs),
        "val_interval": int(cfg.train_cfg.val_interval),
        "checkpoint_interval": int(cfg.default_hooks.checkpoint.interval),
        "save_best": cfg.default_hooks.checkpoint.get("save_best"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nms_patch": str(args.nms_patch.resolve()),
        "nms_patch_sha256": sha256(args.nms_patch),
        "mmcv_version": mmcv.__version__,
        "mmcv_distribution": importlib.metadata.version("mmcv-lite"),
        "mmcv_shim_source": str(args.mmcv_shim.resolve()),
        "mmcv_shim_source_sha256": sha256(args.mmcv_shim),
        "mmcv_shim_installed": str(installed_mmcv_shim),
        "mmcv_shim_installed_sha256": sha256(installed_mmcv_shim),
        "modified_upstream_sources": modified_sources,
        "test_invoked": False,
        "deduplication": False,
        "frame_filtering": False,
        "checkpoint_selection": False,
        "threshold_selection": False,
    }

    checks = {
        "upstream_v030_exact": commit == EXPECTED["upstream_commit"],
        "train_rows_exact": facts["train_rows"] == EXPECTED["train_rows"],
        "val_rows_exact": facts["val_rows"] == EXPECTED["val_rows"],
        "train_hash_exact": facts["train_sha256"] == EXPECTED["train_sha256"],
        "val_hash_exact": facts["val_sha256"] == EXPECTED["val_sha256"],
        "configured_train_exact": Path(facts["configured_train_list"]) == train_list.resolve(),
        "configured_val_exact": Path(facts["configured_val_list"]) == val_list.resolve(),
        "frame_filter_disabled": facts["diff_file"] is None,
        "published_threshold_fixed": facts["threshold"] == EXPECTED["threshold"],
        "nms_compat_patch_exact": (
            facts["nms_patch_sha256"] == EXPECTED["nms_patch_sha256"]
            and reverse_patch.returncode == 0
            and sorted(modified_sources)
            == [
                "libs/models/layers/nms/src/nms.cpp",
                "libs/models/layers/nms/src/nms_kernel.cu",
            ]
        ),
        "mmcv_import_shim_exact": (
            facts["mmcv_version"] == "2.1.0"
            and facts["mmcv_distribution"] == "2.1.0"
            and facts["mmcv_shim_source_sha256"] == EXPECTED["mmcv_shim_sha256"]
            and facts["mmcv_shim_installed_sha256"] == EXPECTED["mmcv_shim_sha256"]
        ),
        "fixed_endpoint": (
            facts["epochs"] == EXPECTED["epochs"]
            and facts["val_interval"] == EXPECTED["epochs"]
            and facts["checkpoint_interval"] == EXPECTED["epochs"]
            and facts["save_best"] is None
        ),
    }
    report = {
        "protocol": "CLRerNet-v0.3-DLA34-all-official-CULane-train",
        "expected": EXPECTED,
        "facts": facts,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("Official CLRerNet protocol contract failed closed")


if __name__ == "__main__":
    main()
