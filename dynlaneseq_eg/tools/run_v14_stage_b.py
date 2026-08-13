from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conditionally execute the predeclared V14 Stage-B gate."
    )
    for name in (
        "stage-a-config",
        "stage-b-config",
        "v7-config",
        "stage-a-checkpoint",
        "source-v7-checkpoint",
        "dataset-root",
        "train-list",
        "heldout-list",
        "heldout-wrong-list",
        "heldout-cross-clip-report",
        "val-list",
        "val-wrong-list",
        "val-cross-clip-report",
        "output-root",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=12)
    parser.add_argument("--amp-dtype", default="bfloat16")
    return parser.parse_args()


def _run(command: list[str], *, log: Path | None = None) -> None:
    print({"command": command})
    if log is None:
        subprocess.run(command, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _checkpoint_iteration(path: Path) -> int:
    from dynlaneseq_eg.engine.checkpoint import _torch_load

    return int(_torch_load(path).get("iteration", -1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if int(args.batch_size) * int(args.grad_accum) != 16:
        raise ValueError("V14 Stage B requires effective batch 16")
    root = Path(args.output_root).resolve()
    audit = root / "audits"
    reports = root / "reports"
    train = root / "train"
    initialization = root / "initialization" / "iter_0227000.pt"
    endpoint = train / "iter_0229000.pt"
    for directory in (audit, reports, train, initialization.parent):
        directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if not initialization.exists():
        _run(
            [
                python,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.initialize_v14_stage_b_checkpoint",
                "--config",
                args.stage_b_config,
                "--source-checkpoint",
                args.stage_a_checkpoint,
                "--seed",
                str(args.seed),
                "--iteration",
                "227000",
                "--output-checkpoint",
                str(initialization),
                "--output-json",
                str(audit / "initialization.json"),
            ]
        )
    contract = audit / "zero_step_contract.json"
    _run(
        [
            python,
            "-u",
            "-m",
            "dynlaneseq_eg.tools.audit_v14_stage_b_contract",
            "--config",
            args.stage_b_config,
            "--source-config",
            args.stage_a_config,
            "--checkpoint",
            str(initialization),
            "--source-checkpoint",
            args.stage_a_checkpoint,
            "--dataset-root",
            args.dataset_root,
            "--device",
            args.device,
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--start-iteration",
            "227000",
            "--output-json",
            str(contract),
        ]
    )
    if json.loads(contract.read_text()).get("passed") is not True:
        minimal_paths = {
            "stage_a_config": Path(args.stage_a_config).resolve(),
            "stage_b_config": Path(args.stage_b_config).resolve(),
            "stage_a_checkpoint": Path(args.stage_a_checkpoint).resolve(),
            "stage_b_initialization": initialization.resolve(),
            "zero_step_contract": contract.resolve(),
        }
        _write_json(
            root / "provenance.json",
            {
                "experiment": "V14 Stage-B Gate-0 failure",
                "git_commit": subprocess.check_output(
                    ("git", "rev-parse", "HEAD"), text=True
                ).strip(),
                "artifacts": {
                    name: {"path": str(path), "sha256": _sha256(path)}
                    for name, path in minimal_paths.items()
                },
                "checkpoint_selection_performed": False,
                "test_set_used": False,
            },
        )
        _write_json(
            root / "v14_completion.json",
            {
                "experiment": "V14 corrected visual-first Stage A/B",
                "stage_a_passed": True,
                "stage_b_gate0_passed": False,
                "stage_b_training_executed": False,
                "full_validation_executed": False,
                "long_training_authorized": False,
                "test_set_used": False,
                "decision": "v14_complete_stage_b_gate0_fail_stop_for_sol",
            },
        )
        print("V14 Stage-B Gate 0 failed; training and later stages closed")
        return

    if not endpoint.exists():
        latest: Path | None = None
        latest_iteration = 227000
        for candidate in sorted(train.glob("iter_*.pt")):
            candidate_iteration = _checkpoint_iteration(candidate)
            if latest_iteration < candidate_iteration < 229000:
                latest = candidate
                latest_iteration = candidate_iteration
        train_command = [
                python,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.train",
                "--config",
                args.stage_b_config,
                "--dataset-root",
                args.dataset_root,
                "--device",
                args.device,
                "--output-dir",
                str(train),
                "--checkpoint-base",
                str(initialization),
                "--checkpoint-interval",
                "500",
                "--seed",
                str(args.seed),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--num-workers",
                str(args.num_workers),
                "--seg-aux-amp-dtype",
                args.amp_dtype,
                "--compile-model",
                "false",
                "--resume-safe-data",
                "true",
                "--train-list",
                args.train_list,
            ]
        if latest is None:
            train_command.extend(
                (
                    "--init-from",
                    str(initialization),
                    "--init-iteration",
                    "227000",
                    "--max-iters",
                    "2000",
                )
            )
        else:
            train_command.extend(
                (
                    "--resume",
                    str(latest),
                    "--max-iters",
                    str(229000 - latest_iteration),
                )
            )
        _run(
            train_command,
            log=train / "train.log",
        )
    if not endpoint.exists() or _checkpoint_iteration(endpoint) != 229000:
        raise FileNotFoundError("fixed V14 Stage-B endpoint is unavailable")

    def official(
        *, split: str, source_list: str, wrong_list: str, crossclip: str, output: Path
    ) -> None:
        _run(
            [
                python,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.audit_v14_stage_b_official",
                "--config",
                args.stage_b_config,
                "--checkpoint",
                str(endpoint),
                "--dataset-root",
                args.dataset_root,
                "--device",
                args.device,
                "--split",
                split,
                "--list-path",
                source_list,
                "--wrong-list-path",
                wrong_list,
                "--cross-clip-report",
                crossclip,
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--metric-workers",
                str(args.metric_workers),
                "--output-json",
                str(output),
            ]
        )

    heldout_official = reports / "heldout_official.json"
    val_official = reports / "validation_official.json"
    official(
        split="train",
        source_list=args.heldout_list,
        wrong_list=args.heldout_wrong_list,
        crossclip=args.heldout_cross_clip_report,
        output=heldout_official,
    )
    official(
        split="val",
        source_list=args.val_list,
        wrong_list=args.val_wrong_list,
        crossclip=args.val_cross_clip_report,
        output=val_official,
    )

    def coverage(
        *, config: str, checkpoint: str, split: str, list_path: str, name: str
    ) -> Path:
        output = reports / f"{name}_coverage.json"
        cache = root / "cache" / name
        _run(
            [
                python,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.analyze_v4_selection_coverage",
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--dataset-root",
                args.dataset_root,
                "--split",
                split,
                "--list-path",
                list_path,
                "--device",
                args.device,
                "--cache-dir",
                str(cache),
                "--max-batches",
                "0",
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--metric-workers",
                str(args.metric_workers),
                "--sample-strategy",
                "sequential",
                "--top-k",
                "4",
                "--iou-thresholds",
                "0.50",
                "0.75",
                "--output-json",
                str(output),
            ]
        )
        return output

    hs = coverage(
        config=args.v7_config,
        checkpoint=args.source_v7_checkpoint,
        split="train",
        list_path=args.heldout_list,
        name="heldout_source",
    )
    ht = coverage(
        config=args.stage_b_config,
        checkpoint=str(endpoint),
        split="train",
        list_path=args.heldout_list,
        name="heldout_treatment",
    )
    vs = coverage(
        config=args.v7_config,
        checkpoint=args.source_v7_checkpoint,
        split="val",
        list_path=args.val_list,
        name="val_source",
    )
    vt = coverage(
        config=args.stage_b_config,
        checkpoint=str(endpoint),
        split="val",
        list_path=args.val_list,
        name="val_treatment",
    )
    bridge = root / "v14_stage_b_bridge_summary.json"
    _run(
        [
            python,
            "-u",
            "-m",
            "dynlaneseq_eg.tools.summarize_v14_stage_b_gate",
            "--mode",
            "bridge",
            "--contract",
            str(contract),
            "--heldout",
            str(heldout_official),
            "--validation",
            str(val_official),
            "--heldout-source-coverage",
            str(hs),
            "--heldout-treatment-coverage",
            str(ht),
            "--val-source-coverage",
            str(vs),
            "--val-treatment-coverage",
            str(vt),
            "--output-json",
            str(bridge),
        ]
    )
    bridge_payload = json.loads(bridge.read_text())

    provenance_paths = {
        "stage_a_config": Path(args.stage_a_config).resolve(),
        "stage_b_config": Path(args.stage_b_config).resolve(),
        "v7_config": Path(args.v7_config).resolve(),
        "source_v7_checkpoint": Path(args.source_v7_checkpoint).resolve(),
        "stage_a_checkpoint": Path(args.stage_a_checkpoint).resolve(),
        "stage_b_initialization": initialization.resolve(),
        "stage_b_fixed_endpoint": endpoint.resolve(),
        "zero_step_contract": contract.resolve(),
        "heldout_official": heldout_official.resolve(),
        "validation_official": val_official.resolve(),
        "bridge_summary": bridge.resolve(),
    }
    provenance = {
        "experiment": "V14 Stage-B fixed gate",
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ("git", "branch", "--show-current"), text=True
        ).strip(),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in provenance_paths.items()
        },
        "endpoint_iteration": _checkpoint_iteration(endpoint),
        "checkpoint_selection_performed": False,
        "test_set_used": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
        "environment": {"python": sys.executable, "cwd": os.getcwd()},
    }
    _write_json(root / "provenance.json", provenance)

    if bridge_payload.get("passed") is not True:
        _write_json(
            root / "v14_completion.json",
            {
                "experiment": "V14 corrected visual-first Stage A/B",
                "stage_a_passed": True,
                "stage_b_executed": True,
                "stage_b_bridge_passed": False,
                "full_validation_executed": False,
                "long_training_authorized": False,
                "test_set_used": False,
                "decision": "v14_complete_fail_stop_for_sol",
            },
        )
        print("V14 Stage B bridge FAIL; full validation and long training closed")
        return

    def full_eval(config: str, checkpoint: str, name: str) -> Path:
        output = reports / f"full_validation_{name}.json"
        pred = root / "predictions" / name
        _run(
            [
                python,
                "-u",
                "-m",
                "dynlaneseq_eg.tools.evaluate_culane",
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--split",
                "val",
                "--dataset-root",
                args.dataset_root,
                "--device",
                args.device,
                "--pred-dir",
                str(pred),
                "--score-thresh",
                "0",
                "--nms-distance-thresh-px",
                "0",
                "--top-k",
                "4",
                "--quality-score-power",
                "0",
                "--score-mode",
                "four_slot",
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--eval-num-workers",
                str(args.num_workers),
                "--iou-thresholds",
                "0.50",
                "0.75",
                "--metric-workers",
                str(args.metric_workers),
                "--no-pretrained-init",
                "--legacy-inference",
                "--amp-dtype",
                "none",
                "--output-json",
                str(output),
            ]
        )
        return output

    source_eval = full_eval(args.v7_config, args.source_v7_checkpoint, "source_v7")
    treatment_eval = full_eval(args.stage_b_config, str(endpoint), "treatment_v14")
    full_summary = root / "v14_stage_b_full_validation_summary.json"
    _run(
        [
            python,
            "-u",
            "-m",
            "dynlaneseq_eg.tools.summarize_v14_stage_b_gate",
            "--mode",
            "full_validation",
            "--source-eval",
            str(source_eval),
            "--treatment-eval",
            str(treatment_eval),
            "--output-json",
            str(full_summary),
        ]
    )
    full_payload = json.loads(full_summary.read_text())
    provenance["artifacts"].update(
        {
            "full_validation_source": {
                "path": str(source_eval.resolve()),
                "sha256": _sha256(source_eval),
            },
            "full_validation_treatment": {
                "path": str(treatment_eval.resolve()),
                "sha256": _sha256(treatment_eval),
            },
            "full_validation_summary": {
                "path": str(full_summary.resolve()),
                "sha256": _sha256(full_summary),
            },
        }
    )
    _write_json(root / "provenance.json", provenance)
    _write_json(
        root / "v14_completion.json",
        {
            "experiment": "V14 corrected visual-first Stage A/B",
            "stage_a_passed": True,
            "stage_b_executed": True,
            "stage_b_bridge_passed": True,
            "full_validation_executed": True,
            "full_validation_passed": full_payload.get("passed") is True,
            "long_training_authorized": False,
            "test_set_used": False,
            "decision": full_payload.get("decision"),
        },
    )


if __name__ == "__main__":
    main()
