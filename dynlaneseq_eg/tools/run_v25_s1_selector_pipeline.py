from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the resumable V25-S1 OOF selector gate.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--v7-config", required=True)
    parser.add_argument("--v7-checkpoint", required=True)
    parser.add_argument("--g0-config", required=True)
    parser.add_argument("--full-g0-checkpoint", required=True)
    parser.add_argument("--wrong-image-list", required=True)
    parser.add_argument("--wrong-image-report", required=True)
    return parser.parse_args()


def _run(command: list[str], *, complete: Path) -> None:
    if complete.is_file():
        print(json.dumps({"status": "skip_complete", "path": str(complete)}), flush=True)
        return
    print(json.dumps({"phase": "launch", "command": command}), flush=True)
    subprocess.run(command, check=True)
    if not complete.is_file():
        raise RuntimeError(f"stage returned success without completion artifact: {complete}")


def _selector_command(
    *,
    caches: list[Path],
    eval_caches: list[Path],
    output: Path,
    arm: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "dynlaneseq_eg.tools.train_v25_s1_immutable_selector",
        "--output-dir",
        str(output),
        "--arm",
        arm,
        "--device",
        "cuda",
        "--epochs",
        "8",
        "--batch-size",
        "128",
        "--num-workers",
        "2",
    ]
    for cache in caches:
        command.extend(("--train-cache", str(cache)))
    for cache in eval_caches:
        command.extend(("--eval-cache", str(cache)))
    return command


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    root = Path(args.output_dir).expanduser().resolve()
    dataset = Path(args.dataset_root).expanduser().resolve()
    folds = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    if int(folds["folds"]) != 2:
        raise ValueError("V25-S1 primary gate is fixed to two OOF folds")
    wrong_report = json.loads(Path(args.wrong_image_report).read_text(encoding="utf-8"))
    if not bool(wrong_report.get("checks", {}).get("all_pass", False)):
        # Older reports store booleans directly rather than an all_pass key.
        checks = wrong_report.get("checks", {})
        if checks and not all(bool(value) for value in checks.values()):
            raise RuntimeError("the predeclared cross-clip wrong-image pairing is invalid")

    cache_roots = []
    for fold in range(2):
        cache = root / f"fold_{fold}" / "bank_holdout"
        cache_roots.append(cache)
        _run(
            [
                sys.executable,
                "-m",
                "dynlaneseq_eg.tools.cache_v25_s1_immutable_bank",
                "--dataset-root",
                str(dataset),
                "--list-path",
                str(root / "folds" / f"fold_{fold}_holdout.txt"),
                "--g0-config",
                args.g0_config,
                "--g0-checkpoint",
                str(root / f"fold_{fold}" / "g0_train" / "v25_g0_endpoint.pt"),
                "--v7-config",
                args.v7_config,
                "--v7-checkpoint",
                args.v7_checkpoint,
                "--output-dir",
                str(cache),
                "--device",
                "cuda",
                "--batch-size",
                "4",
                "--num-workers",
                "2",
                "--label-workers",
                "8",
            ],
            complete=cache / "bank_report.json",
        )

    mechanism = root / "mechanism_gate"
    mechanism_treatment = mechanism / "treatment"
    mechanism_control = mechanism / "geometry_control"
    _run(
        _selector_command(
            caches=[cache_roots[0]],
            eval_caches=[cache_roots[1]],
            output=mechanism_treatment,
            arm="treatment",
        ),
        complete=mechanism_treatment / "selector_training_report.json",
    )
    _run(
        _selector_command(
            caches=[cache_roots[0]],
            eval_caches=[cache_roots[1]],
            output=mechanism_control,
            arm="geometry_control",
        ),
        complete=mechanism_control / "selector_training_report.json",
    )
    treatment_report = json.loads(
        (mechanism_treatment / "selector_training_report.json").read_text(encoding="utf-8")
    )
    control_report = json.loads(
        (mechanism_control / "selector_training_report.json").read_text(encoding="utf-8")
    )
    treatment_eval = treatment_report["eval_metrics"]
    control_eval = control_report["eval_metrics"]
    mechanism_checks = {
        "heldout_gain_f1_50_at_least_0_50_points": float(treatment_eval["gain_f1_50_points"]) >= 0.50,
        "heldout_f1_75_non_regression": float(treatment_eval["gain_f1_75_points"]) >= 0.0,
        "heldout_treatment_beats_geometry_control_0_25_points": (
            float(treatment_eval["gain_f1_50_points"])
            - float(control_eval["gain_f1_50_points"])
            >= 0.25
        ),
        "heldout_source_correct_loss_below_one_percent": (
            int(treatment_eval["source_correct_tp50_lost"])
            / max(int(treatment_eval["source_tp50"]), 1)
            < 0.01
        ),
    }
    mechanism_report = {
        "experiment": "V25-S1 OOF selector mechanism gate",
        "treatment": treatment_eval,
        "geometry_control": control_eval,
        "checks": mechanism_checks,
        "pass": all(mechanism_checks.values()),
    }
    mechanism_path = mechanism / "mechanism_gate_report.json"
    mechanism_path.write_text(json.dumps(mechanism_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not mechanism_report["pass"]:
        print(json.dumps({"status": "mechanism_stop", "report": str(mechanism_path)}, indent=2), flush=True)
        return

    final_treatment = root / "final_selector" / "treatment"
    final_control = root / "final_selector" / "geometry_control"
    _run(
        _selector_command(caches=cache_roots, eval_caches=[], output=final_treatment, arm="treatment"),
        complete=final_treatment / "selector_training_report.json",
    )
    _run(
        _selector_command(caches=cache_roots, eval_caches=[], output=final_control, arm="geometry_control"),
        complete=final_control / "selector_training_report.json",
    )

    val_cache = root / "official_val_bank"
    _run(
        [
            sys.executable,
            "-m",
            "dynlaneseq_eg.tools.cache_v25_s1_immutable_bank",
            "--dataset-root",
            str(dataset),
            "--list-path",
            str(dataset / "list" / "val.txt"),
            "--g0-config",
            args.g0_config,
            "--g0-checkpoint",
            args.full_g0_checkpoint,
            "--v7-config",
            args.v7_config,
            "--v7-checkpoint",
            args.v7_checkpoint,
            "--output-dir",
            str(val_cache),
            "--wrong-image-list",
            args.wrong_image_list,
            "--retain-bank",
            "--device",
            "cuda",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
            "--label-workers",
            "8",
        ],
        complete=val_cache / "bank_report.json",
    )
    official = root / "official_validation"
    _run(
        [
            sys.executable,
            "-m",
            "dynlaneseq_eg.tools.evaluate_v25_s1_immutable_selector",
            "--dataset-root",
            str(dataset),
            "--cache-dir",
            str(val_cache),
            "--treatment-checkpoint",
            str(final_treatment / "selector_endpoint.pt"),
            "--control-checkpoint",
            str(final_control / "selector_endpoint.pt"),
            "--output-dir",
            str(official),
            "--device",
            "cuda",
        ],
        complete=official / "v25_s1_official_validation_report.json",
    )
    final_report = json.loads(
        (official / "v25_s1_official_validation_report.json").read_text(encoding="utf-8")
    )
    state = {
        "status": "complete",
        "mechanism_gate": mechanism_report,
        "official_gate": final_report["gate"],
        "official_gains": final_report["gains_f1_points"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    state_path = root / "v25_s1_pipeline_report.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(state_path), **state}, indent=2), flush=True)


if __name__ == "__main__":
    main()
