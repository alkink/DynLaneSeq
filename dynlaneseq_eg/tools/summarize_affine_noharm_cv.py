"""Summarize repeated sequence-grouped affine no-harm probe runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--feature-name", default="q_ins")
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.input_dir).glob("split_*.json"))
    if not paths:
        raise FileNotFoundError(f"No split_*.json files under {args.input_dir}")

    split_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_seed = int(payload["metadata"]["split_seed"])
        runs = [run for run in payload["runs"] if run["feature"] == args.feature_name]
        if not runs:
            raise RuntimeError(f"No {args.feature_name!r} runs in {path}")
        all_runs.extend(runs)
        row = {
            "split_seed": split_seed,
            "probe_seeds": [int(run["seed"]) for run in runs],
            "calibration_ap": _stats([float(run["calibration"]["ap"]) for run in runs]),
            "calibration_auroc": _stats([float(run["calibration"]["auroc"]) for run in runs]),
            "test_ap": _stats([float(run["test_gate"]["ap"]) for run in runs]),
            "test_auroc": _stats([float(run["test_gate"]["auroc"]) for run in runs]),
            "test_gate_precision": _stats(
                [float(run["test_gate"]["precision"]) for run in runs]
            ),
            "test_gate_recall": _stats([float(run["test_gate"]["recall"]) for run in runs]),
            "official_net_tp": _stats(
                [float(run["test_official"]["net_tp"]) for run in runs]
            ),
            "official_f1_delta": _stats(
                [float(run["test_official"]["f1_delta"]) for run in runs]
            ),
            "positive_net_runs": sum(int(run["test_official"]["net_tp"] > 0) for run in runs),
            "negative_net_runs": sum(int(run["test_official"]["net_tp"] < 0) for run in runs),
            "abstain_runs": sum(int(run["test_official"]["acted"] == 0) for run in runs),
        }
        split_rows.append(row)

    report = {
        "metadata": {
            "input_dir": str(Path(args.input_dir)),
            "feature_name": str(args.feature_name),
            "files": [str(path) for path in paths],
            "sequence_splits": len(split_rows),
            "total_probe_runs": len(all_runs),
        },
        "overall": {
            "calibration_ap": _stats([float(run["calibration"]["ap"]) for run in all_runs]),
            "calibration_auroc": _stats(
                [float(run["calibration"]["auroc"]) for run in all_runs]
            ),
            "test_ap": _stats([float(run["test_gate"]["ap"]) for run in all_runs]),
            "test_auroc": _stats([float(run["test_gate"]["auroc"]) for run in all_runs]),
            "official_net_tp": _stats(
                [float(run["test_official"]["net_tp"]) for run in all_runs]
            ),
            "official_f1_delta": _stats(
                [float(run["test_official"]["f1_delta"]) for run in all_runs]
            ),
            "positive_net_runs": sum(
                int(run["test_official"]["net_tp"] > 0) for run in all_runs
            ),
            "negative_net_runs": sum(
                int(run["test_official"]["net_tp"] < 0) for run in all_runs
            ),
            "zero_net_runs": sum(int(run["test_official"]["net_tp"] == 0) for run in all_runs),
            "abstain_runs": sum(int(run["test_official"]["acted"] == 0) for run in all_runs),
        },
        "splits": split_rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    overall = report["overall"]
    print(
        "sequence CV: "
        f"splits={len(split_rows)} runs={len(all_runs)} "
        f"calAP={overall['calibration_ap']['mean']:.4f}+/-{overall['calibration_ap']['std']:.4f} "
        f"testAP={overall['test_ap']['mean']:.4f}+/-{overall['test_ap']['std']:.4f} "
        f"netTP={overall['official_net_tp']['mean']:.2f} "
        f"positive/negative/zero={overall['positive_net_runs']}/"
        f"{overall['negative_net_runs']}/{overall['zero_net_runs']} "
        f"abstain={overall['abstain_runs']}"
    )
    for row in split_rows:
        print(
            f"split={row['split_seed']}: "
            f"calAP={row['calibration_ap']['mean']:.4f} "
            f"testAP={row['test_ap']['mean']:.4f} "
            f"netTP={row['official_net_tp']['mean']:.2f} "
            f"positive/negative/abstain={row['positive_net_runs']}/"
            f"{row['negative_net_runs']}/{row['abstain_runs']}"
        )
    print(f"output_json: {output_path}")


if __name__ == "__main__":
    main()
