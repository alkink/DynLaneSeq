from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize matched control/candidate row-reference precision audits."
    )
    parser.add_argument("--control-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--lockin-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _comparability(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = control["metadata"]
    right = candidate["metadata"]
    keys = (
        "split",
        "list_sha256",
        "max_batches",
        "num_records",
        "sample_strategy",
        "sampled_dataset_indices",
        "iou_space",
        "top_k",
        "iou_thresholds",
        "near_min_iou",
        "nms_distance",
        "nms_min_overlap_points",
    )
    checks = {key: left.get(key) == right.get(key) for key in keys}
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"precision reports are not matched: {failed}")
    return {"all_checks_pass": True, "checks": checks}


def _metric_delta(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(control[key])
        for key in ("precision", "recall", "f1")
    }


def main() -> None:
    args = parse_args()
    control = _load(args.control_json)
    candidate = _load(args.candidate_json)
    lockin = _load(args.lockin_json)
    comparison = _comparability(control, candidate)
    thresholds: dict[str, Any] = {}
    for threshold in sorted(control["thresholds"], key=float):
        left = control["thresholds"][threshold]
        right = candidate["thresholds"][threshold]
        labels = (
            "duplicate_fp",
            "near_miss_fp",
            "background_fp",
            "empty_scene_fp",
        )
        thresholds[threshold] = {
            "control_metric": left["metric"],
            "candidate_metric": right["metric"],
            "candidate_minus_control": _metric_delta(left["metric"], right["metric"]),
            "control_oracle_ladder": left["oracle_ladder"],
            "candidate_oracle_ladder": right["oracle_ladder"],
            "candidate_minus_control_fp_counts": {
                label: int(right["false_positive_breakdown"][label]["count"])
                - int(left["false_positive_breakdown"][label]["count"])
                for label in labels
            },
            "control_fp_breakdown": left["false_positive_breakdown"],
            "candidate_fp_breakdown": right["false_positive_breakdown"],
        }

    payload = {
        "diagnostic_only": True,
        "comparability": comparison,
        "lockin_gate": lockin["gate"],
        "thresholds": thresholds,
        "inputs": {
            "control_json": args.control_json,
            "candidate_json": args.candidate_json,
            "lockin_json": args.lockin_json,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
