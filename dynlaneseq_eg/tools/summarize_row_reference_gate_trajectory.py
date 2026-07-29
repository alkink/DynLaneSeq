from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired row-reference gate results across checkpoints."
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not bool(payload.get("diagnostic_only", False)):
        raise ValueError(f"{path} is not marked diagnostic_only")
    return payload


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    control = payload["control"]
    candidate = payload["candidate"]
    summary = payload["summary"]
    if int(control["iteration"]) != int(candidate["iteration"]):
        raise ValueError("control and candidate checkpoint iterations differ")
    if control["sampled_dataset_indices"] != candidate["sampled_dataset_indices"]:
        raise ValueError("control and candidate did not use the same images")

    row: dict[str, Any] = {
        "iteration": int(control["iteration"]),
        "images": int(payload["images"]),
        "lanes": int(summary["lanes"]),
        "control_mean_best_iou": float(summary["control_mean_best_iou"]),
        "candidate_mean_best_iou": float(summary["candidate_mean_best_iou"]),
        "mean_best_iou_gain": float(summary["mean_best_iou_gain"]),
    }
    for threshold_key, short_name in (("0.50", "050"), ("0.70", "070")):
        metrics = summary["thresholds"][threshold_key]
        control_recall = float(metrics["control_all_recall"])
        candidate_recall = float(metrics["candidate_all_recall"])
        control_hits = round(control_recall * int(summary["lanes"]))
        lost = int(metrics["control_hits_lost"])
        row.update(
            {
                f"control_all_recall_{short_name}": control_recall,
                f"candidate_all_recall_{short_name}": candidate_recall,
                f"gain_points_{short_name}": float(metrics["candidate_gain_points"]),
                f"image_specificity_points_{short_name}": float(
                    metrics["candidate_image_specificity_points"]
                ),
                f"control_top4_recall_{short_name}": float(
                    metrics["control_scored_topk_recall"]
                ),
                f"candidate_top4_recall_{short_name}": float(
                    metrics["candidate_scored_topk_recall"]
                ),
                f"recovered_{short_name}": int(
                    metrics["unique_control_misses_recovered"]
                ),
                f"lost_{short_name}": lost,
                f"net_unique_hits_{short_name}": int(metrics["net_unique_hits"]),
                f"control_hit_retention_{short_name}": (
                    float(control_hits - lost) / float(control_hits)
                    if control_hits > 0
                    else 1.0
                ),
            }
        )
    return row


def summarize(paths: list[str]) -> dict[str, Any]:
    payloads = [_load(path) for path in paths]
    rows = sorted((_flatten(payload) for payload in payloads), key=lambda row: row["iteration"])
    if not rows:
        raise ValueError("at least one gate result is required")

    reference_indices = payloads[0]["control"]["sampled_dataset_indices"]
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["control"]["sampled_dataset_indices"] != reference_indices:
            raise ValueError(f"{path} used a different validation sample")
        if int(payload["summary"]["lanes"]) != int(payloads[0]["summary"]["lanes"]):
            raise ValueError(f"{path} produced a different lane count")

    checks = {
        "positive_raw_recall_gain_050_at_every_checkpoint": all(
            float(row["gain_points_050"]) > 0.0 for row in rows
        ),
        "positive_top4_recall_gain_050_at_every_checkpoint": all(
            float(row["candidate_top4_recall_050"])
            > float(row["control_top4_recall_050"])
            for row in rows
        ),
        "positive_image_specificity_050_at_every_checkpoint": all(
            float(row["image_specificity_points_050"]) > 5.0 for row in rows
        ),
        "positive_mean_iou_gain_at_every_checkpoint": all(
            float(row["mean_best_iou_gain"]) > 0.0 for row in rows
        ),
        "positive_net_unique_hits_050_at_every_checkpoint": all(
            int(row["net_unique_hits_050"]) > 0 for row in rows
        ),
    }
    return {
        "diagnostic_only": True,
        "warning": (
            "Checkpoint trajectory of raw proposal geometry on a fixed validation "
            "subset; this is not an official CULane F1 result."
        ),
        "sample_strategy": payloads[0]["sample_strategy"],
        "sampled_dataset_indices": reference_indices,
        "trajectory": rows,
        "trend_checks": checks,
        "stable_positive_trajectory": all(checks.values()),
        "source_files": paths,
    }


def main() -> None:
    args = parse_args()
    result = summarize(args.inputs)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = result["trajectory"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result, indent=2))
    print(f"output_json: {output_json}")
    print(f"output_csv: {output_csv}")


if __name__ == "__main__":
    main()
