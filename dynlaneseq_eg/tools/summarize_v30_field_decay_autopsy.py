from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ITERATIONS = (35000, 40000, 45000, 50000)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the exact-paired V30 Field-only decay autopsy."
    )
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--geometry-drift", required=True)
    parser.add_argument("--gradient-35k", required=True)
    parser.add_argument("--gradient-50k", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    row = report["results"][threshold]
    return {
        "f1": float(row["F1"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "tp": int(row["TP"]),
        "fp": int(row["FP"]),
        "fn": int(row["FN"]),
    }


def _oracle(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    row = report["capacity"][threshold]["all_candidate_oracle"]
    return {
        "hits": int(row["hits"]),
        "gt": int(row["gt"]),
        "recall": float(row["recall"]),
    }


def _gradient_focus(report: dict[str, Any]) -> dict[str, Any]:
    pairs = report["summary"]["pairs"]
    output: dict[str, Any] = {}
    for pair_name in (
        "field_vs_legacy",
        "field_vs_four_slot_geometry",
        "field_vs_proposal",
    ):
        output[pair_name] = {
            group: pairs[pair_name][group]
            for group in (
                "backbone_all",
                "backbone_early_p2",
                "backbone_mid_p3",
                "backbone_deep_p4_p5",
                "fpn_all",
                "fpn_p2",
                "fpn_p3_p5",
                "row_image_projection",
                "proposal_row_transformer",
                "slot_decoder",
            )
        }
    return output


def main() -> None:
    args = _parse_args()
    root = Path(args.trajectory_root)
    trajectory: list[dict[str, Any]] = []
    for iteration in ITERATIONS:
        directory = root / f"iter_{iteration:07d}"
        v7_metrics = _load(directory / "v7" / "metrics.json")
        v30_metrics = _load(directory / "v30" / "metrics.json")
        v7_coverage = _load(directory / "v7_uniform256.json")
        v30_coverage = _load(directory / "v30_uniform256.json")
        paired = _load(directory / "paired.json")
        thresholds: dict[str, Any] = {}
        for threshold in ("0.5", "0.75"):
            source = _metric(v7_metrics, threshold)
            candidate = _metric(v30_metrics, threshold)
            bootstrap = paired["thresholds"][threshold]["paired_clip_bootstrap"]
            source_oracle = _oracle(v7_coverage, f"{float(threshold):.2f}")
            candidate_oracle = _oracle(v30_coverage, f"{float(threshold):.2f}")
            thresholds[threshold] = {
                "v7": source,
                "v30_field_only": candidate,
                "v30_minus_v7_f1_points": 100.0
                * (candidate["f1"] - source["f1"]),
                "v30_minus_v7_tp": candidate["tp"] - source["tp"],
                "paired_clip_bootstrap": bootstrap,
                "all32_oracle": {
                    "v7": source_oracle,
                    "v30_field_only": candidate_oracle,
                    "v30_minus_v7_hits": (
                        candidate_oracle["hits"] - source_oracle["hits"]
                    ),
                },
            }
        trajectory.append({"iteration": iteration, "thresholds": thresholds})

    geometry = _load(args.geometry_drift)
    gradient_35 = _load(args.gradient_35k)
    gradient_50 = _load(args.gradient_50k)
    delta_050 = [row["thresholds"]["0.5"]["v30_minus_v7_f1_points"] for row in trajectory]
    delta_075 = [row["thresholds"]["0.75"]["v30_minus_v7_f1_points"] for row in trajectory]
    best_050_index = max(range(len(trajectory)), key=lambda index: delta_050[index])
    best_075_index = max(range(len(trajectory)), key=lambda index: delta_075[index])
    final_050 = trajectory[-1]["thresholds"]["0.5"]
    final_075 = trajectory[-1]["thresholds"]["0.75"]
    checks = {
        "all_endpoints_present": len(trajectory) == len(ITERATIONS),
        "field_050_positive_at_50k": final_050["v30_minus_v7_f1_points"] > 0.0,
        "field_075_non_regression_at_50k": final_075["v30_minus_v7_f1_points"] >= 0.0,
        "field_050_clip_ci_lower_positive_at_50k": float(
            final_050["paired_clip_bootstrap"]["ci_2p5"]
        )
        > 0.0,
        "field_075_clip_ci_lower_positive_at_50k": float(
            final_075["paired_clip_bootstrap"]["ci_2p5"]
        )
        > 0.0,
        "strict_gain_did_not_decay_from_peak": (
            delta_075[-1] >= delta_075[best_075_index] - 0.10
        ),
        "test_split_closed": True,
    }
    passed = all(checks.values())
    payload = {
        "experiment": "V30 Field-only exact-paired 35K-50K decay autopsy",
        "trajectory": trajectory,
        "peak": {
            "0.5": {
                "iteration": trajectory[best_050_index]["iteration"],
                "gain_points": delta_050[best_050_index],
            },
            "0.75": {
                "iteration": trajectory[best_075_index]["iteration"],
                "gain_points": delta_075[best_075_index],
            },
        },
        "geometry_drift": geometry,
        "gradient_conflict": {
            "35k": _gradient_focus(gradient_35),
            "50k": _gradient_focus(gradient_50),
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "field_only_mechanism_persists_authorize_minimal_gradient_routing_ablation"
            if passed
            else "do_not_extend_field_only_blindly_use_autopsy_to_design_one_minimal_fix"
        ),
        "test_split_used": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
