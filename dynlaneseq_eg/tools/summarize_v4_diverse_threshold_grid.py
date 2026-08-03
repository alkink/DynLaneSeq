from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ARM_ORDER = ("source_v4", "c_set_shared", "d_set_unique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the paired V4 threshold/diversity cache grid."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--d", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _f1(row: Mapping[str, Any], threshold: str) -> float:
    return float(row["metrics"][threshold]["f1"])


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    return {
        "key": row["key"],
        "family": row["family"],
        "score_threshold": float(row["score_threshold"]),
        "distance_px": row.get("distance_px"),
        "sigma_px": row.get("sigma_px"),
        "penalty": row.get("penalty"),
        "f1_050": float(metrics["0.50"]["f1"]),
        "precision_050": float(metrics["0.50"]["precision"]),
        "recall_050": float(metrics["0.50"]["recall"]),
        "tp_050": int(metrics["0.50"]["tp"]),
        "fp_050": int(metrics["0.50"]["fp"]),
        "duplicate_fp_050": int(
            metrics["0.50"]["false_positive_breakdown"]["duplicate_fp"][
                "count"
            ]
        ),
        "empty_scene_fp_050": int(
            metrics["0.50"]["false_positive_breakdown"]["empty_scene_fp"][
                "count"
            ]
        ),
        "f1_075": float(metrics["0.75"]["f1"]),
        "precision_075": float(metrics["0.75"]["precision"]),
        "recall_075": float(metrics["0.75"]["recall"]),
        "mean_f1": float(row["objectives"]["mean_f1"]),
    }


def summarize(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = [name for name in ARM_ORDER if name not in reports]
    if missing:
        raise ValueError("missing grid reports: " + ", ".join(missing))

    first = reports[ARM_ORDER[0]]
    reference_meta = first["metadata"]
    reference_indices = reference_meta.get("sampled_dataset_indices")
    reference_list_hash = reference_meta.get("list_sha256")
    reference_oracle = first["all_candidate_oracle"]
    paired_checks: dict[str, dict[str, bool]] = {}
    for name in ARM_ORDER:
        report = reports[name]
        metadata = report["metadata"]
        paired_checks[name] = {
            "same_sample_indices": (
                metadata.get("sampled_dataset_indices") == reference_indices
            ),
            "same_list_sha256": metadata.get("list_sha256") == reference_list_hash,
            "same_geometry_oracle": report["all_candidate_oracle"]
            == reference_oracle,
        }
    paired = all(all(row.values()) for row in paired_checks.values())
    if not paired:
        raise ValueError("grid arms do not share one paired geometry/sample contract")

    arms: dict[str, Any] = {}
    for name in ARM_ORDER:
        report = reports[name]
        family = report["best_by_family"]
        zero_raw = next(
            row
            for row in report["rows"]
            if row["family"] == "score_topk"
            and abs(float(row["score_threshold"])) < 1e-12
        )
        best_score = family["score_topk"]["f1_050"]
        best_hard = family["hard_diversity"]["f1_050"]
        best_mmr = family["mmr"]["f1_050"]
        best_diverse = report["best_overall_diversity"]["f1_050"]
        arms[name] = {
            "score_mode": report["metadata"].get("score_mode"),
            "zero_threshold_score_topk": _compact(zero_raw),
            "best_thresholded_score_topk": _compact(best_score),
            "best_thresholded_hard_diversity": _compact(best_hard),
            "best_thresholded_mmr": _compact(best_mmr),
            "best_diversity": _compact(best_diverse),
            "best_mean_f1_diversity": _compact(
                report["best_overall_diversity"]["mean_f1"]
            ),
            "effects": {
                "threshold_only_f1_050_gain": _f1(best_score, "0.50")
                - _f1(zero_raw, "0.50"),
                "diversity_beyond_best_threshold_f1_050_gain": _f1(
                    best_diverse, "0.50"
                )
                - _f1(best_score, "0.50"),
                "total_f1_050_gain": _f1(best_diverse, "0.50")
                - _f1(zero_raw, "0.50"),
            },
        }

    best_primary_arm = max(
        ARM_ORDER,
        key=lambda name: (
            arms[name]["best_diversity"]["f1_050"],
            arms[name]["best_diversity"]["f1_075"],
        ),
    )
    best_balanced_arm = max(
        ARM_ORDER,
        key=lambda name: (
            arms[name]["best_mean_f1_diversity"]["mean_f1"],
            arms[name]["best_mean_f1_diversity"]["f1_050"],
        ),
    )
    best_primary = arms[best_primary_arm]["best_diversity"]
    clears_primary = float(best_primary["f1_050"]) >= 0.80
    verdict = (
        "cached_teacher_clears_primary_subset_gate_build_structured_selector"
        if clears_primary
        else "diversity_is_causal_but_teacher_below_primary_subset_gate"
    )
    return {
        "diagnostic_only": True,
        "warning": (
            "Grid winners were selected on one uniform validation subset. "
            "They are architecture-teacher settings, not official deployment "
            "parameters."
        ),
        "paired_contract": {
            "passed": paired,
            "checks": paired_checks,
            "all_candidate_oracle": reference_oracle,
        },
        "arms": arms,
        "best_primary_arm": best_primary_arm,
        "best_primary_teacher": best_primary,
        "best_balanced_arm": best_balanced_arm,
        "best_balanced_teacher": arms[best_balanced_arm][
            "best_mean_f1_diversity"
        ],
        "primary_subset_gate_f1_050": 0.80,
        "primary_subset_gate_passed": clears_primary,
        "verdict": verdict,
    }


def main() -> None:
    args = parse_args()
    payload = summarize(
        {
            "source_v4": _load(args.source),
            "c_set_shared": _load(args.c),
            "d_set_unique": _load(args.d),
        }
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
