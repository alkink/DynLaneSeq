from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_spec(value: str) -> tuple[int, Path]:
    iteration, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("report must be ITERATION=PATH")
    return int(iteration), Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the paired V7 legacy/resume-safe data gate."
    )
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--legacy-report", action="append", type=_parse_spec, default=[])
    parser.add_argument("--fixed-report", action="append", type=_parse_spec, default=[])
    parser.add_argument("--preflight-contract", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(iteration: int, path: Path) -> dict[str, Any]:
    report = _load(path)
    methods = report["methods"]
    refined = methods.get("four_slot_refined") or methods[
        "four_slot_global_unique"
    ]
    slot = report["four_slot_diagnostics"]
    cardinality = slot.get("cardinality", {})
    row: dict[str, Any] = {
        "iteration": int(iteration),
        "report": str(path),
        "oracle_recall_050": float(
            report["capacity"]["0.50"]["all_candidate_oracle"]["recall"]
        ),
        "oracle_recall_075": float(
            report["capacity"]["0.75"]["all_candidate_oracle"]["recall"]
        ),
        "semantic_duplicate": float(
            slot["semantic_duplicate_cluster_fraction"]
        ),
        "repair_fraction": float(slot["global_assignment_repair_fraction"]),
        "cardinality_exact": float(cardinality.get("exact_fraction", 0.0)),
        "cardinality_mae": float(cardinality.get("mean_absolute_error", 4.0)),
    }
    for threshold, suffix in (("0.50", "050"), ("0.75", "075")):
        metrics = refined[threshold]
        row[f"f1_{suffix}"] = float(metrics["f1"])
        row[f"precision_{suffix}"] = float(metrics["precision"])
        row[f"recall_{suffix}"] = float(metrics["recall"])
        row[f"tp_{suffix}"] = int(metrics["tp"])
        row[f"fp_{suffix}"] = int(metrics["fp"])
        row[f"fn_{suffix}"] = int(metrics["fn"])
    row["mean_selected"] = float(refined["0.50"]["mean_selected_per_image"])
    return row


def _trajectory(specs: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    return [_row(iteration, path) for iteration, path in sorted(specs)]


def summarize(
    source_path: Path,
    legacy: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if [row["iteration"] for row in legacy] != [130000, 135000]:
        raise ValueError("legacy reports must be exactly 130000 and 135000")
    if [row["iteration"] for row in fixed] != [130000, 135000]:
        raise ValueError("fixed reports must be exactly 130000 and 135000")
    source_row = _row(125000, source_path)
    legacy_final = legacy[-1]
    fixed_final = fixed[-1]
    fixed_minus_legacy = {
        key: float(fixed_final[key]) - float(legacy_final[key])
        for key in (
            "f1_050",
            "f1_075",
            "precision_050",
            "recall_050",
            "oracle_recall_050",
            "oracle_recall_075",
            "mean_selected",
            "cardinality_mae",
            "semantic_duplicate",
        )
    }
    legacy_tail_change = float(legacy[-1]["f1_050"]) - float(
        legacy[0]["f1_050"]
    )
    fixed_tail_change = float(fixed[-1]["f1_050"]) - float(
        fixed[0]["f1_050"]
    )
    integrity_checks = {
        "preflight_passed": bool(preflight.get("passed", False)),
        "fixed_oracle_050_no_collapse": float(fixed_final["oracle_recall_050"])
        >= float(source_row["oracle_recall_050"]) - 0.02,
        "fixed_oracle_075_no_collapse": float(fixed_final["oracle_recall_075"])
        >= float(source_row["oracle_recall_075"]) - 0.03,
        "fixed_cardinality_valid": 2.5
        <= float(fixed_final["mean_selected"])
        <= 3.8,
        "fixed_semantic_duplicates_bounded": float(
            fixed_final["semantic_duplicate"]
        )
        <= 0.05,
        "fixed_global_repair_bounded": float(fixed_final["repair_fraction"])
        <= 0.05,
    }
    margin = float(fixed_minus_legacy["f1_050"])
    if not all(integrity_checks.values()):
        verdict = "fixed_contract_unhealthy"
        next_action = "stop_and_audit_fixed_data_arm"
    elif margin >= 0.002:
        verdict = "fixed_preferred"
        next_action = "continue_resume_safe_only_to_150k_gate"
    elif margin <= -0.005:
        verdict = "legacy_metric_preferred_but_resume_bug_confirmed"
        next_action = "do_not_merge_performance_claim; inspect_data_distribution_shift"
    else:
        verdict = "metric_indistinguishable_fixed_contract_preferred"
        next_action = "adopt_resume_safe_for_future_reproducibility"
    return {
        "experiment": "V7 125k-to-135k legacy versus resume-safe data gate",
        "diagnostic_only": True,
        "source": source_row,
        "legacy": legacy,
        "resume_safe": fixed,
        "fixed_minus_legacy_at_135k": fixed_minus_legacy,
        "resume_boundary_130k_to_135k": {
            "legacy_f1_050_change": legacy_tail_change,
            "fixed_f1_050_change": fixed_tail_change,
            "fixed_minus_legacy_change": fixed_tail_change
            - legacy_tail_change,
        },
        "integrity_checks": integrity_checks,
        "integrity_passed": all(integrity_checks.values()),
        "verdict": verdict,
        "next_action": next_action,
        "full_validation_authorized": False,
        "test_split_closed": True,
        "interpretation_note": (
            "The 125k source was trained with the legacy worker RNG contract. "
            "The fixed arm guarantees exact future resumes from 125k onward; "
            "it cannot reconstruct augmentation states that the old checkpoint "
            "never stored."
        ),
        "preflight_contract": preflight,
    }


def main() -> None:
    args = _parse_args()
    payload = summarize(
        Path(args.source_report),
        _trajectory(args.legacy_report),
        _trajectory(args.fixed_report),
        _load(args.preflight_contract),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
