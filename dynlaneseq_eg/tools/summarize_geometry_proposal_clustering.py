from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAIR_PRECISION_MIN = 0.98
PAIR_RECALL_MIN = 0.50
CATASTROPHIC_CLUSTER_FRACTION_MAX = 0.02
MULTI_MEMBER_CANDIDATE_FRACTION_MIN = 0.50
ORACLE_TP_RETENTION_MIN = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock a geometry-cluster policy on train calibration and verify it on validation."
    )
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--validation-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _structural_pass(row: dict[str, Any]) -> bool:
    return bool(
        float(row["pair_merge_precision"]) >= PAIR_PRECISION_MIN
        and float(row["same_gt_pair_recall"]) >= PAIR_RECALL_MIN
        and float(row["catastrophic_cluster_fraction"])
        <= CATASTROPHIC_CLUSTER_FRACTION_MAX
        and float(row["candidate_multi_member_fraction"])
        >= MULTI_MEMBER_CANDIDATE_FRACTION_MIN
    )


def _metric(report: dict[str, Any], key: str) -> dict[str, Any]:
    value = report.get("official_metrics", {}).get(key)
    if not isinstance(value, dict):
        raise KeyError(f"missing metric {key!r}")
    return value


def _prototype_score(report: dict[str, Any], policy: str, prototype: str) -> float:
    values = [
        float(
            _metric(
                report,
                f"{policy}/{prototype}/oracle_same_count/{threshold}",
            ).get("all32_tp_retention", 0.0)
        )
        for threshold in ("0.50", "0.75")
    ]
    return min(values)


def _select_policy(calibration: dict[str, Any]) -> tuple[str, bool]:
    rows = calibration["structural_clustering"]
    eligible = [name for name, row in rows.items() if _structural_pass(row)]
    if eligible:
        selected = max(
            eligible,
            key=lambda name: (
                float(rows[name]["same_gt_pair_recall"]),
                float(rows[name]["pair_merge_precision"]),
                -float(rows[name]["catastrophic_cluster_fraction"]),
                name,
            ),
        )
        return selected, True
    # A failed calibration still names the least-bad policy for transparent
    # validation reporting; it does not authorize an architecture.
    selected = max(
        rows,
        key=lambda name: (
            float(rows[name]["pair_merge_precision"]),
            float(rows[name]["same_gt_pair_recall"]),
            -float(rows[name]["catastrophic_cluster_fraction"]),
            name,
        ),
    )
    return selected, False


def _select_prototype(calibration: dict[str, Any], policy: str) -> str:
    priority = {"medoid": 2, "median": 1, "mean": 0}
    return max(
        ("medoid", "median", "mean"),
        key=lambda mode: (_prototype_score(calibration, policy, mode), priority[mode]),
    )


def summarize(calibration: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if calibration.get("scope", {}).get("test_set_used") or validation.get("scope", {}).get(
        "test_set_used"
    ):
        raise ValueError("test results are forbidden in the clustering feasibility audit")
    selected_policy, calibration_structural_pass = _select_policy(calibration)
    selected_prototype = _select_prototype(calibration, selected_policy)
    validation_row = validation["structural_clustering"][selected_policy]
    validation_structural_pass = _structural_pass(validation_row)
    retention = {
        threshold: float(
            _metric(
                validation,
                f"{selected_policy}/{selected_prototype}/oracle_same_count/{threshold}",
            ).get("all32_tp_retention", 0.0)
        )
        for threshold in ("0.50", "0.75")
    }
    prototype_capacity_pass = all(
        value >= ORACLE_TP_RETENTION_MIN for value in retention.values()
    )
    passed = bool(
        calibration_structural_pass
        and validation_structural_pass
        and prototype_capacity_pass
    )
    bottom_cal = calibration["bottom_guard_ablation"]
    bottom_val = validation["bottom_guard_ablation"]
    return {
        "experiment": "geometry proposal clustering calibration/validation decision",
        "predeclared_gates": {
            "pair_merge_precision_min": PAIR_PRECISION_MIN,
            "same_gt_pair_recall_min": PAIR_RECALL_MIN,
            "catastrophic_cluster_fraction_max": CATASTROPHIC_CLUSTER_FRACTION_MAX,
            "candidate_multi_member_fraction_min": MULTI_MEMBER_CANDIDATE_FRACTION_MIN,
            "same_count_oracle_tp_retention_min": ORACLE_TP_RETENTION_MIN,
        },
        "selection_contract": {
            "policy_selected_on_calibration_only": True,
            "prototype_selected_on_calibration_only": True,
            "selected_policy": selected_policy,
            "selected_prototype": selected_prototype,
            "checkpoint_selection_performed": False,
            "threshold_search_performed": False,
        },
        "calibration": {
            "path": calibration.get("metadata", {}).get("cache_path", ""),
            "structural": calibration["structural_clustering"][selected_policy],
            "structural_pass": calibration_structural_pass,
            "prototype_min_retention": _prototype_score(
                calibration, selected_policy, selected_prototype
            ),
            "bottom_guard": bottom_cal,
        },
        "validation": {
            "path": validation.get("metadata", {}).get("cache_path", ""),
            "structural": validation_row,
            "structural_pass": validation_structural_pass,
            "prototype_retention": retention,
            "prototype_capacity_pass": prototype_capacity_pass,
            "bottom_guard": bottom_val,
        },
        "passed": passed,
        "decision": (
            "geometry_clustering_supported_stop_and_plan_next_architecture"
            if passed
            else "geometry_clustering_not_supported_as_current_next_architecture"
        ),
        "new_model_version_authorized": False,
        "long_training_authorized": False,
        "test_set_used": False,
        "required_next_action": "stop_for_user_and_sol_review",
    }


def write_markdown(path: str, report: dict[str, Any]) -> None:
    policy = report["selection_contract"]["selected_policy"]
    prototype = report["selection_contract"]["selected_prototype"]
    cal = report["calibration"]
    val = report["validation"]
    lines = [
        "# Geometry proposal clustering decision",
        "",
        f"**Formal result: {'PASS' if report['passed'] else 'FAIL'}**",
        "",
        f"Calibration-selected policy: `{policy}`  ",
        f"Calibration-selected prototype: `{prototype}`",
        "",
        (
            "| Domain | Pair precision | Same-lane pair recall | Catastrophic "
            "clusters | Multi-member candidate fraction | Structural gate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, row in (("Calibration", cal), ("Validation", val)):
        structural = row["structural"]
        lines.append(
            f"| {label} | {structural['pair_merge_precision']:.4f} | "
            f"{structural['same_gt_pair_recall']:.4f} | "
            f"{structural['catastrophic_cluster_fraction']:.4f} | "
            f"{structural['candidate_multi_member_fraction']:.4f} | "
            f"{'PASS' if row['structural_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Validation prototype retention against the all-32 same-count oracle:",
            "",
            f"- IoU .50: `{val['prototype_retention']['0.50']:.4f}`",
            f"- IoU .75: `{val['prototype_retention']['0.75']:.4f}`",
            "",
            "Bottom-separation check:",
            "",
            (
                "- Calibration different-GT precision among labeled removed pairs: "
                f"`{cal['bottom_guard']['different_gt_precision_among_labeled_removed']:.4f}`"
            ),
            (
                "- Validation different-GT precision among labeled removed pairs: "
                f"`{val['bottom_guard']['different_gt_precision_among_labeled_removed']:.4f}`"
            ),
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                "This report does not authorize a new V version or long training. "
                "Stop and review with the user and Sol Pro."
            ),
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = summarize(_load(args.calibration_json), _load(args.validation_json))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps({
        "passed": report["passed"],
        "decision": report["decision"],
        "output_json": str(output.resolve()),
        "output_md": str(Path(args.output_md).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
