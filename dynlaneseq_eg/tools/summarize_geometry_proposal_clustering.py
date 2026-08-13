from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_POLICY = "perspective_balanced_48"
PRIMARY_PROTOTYPE = "medoid"
PRIMARY_SELECTION = "slot_cluster_mass_unique"
POLICIES = (
    "flat_complete_48",
    "perspective_balanced_48",
    "perspective_conservative_36",
)
PROTOTYPES = ("medoid", "median", "mean")
DEPLOYABLE_SELECTIONS = (
    "slot_cluster_mass_unique",
    "routed_consensus",
    "score_topk",
)

PAIR_PRECISION_MIN = 0.98
PAIR_RECALL_MIN = 0.50
CATASTROPHIC_CLUSTER_FRACTION_MAX = 0.02
MULTI_MEMBER_CANDIDATE_FRACTION_MIN = 0.50
ORACLE_TP_RETENTION_MIN = 0.95
CALIBRATION_DELTA_TP_MIN = {"0.50": 0, "0.75": 0}
VALIDATION_DELTA_TP_MIN = {"0.50": 3, "0.75": 0}
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the predeclared GT-free bottom-aware cluster-mass medoid "
            "policy. Secondary policies are reported only as diagnostics and "
            "cannot retroactively pass the confirmatory gate."
        )
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


def _deploy_delta(
    report: dict[str, Any],
    *,
    policy: str,
    prototype: str,
    selection: str,
    threshold: str,
) -> dict[str, Any]:
    baseline = _metric(report, f"current_v7_refined/{threshold}")
    treatment = _metric(
        report,
        f"{policy}/{prototype}/{selection}/{threshold}",
    )
    return {
        "baseline": baseline,
        "treatment": treatment,
        "delta_tp": int(treatment["tp"]) - int(baseline["tp"]),
        "delta_f1_points": 100.0
        * (float(treatment["f1"]) - float(baseline["f1"])),
        "exact_count_parity": bool(treatment.get("exact_count_parity", False))
        and int(treatment["predictions"]) == int(baseline["predictions"]),
    }


def _primary_delta(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    return _deploy_delta(
        report,
        policy=PRIMARY_POLICY,
        prototype=PRIMARY_PROTOTYPE,
        selection=PRIMARY_SELECTION,
        threshold=threshold,
    )


def _prototype_retention(report: dict[str, Any], threshold: str) -> float:
    return float(
        _metric(
            report,
            f"{PRIMARY_POLICY}/{PRIMARY_PROTOTYPE}/oracle_same_count/{threshold}",
        ).get("all32_tp_retention", 0.0)
    )


def _f1(tp: int, predictions: int, gt: int) -> float:
    denominator = int(predictions) + int(gt)
    return 0.0 if denominator <= 0 else float(2 * int(tp)) / float(denominator)


def _paired_bootstrap(report: dict[str, Any], threshold: str) -> dict[str, Any]:
    rows = report.get("per_image", [])
    if not rows:
        raise ValueError("paired bootstrap requires per_image rows")
    baseline_tp = np.asarray(
        [row["current_v7_refined"][f"hits_{threshold}"] for row in rows],
        dtype=np.int64,
    )
    treatment_tp = np.asarray(
        [
            row["policies"][PRIMARY_POLICY]["prototypes"][PRIMARY_PROTOTYPE][
                f"{PRIMARY_SELECTION}_hits_{threshold}"
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    baseline_predictions = np.asarray(
        [row["current_writer_count"] for row in rows], dtype=np.int64
    )
    treatment_predictions = np.asarray(
        [
            row["policies"][PRIMARY_POLICY]["prototypes"][PRIMARY_PROTOTYPE][
                f"{PRIMARY_SELECTION}_predictions_{threshold}"
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    gt = np.asarray([row["gt_count"] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(round(float(threshold) * 100)))
    deltas = np.empty((BOOTSTRAP_REPLICATES,), dtype=np.float64)
    image_count = int(len(rows))
    for replicate in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, image_count, size=image_count)
        baseline_f1 = _f1(
            int(baseline_tp[sample].sum()),
            int(baseline_predictions[sample].sum()),
            int(gt[sample].sum()),
        )
        treatment_f1 = _f1(
            int(treatment_tp[sample].sum()),
            int(treatment_predictions[sample].sum()),
            int(gt[sample].sum()),
        )
        deltas[replicate] = 100.0 * (treatment_f1 - baseline_f1)
    observed = 100.0 * (
        _f1(int(treatment_tp.sum()), int(treatment_predictions.sum()), int(gt.sum()))
        - _f1(int(baseline_tp.sum()), int(baseline_predictions.sum()), int(gt.sum()))
    )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "observed_delta_f1_points": observed,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_positive": float((deltas > 0.0).mean()),
    }


def _secondary_diagnostics(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for prototype in PROTOTYPES:
            for selection in DEPLOYABLE_SELECTIONS:
                row = {
                    "policy": policy,
                    "prototype": prototype,
                    "selection": selection,
                    "confirmatory_primary": bool(
                        policy == PRIMARY_POLICY
                        and prototype == PRIMARY_PROTOTYPE
                        and selection == PRIMARY_SELECTION
                    ),
                }
                try:
                    row["0.50"] = _deploy_delta(
                        report,
                        policy=policy,
                        prototype=prototype,
                        selection=selection,
                        threshold="0.50",
                    )
                    row["0.75"] = _deploy_delta(
                        report,
                        policy=policy,
                        prototype=prototype,
                        selection=selection,
                        threshold="0.75",
                    )
                except KeyError:
                    continue
                rows.append(row)
    return rows


def summarize(calibration: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if calibration.get("scope", {}).get("test_set_used") or validation.get(
        "scope", {}
    ).get("test_set_used"):
        raise ValueError("test results are forbidden in the clustering feasibility audit")
    for label, report in (("calibration", calibration), ("validation", validation)):
        scope = report.get("scope", {})
        if scope.get("optimizer_steps") not in (0, None):
            raise ValueError(f"{label} unexpectedly performed optimizer steps")
        contract = report.get("primary_confirmatory_contract", {})
        expected = (PRIMARY_POLICY, PRIMARY_PROTOTYPE, PRIMARY_SELECTION)
        observed = (
            contract.get("policy"),
            contract.get("prototype"),
            contract.get("selection"),
        )
        if observed != expected:
            raise ValueError(f"{label} primary contract mismatch: {observed} != {expected}")

    calibration_structural = calibration["structural_clustering"][PRIMARY_POLICY]
    validation_structural = validation["structural_clustering"][PRIMARY_POLICY]
    calibration_structural_pass = _structural_pass(calibration_structural)
    validation_structural_pass = _structural_pass(validation_structural)
    calibration_delta = {
        threshold: _primary_delta(calibration, threshold)
        for threshold in ("0.50", "0.75")
    }
    validation_delta = {
        threshold: _primary_delta(validation, threshold)
        for threshold in ("0.50", "0.75")
    }
    validation_retention = {
        threshold: _prototype_retention(validation, threshold)
        for threshold in ("0.50", "0.75")
    }
    calibration_deploy_pass = all(
        row["exact_count_parity"]
        and int(row["delta_tp"]) >= CALIBRATION_DELTA_TP_MIN[threshold]
        for threshold, row in calibration_delta.items()
    )
    validation_deploy_pass = all(
        row["exact_count_parity"]
        and int(row["delta_tp"]) >= VALIDATION_DELTA_TP_MIN[threshold]
        for threshold, row in validation_delta.items()
    )
    prototype_capacity_pass = all(
        value >= ORACLE_TP_RETENTION_MIN for value in validation_retention.values()
    )
    passed = bool(
        calibration_structural_pass
        and validation_structural_pass
        and calibration_deploy_pass
        and validation_deploy_pass
        and prototype_capacity_pass
    )
    return {
        "experiment": "confirmatory training-free bottom-aware proposal clustering",
        "formal_result": "PASS" if passed else "FAIL",
        "predeclared_primary": {
            "policy": PRIMARY_POLICY,
            "prototype": PRIMARY_PROTOTYPE,
            "selection": PRIMARY_SELECTION,
            "interpretation": (
                "sum each active V7 slot's proposal probabilities inside each "
                "GT-free geometry cluster, assign distinct clusters, emit the medoid"
            ),
        },
        "predeclared_gates": {
            "pair_merge_precision_min": PAIR_PRECISION_MIN,
            "same_gt_pair_recall_min": PAIR_RECALL_MIN,
            "catastrophic_cluster_fraction_max": CATASTROPHIC_CLUSTER_FRACTION_MAX,
            "candidate_multi_member_fraction_min": MULTI_MEMBER_CANDIDATE_FRACTION_MIN,
            "same_count_oracle_tp_retention_min": ORACLE_TP_RETENTION_MIN,
            "calibration_delta_tp_min": CALIBRATION_DELTA_TP_MIN,
            "validation_delta_tp_min": VALIDATION_DELTA_TP_MIN,
        },
        "selection_contract": {
            "primary_fixed_before_results": True,
            "secondary_policy_selection_performed": False,
            "checkpoint_selection_performed": False,
            "threshold_search_performed": False,
            "test_set_used": False,
        },
        "calibration": {
            "structural": calibration_structural,
            "structural_pass": calibration_structural_pass,
            "primary_deployable": calibration_delta,
            "deployable_pass": calibration_deploy_pass,
            "bottom_guard": calibration["bottom_guard_ablation"],
        },
        "validation": {
            "structural": validation_structural,
            "structural_pass": validation_structural_pass,
            "primary_deployable": validation_delta,
            "deployable_pass": validation_deploy_pass,
            "prototype_retention": validation_retention,
            "prototype_capacity_pass": prototype_capacity_pass,
            "paired_bootstrap": {
                threshold: _paired_bootstrap(validation, threshold)
                for threshold in ("0.50", "0.75")
            },
            "bottom_guard": validation["bottom_guard_ablation"],
            "secondary_diagnostics_posthoc_selection_forbidden": _secondary_diagnostics(
                validation
            ),
        },
        "passed": passed,
        "decision": (
            "training_free_bottom_aware_clustering_supported_stop_and_review"
            if passed
            else "training_free_bottom_aware_clustering_not_confirmed_stop_and_review"
        ),
        "training_performed": False,
        "new_model_version_authorized": False,
        "long_training_authorized": False,
        "full_validation_authorized": False,
        "test_set_used": False,
        "required_next_action": "stop_for_user_review",
    }


def write_markdown(path: str, report: dict[str, Any]) -> None:
    cal = report["calibration"]
    val = report["validation"]
    lines = [
        "# Training-free bottom-aware proposal clustering decision",
        "",
        f"**Formal result: {report['formal_result']}**",
        "",
        "No optimizer, backward pass, checkpoint update, test evaluation, or threshold search was used.",
        "",
        "Primary policy fixed before results:",
        "",
        f"- cluster policy: `{PRIMARY_POLICY}`",
        f"- representative: `{PRIMARY_PROTOTYPE}`",
        f"- slot selection: `{PRIMARY_SELECTION}`",
        "",
        "| Domain | Pair precision | Same-lane pair recall | Catastrophic clusters | Multi-member candidate fraction | Structural gate |",
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
            "## Deployable same-count result",
            "",
            "| Domain | IoU | V7 TP/F1 | Cluster TP/F1 | Delta TP | Delta F1 points | Exact count |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label, row in (("Calibration", cal), ("Validation", val)):
        for threshold in ("0.50", "0.75"):
            value = row["primary_deployable"][threshold]
            baseline = value["baseline"]
            treatment = value["treatment"]
            lines.append(
                f"| {label} | {threshold} | {baseline['tp']} / {100.0 * float(baseline['f1']):.3f} | "
                f"{treatment['tp']} / {100.0 * float(treatment['f1']):.3f} | "
                f"{value['delta_tp']:+d} | {value['delta_f1_points']:+.3f} | "
                f"{'yes' if value['exact_count_parity'] else 'NO'} |"
            )
    lines.extend(
        [
            "",
            "Validation paired bootstrap (diagnostic, not a post-hoc gate):",
            "",
        ]
    )
    for threshold in ("0.50", "0.75"):
        boot = val["paired_bootstrap"][threshold]
        lines.append(
            f"- IoU {threshold}: observed `{boot['observed_delta_f1_points']:+.3f}`, "
            f"95% CI `[{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}]`, "
            f"P(delta>0) `{boot['probability_delta_positive']:.3f}`"
        )
    lines.extend(
        [
            "",
            "Prototype retention against the all-32 same-count oracle:",
            "",
            f"- IoU .50: `{val['prototype_retention']['0.50']:.4f}`",
            f"- IoU .75: `{val['prototype_retention']['0.75']:.4f}`",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "Secondary combinations are diagnostic only and cannot replace the predeclared primary after seeing validation.",
            "No training, full validation, new model version, or test run is authorized.",
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
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "decision": report["decision"],
                "output_json": str(output.resolve()),
                "output_md": str(Path(args.output_md).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
