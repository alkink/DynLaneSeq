from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.tools.summarize_v5_protected_ownership_gate import _arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize exact shared-trunk V5.1 ownership forks."
    )
    parser.add_argument("--trunk-report", required=True)
    parser.add_argument("--trunk-representative", required=True)
    parser.add_argument("--trunk-contract", required=True)
    parser.add_argument("--control-reports", nargs="+", required=True)
    parser.add_argument("--assignment-reports", nargs="+", required=True)
    parser.add_argument("--control-representatives", nargs="+", required=True)
    parser.add_argument("--assignment-representatives", nargs="+", required=True)
    parser.add_argument("--control-stability", required=True)
    parser.add_argument("--assignment-stability", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta(right: dict[str, Any], left: dict[str, Any], key: str) -> float:
    if key == "f1_050":
        return 100.0 * (
            float(right["metrics"]["0.50"]["direct"]["f1"])
            - float(left["metrics"]["0.50"]["direct"]["f1"])
        )
    if key == "top1":
        return 100.0 * (
            float(right["representative"]["top1_rate"])
            - float(left["representative"]["top1_rate"])
        )
    if key == "oracle_050":
        return 100.0 * (
            float(right["metrics"]["0.50"]["oracle_top4_recall"])
            - float(left["metrics"]["0.50"]["oracle_top4_recall"])
        )
    raise ValueError(f"unsupported V5.1 delta key: {key}")


def main() -> None:
    args = parse_args()
    control = _arm(
        args.control_reports,
        args.control_representatives,
        args.control_stability,
    )
    assignment = _arm(
        args.assignment_reports,
        args.assignment_representatives,
        args.assignment_stability,
    )
    # Reuse the control stability file only as a required _arm parser input;
    # the shared branch-point row itself comes from one physical checkpoint.
    trunk = _arm(
        [args.trunk_report],
        [args.trunk_representative],
        args.control_stability,
    )["trajectory"][0]
    if int(trunk["iteration"]) != 10000:
        raise ValueError("V5.1 shared branch point must be iteration 10000")

    control_by_iteration = {
        int(row["iteration"]): row for row in control["trajectory"]
    }
    assignment_by_iteration = {
        int(row["iteration"]): row for row in assignment["trajectory"]
    }
    expected_iterations = {15000, 20000, 25000}
    if set(control_by_iteration) != expected_iterations:
        raise ValueError("control fork must contain 15k, 20k, and 25k")
    if set(assignment_by_iteration) != expected_iterations:
        raise ValueError("assignment fork must contain 15k, 20k, and 25k")

    paired = []
    for iteration in sorted(expected_iterations):
        left = control_by_iteration[iteration]
        right = assignment_by_iteration[iteration]
        paired.append(
            {
                "iteration": iteration,
                "f1_050_delta_points": _delta(right, left, "f1_050"),
                "representative_top1_delta_points": _delta(right, left, "top1"),
                "oracle_top4_recall_050_delta_points": _delta(
                    right,
                    left,
                    "oracle_050",
                ),
            }
        )

    assignment_best = assignment["best"]
    paired_control = control_by_iteration[int(assignment_best["iteration"])]
    selected_pair = {
        "iteration": int(assignment_best["iteration"]),
        "f1_050_delta_points": _delta(
            assignment_best,
            paired_control,
            "f1_050",
        ),
        "representative_top1_delta_points": _delta(
            assignment_best,
            paired_control,
            "top1",
        ),
        "oracle_top4_recall_050_delta_points": _delta(
            assignment_best,
            paired_control,
            "oracle_050",
        ),
    }
    trunk_contract = _load(args.trunk_contract)
    exact_branch_point_pass = bool(
        trunk_contract.get("passed")
        and trunk_contract.get("checkpoint", {}).get("passed")
        and trunk_contract.get("checkpoint", {}).get("sha256")
    )
    representation_pass = all(
        bool(value)
        for value in assignment_best["representative_gate"].values()
    )
    positive_f1_checkpoints = sum(
        int(float(row["f1_050_delta_points"]) > 0.0) for row in paired
    )
    positive_top1_checkpoints = sum(
        int(float(row["representative_top1_delta_points"]) > 0.0)
        for row in paired
    )
    causal_assignment_pass = bool(
        exact_branch_point_pass
        and representation_pass
        and positive_f1_checkpoints >= 2
        and positive_top1_checkpoints >= 2
        and float(selected_pair["f1_050_delta_points"]) >= 2.0
        and float(selected_pair["representative_top1_delta_points"]) >= 5.0
        and float(selected_pair["oracle_top4_recall_050_delta_points"]) >= -0.5
    )
    if causal_assignment_pass:
        verdict = "promote_shared_trunk_v5_1_assignment_to_full_validation"
    elif representation_pass:
        verdict = "ownership_formed_but_shared_trunk_causal_gate_failed"
    else:
        verdict = "shared_trunk_v5_1_not_supported"

    payload = {
        "diagnostic_only": True,
        "experiment": "V5.1 exact shared-trunk ownership fork",
        "shared_branch_point": trunk,
        "shared_trunk_checkpoint": trunk_contract.get("checkpoint"),
        "control": control,
        "assignment": assignment,
        "paired_deltas": paired,
        "paired_at_assignment_best": selected_pair,
        "gates": {
            "exact_shared_branch_point": exact_branch_point_pass,
            "assignment_representation_pass": representation_pass,
            "positive_f1_checkpoints": positive_f1_checkpoints,
            "positive_top1_checkpoints": positive_top1_checkpoints,
            "causal_assignment_pass": causal_assignment_pass,
        },
        "verdict": verdict,
        "warning": (
            "Uniform-256 is a causal gate only. A pass licenses one full "
            "validation run and never licenses test tuning."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
