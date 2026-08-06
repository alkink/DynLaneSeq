from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired V5-A/V5-B uniform trajectory diagnostics."
    )
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


def _iteration(payload: dict[str, Any]) -> int:
    checkpoint = str(payload.get("metadata", {}).get("checkpoint", ""))
    match = re.search(r"iter_(\d+)", checkpoint)
    if match is None:
        raise ValueError(f"cannot infer iteration from checkpoint {checkpoint!r}")
    return int(match.group(1))


def _close(left: Any, right: float, atol: float = 1e-8) -> bool:
    return left is not None and abs(float(left) - float(right)) <= atol


def _row(
    payload: dict[str, Any],
    *,
    strategy: str,
    iou: float,
    score_threshold: float | None,
) -> dict[str, Any]:
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("stage") == "main"
        and row.get("strategy") == strategy
        and int(row.get("top_k", 0)) == 4
        and _close(row.get("iou_threshold"), iou)
        and (
            (score_threshold is None and row.get("score_threshold") is None)
            or (
                score_threshold is not None
                and _close(row.get("score_threshold"), score_threshold)
            )
        )
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one row for {strategy=}, {iou=}, {score_threshold=}; "
            f"found {len(rows)}"
        )
    return rows[0]


def _arm(
    report_paths: list[str],
    representative_paths: list[str],
    stability_path: str,
) -> dict[str, Any]:
    reports = {_iteration(payload): payload for payload in map(_load, report_paths)}
    representatives = {}
    for path in representative_paths:
        payload = _load(path)
        checkpoint = str(payload.get("checkpoint", ""))
        match = re.search(r"iter_(\d+)", checkpoint)
        if match is None:
            raise ValueError(f"cannot infer representative iteration from {checkpoint!r}")
        representatives[int(match.group(1))] = payload
    if set(reports) != set(representatives):
        raise ValueError("oracle and representative trajectories do not align")

    trajectory = []
    for iteration in sorted(reports):
        report = reports[iteration]
        representative = representatives[iteration]
        metrics = {}
        for iou in (0.5, 0.75):
            direct = _row(
                report,
                strategy="model_topk_nms",
                iou=iou,
                score_threshold=0.5,
            )
            no_threshold = _row(
                report,
                strategy="model_topk_nms",
                iou=iou,
                score_threshold=-1.0,
            )
            oracle = _row(
                report,
                strategy="oracle_topk",
                iou=iou,
                score_threshold=None,
            )
            exact = direct.get(f"official_iou_{iou:g}")
            if not isinstance(exact, dict):
                raise ValueError("direct ownership row has no exact official metric")
            metrics[f"{iou:.2f}"] = {
                "direct": exact,
                "no_threshold_top4_recall": float(no_threshold["recall"]),
                "oracle_top4_recall": float(oracle["recall"]),
            }
        rep = representative["representative"]
        trajectory.append(
            {
                "iteration": iteration,
                "metrics": metrics,
                "representative": rep,
                "representative_gate": representative["predeclared_gate"],
            }
        )

    best = max(
        trajectory,
        key=lambda row: (
            float(row["metrics"]["0.50"]["direct"]["f1"]),
            float(row["metrics"]["0.75"]["direct"]["f1"]),
            -int(row["iteration"]),
        ),
    )
    stability = _load(stability_path)
    return {
        "trajectory": trajectory,
        "best": best,
        "stability_gate": stability.get("gate", {}),
        "final_training_official_owner_agreement": stability.get(
            "checkpoints",
            [{}],
        )[-1]
        .get("summary", {})
        .get("training_vs_official_owner_agreement", 0.0),
    }


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
    control_best = control["best"]
    assignment_best = assignment["best"]
    selected_best_delta_f1_050 = 100.0 * (
        float(assignment_best["metrics"]["0.50"]["direct"]["f1"])
        - float(control_best["metrics"]["0.50"]["direct"]["f1"])
    )
    selected_best_delta_top1 = 100.0 * (
        float(assignment_best["representative"]["top1_rate"])
        - float(control_best["representative"]["top1_rate"])
    )
    selected_best_delta_oracle_050 = 100.0 * (
        float(assignment_best["metrics"]["0.50"]["oracle_top4_recall"])
        - float(control_best["metrics"]["0.50"]["oracle_top4_recall"])
    )

    control_by_iteration = {
        int(row["iteration"]): row for row in control["trajectory"]
    }
    assignment_by_iteration = {
        int(row["iteration"]): row for row in assignment["trajectory"]
    }
    if set(control_by_iteration) != set(assignment_by_iteration):
        raise ValueError("control and assignment checkpoint trajectories do not align")
    paired_control = control_by_iteration[int(assignment_best["iteration"])]
    paired_delta_f1_050 = 100.0 * (
        float(assignment_best["metrics"]["0.50"]["direct"]["f1"])
        - float(paired_control["metrics"]["0.50"]["direct"]["f1"])
    )
    paired_delta_top1 = 100.0 * (
        float(assignment_best["representative"]["top1_rate"])
        - float(paired_control["representative"]["top1_rate"])
    )
    paired_delta_oracle_050 = 100.0 * (
        float(assignment_best["metrics"]["0.50"]["oracle_top4_recall"])
        - float(paired_control["metrics"]["0.50"]["oracle_top4_recall"])
    )

    pre_ramp_rows = [
        (
            control_by_iteration[iteration],
            assignment_by_iteration[iteration],
        )
        for iteration in sorted(control_by_iteration)
        if iteration <= 10000
    ]
    if not pre_ramp_rows:
        raise ValueError("V5 trajectory must include a pre-ramp checkpoint")
    pre_ramp_max_abs_f1_delta = max(
        abs(
            100.0
            * (
                float(right["metrics"]["0.50"]["direct"]["f1"])
                - float(left["metrics"]["0.50"]["direct"]["f1"])
            )
        )
        for left, right in pre_ramp_rows
    )
    pre_ramp_max_abs_top1_delta = max(
        abs(
            100.0
            * (
                float(right["representative"]["top1_rate"])
                - float(left["representative"]["top1_rate"])
            )
        )
        for left, right in pre_ramp_rows
    )
    pre_ramp_max_abs_oracle_delta = max(
        abs(
            100.0
            * (
                float(right["metrics"]["0.50"]["oracle_top4_recall"])
                - float(left["metrics"]["0.50"]["oracle_top4_recall"])
            )
        )
        for left, right in pre_ramp_rows
    )
    pre_ramp_equivalence_pass = bool(
        pre_ramp_max_abs_f1_delta <= 0.05
        and pre_ramp_max_abs_top1_delta <= 0.50
        and pre_ramp_max_abs_oracle_delta <= 0.05
    )
    assignment_rep_gate = assignment_best["representative_gate"]
    representation_pass = all(bool(value) for value in assignment_rep_gate.values())
    causal_assignment_pass = bool(
        pre_ramp_equivalence_pass
        and paired_delta_f1_050 >= 0.50
        and paired_delta_top1 >= 3.0
        and paired_delta_oracle_050 >= -0.50
    )
    if representation_pass and causal_assignment_pass:
        verdict = "promote_v5_b_to_full_validation"
    elif representation_pass:
        verdict = "ownership_representation_formed_but_assignment_edge_not_proven"
    elif causal_assignment_pass:
        verdict = "assignment_signal_positive_but_representation_gate_not_met"
    else:
        verdict = "v5_gate_not_yet_supported"

    payload = {
        "diagnostic_only": True,
        "experiment": "V5 protected dual-state ownership gate",
        "control": control,
        "assignment": assignment,
        "independently_selected_best_delta_points": {
            "f1_050": selected_best_delta_f1_050,
            "representative_top1": selected_best_delta_top1,
            "oracle_top4_recall_050": selected_best_delta_oracle_050,
        },
        "paired_at_assignment_best": {
            "iteration": int(assignment_best["iteration"]),
            "f1_050_delta_points": paired_delta_f1_050,
            "representative_top1_delta_points": paired_delta_top1,
            "oracle_top4_recall_050_delta_points": paired_delta_oracle_050,
        },
        "pre_ramp_equivalence": {
            "iterations": [
                int(left["iteration"]) for left, _right in pre_ramp_rows
            ],
            "max_abs_f1_050_delta_points": pre_ramp_max_abs_f1_delta,
            "max_abs_representative_top1_delta_points": (
                pre_ramp_max_abs_top1_delta
            ),
            "max_abs_oracle_top4_recall_050_delta_points": (
                pre_ramp_max_abs_oracle_delta
            ),
            "passed": pre_ramp_equivalence_pass,
        },
        "gates": {
            "assignment_representation_pass": representation_pass,
            "pre_ramp_equivalence_pass": pre_ramp_equivalence_pass,
            "causal_assignment_pass": causal_assignment_pass,
        },
        "verdict": verdict,
        "warning": (
            "Uniform-subset checkpoint selection is diagnostic. A positive "
            "verdict licenses full validation only, never test tuning."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
