from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the validation-only V8 anchor-neighborhood gate."
    )
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="ITER=JSON",
    )
    parser.add_argument("--source-full-validation")
    parser.add_argument("--candidate-full-validation")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], threshold: str) -> dict[str, float | int]:
    row = report["methods"]["four_slot_refined"][threshold]
    return {
        name: row[name]
        for name in (
            "f1",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "mean_selected_per_image",
        )
    }


def _anchor_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_unique_050": report["methods"]["four_slot_global_unique"]["0.50"],
        "global_unique_075": report["methods"]["four_slot_global_unique"]["0.75"],
        "oracle_050": report["capacity"]["0.50"]["all_candidate_oracle"],
        "oracle_075": report["capacity"]["0.75"]["all_candidate_oracle"],
        "cardinality": report["four_slot_diagnostics"]["cardinality"],
    }


def _same(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _full_validation_metric(
    report: dict[str, Any], threshold: str
) -> dict[str, float | int]:
    row = report["results"][threshold]
    return {
        "f1": float(row["F1"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "tp": int(row["TP"]),
        "fp": int(row["FP"]),
        "fn": int(row["FN"]),
        "prediction_count": int(row["TP"]) + int(row["FP"]),
    }


def _full_validation_protocol(report: dict[str, Any]) -> dict[str, Any]:
    return {
        name: report[name]
        for name in (
            "split",
            "score_thresh",
            "lane_nms_distance_thresh_px",
            "top_k",
            "row_visibility_thresh",
            "quality_score_power",
            "score_mode",
            "eval_batch_size",
            "channels_last",
            "inference_only",
            "amp_dtype",
            "compile_model",
            "no_pretrained_init",
        )
    }


def _summarize_full_validation(
    source: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    source_protocol = _full_validation_protocol(source)
    candidate_protocol = _full_validation_protocol(candidate)
    protocol_equal = _same(source_protocol, candidate_protocol)
    source_metrics = {
        threshold: _full_validation_metric(source, threshold)
        for threshold in ("0.5", "0.75")
    }
    candidate_metrics = {
        threshold: _full_validation_metric(candidate, threshold)
        for threshold in ("0.5", "0.75")
    }
    delta = {
        threshold: {
            name: float(candidate_metrics[threshold][name])
            - float(source_metrics[threshold][name])
            for name in ("f1", "precision", "recall", "tp", "fp", "fn")
        }
        for threshold in ("0.5", "0.75")
    }
    primary_improved = float(delta["0.5"]["f1"]) > 0.0
    strict_improved = float(delta["0.75"]["f1"]) > 0.0
    if primary_improved and strict_improved:
        verdict = "both_metrics_improved"
    elif primary_improved:
        verdict = "primary_only_improved"
    elif strict_improved:
        verdict = "strict_only_improved_primary_regressed"
    else:
        verdict = "both_metrics_regressed"
    return {
        "protocol_equal": protocol_equal,
        "source_protocol": source_protocol,
        "source": source_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "uniform_primary_signal_replicated": (
            protocol_equal and primary_improved and strict_improved
        ),
        "verdict": verdict,
        "long_training_authorized": False,
    }


def summarize(
    source: dict[str, Any],
    reports: list[tuple[int, dict[str, Any]]],
    contract: dict[str, Any],
    source_full_validation: dict[str, Any] | None = None,
    candidate_full_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one V8 checkpoint report is required")
    reports = sorted(reports)
    source_metrics = {
        threshold: _metric(source, threshold)
        for threshold in ("0.50", "0.75")
    }
    source_anchor = _anchor_signature(source)
    trajectory = []
    anchor_invariant = True
    for iteration, report in reports:
        metrics = {
            threshold: _metric(report, threshold)
            for threshold in ("0.50", "0.75")
        }
        neighborhood = report["four_slot_diagnostics"].get("neighborhood")
        trajectory.append(
            {
                "iteration": int(iteration),
                "metrics": metrics,
                "gain_from_source": {
                    threshold: float(metrics[threshold]["f1"])
                    - float(source_metrics[threshold]["f1"])
                    for threshold in ("0.50", "0.75")
                },
                "neighborhood": neighborhood,
                "refinement": report["four_slot_diagnostics"].get("refinement"),
            }
        )
        anchor_invariant &= _same(source_anchor, _anchor_signature(report))

    best_050 = max(trajectory, key=lambda row: row["metrics"]["0.50"]["f1"])
    best_075 = max(trajectory, key=lambda row: row["metrics"]["0.75"]["f1"])
    final = trajectory[-1]
    final_gain_050 = float(final["gain_from_source"]["0.50"])
    final_gain_075 = float(final["gain_from_source"]["0.75"])
    safety = {
        "gradient_contract": contract.get("passed") is True,
        "frozen_anchor_route_and_proposal_capacity_exact": anchor_invariant,
        "final_f1_050_no_material_regression": final_gain_050 >= -0.002,
        "final_neighborhood_is_open": (
            isinstance(final.get("neighborhood"), dict)
            and abs(float(final["neighborhood"].get("mean_mix", 0.0))) > 1.0e-6
        ),
    }
    primary_pass = (
        all(safety.values())
        and final_gain_050 >= 0.0
        and final_gain_075 >= 0.005
    )
    strong_pass = (
        all(safety.values())
        and final_gain_050 >= 0.003
        and final_gain_075 >= 0.010
    )
    any_positive_signal = (
        all(
            safety[name]
            for name in (
                "gradient_contract",
                "frozen_anchor_route_and_proposal_capacity_exact",
            )
        )
        and (
            float(best_050["gain_from_source"]["0.50"]) >= 0.002
            or float(best_075["gain_from_source"]["0.75"]) >= 0.005
        )
    )
    if strong_pass:
        verdict = "strong_pass"
        next_action = "run_one_predeclared_full_validation_before_any_long_training"
    elif primary_pass:
        verdict = "pass"
        next_action = "run_one_predeclared_full_validation_before_any_long_training"
    elif any_positive_signal and all(safety.values()):
        verdict = "conditional"
        next_action = "inspect_learning_dynamics_then_extend_only_1000_steps"
    else:
        verdict = "fail"
        next_action = "stop_neighborhood_mixture_and_do_not_open_long_training"
    payload = {
        "experiment": "V8 route-anchored sparse slot-owned geometry gate",
        "diagnostic_only": True,
        "test_set_used": False,
        "source": source_metrics,
        "trajectory": trajectory,
        "best": {
            "f1_050": best_050,
            "f1_075": best_075,
        },
        "final": final,
        "safety": safety,
        "predeclared_gates": {
            "primary": {
                "final_f1_050_gain_min": 0.0,
                "final_f1_075_gain_min": 0.005,
                "passed": primary_pass,
            },
            "strong": {
                "final_f1_050_gain_min": 0.003,
                "final_f1_075_gain_min": 0.010,
                "passed": strong_pass,
            },
        },
        "verdict": verdict,
        "next_action": next_action,
        "full_validation_authorized": verdict in {"pass", "strong_pass"},
        "long_training_authorized": False,
    }
    if (source_full_validation is None) != (candidate_full_validation is None):
        raise ValueError(
            "source and candidate full-validation reports must be supplied together"
        )
    if source_full_validation is not None and candidate_full_validation is not None:
        payload["full_validation"] = _summarize_full_validation(
            source_full_validation,
            candidate_full_validation,
        )
        payload["long_training_authorized"] = False
        if not payload["full_validation"]["uniform_primary_signal_replicated"]:
            payload["next_action"] = (
                "stop_long_training_and_audit_why_uniform_gain_did_not_generalize"
            )
    return payload


def main() -> None:
    args = parse_args()
    reports: list[tuple[int, dict[str, Any]]] = []
    for value in args.report:
        iteration_text, path = value.split("=", 1)
        reports.append((int(iteration_text), _load(path)))
    payload = summarize(
        _load(args.source_report),
        reports,
        _load(args.contract),
        (
            _load(args.source_full_validation)
            if args.source_full_validation is not None
            else None
        ),
        (
            _load(args.candidate_full_validation)
            if args.candidate_full_validation is not None
            else None
        ),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
