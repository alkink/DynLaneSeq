from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the V33 private-head cold-start firewall gate."
    )
    parser.add_argument("--a-vs-d-report", required=True)
    parser.add_argument("--b-vs-d-report", required=True)
    parser.add_argument("--private-training-report", required=True)
    parser.add_argument("--pretrain-gradient-audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(report: dict[str, Any], policy: str, threshold: str) -> dict[str, Any]:
    values = report["metrics"][policy]
    return values[threshold] if threshold in values else values[str(float(threshold))]


def _f1_points(values: dict[str, Any]) -> float:
    return 100.0 * float(values["F1"])


def _compact(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "F1_points": _f1_points(values),
        "TP": int(values["TP"]),
        "FP": int(values["FP"]),
        "FN": int(values["FN"]),
        "precision": float(values.get("Precision", values.get("precision"))),
        "recall": float(values.get("Recall", values.get("recall"))),
    }


def build(
    a_vs_d: dict[str, Any],
    b_vs_d: dict[str, Any],
    private_report: dict[str, Any],
    gradient_audit: dict[str, Any],
) -> dict[str, Any]:
    a = {threshold: _metric(a_vs_d, "control", threshold) for threshold in ("0.5", "0.75")}
    d_first = {
        threshold: _metric(a_vs_d, "image_ownership", threshold)
        for threshold in ("0.5", "0.75")
    }
    b = {threshold: _metric(b_vs_d, "control", threshold) for threshold in ("0.5", "0.75")}
    d_second = {
        threshold: _metric(b_vs_d, "image_ownership", threshold)
        for threshold in ("0.5", "0.75")
    }
    d_replay = {
        threshold: {
            name: d_first[threshold][name] == d_second[threshold][name]
            for name in ("TP", "FP", "FN")
        }
        for threshold in ("0.5", "0.75")
    }
    if not all(value for row in d_replay.values() for value in row.values()):
        raise ValueError("V33-PH D replay was not exact: " + json.dumps(d_replay))

    delta_d_a = {
        threshold: _f1_points(d_first[threshold]) - _f1_points(a[threshold])
        for threshold in ("0.5", "0.75")
    }
    delta_d_b = {
        threshold: _f1_points(d_second[threshold]) - _f1_points(b[threshold])
        for threshold in ("0.5", "0.75")
    }
    hybrid = gradient_audit["hybrid_parent_with_trained_private_auxiliary"]
    hybrid_verdict = hybrid["verdict"]
    hybrid_shared = hybrid["summary"]["gradient_groups"]["all_shared"]
    largest_scale = sorted(
        hybrid["summary"]["cross_batch_virtual_steps"], key=float
    )[-1]
    hybrid_virtual = hybrid["summary"]["cross_batch_virtual_steps"][largest_scale]
    mechanism_checks = {
        "frozen_parent_state_exact": private_report.get("frozen_parent_state_exact") is True,
        "shared_updates_zero": int(private_report.get("shared_updates", -1)) == 0,
        "private_gradient_ownership_passed": bool(
            private_report.get("gradient_ownership_gate", {}).get("passed", False)
        ),
        "pretrained_head_shared_cosine_above_0p10": float(
            hybrid_shared["cosine"]["median"]
        ) > 0.10,
        "pretrained_head_aux_harm_below_0p60": float(
            hybrid_virtual["auxiliary_direction_harm_fraction"]
        ) < 0.60,
        "pretrained_head_classified_aligned": hybrid_verdict["label"]
        == "aligned_or_redundant",
    }
    f1_checks = {
        "d_minus_a_f1_50_at_least_0p30": delta_d_a["0.5"] >= 0.30,
        "d_minus_a_f1_75_non_regression": delta_d_a["0.75"] >= 0.0,
        "d_minus_b_f1_50_at_least_0p20": delta_d_b["0.5"] >= 0.20,
        "d_minus_b_f1_75_non_regression": delta_d_b["0.75"] >= 0.0,
        "d_tp50_not_lower_than_a": int(d_first["0.5"]["TP"])
        >= int(a["0.5"]["TP"]),
        "d_tp75_not_lower_than_a": int(d_first["0.75"]["TP"])
        >= int(a["0.75"]["TP"]),
    }
    mechanism_pass = all(mechanism_checks.values())
    f1_pass = all(f1_checks.values())
    if mechanism_pass and f1_pass:
        verdict = "private_head_firewall_improves_primary"
    elif mechanism_pass and delta_d_a["0.5"] < 0.30:
        verdict = "cold_start_fixed_but_auxiliary_redundant"
    elif not mechanism_pass:
        verdict = "private_head_pretraining_did_not_reproduce_alignment"
    else:
        verdict = "mixed_f1_outcome"
    return {
        "experiment": "V33-PH private auxiliary cold-start firewall gate",
        "question": (
            "Does pretraining only proposal_memory with the causal parent "
            "frozen remove cold-start harm and convert auxiliary supervision "
            "into official validation F1 gain?"
        ),
        "arms": {
            "A_primary_only": {key: _compact(value) for key, value in a.items()},
            "B_random_head_joint": {key: _compact(value) for key, value in b.items()},
            "D_pretrained_head_joint": {
                key: _compact(value) for key, value in d_first.items()
            },
        },
        "deltas_F1_points": {"D_minus_A": delta_d_a, "D_minus_B": delta_d_b},
        "d_replay_exact": d_replay,
        "mechanism_gate": {
            "passed": mechanism_pass,
            "checks": mechanism_checks,
            "pretrained_head_shared_cosine_median": float(
                hybrid_shared["cosine"]["median"]
            ),
            "pretrained_head_auxiliary_harm_fraction": float(
                hybrid_virtual["auxiliary_direction_harm_fraction"]
            ),
            "relative_step_scale": float(largest_scale),
        },
        "f1_gate": {"passed": f1_pass, "checks": f1_checks},
        "verdict": verdict,
        "decision_contract": {
            "private_head_firewall_improves_primary": (
                "Repeat at a longer exact-paired endpoint before any test use."
            ),
            "cold_start_fixed_but_auxiliary_redundant": (
                "Close the direct-primary auxiliary family; alignment alone "
                "does not provide useful new information."
            ),
            "private_head_pretraining_did_not_reproduce_alignment": (
                "Close V33-PH because its required mechanism failed."
            ),
            "mixed_f1_outcome": (
                "Do not open a long run; inspect threshold-specific TP/FP changes."
            ),
            "test_set_used": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
        },
    }


def main() -> None:
    args = parse_args()
    paths = {
        "a_vs_d": Path(args.a_vs_d_report).expanduser().resolve(),
        "b_vs_d": Path(args.b_vs_d_report).expanduser().resolve(),
        "private_training": Path(args.private_training_report).expanduser().resolve(),
        "pretrain_gradient_audit": Path(args.pretrain_gradient_audit).expanduser().resolve(),
    }
    result = build(
        _read(paths["a_vs_d"]),
        _read(paths["b_vs_d"]),
        _read(paths["private_training"]),
        _read(paths["pretrain_gradient_audit"]),
    )
    result["sources"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
