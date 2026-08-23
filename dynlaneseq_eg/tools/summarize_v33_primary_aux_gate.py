from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the predeclared V33 direct-primary / auxiliary "
            "supervision / auxiliary-memory causal gate."
        )
    )
    parser.add_argument("--primary-vs-aux-report", required=True)
    parser.add_argument("--aux-vs-memory-report", required=True)
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


def build(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    # evaluate_v25_g2_component_pair uses generic historical policy names.
    # In V33 pair one, control=A and image_ownership=B. In pair two,
    # control=B and image_ownership=C.
    a = {
        threshold: _metric(first, "control", threshold)
        for threshold in ("0.5", "0.75")
    }
    b_first = {
        threshold: _metric(first, "image_ownership", threshold)
        for threshold in ("0.5", "0.75")
    }
    b_second = {
        threshold: _metric(second, "control", threshold)
        for threshold in ("0.5", "0.75")
    }
    c = {
        threshold: _metric(second, "image_ownership", threshold)
        for threshold in ("0.5", "0.75")
    }
    c_wrong = {
        threshold: _metric(second, "image_ownership_wrong_image", threshold)
        for threshold in ("0.5", "0.75")
    }
    b_replay_checks = {
        threshold: {
            name: b_first[threshold][name] == b_second[threshold][name]
            for name in ("TP", "FP", "FN")
        }
        for threshold in ("0.5", "0.75")
    }
    if not all(
        passed
        for threshold in b_replay_checks.values()
        for passed in threshold.values()
    ):
        raise ValueError(
            "V33 Arm-B official replay was not exact: "
            + json.dumps(b_replay_checks, sort_keys=True)
        )

    delta_b_a = {
        threshold: _f1_points(b_first[threshold]) - _f1_points(a[threshold])
        for threshold in ("0.5", "0.75")
    }
    delta_c_b = {
        threshold: _f1_points(c[threshold]) - _f1_points(b_first[threshold])
        for threshold in ("0.5", "0.75")
    }
    delta_c_a = {
        threshold: _f1_points(c[threshold]) - _f1_points(a[threshold])
        for threshold in ("0.5", "0.75")
    }
    auxiliary_checks = {
        "arm_b_minus_a_f1_50_at_least_0p30": delta_b_a["0.5"] >= 0.30,
        "arm_b_f1_75_non_regression": delta_b_a["0.75"] >= 0.0,
        "arm_b_tp50_not_lower": int(b_first["0.5"]["TP"]) >= int(a["0.5"]["TP"]),
        "arm_b_tp75_not_lower": int(b_first["0.75"]["TP"]) >= int(a["0.75"]["TP"]),
    }
    memory_checks = {
        "arm_c_minus_b_f1_50_at_least_0p30": delta_c_b["0.5"] >= 0.30,
        "arm_c_f1_75_non_regression": delta_c_b["0.75"] >= 0.0,
        "correct_image_beats_wrong_tp50": int(c["0.5"]["TP"])
        > int(c_wrong["0.5"]["TP"]),
        "correct_image_beats_wrong_tp75": int(c["0.75"]["TP"])
        > int(c_wrong["0.75"]["TP"]),
    }
    best_delta_50 = max(delta_b_a["0.5"], delta_c_a["0.5"])
    best_delta_75 = (
        delta_b_a["0.75"]
        if delta_b_a["0.5"] >= delta_c_a["0.5"]
        else delta_c_a["0.75"]
    )
    overall_checks = {
        "some_auxiliary_arm_beats_primary_by_0p30_at_50": best_delta_50 >= 0.30,
        "the_same_best_50_arm_is_non_regressive_at_75": best_delta_75 >= 0.0,
    }
    return {
        "experiment": "V33 direct-primary auxiliary sufficiency gate",
        "question": (
            "Does 32-query one-to-many auxiliary supervision improve an "
            "otherwise identical four-query direct-primary detector, and "
            "does reading the auxiliary spatial bank add further value?"
        ),
        "arms": {
            "A_primary_only": {key: _compact(value) for key, value in a.items()},
            "B_primary_plus_aux_training_only": {
                key: _compact(value) for key, value in b_first.items()
            },
            "C_primary_plus_aux_memory": {
                key: _compact(value) for key, value in c.items()
            },
            "C_cross_clip_wrong_image": {
                key: _compact(value) for key, value in c_wrong.items()
            },
        },
        "deltas_F1_points": {
            "B_minus_A": delta_b_a,
            "C_minus_B": delta_c_b,
            "C_minus_A": delta_c_a,
        },
        "gates": {
            "auxiliary_training_only": {
                "passed": all(auxiliary_checks.values()),
                "checks": auxiliary_checks,
            },
            "auxiliary_memory": {
                "passed": all(memory_checks.values()),
                "checks": memory_checks,
            },
            "overall_primary_auxiliary_mechanism": {
                "passed": all(overall_checks.values()),
                "checks": overall_checks,
            },
        },
        "arm_b_replay_exact": b_replay_checks,
        "interpretation_contract": {
            "this_is_a_short_mechanism_gate_not_a_final_v7_comparison": True,
            "a_pass_does_not_authorize_test_or_claim_81_plus": True,
            "a_fail_closes_the_current_direct_primary_auxiliary_implementation": True,
            "test_set_used": False,
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
        },
    }


def main() -> None:
    args = parse_args()
    first_path = Path(args.primary_vs_aux_report).expanduser().resolve()
    second_path = Path(args.aux_vs_memory_report).expanduser().resolve()
    result = build(_read(first_path), _read(second_path))
    result["sources"] = {
        "primary_vs_aux": str(first_path),
        "primary_vs_aux_sha256": _sha256(first_path),
        "aux_vs_memory": str(second_path),
        "aux_vs_memory_sha256": _sha256(second_path),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
