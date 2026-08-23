from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICIES = (
    "control_selection_control_refiner",
    "control_selection_treatment_refiner",
    "treatment_selection_control_refiner",
    "treatment_selection_treatment_refiner",
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(payload: dict, threshold: float) -> dict:
    key = f"{threshold:.2f}"
    results = payload["results"]
    if key in results:
        return results[key]
    # Historical evaluator payloads occasionally used float-like keys without
    # fixed precision. Keep the replay compatible without relaxing semantics.
    for candidate, row in results.items():
        if abs(float(candidate) - threshold) < 1e-8:
            return row
    raise KeyError(f"missing IoU threshold {threshold:.2f}")


def summarize(metrics: dict[str, dict]) -> dict:
    output: dict[str, object] = {
        "experiment": "V31 50K training-free refiner 2x2 replay",
        "contract": {
            "selection_and_all_upstream_state_come_from_base_checkpoint": True,
            "only_slot_refinement_parameters_are_swapped": True,
            "test_split_opened": False,
        },
        "thresholds": {},
    }
    for threshold in (0.50, 0.75):
        rows = {name: _row(metrics[name], threshold) for name in POLICIES}
        f1 = {name: float(row["F1"]) for name, row in rows.items()}
        tp = {name: int(row["TP"]) for name, row in rows.items()}
        cc, ct, tc, tt = (f1[name] for name in POLICIES)
        cc_tp, ct_tp, tc_tp, tt_tp = (tp[name] for name in POLICIES)
        output["thresholds"][f"{threshold:.2f}"] = {
            "policies": {
                name: {
                    "f1": f1[name],
                    "f1_points": 100.0 * f1[name],
                    "tp": tp[name],
                    "fp": int(rows[name]["FP"]),
                    "fn": int(rows[name]["FN"]),
                }
                for name in POLICIES
            },
            "causal_deltas": {
                "selection_bridge_with_control_refiner_f1_points": 100.0 * (tc - cc),
                "selection_bridge_with_control_refiner_tp": tc_tp - cc_tp,
                "treatment_refiner_on_control_base_f1_points": 100.0 * (ct - cc),
                "treatment_refiner_on_control_base_tp": ct_tp - cc_tp,
                "treatment_refiner_on_treatment_base_f1_points": 100.0 * (tt - tc),
                "treatment_refiner_on_treatment_base_tp": tt_tp - tc_tp,
                "refiner_interaction_f1_points": 100.0 * ((tt - tc) - (ct - cc)),
            },
        }

    strict = output["thresholds"]["0.75"]["causal_deltas"]
    bridge_survives = strict["selection_bridge_with_control_refiner_tp"] > 0
    treatment_refiner_harms_treatment = (
        strict["treatment_refiner_on_treatment_base_tp"] < 0
    )
    if bridge_survives and treatment_refiner_harms_treatment:
        verdict = "bridge_gain_survives_but_treatment_refiner_erases_it"
        next_action = "freeze_or_decouple_refiner_then_repeat_short_exact_pair"
    elif bridge_survives:
        verdict = "bridge_gain_survives_with_both_refiners"
        next_action = "inspect_activity_and_50_threshold_regression_before_training"
    else:
        verdict = "control_refiner_does_not_recover_bridge_gain"
        next_action = "do_not_train_longer_and_close_current_bridge_form"
    output["verdict"] = {
        "status": verdict,
        "next_action": next_action,
        "basis": "official full-validation raster TP at IoU 0.75",
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V31 refiner 2x2 replay.")
    for name in POLICIES:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    metrics = {
        name: _load(getattr(args, name))
        for name in POLICIES
    }
    output = summarize(metrics)
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
