from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


THRESHOLDS = ("0.50", "0.75")
PRIMARY_DOMAINS = ("heldout_clip", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the predeclared clip-disjoint V11 bridge gate."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _coverage(payload: dict[str, Any]) -> dict[str, Any]:
    methods = payload["methods"]
    key = "four_slot_refined"
    if key not in methods:
        raise KeyError("coverage report has no four_slot_refined method")
    metric = {threshold: methods[key][threshold] for threshold in THRESHOLDS}
    capacity = {
        threshold: payload["capacity"][threshold]["all_candidate_oracle"]
        for threshold in THRESHOLDS
    }
    slots = payload.get("four_slot_diagnostics") or {}
    cardinality = slots.get("cardinality") or {}
    return {
        "metric": metric,
        "proposal_oracle": capacity,
        "cardinality_exact": float(cardinality.get("exact_fraction", 0.0)),
        "cardinality_mae": float(cardinality.get("mean_absolute_error", 0.0)),
        "semantic_duplicate_fraction": float(
            slots.get("semantic_duplicate_cluster_fraction", 0.0)
        ),
        "mean_selected": float(metric["0.50"]["mean_selected_per_image"]),
        "refinement": slots.get("refinement"),
    }


def _state(payload: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value["mean"])
        for name, value in payload["statistics"].items()
    }


def _finite_metric(coverage: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(coverage["metric"][threshold][name]))
        for threshold in THRESHOLDS
        for name in ("precision", "recall", "f1")
    )


def _oracle_equal(*coverages: dict[str, Any]) -> bool:
    reference = coverages[0]["proposal_oracle"]
    return all(
        int(coverage["proposal_oracle"][threshold]["hits"])
        == int(reference[threshold]["hits"])
        for coverage in coverages[1:]
        for threshold in THRESHOLDS
    )


def _metric_delta(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for threshold in THRESHOLDS:
        left_metric = left["metric"][threshold]
        right_metric = right["metric"][threshold]
        result[threshold] = {
            "tp": int(right_metric["tp"]) - int(left_metric["tp"]),
            "fp": int(right_metric["fp"]) - int(left_metric["fp"]),
            "fn": int(right_metric["fn"]) - int(left_metric["fn"]),
            "precision": float(right_metric["precision"])
            - float(left_metric["precision"]),
            "recall": float(right_metric["recall"])
            - float(left_metric["recall"]),
            "f1": float(right_metric["f1"]) - float(left_metric["f1"]),
        }
    return result


def _domain(
    entry: dict[str, Any],
) -> dict[str, Any]:
    source_payload = _read(entry["source_coverage"])
    init_payload = _read(entry["init_coverage"])
    end_payload = _read(entry["end_coverage"])
    init_state_payload = _read(entry["init_state"])
    end_state_payload = _read(entry["end_state"])
    for payload in (
        source_payload,
        init_payload,
        end_payload,
        init_state_payload,
        end_state_payload,
    ):
        if payload.get("test_set_used") is True or payload.get("split") == "test":
            raise RuntimeError("test split entered the V11 bridge gate")
    source = _coverage(source_payload)
    initialization = _coverage(init_payload)
    endpoint = _coverage(end_payload)
    init_state = _state(init_state_payload)
    end_state = _state(end_state_payload)
    return {
        "split": entry["split"],
        "list_path": entry["list_path"],
        "source": source,
        "initialization": initialization,
        "endpoint": endpoint,
        "initialization_state": init_state,
        "endpoint_state": end_state,
        "endpoint_minus_source": _metric_delta(source, endpoint),
        "endpoint_minus_initialization": _metric_delta(
            initialization, endpoint
        ),
        "target_support_mass_gain": (
            end_state["target_support_mass"]
            - init_state["target_support_mass"]
        ),
        "active_count_drift_from_source": (
            endpoint["mean_selected"] - source["mean_selected"]
        ),
        "proposal_oracle_unchanged": _oracle_equal(
            source, initialization, endpoint
        ),
        "metrics_finite": all(
            _finite_metric(value)
            for value in (source, initialization, endpoint)
        ),
    }


def _f1_not_worse(
    endpoint: dict[str, Any],
    reference: dict[str, Any],
    threshold: str,
) -> bool:
    return float(endpoint["metric"][threshold]["f1"]) >= float(
        reference["metric"][threshold]["f1"]
    )


def _classify(domains: dict[str, dict[str, Any]], passed: bool) -> str:
    if passed:
        return "clip_disjoint_and_validation_transfer"
    seen = domains["seen_train"]
    same = domains["same_clip_unseen"]
    heldout = domains["heldout_clip"]
    val = domains["val"]

    def signal(domain: dict[str, Any]) -> bool:
        delta = domain["endpoint_minus_source"]
        return (
            float(delta["0.50"]["f1"]) > 0.0
            and float(delta["0.75"]["f1"]) > 0.0
            and float(domain["target_support_mass_gain"]) > 0.0
        )

    if signal(seen) and signal(same) and not (signal(heldout) and signal(val)):
        return "scene_or_clip_memorization_without_independent_transfer"
    if signal(seen) and not signal(heldout):
        return "training_subset_memorization_without_clip_transfer"
    if signal(heldout) and not signal(val):
        return "train_domain_transfer_without_validation_transfer"
    return "no_predeclared_generalization_signal"


def main() -> None:
    args = parse_args()
    manifest = _read(args.manifest)
    protocol = _read(manifest["protocol"])
    contract = _read(manifest["contract"])
    expected_domains = {
        "seen_train",
        "same_clip_unseen",
        "heldout_clip",
        "val",
    }
    if set(manifest["domains"]) != expected_domains:
        raise ValueError(
            "bridge manifest must contain exactly: "
            + ", ".join(sorted(expected_domains))
        )
    domains = {
        name: _domain(entry) for name, entry in manifest["domains"].items()
    }

    common_checks = {
        "list_protocol_passed": protocol.get("passed") is True,
        "initialization_contract_passed": contract.get("passed") is True,
        "test_set_not_used": (
            protocol.get("test_set_used") is False
            and manifest.get("test_set_used") is False
        ),
        "all_metrics_finite": all(
            domain["metrics_finite"] for domain in domains.values()
        ),
        "proposal_oracle_unchanged_all_domains": all(
            domain["proposal_oracle_unchanged"] for domain in domains.values()
        ),
        "semantic_duplicate_at_most_2pct_all_domains": all(
            domain["endpoint"]["semantic_duplicate_fraction"] <= 0.02
            for domain in domains.values()
        ),
    }
    heldout = domains["heldout_clip"]
    val = domains["val"]
    heldout_checks = {
        "heldout_support_mass_gain_at_least_2_points": heldout[
            "target_support_mass_gain"
        ]
        >= 0.02,
        "heldout_tp_050_gain_at_least_1": int(
            heldout["endpoint_minus_source"]["0.50"]["tp"]
        )
        >= 1,
        "heldout_tp_075_gain_at_least_2": int(
            heldout["endpoint_minus_source"]["0.75"]["tp"]
        )
        >= 2,
        "heldout_f1_050_not_below_source": _f1_not_worse(
            heldout["endpoint"], heldout["source"], "0.50"
        ),
        "heldout_f1_075_not_below_source": _f1_not_worse(
            heldout["endpoint"], heldout["source"], "0.75"
        ),
        "heldout_f1_050_not_below_initialization": _f1_not_worse(
            heldout["endpoint"], heldout["initialization"], "0.50"
        ),
        "heldout_f1_075_not_below_initialization": _f1_not_worse(
            heldout["endpoint"], heldout["initialization"], "0.75"
        ),
        "heldout_active_count_drift_at_most_0p05": abs(
            heldout["active_count_drift_from_source"]
        )
        <= 0.05,
    }
    val_checks = {
        "val_support_mass_gain_at_least_5_points": val[
            "target_support_mass_gain"
        ]
        >= 0.05,
        "val_tp_050_gain_at_least_3": int(
            val["endpoint_minus_source"]["0.50"]["tp"]
        )
        >= 3,
        "val_tp_075_gain_at_least_6": int(
            val["endpoint_minus_source"]["0.75"]["tp"]
        )
        >= 6,
        "val_f1_050_not_below_source": _f1_not_worse(
            val["endpoint"], val["source"], "0.50"
        ),
        "val_f1_075_not_below_source": _f1_not_worse(
            val["endpoint"], val["source"], "0.75"
        ),
        "val_f1_050_not_below_initialization": _f1_not_worse(
            val["endpoint"], val["initialization"], "0.50"
        ),
        "val_f1_075_not_below_initialization": _f1_not_worse(
            val["endpoint"], val["initialization"], "0.75"
        ),
        "val_active_count_drift_at_most_0p05": abs(
            val["active_count_drift_from_source"]
        )
        <= 0.05,
    }
    checks = {**common_checks, **heldout_checks, **val_checks}
    passed = all(checks.values())
    diagnosis = _classify(domains, passed)
    report = {
        "experiment": "V11 clip-disjoint 4096-image bridge gate",
        "iteration": int(manifest["iteration"]),
        "training_contract": {
            "train_images": 4096,
            "train_clips": 512,
            "effective_batch_size": 16,
            "optimizer_steps": 3000,
            "sample_exposures": 48000,
            "approximate_subset_epochs": 11.71875,
            "checkpoint_selection": "fixed_predeclared_endpoint_only",
        },
        "decision_contract": {
            "primary_domains": list(PRIMARY_DOMAINS),
            "seen_train_is_diagnostic_only": True,
            "same_clip_unseen_is_diagnostic_only": True,
            "score_threshold_sweep_used": False,
            "official_iou_thresholds": list(THRESHOLDS),
            "test_set_used": False,
        },
        "protocol": protocol,
        "contract_passed": contract.get("passed") is True,
        "domains": domains,
        "checks": checks,
        "diagnosis": diagnosis,
        "passed": passed,
        "long_training_authorized": False,
        "next_action": (
            "Run one all-train 3k validation-only pilot from the same V7 "
            "source; long training remains unauthorized."
            if passed
            else (
                "STOP V11. The clip-disjoint bridge did not establish robust "
                "transfer; do not use test, relax gates, or extend training."
            )
        ),
    }
    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
