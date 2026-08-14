from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed paired V20 treatment/control gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--heldout-treatment", required=True)
    parser.add_argument("--validation-treatment", required=True)
    parser.add_argument("--heldout-control", required=True)
    parser.add_argument("--validation-control", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row(report: dict[str, Any], policy: str, threshold: str) -> dict[str, Any]:
    return report["metrics"][policy]["writer_valid"]["thresholds"][threshold]


def main() -> None:
    args = parse_args()
    contract = _read(args.contract)
    cache = _read(args.cache_manifest)
    reports = {
        "heldout": {
            "treatment": _read(args.heldout_treatment),
            "control": _read(args.heldout_control),
        },
        "validation": {
            "treatment": _read(args.validation_treatment),
            "control": _read(args.validation_control),
        },
    }
    domains: dict[str, Any] = {}
    all_pass = bool(contract.get("passed")) and bool(
        cache.get("contract", {}).get("passed")
    )
    for domain, arms in reports.items():
        treatment = arms["treatment"]
        control = arms["control"]
        threshold_rows: dict[str, Any] = {}
        domain_pass = bool(treatment["parity"]["passed"]) and bool(
            control["parity"]["passed"]
        )
        for threshold in ("0.50", "0.75"):
            source = _row(treatment, "source_v7", threshold)
            treated = _row(treatment, "v20_deployed", threshold)
            controlled = _row(control, "v20_deployed", threshold)
            treatment_gain = int(treated["tp"]) - int(source["tp"])
            causal_gain = int(treated["tp"]) - int(controlled["tp"])
            row_pass = all(
                (
                    treatment_gain >= 5,
                    causal_gain >= 5,
                    float(treated["f1"]) >= float(source["f1"]),
                    int(treated["predictions"]) == int(source["predictions"]),
                )
            )
            domain_pass &= row_pass
            threshold_rows[threshold] = {
                "source": source,
                "treatment": treated,
                "control": controlled,
                "treatment_minus_source_tp": treatment_gain,
                "treatment_minus_control_tp": causal_gain,
                "passed": row_pass,
            }
        replacement = treatment["replacement"]
        degradation = treatment["degradation"]
        safety_pass = all(
            (
                float(replacement["precision"]) >= 0.70,
                int(replacement.get("beneficial", 0))
                >= 2 * int(replacement.get("harmful", 0)),
                float(degradation["source_correct_loss_fraction_50"]) < 0.01,
                float(degradation["source_correct_loss_fraction_75"]) < 0.01,
            )
        )
        domain_pass &= safety_pass
        domains[domain] = {
            "thresholds": threshold_rows,
            "replacement": replacement,
            "degradation": degradation,
            "safety_passed": safety_pass,
            "passed": domain_pass,
        }
        all_pass &= domain_pass
    report = {
        "experiment": "V20 paired slot-owned safe replacement fixed gate",
        "passed": bool(all_pass),
        "decision": (
            "V20_ONE_EDIT_PASS_STOP_FOR_PLANNING"
            if all_pass
            else "V20_ONE_EDIT_FAIL_STOP"
        ),
        "contract_passed": bool(contract.get("passed")),
        "cache_contract_passed": bool(cache.get("contract", {}).get("passed")),
        "domains": domains,
        "next_version_authorized": False,
        "second_edit_authorized": False,
        "long_training_authorized": False,
        "full_validation_authorized": False,
        "test_authorized": False,
        "checkpoint_selection_performed": False,
        "threshold_search_performed": False,
        "nms_search_performed": False,
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

