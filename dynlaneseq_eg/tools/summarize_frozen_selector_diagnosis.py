from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine unified-selector ownership churn and frozen-selector "
            "capacity probes into one bounded diagnosis."
        )
    )
    parser.add_argument("--ownership-json", required=True)
    parser.add_argument("--selector-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(
    ownership: dict[str, Any],
    selector: dict[str, Any],
) -> dict[str, Any]:
    ownership_unstable = bool(ownership["gate"]["ownership_unstable"])
    selector_gate = selector["gate"]
    frozen_positive = bool(selector_gate["stationary_frozen_selector_positive"])
    matcher_positive = bool(
        selector_gate.get("stationary_original_matcher_target_positive", frozen_positive)
    )
    official_positive = bool(
        selector_gate.get("stationary_official_target_positive", False)
    )
    if matcher_positive and ownership_unstable:
        diagnosis = "moving_owner_targets_are_primary"
        next_action = (
            "use a stable canonical owner/teacher or freeze geometry while "
            "warming the unified selector before joint optimization"
        )
        confidence = "strong"
    elif matcher_positive:
        diagnosis = "selector_capacity_exists_but_joint_optimization_fails"
        next_action = (
            "separate selector warmup from joint geometry optimization and "
            "audit selector-to-decoder gradient coupling"
        )
        confidence = "moderate"
    elif official_positive:
        diagnosis = "training_selection_target_is_misaligned_with_official_metric"
        next_action = (
            "replace the range-aware matcher membership target with a stable "
            "official-metric-aligned surrogate before another joint run"
        )
        confidence = "strong"
    elif ownership_unstable:
        diagnosis = "owner_churn_exists_but_scalar_selector_remains_insufficient"
        next_action = (
            "replace independent scalar membership with explicit diverse-set "
            "selection (selection without replacement or coverage slots)"
        )
        confidence = "moderate"
    else:
        diagnosis = "scalar_set_selector_or_its_features_are_insufficient"
        next_action = (
            "replace independent scalar membership with explicit diverse-set "
            "selection; do not extend the current full training"
        )
        confidence = "strong"
    return {
        "diagnostic_only": True,
        "ownership_unstable": ownership_unstable,
        "stationary_frozen_selector_positive": frozen_positive,
        "stationary_original_matcher_target_positive": matcher_positive,
        "stationary_official_target_positive": official_positive,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "next_action": next_action,
        "limits": (
            "This localizes the selection failure family. It does not prove a "
            "specific full-training F1 gain or guarantee 80+ CULane F1."
        ),
        "ownership_gate": ownership["gate"],
        "selector_gate": selector_gate,
    }


def main() -> None:
    args = parse_args()
    result = summarize(_load(args.ownership_json), _load(args.selector_json))
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {path}")


if __name__ == "__main__":
    main()
