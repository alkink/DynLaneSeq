from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY = "adaptive_voronoi_060"
DOMAINS = ("heldout_clip_256", "validation_256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the predeclared V16 group-capacity gate.")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _domain(report: dict[str, Any]) -> dict[str, Any]:
    policy = report["policies"][PRIMARY]
    metric50 = report["official_metrics"][f"{PRIMARY}/fixed_assignment_oracle/0.50"]
    metric75 = report["official_metrics"][f"{PRIMARY}/fixed_assignment_oracle/0.75"]
    checks = {
        "test_closed": report["scope"].get("test_set_used") is False,
        "training_not_performed": report["scope"].get("training_performed") is False,
        "no_coordinate_averaging": report["scope"].get("proposal_coordinates_averaged") is False,
        "no_fixed_k_or_padding": report["scope"].get("fixed_k_or_padding_used") is False,
        "anchors_retained_exactly": abs(float(policy["anchor_retention"]) - 1.0) <= 1.0e-12,
        "groups_disjoint": int(policy["duplicate_memberships"]) == 0,
        "target_support_any_coverage_at_least_0p90": float(
            policy["target_support_any_coverage"]
        ) >= 0.90,
        "near_equivalent_coverage_at_least_0p80": float(
            policy["near_equivalent_coverage"]
        ) >= 0.80,
        "fixed_assignment_gap_closure_050_at_least_0p65": float(
            metric50["global_gap_closure"]
        ) >= 0.65,
        "fixed_assignment_gap_closure_075_at_least_0p70": float(
            metric75["global_gap_closure"]
        ) >= 0.70,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "policy": policy,
        "fixed_assignment_oracle": {"0.50": metric50, "0.75": metric75},
        "current_reference": {
            "0.50": report["official_metrics"]["current_reference/0.50"],
            "0.75": report["official_metrics"]["current_reference/0.75"],
        },
        "all32_same_count_oracle": {
            "0.50": report["official_metrics"]["all32_same_count_oracle/0.50"],
            "0.75": report["official_metrics"]["all32_same_count_oracle/0.75"],
        },
        "metadata": report["metadata"],
    }


def summarize(heldout: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    domains = {
        "heldout_clip_256": _domain(heldout),
        "validation_256": _domain(validation),
    }
    passed = all(domain["passed"] for domain in domains.values())
    return {
        "experiment": "V16 variable-size anchor candidate-group preflight gate",
        "formal_result": "PASS" if passed else "FAIL",
        "passed": passed,
        "primary_policy": PRIMARY,
        "predeclared_thresholds": {
            "target_support_any_coverage_min": 0.90,
            "near_equivalent_coverage_min": 0.80,
            "fixed_assignment_gap_closure_min": {"0.50": 0.65, "0.75": 0.70},
            "anchor_retention": 1.0,
            "duplicate_memberships": 0,
        },
        "domains": domains,
        "stage_a_authorized": passed,
        "long_training_authorized": False,
        "full_validation_authorized": False,
        "test_set_used": False,
        "decision": (
            "implement_and_run_short_v16_candidate_reranker_stage_a_then_stop"
            if passed
            else "stop_v16_before_training_candidate_groups_do_not_retain_enough_headroom"
        ),
    }


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# V16 candidate-group preflight decision",
        "",
        f"Formal result: **{report['formal_result']}**.",
        "",
        "| Domain | Support coverage | Near-equivalent | Gap closure .50 | Gap closure .75 | PASS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in DOMAINS:
        domain = report["domains"][name]
        policy = domain["policy"]
        metric = domain["fixed_assignment_oracle"]
        lines.append(
            f"| `{name}` | {policy['target_support_any_coverage']:.4f} | "
            f"{policy['near_equivalent_coverage']:.4f} | "
            f"{metric['0.50']['global_gap_closure']:.4f} | "
            f"{metric['0.75']['global_gap_closure']:.4f} | "
            f"{domain['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{report['decision']}`.",
            "",
            "This gate never authorizes full validation, long training, test, or threshold search.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = summarize(_load(args.heldout), _load(args.validation))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    print(json.dumps({"passed": report["passed"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
