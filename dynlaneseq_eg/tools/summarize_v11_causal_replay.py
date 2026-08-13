from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DOMAINS = ("heldout_clip", "val")
PRIMARY = (
    "v7_geometry__v7_activity",
    "v11_coarse__v7_activity",
    "v11_final__v7_activity",
    "v7_geometry__v11_activity",
    "v11_final__v11_activity",
    "v7_anchor_plus_v11_residual__v7_activity",
    "v7_anchor_plus_v11_residual__v11_activity",
)
ABLATIONS = (
    "p2_zero_content",
    "p2_wrong_image",
    "p2_x_reversed",
    "p2_row_reversed",
    "proposal_tokens_zero",
    "proposal_tokens_wrong_image",
    "proposal_all_wrong_image",
    "legacy_route_prior_zero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine held-out and validation V11 causal replays."
    )
    parser.add_argument("--heldout-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _metric(
    report: dict[str, Any],
    policy: str,
    threshold: str,
) -> dict[str, Any]:
    return report["metrics"][policy]["writer_valid"]["thresholds"][threshold]


def _delta(
    report: dict[str, Any],
    treatment: str,
    control: str,
    threshold: str,
) -> dict[str, float | int]:
    lhs = _metric(report, treatment, threshold)
    rhs = _metric(report, control, threshold)
    return {
        "delta_tp": int(lhs["tp"]) - int(rhs["tp"]),
        "delta_fp": int(lhs["fp"]) - int(rhs["fp"]),
        "delta_fn": int(lhs["fn"]) - int(rhs["fn"]),
        "delta_predictions": int(lhs["predictions"])
        - int(rhs["predictions"]),
        "delta_f1_points": 100.0 * (float(lhs["f1"]) - float(rhs["f1"])),
    }


def _domain_summary(report: dict[str, Any]) -> dict[str, Any]:
    thresholds = tuple(f"{float(value):.2f}" for value in report["thresholds"])
    primary = {
        policy: {threshold: _metric(report, policy, threshold) for threshold in thresholds}
        for policy in PRIMARY
    }
    comparisons = {
        "fixed_activity_final_geometry_minus_v7": {
            threshold: _delta(
                report,
                "v11_final__v7_activity",
                "v7_geometry__v7_activity",
                threshold,
            )
            for threshold in thresholds
        },
        "learned_activity_on_v11_geometry": {
            threshold: _delta(
                report,
                "v11_final__v11_activity",
                "v11_final__v7_activity",
                threshold,
            )
            for threshold in thresholds
        },
        "coarse_geometry_minus_v7": {
            threshold: _delta(
                report,
                "v11_coarse__v7_activity",
                "v7_geometry__v7_activity",
                threshold,
            )
            for threshold in thresholds
        },
        "final_decoder_minus_coarse": {
            threshold: _delta(
                report,
                "v11_final__v7_activity",
                "v11_coarse__v7_activity",
                threshold,
            )
            for threshold in thresholds
        },
        "learned_residual_on_v7_anchor_minus_v7": {
            threshold: _delta(
                report,
                "v7_anchor_plus_v11_residual__v7_activity",
                "v7_geometry__v7_activity",
                threshold,
            )
            for threshold in thresholds
        },
    }
    evidence: dict[str, Any] = {}
    for variant in ABLATIONS:
        policy = f"ablation_{variant}__v7_activity"
        if policy not in report["metrics"]:
            continue
        # Positive means the correctly paired evidence is better than the
        # intervention, which is the intuitive direction for causality.
        evidence[variant] = {
            threshold: _delta(
                report,
                "v11_final__v7_activity",
                policy,
                threshold,
            )
            for threshold in thresholds
        }
    return {
        "images": int(report["images"]),
        "split": report["split"],
        "list_path": report["list_path"],
        "list_sha256": report["list_sha256"],
        "source_iteration": int(report["source_iteration"]),
        "v11_iteration": int(report["v11_iteration"]),
        "alignment_and_replay_contract": report[
            "alignment_and_replay_contract"
        ],
        "primary_metrics": primary,
        "comparisons": comparisons,
        "evidence_correct_minus_intervention": evidence,
        "geometry_activity_shapley": report["geometry_activity_shapley"][
            "writer_valid"
        ],
        "single_domain_diagnosis": report["diagnosis"],
    }


def main() -> None:
    args = parse_args()
    reports = {
        "heldout_clip": _load(args.heldout_json),
        "val": _load(args.val_json),
    }
    for name, report in reports.items():
        if report.get("test_set_used") is not False:
            raise ValueError(f"test-set contract is not closed for {name}")
        if report.get("optimizer_steps") != 0:
            raise ValueError(f"causal replay unexpectedly optimized in {name}")
        contract = report.get("alignment_and_replay_contract", {})
        if contract.get("slot_alignment_passed") is not True:
            raise ValueError(f"slot-alignment contract failed for {name}")
    summaries = {
        name: _domain_summary(reports[name]) for name in DOMAINS
    }
    report = {
        "experiment": "V11 heldout+validation causal replay summary",
        "domains": summaries,
        "test_set_used": False,
        "optimizer_steps": 0,
        "current_v11_long_training_authorized": False,
        "v11_1_short_training_automatically_authorized": False,
        "decision_status": "requires_causal_result_review",
        "decision_rule": (
            "Review fixed-activity final geometry, final-minus-coarse, and "
            "correct-minus-evidence-intervention jointly.  This summary does "
            "not convert a post-hoc autopsy into an automatic training pass."
        ),
    }
    output = Path(args.output_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
