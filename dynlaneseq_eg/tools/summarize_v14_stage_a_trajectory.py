from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate fixed, debug-only V14 Stage-A checkpoints."
    )
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.report_root)
    rows = []
    for iteration in args.iterations:
        row = {"iteration": int(iteration), "domains": {}}
        for domain in ("heldout", "validation"):
            path = root / f"{domain}_{int(iteration):07d}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            mean = lambda name: float(report["summaries"][name]["mean"])
            row["domains"][domain] = {
                "correct_support_mass": mean("correct_p2_support_mass"),
                "v7_support_mass": mean("v7_support_mass"),
                "correct_hard_target_top1": mean(
                    "correct_p2_hard_target_top1"
                ),
                "v7_hard_target_top1": mean("v7_hard_target_top1"),
                "cross_clip_wrong_support_mass": mean(
                    "cross_clip_wrong_p2_support_mass"
                ),
                "zero_content_support_mass": mean(
                    "zero_content_support_mass"
                ),
                "position_only_support_mass": mean(
                    "position_only_support_mass"
                ),
                "correct_visual_dfl": mean("correct_p2_visual_dfl"),
                "cross_clip_wrong_visual_dfl": mean(
                    "cross_clip_wrong_p2_visual_dfl"
                ),
                "position_only_visual_dfl": mean(
                    "position_only_visual_dfl"
                ),
            }
        rows.append(row)
    report = {
        "experiment": "V14 Stage-A fixed-checkpoint trajectory",
        "checkpoint_selection_performed": False,
        "fixed_endpoint_iteration": max(args.iterations),
        "intermediate_checkpoint_role": "debug_only",
        "trajectory": rows,
        "test_set_used": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
