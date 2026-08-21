from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynlaneseq_eg.tools.summarize_v29_oof_belief_gate import _direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize one predeclared V29 OOF direction as an early stopping "
            "gate. A pass authorizes the reverse direction; it is not a final "
            "two-direction mechanism claim."
        )
    )
    parser.add_argument("--direction", required=True)
    parser.add_argument("--support-fold", choices=("a", "b"), required=True)
    parser.add_argument("--belief-fold", choices=("a", "b"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.support_fold == args.belief_fold:
        raise ValueError("support and belief folds must be clip-disjoint")
    expected_name = f"support_{args.support_fold}_to_fold_{args.belief_fold}"
    row = _direction(
        Path(args.direction).expanduser().resolve(),
        expected_name=expected_name,
        expected_belief_fold=args.belief_fold,
    )
    report = {
        "experiment": "V29 one-direction early OOF belief gate",
        "passed": bool(row["passed"]),
        "direction": row,
        "contract": {
            "checkpoint_selection_performed": False,
            "threshold_selection_performed": False,
            "full_official_validation": True,
            "test_set_used": False,
            "requires_reverse_direction_for_final_claim": True,
        },
        "recommendation": (
            "authorize_reverse_oof_direction"
            if row["passed"]
            else "stop_rbf_family_before_second_support"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
