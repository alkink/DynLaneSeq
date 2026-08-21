from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dynlaneseq_eg.evaluation.candidate_diagnostics import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predeclared training-fold-only capacity gate for a shortened V29 "
            "support endpoint. It never opens official validation or test."
        )
    )
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-records", type=int, default=1024)
    parser.add_argument("--min-all32-recall-50", type=float, default=0.90)
    parser.add_argument("--min-all32-recall-75", type=float, default=0.60)
    parser.add_argument("--min-recoverable-gap-50", type=float, default=0.05)
    return parser.parse_args()


def _row(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    threshold: float,
    top_k: int | None = None,
) -> dict[str, Any]:
    found = [
        row
        for row in rows
        if str(row.get("stage")) == "main"
        and str(row.get("strategy")) == strategy
        and abs(float(row.get("iou_threshold", -1.0)) - threshold) <= 1.0e-9
        and (top_k is None or int(row.get("top_k", -1)) == top_k)
    ]
    if len(found) != 1:
        raise ValueError(
            f"expected one {strategy} row at IoU {threshold}, found {len(found)}"
        )
    return found[0]


def main() -> None:
    args = parse_args()
    source = Path(args.oracle_report).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    all50 = _row(rows, strategy="all_raw", threshold=0.50)
    all75 = _row(rows, strategy="all_raw", threshold=0.75)
    deployed50 = _row(
        rows,
        strategy="model_topk_nms",
        threshold=0.50,
        top_k=4,
    )
    oracle50 = _row(
        rows,
        strategy="oracle_topk",
        threshold=0.50,
        top_k=4,
    )
    records = int(payload.get("metadata", {}).get("num_records", 0))
    recoverable_gap = float(oracle50["recall"]) - float(deployed50["recall"])
    checks = {
        "training_fold_only": str(payload.get("metadata", {}).get("split"))
        == "train",
        "sample_population_large_enough": records >= int(args.min_records),
        "all32_recall_0p50_sufficient": float(all50["recall"])
        >= float(args.min_all32_recall_50),
        "all32_recall_0p75_sufficient": float(all75["recall"])
        >= float(args.min_all32_recall_75),
        "recoverable_gap_0p50_sufficient": recoverable_gap
        >= float(args.min_recoverable_gap_50),
    }
    report = {
        "experiment": "V29 shortened-support OOF bank sufficiency gate",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "records": records,
            "all32_recall_0p50": float(all50["recall"]),
            "all32_recall_0p75": float(all75["recall"]),
            "deployed_top4_recall_0p50": float(deployed50["recall"]),
            "oracle_top4_recall_0p50": float(oracle50["recall"]),
            "recoverable_gap_0p50": recoverable_gap,
        },
        "oracle_report": str(source),
        "oracle_report_sha256": sha256_file(source),
        "contract": {
            "checkpoint_selection_performed": False,
            "validation_used": False,
            "test_set_used": False,
            "failure_action": "extend_same_fold_support_before_belief_training",
            "pass_action": "authorize_first_oof_belief_direction",
        },
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
