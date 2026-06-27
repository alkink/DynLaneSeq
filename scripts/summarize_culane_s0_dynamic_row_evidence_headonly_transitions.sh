#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-outputs/culane_s0_dynamic_row_evidence_headonly_25k/transition_vs_s0_val_10k}"

python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", "outputs/culane_s0_dynamic_row_evidence_headonly_25k/transition_vs_s0_val_10k"))
for name in ("transitions_iou0p5.json", "transitions_iou0p7.json"):
    path = out_dir / name
    if not path.exists():
        print(f"missing: {path}")
        continue
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n== {name} ==")
    for result in data.get("results", []):
        print(
            f"q={result.get('quality_power')} thr={result.get('score_threshold')} "
            f"K={result.get('top_k')}"
        )
        for stage in result.get("stages", []):
            print(
                f"  {stage['stage']}: selected_tp={stage['selected_tp']} "
                f"selected_fp={stage['selected_fp']}"
            )
        for pair in result.get("pairs", []):
            summary = pair.get("summary", {})
            print(f"  {pair['from']} -> {pair['to']}:")
            for key in (
                "rescued_tp",
                "killed_by_geometry",
                "killed_by_score",
                "killed_by_nms",
                "killed_by_topk",
                "selected_tp_before",
                "selected_tp_after",
                "selected_fp_before",
                "selected_fp_after",
                "fp_removed",
            ):
                print(f"    {key}: {summary.get(key, 0)}")
            transitions = pair.get("gt_state_transitions", {})
            interesting = sorted(transitions.items(), key=lambda item: item[1], reverse=True)[:12]
            print("    top_gt_state_transitions:")
            for transition, count in interesting:
                print(f"      {transition}: {count}")
PY
