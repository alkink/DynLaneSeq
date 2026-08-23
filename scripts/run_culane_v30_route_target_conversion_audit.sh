#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-outputs/diagnostics/v30_field_only_exact_pair_30k_to35k/stage_autopsy}"
CONTROL_CACHE="${CONTROL_CACHE:-$ROOT/cache/control/iter_0035000_44b5a866fcbbfa2e.pt}"
TREATMENT_CACHE="${TREATMENT_CACHE:-$ROOT/cache/field_only/iter_0035000_f70077edecefddc8.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT/route_target_conversion_strict075.json}"
PYTHON="${PYTHON:-python}"

"$PYTHON" -u -m dynlaneseq_eg.tools.audit_v30_route_target_conversion \
  --control-cache "$CONTROL_CACHE" \
  --treatment-cache "$TREATMENT_CACHE" \
  --iou-threshold 0.75 \
  --line-width 30.0 \
  --min-valid-rows 5 \
  --cluster-min 0.0 \
  --cluster-delta 0.10 \
  --cluster-temperature 0.03 \
  --sample-limit 50 \
  --output-json "$OUTPUT_JSON"
