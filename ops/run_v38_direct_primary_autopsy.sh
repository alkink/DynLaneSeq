#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/DynLaneSeq}"
PYTHON="${PYTHON:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/diagnostics/v38_direct_primary_maturity/autopsy_50k}"

cd "${ROOT}"

"${PYTHON}" -m dynlaneseq_eg.tools.audit_v38_direct_primary_autopsy \
  --config dynlaneseq_eg/configs/culane_v38_direct_primary_maturity_50k.yaml \
  --checkpoint outputs/diagnostics/v38_direct_primary_maturity/train/iter_0050000.pt \
  --v38-report outputs/diagnostics/v38_direct_primary_maturity/official_val_50k/v38_direct_primary_50k_official_val.json \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --cache-dir "${OUTPUT_ROOT}/cache" \
  --device cuda \
  --eval-batch-size 16 \
  --num-workers 6 \
  --official-workers 12 \
  --channels-last \
  "$@"
