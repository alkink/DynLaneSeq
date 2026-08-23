#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DynLaneSeq
exec env \
  PYTHON=/venv/clrernet/bin/python \
  DATA_ROOT=/workspace/CULane \
  NUM_WORKERS=12 \
  EVAL_BATCH_SIZE=16 \
  METRIC_WORKERS=16 \
  bash scripts/run_culane_v31_exact_pair_35k_to50k.sh
