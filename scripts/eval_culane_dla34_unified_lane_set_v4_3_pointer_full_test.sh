#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# STOP is the cardinality decision.  A second scalar threshold would silently
# change the learned deployment contract, so the pointer path uses zero.
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_3_pointer_stop.yaml}" \
CKPT="${CKPT:-outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer_stop/iter_0065000.pt}" \
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
SCORE_THRESH=0.0 \
QUALITY_POWER=0.0 \
TOP_K=4 \
NMS_DISTANCE_THRESH_PX=0.0 \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}" \
AMP_DTYPE="${AMP_DTYPE:-none}" \
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.75}" \
CATEGORIES="${CATEGORIES:---categories}" \
  bash scripts/eval_culane_dla34_row_reference_full_test.sh
