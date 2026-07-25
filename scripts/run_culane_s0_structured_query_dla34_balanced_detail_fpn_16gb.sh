#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Safe measured setting for a 16 GB accelerator.  The effective batch remains
# 16, while the explicit output name records the changed micro-batch.
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep}"
export SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export COMPILE_MODE="${COMPILE_MODE:-default}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-default}"

exec bash scripts/run_culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep.sh
