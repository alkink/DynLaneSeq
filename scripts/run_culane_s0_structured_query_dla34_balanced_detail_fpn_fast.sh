#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# The optimized structured path keeps the original 8x2 experiment below the
# audited 24 GB memory cliff, so micro-batch statistics remain unchanged.
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"

# Compilation is the fastest measured path on Ampere and also reduces peak
# memory.  Once compiled, PyTorch's default attention backend is faster than
# explicitly forcing Flash for this complete model.
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export COMPILE_MODE="${COMPILE_MODE:-default}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-default}"

exec bash scripts/run_culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep.sh
