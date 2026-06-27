#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRED_DIR="${PRED_DIR:-outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_thr0p40_q0p25}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25}"
DATASET_ROOT="${DATASET_ROOT:-dataset}"
SPLIT_DIR="${SPLIT_DIR:-dataset/list/test_split}"
NUM_PER_CATEGORY="${NUM_PER_CATEGORY:-10}"
WORKERS="${WORKERS:-12}"

python -u -m dynlaneseq_eg.tools.visualize_culane_failure_overlays \
  --dataset-root "${DATASET_ROOT}" \
  --pred-dir "${PRED_DIR}" \
  --split-dir "${SPLIT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --num-per-category "${NUM_PER_CATEGORY}" \
  --workers "${WORKERS}"
