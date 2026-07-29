#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_reference_gate_control_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_row_reference_gate_10k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_reference_gate_control_10k/iter_0010000.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_row_reference_gate_10k/iter_0010000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_reference_gate/dla34_control_vs_candidate_10k_uniform64.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

exec "${PYTHON}" -u -m dynlaneseq_eg.tools.evaluate_row_reference_gate \
  --control-config "${CONTROL_CONFIG}" \
  --control-checkpoint "${CONTROL_CHECKPOINT}" \
  --candidate-config "${CANDIDATE_CONFIG}" \
  --candidate-checkpoint "${CANDIDATE_CHECKPOINT}" \
  --dataset-root "${DATA_ROOT}" \
  --split val \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --sample-strategy uniform \
  --line-width 30.0 \
  --top-k 4 \
  --amp-dtype "${AMP_DTYPE}" \
  --min-recovered-lanes 5 \
  --max-lost-lanes 2 \
  --min-image-specificity-points 5.0 \
  --output-json "${OUTPUT_JSON}"

