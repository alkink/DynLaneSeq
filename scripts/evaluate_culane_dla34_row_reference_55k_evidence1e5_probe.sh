#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_floor_to75k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_probe_5k.yaml}"
CONTROL_CHECKPOINT="${CONTROL_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k/iter_0060000.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-outputs/diagnostics/dla34_rowref_from55k_evidence1e5_probe_5k/iter_0060000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_reference_evidence_lr_probe/floor2e6_vs_evidence1e5_iter60000_uniform64.json}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

for checkpoint in "${CONTROL_CHECKPOINT}" "${CANDIDATE_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing comparison checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done
mkdir -p "$(dirname "${OUTPUT_JSON}")"

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
