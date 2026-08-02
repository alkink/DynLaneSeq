#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.20 0.30}"
ITERATIONS="${ITERATIONS:-}"
REUSE_REPORTS="${REUSE_REPORTS:-0}"
RESULT_DIR="${RESULT_DIR:-outputs/diagnostics/unified_lane_set_v3_checkpoint_trajectory}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
mkdir -p "${RESULT_DIR}"

checkpoints=()
if [[ -n "${ITERATIONS}" ]]; then
  for iteration in ${ITERATIONS}; do
    checkpoints+=("${CHECKPOINT_DIR}/iter_$(printf '%07d' "${iteration}").pt")
  done
else
  mapfile -t checkpoints < <(
    find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' | sort -V
  )
fi
if (( ${#checkpoints[@]} == 0 )); then
  echo "No iter_*.pt checkpoints found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi
for checkpoint in "${checkpoints[@]}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing requested checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

echo "Unified lane-set V3 checkpoint trajectory"
echo "checkpoints: ${#checkpoints[@]}"
printf '  %s\n' "${checkpoints[@]}"
echo "sample/checkpoint: $((EVAL_BATCH_SIZE * MAX_BATCHES)) validation images"
echo "result directory: ${RESULT_DIR}"

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_unified_lane_set_checkpoint_parameters \
  --inputs "${checkpoints[@]}" \
  --input-width 1600 \
  --output-json "${RESULT_DIR}/parameter_trajectory.json" \
  --output-csv "${RESULT_DIR}/parameter_trajectory.csv"

reports=()
for checkpoint in "${checkpoints[@]}"; do
  iteration="$(${PYTHON} - "${checkpoint}" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
PY
)"
  if (( iteration < 0 )); then
    echo "Checkpoint has no valid iteration: ${checkpoint}" >&2
    exit 1
  fi
  report="${RESULT_DIR}/contract_iter_$(printf '%07d' "${iteration}")_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  if [[ "${REUSE_REPORTS}" != "1" || ! -f "${report}" ]]; then
    DATA_ROOT="${DATA_ROOT}" \
    CONFIG="${CONFIG}" \
    CKPT="${checkpoint}" \
    DEVICE="${DEVICE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_BATCHES="${MAX_BATCHES}" \
    GRADIENT_IMAGES=0 \
    AMP_DTYPE="${AMP_DTYPE}" \
    SCORE_THRESHOLDS="${SCORE_THRESHOLDS}" \
    TOP_K=4 \
    OUTPUT_JSON="${report}" \
      bash scripts/audit_culane_dla34_unified_lane_set_v3_short.sh
  else
    echo "Reusing trajectory report: ${report}"
  fi
  reports+=("${report}")
done

"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_unified_lane_set_contract_trajectory \
  --inputs "${reports[@]}" \
  --output-json "${RESULT_DIR}/trajectory_summary.json" \
  --output-csv "${RESULT_DIR}/trajectory_summary.csv"

echo "V3 checkpoint trajectory completed."
