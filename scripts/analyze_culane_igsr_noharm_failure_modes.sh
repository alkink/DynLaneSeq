#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Read-only model diagnostics. This does not train or write CULane predictions.
# It runs model inference once to cache raw coarse/final candidates, then reuses
# that cache for Oracle Top-K, score/duplicate, and stage-transition analyses.

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_igsr_joint_controlled_noharm_25k.yaml}"
CKPT="${CKPT:-outputs/culane_igsr_joint_controlled_noharm_25k/iter_0025000.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
OUT_DIR="${OUT_DIR:-outputs/culane_igsr_joint_controlled_noharm_25k/failure_diagnostics_val}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RUN_TRANSITIONS="${RUN_TRANSITIONS:-1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_LIST}" ]]; then
  echo "Missing evaluation list: ${EVAL_LIST}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}" "${CACHE_DIR}"

COMMON=(
  --config "${CONFIG}"
  --checkpoint "${CKPT}"
  --split "${SPLIT}"
  --list-path "${EVAL_LIST}"
  --device "${DEVICE}"
  --line-width 30
  --min-valid-rows 5
  --cache-dir "${CACHE_DIR}"
  --reuse-cache
  --exact-postprocess
  --max-batches "${MAX_BATCHES}"
)

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  "${COMMON[@]}" \
  --top-k-values 4 6 8 \
  --iou-thresholds 0.3 0.5 0.7 \
  --quality-powers 0.0 \
  --score-thresholds 0.5 \
  --output-json "${OUT_DIR}/oracle_topk.json"

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_score_distribution \
  "${COMMON[@]}" \
  --iou-thresh 0.5 \
  --quality-powers 0.0 \
  --score-thresholds 0.4 0.5 0.55 \
  --top-k-values 4 6 8 \
  --output-json "${OUT_DIR}/score_duplicates.json"

if [[ "${RUN_TRANSITIONS}" == "1" ]]; then
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_stage_transitions \
    "${COMMON[@]}" \
    --iou-thresh 0.5 \
    --quality-powers 0.0 \
    --score-thresholds 0.5 \
    --top-k-values 4 \
    --output-json "${OUT_DIR}/coarse_to_final_transitions.json"
fi

echo "Diagnostics written to ${OUT_DIR}"
