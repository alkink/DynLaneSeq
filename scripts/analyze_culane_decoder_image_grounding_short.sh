#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/home/alki/projects/CULane}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/alki/projects/DynLaneSeq/outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_ROOT}/diagnostics/decoder_image_grounding}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-32}"
SHIFT_COLS="${SHIFT_COLS:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-uniform}"

R34_CONFIG="${R34_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
R34_CKPT="${R34_CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
DLA_CONFIG="${DLA_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
DLA_CKPT="${DLA_CKPT:-${CHECKPOINT_ROOT}/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"

mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 2
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  python -u -m dynlaneseq_eg.tools.analyze_decoder_image_grounding \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-batches "${MAX_BATCHES}" \
    --sample-strategy "${SAMPLE_STRATEGY}" \
    --shift-cols "${SHIFT_COLS}" \
    --line-width 30.0 \
    --iou-thresholds 0.5 0.7 \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${OUTPUT_ROOT}/${name}.json"
}

run_one r34_225k "${R34_CONFIG}" "${R34_CKPT}"
run_one dla34_225k "${DLA_CONFIG}" "${DLA_CKPT}"
