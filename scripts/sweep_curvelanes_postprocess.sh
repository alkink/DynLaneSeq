#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CKPT:?Set CKPT to one CurveLanes checkpoint}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/Datasets/CurveLanes/Curvelanes}"
SPLIT="${SPLIT:-val}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.30}"
QUALITY_POWERS="${QUALITY_POWERS:-0.50}"
TOP_KS="${TOP_KS:-5 8 0}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
RESULT_DIR="${RESULT_DIR:-${CKPT_DIR}/${SPLIT}_postprocess_topk_${CKPT_TAG}}"

read -r -a SCORE_ARGS <<< "${SCORE_THRESHOLDS}"
read -r -a QUALITY_ARGS <<< "${QUALITY_POWERS}"
read -r -a TOPK_ARGS <<< "${TOP_KS}"

PIN_MEMORY_ARGS=()
if [[ "${PIN_MEMORY:-0}" == "1" ]]; then
  PIN_MEMORY_ARGS+=(--pin-memory)
fi

python -u -m dynlaneseq_eg.tools.sweep_curvelanes_postprocess \
  --config "${CONFIG}" \
  --checkpoints "${CKPT}" \
  --dataset-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --device "${DEVICE:-cuda}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --score-thresholds "${SCORE_ARGS[@]}" \
  --quality-powers "${QUALITY_ARGS[@]}" \
  --top-ks "${TOPK_ARGS[@]}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --cache-dir "${RESULT_DIR}/cache" \
  --reuse-cache \
  --output-json "${RESULT_DIR}/sweep.json" \
  --output-csv "${RESULT_DIR}/sweep.csv" \
  "${PIN_MEMORY_ARGS[@]}"
