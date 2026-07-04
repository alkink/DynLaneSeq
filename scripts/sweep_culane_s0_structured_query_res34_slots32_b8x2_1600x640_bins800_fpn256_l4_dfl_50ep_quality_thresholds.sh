#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
TOP_K_VALUES="${TOP_K_VALUES:-4}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5 0.7}"
QUALITY_POWERS="${QUALITY_POWERS:-0.25 0.50 0.75}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.28 0.29 0.30 0.31 0.32}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
EXACT_POSTPROCESS="${EXACT_POSTPROCESS:-0}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
OUT_ROOT="${OUT_ROOT:-${CKPT_DIR}}"
OUT_JSON="${OUT_JSON:-${OUT_ROOT}/quality_threshold_sweep_${CKPT_TAG}_val.json}"

EXTRA_ARGS=()
if [[ "${EXACT_POSTPROCESS}" == "1" || "${EXACT_POSTPROCESS}" == "true" || "${EXACT_POSTPROCESS}" == "TRUE" ]]; then
  EXTRA_ARGS+=(--exact-postprocess)
fi

python -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --reuse-cache \
  --top-k-values ${TOP_K_VALUES} \
  --iou-thresholds ${IOU_THRESHOLDS} \
  --quality-powers ${QUALITY_POWERS} \
  --score-thresholds ${SCORE_THRESHOLDS} \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --output-json "${OUT_JSON}" \
  "${EXTRA_ARGS[@]}"
