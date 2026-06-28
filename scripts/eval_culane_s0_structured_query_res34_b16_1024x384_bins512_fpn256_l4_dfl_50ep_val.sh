#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_1024x384_bins512_fpn256_l4_dfl_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_structured_query_res34_b16_1024x384_bins512_fpn256_l4_dfl_50ep/iter_0025000.pt}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.40}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"
TOP_K="${TOP_K:-4}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"

CKPT_TAG="$(basename "${CKPT%.pt}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
NMS_TAG="${NMS_DISTANCE_THRESH_PX/./p}"
PRED_DIR="${PRED_DIR:-outputs/culane_s0_structured_query_res34_b16_1024x384_bins512_fpn256_l4_dfl_50ep/val_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms${NMS_TAG}}"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --pred-dir "${PRED_DIR}"
