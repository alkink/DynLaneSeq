#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep/iter_0025000.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep/failure_diagnostics_val}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"
SCORE_THRESH="${SCORE_THRESH:-0.40}"

mkdir -p "${OUT_DIR}"

python -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT}" \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --cache-dir "${CACHE_DIR}" \
  --reuse-cache \
  --exact-postprocess \
  --top-k-values 4 \
  --iou-thresholds 0.5 0.7 \
  --quality-powers "${QUALITY_POWER}" \
  --score-thresholds "${SCORE_THRESH}" \
  --output-json "${OUT_DIR}/oracle_topk.json"
