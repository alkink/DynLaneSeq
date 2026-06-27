#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep.yaml}"
CKPT="${CKPT:-outputs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep/iter_0025000.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
OUT_JSON="${OUT_JSON:-outputs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep/quality_threshold_sweep.json}"

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
  --quality-powers 0.0 0.1 0.25 0.5 0.75 1.0 1.5 \
  --score-thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
  --output-json "${OUT_JSON}"
