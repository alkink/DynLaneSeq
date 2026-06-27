#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_CONFIG="${BASE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_50ep.yaml}"
BASE_CKPT="${BASE_CKPT:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}"
NEW_CONFIG="${NEW_CONFIG:-dynlaneseq_eg/configs/culane_s0_dynamic_row_evidence_headonly_25k.yaml}"
NEW_CKPT="${NEW_CKPT:-outputs/culane_s0_dynamic_row_evidence_headonly_25k/iter_0010000.pt}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_dynamic_row_evidence_headonly_25k/transition_vs_s0_val_10k}"
QUALITY_POWER="${QUALITY_POWER:-0.25}"
SCORE_THRESH="${SCORE_THRESH:-0.40}"
TOP_K="${TOP_K:-4}"

mkdir -p "${OUT_DIR}"

for IOU in 0.5 0.7; do
  python -u -m dynlaneseq_eg.tools.analyze_stage_transitions \
    --stage-configs "${BASE_CONFIG}" "${NEW_CONFIG}" \
    --stage-checkpoints "${BASE_CKPT}" "${NEW_CKPT}" \
    --stage-names s0_base dyn_row_evidence_headonly \
    --split "${SPLIT}" \
    --list-path "${EVAL_LIST}" \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_DIR}" \
    --reuse-cache \
    --exact-postprocess \
    --iou-thresh "${IOU}" \
    --score-thresholds "${SCORE_THRESH}" \
    --quality-powers "${QUALITY_POWER}" \
    --top-k-values "${TOP_K}" \
    --output-json "${OUT_DIR}/transitions_iou${IOU/./p}.json"
done
