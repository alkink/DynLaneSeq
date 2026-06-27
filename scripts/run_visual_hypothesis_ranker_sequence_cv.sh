#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_50ep.yaml}
CKPT=${CKPT:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}
CANDIDATE_CACHE=${CANDIDATE_CACHE:-outputs/diagnostic_cache/iter_0175000_267b14df707bf94c.pt}
BASE_OUT_DIR=${BASE_OUT_DIR:-outputs/culane_s0_structured_query_res34_b16_50ep/failure_diagnostics_val_175k/visual_hypothesis_ranker}
FEATURE_CACHE=${FEATURE_CACHE:-${BASE_OUT_DIR}/visual_rank_features.pt}
CV_OUT_DIR=${CV_OUT_DIR:-${BASE_OUT_DIR}/sequence_cv}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-80}
SPLIT_SEEDS=${SPLIT_SEEDS:-"2022 2023 2024 2025 2026"}
ROW_SEGMENTS=${ROW_SEGMENTS:-8}
HIDDEN_DIM=${HIDDEN_DIM:-128}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-128}

mkdir -p "${CV_OUT_DIR}"

for SPLIT_SEED in ${SPLIT_SEEDS}; do
  echo "visual hypothesis-ranker sequence split seed: ${SPLIT_SEED}"
  python -u -m dynlaneseq_eg.tools.probe_visual_hypothesis_ranker \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --candidate-cache "${CANDIDATE_CACHE}" \
    --split val \
    --list-path dataset/list/val.txt \
    --device "${DEVICE}" \
    --stage main \
    --score-thresh 0.40 \
    --quality-power 0.25 \
    --top-k 4 \
    --near-min-iou 0.30 \
    --near-max-iou 0.50 \
    --max-affine-displacement 64 \
    --train-fraction 0.60 \
    --calibration-fraction 0.20 \
    --split-seed "${SPLIT_SEED}" \
    --split-group sequence \
    --probe-seeds 2024 2025 2026 \
    --feature-kinds visual_rank visual_rank_geometry \
    --feature-cache "${FEATURE_CACHE}" \
    --reuse-feature-cache \
    --feature-dtype float16 \
    --translations-px -32 -24 -16 -8 0 8 16 24 32 \
    --tilts-px -24 -12 0 12 24 \
    --side-offsets-px -16 -8 0 8 16 \
    --visual-map-kinds seg_prob center_prob feature_abs image_luma \
    --row-segments "${ROW_SEGMENTS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --epochs "${EPOCHS}" \
    --probe-batch-size "${PROBE_BATCH_SIZE}" \
    --lr 0.002 \
    --weight-decay 0.001 \
    --action-loss-weight 1.0 \
    --dropout 0.1 \
    --max-tp-action-rate 0.005 \
    --min-gate-precision 0.50 \
    --output-json "${CV_OUT_DIR}/split_${SPLIT_SEED}.json"
done

for SUMMARY_FEATURE in visual_rank visual_rank_geometry; do
  python -u -m dynlaneseq_eg.tools.summarize_affine_noharm_cv \
    --input-dir "${CV_OUT_DIR}" \
    --feature-name "${SUMMARY_FEATURE}" \
    --output-json "${CV_OUT_DIR}/summary_${SUMMARY_FEATURE}.json"
done
