#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_50ep.yaml}
CKPT=${CKPT:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}
CANDIDATE_CACHE=${CANDIDATE_CACHE:-outputs/diagnostic_cache/iter_0175000_267b14df707bf94c.pt}
OUT_DIR=${OUT_DIR:-outputs/culane_s0_structured_query_res34_b16_50ep/failure_diagnostics_val_175k}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-100}
REUSE_FEATURE_CACHE=${REUSE_FEATURE_CACHE:-1}

EXTRA_ARGS=()
if [[ "${REUSE_FEATURE_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-feature-cache)
fi

python -u -m dynlaneseq_eg.tools.probe_affine_noharm \
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
  --split-seed 2026 \
  --split-group sequence \
  --probe-seeds 2024 2025 2026 \
  --feature-kinds geometry q_ins segmented \
  --feature-cache "${OUT_DIR}/affine_noharm_features.pt" \
  --hidden-dim 128 \
  --epochs "${EPOCHS}" \
  --probe-batch-size 256 \
  --lr 0.003 \
  --weight-decay 0.001 \
  --regression-weight 1.0 \
  --max-tp-action-rate 0.005 \
  --min-gate-precision 0.50 \
  --output-json "${OUT_DIR}/affine_noharm_probe_sequence.json" \
  "${EXTRA_ARGS[@]}"
