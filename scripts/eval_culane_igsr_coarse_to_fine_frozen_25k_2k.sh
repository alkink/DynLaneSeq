#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="dynlaneseq_eg/configs/culane_igsr_coarse_to_fine_frozen_25k.yaml"
OUT_DIR="outputs/culane_igsr_coarse_to_fine_frozen_25k"
CKPT_NAME="${CKPT_NAME:-last.pt}"
CKPT="${CKPT:-${OUT_DIR}/${CKPT_NAME}}"
EVAL_LIST="${EVAL_LIST:-dataset/list/test_2k.txt}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

ckpt_label="${CKPT_NAME%.pt}"
for threshold in 0.40 0.45 0.50 0.55; do
  threshold_label="${threshold//./p}"
  "${PYTHON_BIN}" -m dynlaneseq_eg.tools.evaluate_culane \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --split val \
    --device "${DEVICE}" \
    --score-thresh "${threshold}" \
    --quality-score-power 0.0 \
    --pred-dir "${OUT_DIR}/sweep_${ckpt_label}_thr${threshold_label}_q0p0" \
    --sequential
done

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_active_corridor_evidence \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --shuffle-modes offset_reverse lane_roll \
  --output-json "outputs/active_corridor_diagnostics/coarse_to_fine_fulltrain_${ckpt_label}.json"

