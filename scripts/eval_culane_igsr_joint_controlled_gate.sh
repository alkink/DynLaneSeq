#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_igsr_joint_controlled_25k.yaml}"
CKPT="${CKPT:-outputs/culane_igsr_joint_controlled_25k/iter_0025000.pt}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
OUT_DIR="${OUT_DIR:-outputs/culane_igsr_joint_controlled_25k/gate_eval}"
THRESHOLDS="${THRESHOLDS:-0.40 0.45 0.50 0.55}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing joint checkpoint: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_LIST}" ]]; then
  echo "Missing evaluation list: ${EVAL_LIST}" >&2
  exit 1
fi

for stage in coarse final; do
  for threshold in ${THRESHOLDS}; do
    threshold_label="${threshold//./p}"
    "${PYTHON_BIN}" -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${CONFIG}" \
      --checkpoint "${CKPT}" \
      --split val \
      --list-path "${EVAL_LIST}" \
      --output-stage "${stage}" \
      --device "${DEVICE}" \
      --score-thresh "${threshold}" \
      --quality-score-power 0.0 \
      --pred-dir "${OUT_DIR}/${stage}_thr${threshold_label}_q0p0" \
      --sequential
  done
done

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_proposal_recall \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --top-k 0 \
  --rank-by none \
  --iou-thresholds 0.3 0.5 0.7

"${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.analyze_active_corridor_evidence \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --list-path "${EVAL_LIST}" \
  --device "${DEVICE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --shuffle-modes offset_reverse lane_roll \
  --output-json "${OUT_DIR}/active_corridor_diagnostics.json"
