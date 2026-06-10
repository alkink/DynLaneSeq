#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_dynevidence_2k.yaml}"
OUT_DIR="${OUT_DIR:-outputs/debug_s0_dynevidence_2k}"
DEVICE="${DEVICE:-cuda}"
QUALITY_POWER="${QUALITY_POWER:-0.0}"

if [[ -n "${SCORE_THRESH:-}" ]]; then
  SCORE_THRESHOLDS=("${SCORE_THRESH}")
else
  SCORE_THRESHOLDS=("0.40" "0.45" "0.50" "0.55")
fi

if [[ -n "${CKPTS:-}" ]]; then
  read -r -a CKPT_LIST <<< "${CKPTS}"
else
  mapfile -t CKPT_LIST < <(find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' -printf '%f\n' | sort)
  if [[ -f "${OUT_DIR}/last.pt" ]]; then
    CKPT_LIST+=("last.pt")
  fi
fi

if [[ "${#CKPT_LIST[@]}" -eq 0 ]]; then
  echo "No checkpoints found in ${OUT_DIR}" >&2
  exit 1
fi

q_label="${QUALITY_POWER//./p}"
for ckpt_name in "${CKPT_LIST[@]}"; do
  ckpt="${OUT_DIR}/${ckpt_name}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt}" >&2
    continue
  fi
  ckpt_label="${ckpt_name%.pt}"
  for thresh in "${SCORE_THRESHOLDS[@]}"; do
    thr_label="${thresh//./p}"
    pred_dir="${OUT_DIR}/sweep_${ckpt_label}_thr${thr_label}_q${q_label}"
    echo
    echo "== checkpoint=${ckpt_name} score_thresh=${thresh} quality_score_power=${QUALITY_POWER} =="
    python -m dynlaneseq_eg.tools.evaluate_culane \
      --config "${CONFIG}" \
      --checkpoint "${ckpt}" \
      --split val \
      --device "${DEVICE}" \
      --score-thresh "${thresh}" \
      --quality-score-power "${QUALITY_POWER}" \
      --pred-dir "${pred_dir}" \
      --sequential
  done
done
