#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PHASE1_CKPT="${PHASE1_CKPT:-outputs/debug_s3_unified_fair_2k_from_s0continue_phase1_12k/last.pt}"

PHASE2_CONFIG="dynlaneseq_eg/configs/debug/culane_s3_unified_fair_2k_from_s0continue_phase2_12k.yaml"
PHASE2_OUT="outputs/debug_s3_unified_fair_2k_from_s0continue_phase2_12k"
PHASE3_CONFIG="dynlaneseq_eg/configs/debug/culane_s3_unified_fair_2k_from_s0continue_phase3_12k.yaml"
PHASE3_OUT="outputs/debug_s3_unified_fair_2k_from_s0continue_phase3_12k"

if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "Missing Phase-1 checkpoint: ${PHASE1_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_COMPLETED}" == "1" && -f "${PHASE2_OUT}/last.pt" ]]; then
  echo "== Phase-2 already complete; reusing ${PHASE2_OUT}/last.pt =="
else
  echo "== Phase-2: fresh optimizer, init from Phase-1 =="
  python -u -m dynlaneseq_eg.tools.train \
    --config "${PHASE2_CONFIG}" \
    --device "${DEVICE}" \
    --init-from "${PHASE1_CKPT}"
fi

PHASE2_CKPT="${PHASE2_OUT}/last.pt"
if [[ ! -f "${PHASE2_CKPT}" ]]; then
  echo "Missing Phase-2 checkpoint: ${PHASE2_CKPT}" >&2
  exit 1
fi

if [[ "${SKIP_COMPLETED}" == "1" && -f "${PHASE3_OUT}/last.pt" ]]; then
  echo "== Phase-3 already complete; reusing ${PHASE3_OUT}/last.pt =="
else
  echo "== Phase-3: fresh optimizer, init from Phase-2 =="
  python -u -m dynlaneseq_eg.tools.train \
    --config "${PHASE3_CONFIG}" \
    --device "${DEVICE}" \
    --init-from "${PHASE2_CKPT}"
fi
