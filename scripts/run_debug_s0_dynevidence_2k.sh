#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/debug/culane_s0_dynevidence_2k.yaml}"
INIT_FROM="${INIT_FROM-outputs/debug_s0_strong_2k_giou_stable/last.pt}"

args=(--config "${CONFIG}" --device "${DEVICE}")
if [[ -n "${INIT_FROM}" ]]; then
  if [[ ! -f "${INIT_FROM}" ]]; then
    echo "Missing S0 init checkpoint: ${INIT_FROM}" >&2
    echo "Set INIT_FROM='' to train this S0 dynamic-evidence ablation without compatible init." >&2
    exit 1
  fi
  args+=(--init-from "${INIT_FROM}")
fi

python -m dynlaneseq_eg.tools.train "${args[@]}"
