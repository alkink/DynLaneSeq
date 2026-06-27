#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep.yaml}"
INIT_FROM="${INIT_FROM:-}"
RESUME="${RESUME:-}"
MAX_ITERS="${MAX_ITERS:-0}"

if [[ -n "${RESUME}" && -n "${INIT_FROM}" ]]; then
  echo "Use either RESUME or INIT_FROM, not both." >&2
  exit 1
fi

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  exit 1
fi

if [[ -n "${INIT_FROM}" && ! -f "${INIT_FROM}" ]]; then
  echo "Missing init checkpoint: ${INIT_FROM}" >&2
  exit 1
fi

ARGS=()
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
elif [[ -n "${INIT_FROM}" ]]; then
  ARGS+=(--init-from "${INIT_FROM}")
fi
if [[ "${MAX_ITERS}" != "0" ]]; then
  ARGS+=(--max-iters "${MAX_ITERS}")
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "${ARGS[@]}"
