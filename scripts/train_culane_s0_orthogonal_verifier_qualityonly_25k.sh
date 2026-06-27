#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_orthogonal_verifier_qualityonly_25k.yaml}"
INIT_FROM="${INIT_FROM:-outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt}"
RESUME="${RESUME:-}"
MAX_ITERS="${MAX_ITERS:-0}"

if [[ -n "${RESUME}" && -n "${INIT_FROM}" && "${INIT_FROM}" != "outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt" ]]; then
  echo "Use either RESUME or INIT_FROM, not both." >&2
  exit 1
fi

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Missing resume checkpoint: ${RESUME}" >&2
  exit 1
fi

if [[ -z "${RESUME}" && ! -f "${INIT_FROM}" ]]; then
  echo "Missing S0 structured checkpoint: ${INIT_FROM}" >&2
  echo "Set INIT_FROM=/path/to/checkpoint.pt." >&2
  exit 1
fi

ARGS=()
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
else
  ARGS+=(--init-from "${INIT_FROM}")
fi
if [[ "${MAX_ITERS}" != "0" ]]; then
  ARGS+=(--max-iters "${MAX_ITERS}")
fi

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "${ARGS[@]}"

