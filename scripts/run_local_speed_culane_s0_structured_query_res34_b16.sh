#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
MAX_ITERS="${MAX_ITERS:-200}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_local_speedtest.yaml}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "clrernet"; then
    conda activate clrernet
  fi
fi

echo "local speed test config=${CONFIG}"
echo "device=${DEVICE} max_iters=${MAX_ITERS}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --max-iters "${MAX_ITERS}"
