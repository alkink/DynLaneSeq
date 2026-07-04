#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_crossgate_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_crossgate_50ep}"
TARGET_ITERS_TOTAL="${TARGET_ITERS_TOTAL:-278000}"

if [[ -z "${RESUME:-}" ]]; then
  echo "RESUME must be set, e.g. RESUME=${OUT_DIR}/iter_0125000.pt $0" >&2
  exit 2
fi

CKPT_NAME="$(basename "${RESUME}")"
START_ITER="$(python - "${CKPT_NAME}" <<'PY'
import re
import sys
name = sys.argv[1]
m = re.search(r"iter_(\d+)\.pt$", name)
if not m:
    raise SystemExit(f"cannot parse iteration from checkpoint name: {name}")
print(int(m.group(1)))
PY
)"
REMAINING_ITERS=$((TARGET_ITERS_TOTAL - START_ITER))
if (( REMAINING_ITERS <= 0 )); then
  echo "checkpoint iteration ${START_ITER} is already >= target ${TARGET_ITERS_TOTAL}" >&2
  exit 2
fi

echo "config: ${CONFIG}"
echo "out_dir: ${OUT_DIR}"
echo "resume: ${RESUME}"
echo "start_iter: ${START_ITER}"
echo "target_iters_total: ${TARGET_ITERS_TOTAL}"
echo "remaining_iters_this_run: ${REMAINING_ITERS}"

python -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --resume "${RESUME}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${REMAINING_ITERS}"
