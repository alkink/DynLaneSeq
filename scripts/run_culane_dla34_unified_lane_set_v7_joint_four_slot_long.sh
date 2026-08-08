#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_278k.yaml}"
SEED="${SEED:-3407}"
TARGET_ITERATION="${TARGET_ITERATION:-50000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_CONTRACT_AUDIT="${RUN_CONTRACT_AUDIT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diagnostics/unified_lane_set_v7_joint_four_slot_long/seed_${SEED}/joint_four_slot}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing V7 config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing CULane root: ${DATA_ROOT}" >&2
  exit 1
fi
if (( TARGET_ITERATION < 1 || TARGET_ITERATION > 278000 )); then
  echo "TARGET_ITERATION must be in [1, 278000]." >&2
  exit 1
fi
if (( CHECKPOINT_INTERVAL < 1 )); then
  echo "CHECKPOINT_INTERVAL must be positive." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V7 requires effective batch 16; got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "V7 long training requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_CONTRACT_AUDIT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_joint_four_slot_contract \
    --config "${CONFIG}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype none \
    --output-json "${OUTPUT_DIR}/contract_preflight.json"
fi

"${PYTHON}" - "${OUTPUT_DIR}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
free = shutil.disk_usage(path).free / 1024 ** 3
minimum = float(sys.argv[2])
print({"checkpoint_filesystem_free_gib": round(free, 2)})
if free < minimum:
    raise SystemExit(
        f"V7 long training requires at least {minimum:.2f} GiB free; "
        f"found {free:.2f} GiB"
    )
PY

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "V7 contract audit complete; RUN_TRAIN=${RUN_TRAIN}."
  exit 0
fi

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

latest_checkpoint=""
latest_iteration=0
for candidate in "${OUTPUT_DIR}"/iter_*.pt; do
  [[ -f "${candidate}" ]] || continue
  iteration="$(checkpoint_iteration "${candidate}")"
  if (( iteration > latest_iteration && iteration <= TARGET_ITERATION )); then
    latest_checkpoint="${candidate}"
    latest_iteration="${iteration}"
  fi
done

if (( latest_iteration == TARGET_ITERATION )); then
  echo "[SKIP] V7 target checkpoint already exists: ${latest_checkpoint}"
  exit 0
fi

remaining=$((TARGET_ITERATION - latest_iteration))
train_args=(
  --config "${CONFIG}"
  --dataset-root "${DATA_ROOT}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
  --max-iters "${remaining}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --num-workers "${NUM_WORKERS}"
  --seg-aux-amp-dtype "${AMP_DTYPE}"
  --compile-model false
)
if [[ -n "${latest_checkpoint}" ]]; then
  echo "Resuming coherent V7: ${latest_iteration} -> ${TARGET_ITERATION}"
  train_args+=(--resume "${latest_checkpoint}")
else
  echo "Starting coherent V7 from scratch: 0 -> ${TARGET_ITERATION}"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
"${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"

final_tag="$(printf '%07d' "${TARGET_ITERATION}")"
final_checkpoint="${OUTPUT_DIR}/iter_${final_tag}.pt"
if [[ ! -f "${final_checkpoint}" ]]; then
  echo "V7 did not produce ${final_checkpoint}." >&2
  exit 1
fi
echo "V7 checkpoint ready: ${final_checkpoint}"
