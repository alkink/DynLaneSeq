#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-3407}"
ARMS="${ARMS:-a b}"
TRAIN_STEPS="${TRAIN_STEPS:-25000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
COMPILE_MODEL="${COMPILE_MODEL:-config}"
RUN_TRAIN="${RUN_TRAIN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-3.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_protected_ownership_gate}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_a_protected_ownership_sidecar_25k.yaml}"
ASSIGNMENT_CONFIG="${ASSIGNMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_b_protected_ownership_assignment_25k.yaml}"

if (( TRAIN_STEPS != 25000 || CHECKPOINT_INTERVAL != 5000 )); then
  echo "V5 causal gate is predeclared at 25k with 5k checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V5 requires effective batch 16; got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "V5 gate is predeclared with AMP_DTYPE=bfloat16." >&2
  exit 1
fi
if [[ "${COMPILE_MODEL}" != "config" && "${COMPILE_MODEL}" != "true" && "${COMPILE_MODEL}" != "false" ]]; then
  echo "COMPILE_MODEL accepts only config, true, or false; got ${COMPILE_MODEL}." >&2
  exit 1
fi
for arm in ${ARMS}; do
  if [[ "${arm}" != "a" && "${arm}" != "b" ]]; then
    echo "ARMS accepts only 'a' and/or 'b'; got ${arm}." >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
CONTRACT_JSON="${OUTPUT_ROOT}/contract.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v5_protected_ownership_contract \
  --control-config "${CONTROL_CONFIG}" \
  --assignment-config "${ASSIGNMENT_CONFIG}" \
  --output-json "${CONTRACT_JSON}"

"${PYTHON}" - "${OUTPUT_ROOT}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
free = shutil.disk_usage(path).free / 1024 ** 3
minimum = float(sys.argv[2])
print({"checkpoint_filesystem_free_gib": round(free, 2)})
if free < minimum:
    raise SystemExit(
        f"V5 model-only trajectories require at least {minimum:.2f} GiB free; "
        f"found {free:.2f} GiB"
    )
PY

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "V5 contract audit complete; RUN_TRAIN=${RUN_TRAIN}."
  exit 0
fi

# PyTorch 2.1 + CUDA-on-WSL can report a driver initialization error when
# ``expandable_segments`` is requested.  A bounded split size is supported by
# both the local RTX 3090 stack and the newer server allocator.  Callers may
# still provide their own allocator contract explicitly.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    if [[ "${arm}" == "a" ]]; then
      config="${CONTROL_CONFIG}"
      arm_name="v5_a_sidecar"
    else
      config="${ASSIGNMENT_CONFIG}"
      arm_name="v5_b_assignment"
    fi
    output_dir="${OUTPUT_ROOT}/seed_${seed}/${arm_name}"
    final_checkpoint="${output_dir}/iter_0025000.pt"
    mkdir -p "${output_dir}"

    if [[ -f "${final_checkpoint}" ]]; then
      echo "[SKIP] existing ${final_checkpoint}"
      continue
    fi
    if find "${output_dir}" -maxdepth 1 -type f -name 'iter_*.pt' | grep -q .; then
      echo "Partial model-only V5 trajectory exists in ${output_dir}." >&2
      echo "Exact optimizer resume is intentionally unavailable; choose a new OUTPUT_ROOT or clear only this arm after preserving it." >&2
      exit 1
    fi

    echo "V5 protected ownership ${arm_name}, seed=${seed}: 0 -> 25000"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${config}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${output_dir}" \
      --max-iters "${TRAIN_STEPS}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --seed "${seed}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --num-workers "${NUM_WORKERS}" \
      --seg-aux-amp-dtype "${AMP_DTYPE}" \
      --compile-model "${COMPILE_MODEL}" \
      2>&1 | tee "${output_dir}/train.log"

    if [[ ! -f "${final_checkpoint}" ]]; then
      echo "V5 arm failed to produce ${final_checkpoint}." >&2
      exit 1
    fi
  done
done

echo "V5-A/V5-B 25k training trajectories are ready under ${OUTPUT_ROOT}."
