#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
COMPILE_MODEL="${COMPILE_MODEL:-config}"
TRUNK_STEPS="${TRUNK_STEPS:-10000}"
FORK_STEPS="${FORK_STEPS:-15000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
RUN_TRAIN="${RUN_TRAIN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate}"

TRUNK_CONFIG="${TRUNK_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_10k.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_control_10k_to25k.yaml}"
ASSIGNMENT_CONFIG="${ASSIGNMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_assignment_10k_to25k.yaml}"

if (( TRUNK_STEPS != 10000 || FORK_STEPS != 15000 || CHECKPOINT_INTERVAL != 5000 )); then
  echo "V5.1 is predeclared as a 10k trunk plus two 15k forks with 5k checkpoints." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V5.1 requires effective batch 16; got $((BATCH_SIZE * GRAD_ACCUM))." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "V5.1 requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi
if [[ "${COMPILE_MODEL}" != "config" && "${COMPILE_MODEL}" != "true" && "${COMPILE_MODEL}" != "false" ]]; then
  echo "COMPILE_MODEL accepts only config, true, or false; got ${COMPILE_MODEL}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v5_1_shared_trunk_contract \
  --trunk-config "${TRUNK_CONFIG}" \
  --control-config "${CONTROL_CONFIG}" \
  --assignment-config "${ASSIGNMENT_CONFIG}" \
  --output-json "${OUTPUT_ROOT}/contract_preflight.json"

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
        f"V5.1 requires at least {minimum:.2f} GiB free; found {free:.2f} GiB"
    )
PY

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "V5.1 contract preflight complete; RUN_TRAIN=${RUN_TRAIN}."
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  trunk_dir="${seed_root}/shared_trunk"
  trunk_checkpoint="${trunk_dir}/iter_0010000.pt"
  mkdir -p "${trunk_dir}"

  if [[ ! -f "${trunk_checkpoint}" ]]; then
    if find "${trunk_dir}" -maxdepth 1 -type f -name 'iter_*.pt' | grep -q .; then
      echo "Partial shared trunk exists in ${trunk_dir}; exact restart requires a new OUTPUT_ROOT." >&2
      exit 1
    fi
    echo "V5.1 shared trunk, seed=${seed}: 0 -> 10000"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${TRUNK_CONFIG}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${trunk_dir}" \
      --max-iters "${TRUNK_STEPS}" \
      --checkpoint-interval 10000 \
      --seed "${seed}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --num-workers "${NUM_WORKERS}" \
      --seg-aux-amp-dtype "${AMP_DTYPE}" \
      --compile-model "${COMPILE_MODEL}" \
      2>&1 | tee "${trunk_dir}/train.log"
  fi

  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v5_1_shared_trunk_contract \
    --trunk-config "${TRUNK_CONFIG}" \
    --control-config "${CONTROL_CONFIG}" \
    --assignment-config "${ASSIGNMENT_CONFIG}" \
    --trunk-checkpoint "${trunk_checkpoint}" \
    --output-json "${seed_root}/shared_trunk_contract.json"

  for arm in control assignment; do
    if [[ "${arm}" == "control" ]]; then
      config="${CONTROL_CONFIG}"
      arm_name="control_fork"
    else
      config="${ASSIGNMENT_CONFIG}"
      arm_name="assignment_fork"
    fi
    arm_dir="${seed_root}/${arm_name}"
    final_checkpoint="${arm_dir}/iter_0025000.pt"
    mkdir -p "${arm_dir}"

    if [[ -f "${final_checkpoint}" ]]; then
      echo "[SKIP] existing ${final_checkpoint}"
      continue
    fi
    if find "${arm_dir}" -maxdepth 1 -type f -name 'iter_*.pt' | grep -q .; then
      echo "Partial V5.1 fork exists in ${arm_dir}; exact resume is intentionally disabled." >&2
      exit 1
    fi

    echo "V5.1 ${arm_name}, seed=${seed}: shared 10000 -> 25000"
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${config}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --resume "${trunk_checkpoint}" \
      --output-dir "${arm_dir}" \
      --max-iters "${FORK_STEPS}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --seed "${seed}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --num-workers "${NUM_WORKERS}" \
      --seg-aux-amp-dtype "${AMP_DTYPE}" \
      --compile-model "${COMPILE_MODEL}" \
      2>&1 | tee "${arm_dir}/train.log"

    if [[ ! -f "${final_checkpoint}" ]]; then
      echo "V5.1 fork failed to produce ${final_checkpoint}." >&2
      exit 1
    fi
  done
done

echo "V5.1 exact shared-trunk trajectories are ready under ${OUTPUT_ROOT}."
