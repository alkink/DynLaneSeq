#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from50k_cooldown_5k.yaml}"
SOURCE_OUT="${SOURCE_OUT:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${SOURCE_OUT}/iter_0050000.pt}"
OUT_DIR="${OUT_DIR:-outputs/diagnostics/dla34_rowref_from50k_evidence_lr2e5_cooldown_5k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
START_ITERATION=50000
TARGET_ITERATION=55000
FINAL_CHECKPOINT="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERATION}").pt"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Refusing unmatched effective batch size: $((BATCH_SIZE * GRAD_ACCUM)); expected 16." >&2
  exit 1
fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing healthy 50k checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

checkpoint_iteration="$(
  "${PYTHON}" -c \
    'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", -1)))' \
    "${INIT_CHECKPOINT}"
)"
if (( checkpoint_iteration != START_ITERATION )); then
  echo "Expected an exact 50k checkpoint, but ${INIT_CHECKPOINT} stores iteration ${checkpoint_iteration}." >&2
  exit 1
fi
if [[ -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Cooldown target already exists: ${FINAL_CHECKPOINT}"
  exit 0
fi

train_args=()
run_iters=5000
resume_checkpoint=""
if [[ -d "${OUT_DIR}" ]]; then
  resume_checkpoint="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' \
      | sort -V \
      | tail -n 1
  )"
fi
if [[ -n "${resume_checkpoint}" && -f "${resume_checkpoint}" ]]; then
  resume_iteration="$(
    "${PYTHON}" -c \
      'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", -1)))' \
      "${resume_checkpoint}"
  )"
  if (( resume_iteration < START_ITERATION || resume_iteration >= TARGET_ITERATION )); then
    echo "Unexpected checkpoint in cooldown output: ${resume_checkpoint} (iteration ${resume_iteration})." >&2
    exit 1
  fi
  run_iters="$((TARGET_ITERATION - resume_iteration))"
  train_args=(--resume "${resume_checkpoint}")
  echo "Resuming the isolated cooldown optimizer: ${resume_iteration} -> ${TARGET_ITERATION}"
else
  train_args=(
    --init-from "${INIT_CHECKPOINT}"
    --init-iteration "${START_ITERATION}"
  )
  echo "Rollback-and-cool: model 50k -> 55k; optimizer/scheduler start fresh."
fi

echo "Config: ${CONFIG}"
echo "Source: ${INIT_CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Batch/accum/effective: ${BATCH_SIZE}/${GRAD_ACCUM}/$((BATCH_SIZE * GRAD_ACCUM))"
echo "Intervention: evidence/structured-head LR 2e-4 -> 2e-5; cosine floor 2e-6."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.train \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --max-iters "${run_iters}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --seg-aux-amp-dtype "${SEG_AUX_AMP_DTYPE}" \
  "${train_args[@]}"
