#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep.yaml}"
OUT_DIR="${OUT_DIR:-outputs/culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_fpn256_l4_dfl_rowref_r15_deepsup_50ep}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
TARGET_ITERS="${TARGET_ITERS:-278000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEG_AUX_AMP_DTYPE="${SEG_AUX_AMP_DTYPE:-bfloat16}"
AUTO_RESUME="${AUTO_RESUME:-1}"

if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "Warning: effective batch size is $((BATCH_SIZE * GRAD_ACCUM)); the validated setting is 16." >&2
fi

final_checkpoint="${OUT_DIR}/iter_$(printf '%07d' "${TARGET_ITERS}").pt"
if [[ -f "${final_checkpoint}" ]]; then
  echo "Training target already exists: ${final_checkpoint}"
  exit 0
fi

resume_args=()
run_iters="${TARGET_ITERS}"
if [[ "${AUTO_RESUME}" == "1" && -d "${OUT_DIR}" ]]; then
  resume_checkpoint="$(
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'iter_*.pt' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -n "${resume_checkpoint}" && -f "${resume_checkpoint}" ]]; then
    start_iter="$(
      "${PYTHON}" -c \
        'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu").get("iteration", 0)))' \
        "${resume_checkpoint}"
    )"
    if (( start_iter >= TARGET_ITERS )); then
      echo "Checkpoint already reached target: ${resume_checkpoint}"
      exit 0
    fi
    run_iters="$((TARGET_ITERS - start_iter))"
    resume_args=(--resume "${resume_checkpoint}")
    echo "Auto-resume: ${resume_checkpoint}"
    echo "Iterations: ${start_iter} -> ${TARGET_ITERS} (${run_iters} remaining)"
  fi
fi

if (( ${#resume_args[@]} == 0 )); then
  echo "Fresh full-schedule training; the 10k diagnostic checkpoint is intentionally not reused."
  echo "Iterations: 0 -> ${TARGET_ITERS}"
fi
echo "Config: ${CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Batch/accum/effective: ${BATCH_SIZE}/${GRAD_ACCUM}/$((BATCH_SIZE * GRAD_ACCUM))"

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
  "${resume_args[@]}"
