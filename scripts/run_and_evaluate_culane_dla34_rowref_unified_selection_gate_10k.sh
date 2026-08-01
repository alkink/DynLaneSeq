#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
MAX_ITERS="${MAX_ITERS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RUN_CONTROL="${RUN_CONTROL:-auto}"
RUN_CANDIDATE="${RUN_CANDIDATE:-1}"
RUN_GATE="${RUN_GATE:-1}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_hybrid_control_gate_10k.yaml}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml}"
CONTROL_OUT="${CONTROL_OUT:-outputs/diagnostics/dla34_rowref_hybrid_control_gate_10k}"
CANDIDATE_OUT="${CANDIDATE_OUT:-outputs/diagnostics/dla34_rowref_unified_selection_gate_10k}"
ITER_TAG="$(printf '%07d' "${MAX_ITERS}")"

train_to_target() {
  local config="$1"
  local output_dir="$2"
  local final_checkpoint="${output_dir}/iter_${ITER_TAG}.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "target checkpoint already exists; skipping: ${final_checkpoint}"
    return
  fi

  local resume_args=()
  local run_iters="${MAX_ITERS}"
  local resume_checkpoint=""
  if [[ "${AUTO_RESUME}" == "1" && -d "${output_dir}" ]]; then
    resume_checkpoint="$(
      find "${output_dir}" -maxdepth 1 -type f -name 'iter_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
  fi
  if [[ -n "${resume_checkpoint}" && -f "${resume_checkpoint}" ]]; then
    local start_iter
    start_iter="$(
      "${PYTHON}" -c \
        "import torch; print(int(torch.load('${resume_checkpoint}', map_location='cpu').get('iteration', 0)))"
    )"
    if (( start_iter >= MAX_ITERS )); then
      echo "latest checkpoint is already iteration ${start_iter}: ${resume_checkpoint}"
      return
    fi
    run_iters="$((MAX_ITERS - start_iter))"
    resume_args=(--resume "${resume_checkpoint}")
    echo "resuming ${resume_checkpoint}: ${start_iter} -> ${MAX_ITERS}"
  fi

  "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
    --config "${config}" \
    --dataset-root "${DATA_ROOT}" \
    --output-dir "${output_dir}" \
    --max-iters "${run_iters}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    "${resume_args[@]}"
}

if [[ "${RUN_CONTROL}" == "auto" ]]; then
  if [[ -f "${CONTROL_OUT}/iter_${ITER_TAG}.pt" ]]; then
    RUN_CONTROL=0
  else
    RUN_CONTROL=1
  fi
fi

if [[ "${RUN_CONTROL}" == "1" ]]; then
  echo "===== matched legacy-hybrid control ====="
  train_to_target "${CONTROL_CONFIG}" "${CONTROL_OUT}"
fi

if [[ "${RUN_CANDIDATE}" == "1" ]]; then
  echo "===== unified matcher/selection candidate ====="
  train_to_target "${CANDIDATE_CONFIG}" "${CANDIDATE_OUT}"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT}" \
  CONTROL_CONFIG="${CONTROL_CONFIG}" \
  CANDIDATE_CONFIG="${CANDIDATE_CONFIG}" \
  CONTROL_OUT="${CONTROL_OUT}" \
  CANDIDATE_OUT="${CANDIDATE_OUT}" \
  bash scripts/evaluate_culane_dla34_rowref_unified_selection_gate_10k.sh
fi
