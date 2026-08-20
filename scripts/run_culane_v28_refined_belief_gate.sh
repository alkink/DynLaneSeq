#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/DynLaneSeq}"
PYTHON_BIN="${PYTHON_BIN:-/venv/clrernet/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-${REPO_ROOT}/dynlaneseq_eg/configs/culane_v28_refined_belief_gate.yaml}"
V7_CHECKPOINT="${V7_CHECKPOINT:-${REPO_ROOT}/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diagnostics/v28_refined_belief_gate/scientific_gate_6k}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
RESUME_INTERVAL="${RESUME_INTERVAL:-500}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_arm() {
  local arm="$1"
  local lower
  lower="$(printf '%s' "${arm}" | tr '[:upper:]' '[:lower:]')"
  local output_dir="${OUTPUT_ROOT}/arm_${lower}"
  local resume_args=()
  if [[ -f "${output_dir}/resume_latest.pt" && ! -f "${output_dir}/v28_gate_endpoint.pt" ]]; then
    resume_args=(--resume "${output_dir}/resume_latest.pt")
  fi
  if [[ -f "${output_dir}/v28_gate_endpoint.pt" ]]; then
    echo "V28 arm ${arm} endpoint already exists; preserving it."
    return
  fi
  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.train_v28_refined_belief_router \
    --config "${CONFIG}" \
    --v7-checkpoint "${V7_CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${output_dir}" \
    --arm "${arm}" \
    --mode gate \
    --num-workers "${NUM_WORKERS}" \
    --log-interval "${LOG_INTERVAL}" \
    --resume-interval "${RESUME_INTERVAL}" \
    "${resume_args[@]}"
}

# One 16-GiB GPU cannot safely host the two 9.5-GiB arms concurrently. Their
# seed, stream, initialization, and endpoint remain identical by running the
# predeclared B control first and C treatment second.
run_arm B
run_arm C

echo "V28 B/C scientific training endpoints completed."

