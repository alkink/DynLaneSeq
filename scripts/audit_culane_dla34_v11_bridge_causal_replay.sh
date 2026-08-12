#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
V11_CONFIG="${V11_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_bridge4096_225k_to228k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
BRIDGE_ROOT="${BRIDGE_ROOT:-outputs/diagnostics/v11_bridge4096_gate_225k}"
V11_CHECKPOINT="${V11_CHECKPOINT:-${BRIDGE_ROOT}/train/v11/iter_0228000.pt}"
HELDOUT_LIST="${HELDOUT_LIST:-${BRIDGE_ROOT}/lists/heldout_clip_image256.txt}"
VAL_LIST="${VAL_LIST:-${BRIDGE_ROOT}/lists/val_clip_balanced_image256.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BRIDGE_ROOT}/causal_replay}"

mkdir -p "${OUTPUT_ROOT}"

run_domain() {
  local name="$1"
  local split="$2"
  local list_path="$3"
  local output="${OUTPUT_ROOT}/${name}_v11_causal_replay.json"
  if [[ -f "${output}" ]]; then
    echo "Reusing ${output}"
    return
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v11_causal_replay \
    --source-config "${V7_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --v11-config "${V11_CONFIG}" \
    --v11-checkpoint "${V11_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --sample-strategy sequential \
    --max-images 0 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --iou-thresholds 0.50 0.75 \
    --line-width 30 \
    --min-valid-rows 5 \
    --output-json "${output}"
}

run_domain heldout_clip train "${HELDOUT_LIST}"
run_domain val val "${VAL_LIST}"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v11_causal_replay \
  --heldout-json "${OUTPUT_ROOT}/heldout_clip_v11_causal_replay.json" \
  --val-json "${OUTPUT_ROOT}/val_v11_causal_replay.json" \
  --output-json "${OUTPUT_ROOT}/v11_causal_replay_summary.json"

echo "V11 causal replay complete. No optimizer step or test-set access occurred."
