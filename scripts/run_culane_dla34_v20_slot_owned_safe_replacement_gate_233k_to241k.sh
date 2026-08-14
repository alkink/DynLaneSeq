#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_V7_ITERATION=225000
SOURCE_V19_ITERATION=233000
ENDPOINT_ITERATION=241000
TRAINING_STEPS=8000
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-1}"
CACHE_SHARD_SIZE="${CACHE_SHARD_SIZE:-64}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_CACHE="${RUN_CACHE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

V7_CONFIG="${V7_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
TREATMENT_CONFIG="${TREATMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_233k_to241k.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v20_slot_owned_safe_replacement_control_233k_to241k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-/workspace/DynLaneSeq/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
SOURCE_V19_CHECKPOINT="${SOURCE_V19_CHECKPOINT:-/workspace/DynLaneSeq_v19/outputs/diagnostics/v19_frozen_counterfactual_fidelity_225k/train/iter_0233000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v20_slot_owned_safe_replacement_233k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache/train8192_exact_raster}"
TREATMENT_INITIAL="${OUTPUT_ROOT}/initialization/treatment_iter_0233000.pt"
CONTROL_INITIAL="${OUTPUT_ROOT}/initialization/control_iter_0233000.pt"
TREATMENT_TRAIN="${OUTPUT_ROOT}/train_treatment"
CONTROL_TRAIN="${OUTPUT_ROOT}/train_control"
TREATMENT_ENDPOINT="${TREATMENT_TRAIN}/iter_0241000.pt"
CONTROL_ENDPOINT="${CONTROL_TRAIN}/iter_0241000.pt"

mkdir -p "${OUTPUT_ROOT}/audits" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/initialization" "${LIST_ROOT}" \
  "${TREATMENT_TRAIN}" "${CONTROL_TRAIN}"

for required in \
  "${V7_CONFIG}" "${TREATMENT_CONFIG}" "${CONTROL_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" "${SOURCE_V19_CHECKPOINT}" \
  "${DATA_ROOT}/list/train_gt.txt" "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V20 artifact: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.build_v11_bridge_lists \
  --train-list "${DATA_ROOT}/list/train_gt.txt" \
  --val-list "${DATA_ROOT}/list/val.txt" \
  --output-dir "${LIST_ROOT}" \
  --seed "${SEED}" \
  --train-clips 640 \
  --train-images 8192 \
  --balanced-train-remainder \
  --seen-images 256 \
  --same-clip-unseen-images 256 \
  --heldout-clips 64 \
  --heldout-images 256 \
  --val-images 256 \
  --optimizer-steps "${TRAINING_STEPS}" \
  --effective-batch-size "${BATCH_SIZE}" \
  --experiment-name "V20 frozen slot-owned safe replacement fixed gate" \
  --output-json "${OUTPUT_ROOT}/audits/list_protocol.json"

TRAIN_LIST="${LIST_ROOT}/train_clip640_image8192.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"

initialize() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  [[ -f "${checkpoint}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.initialize_v20_slot_owned_replacement_checkpoint \
    --config "${config}" \
    --source-checkpoint "${SOURCE_V19_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_V19_ITERATION}" \
    --output-checkpoint "${checkpoint}" \
    --output-json "${report}"
}
initialize "${TREATMENT_CONFIG}" "${TREATMENT_INITIAL}" "${OUTPUT_ROOT}/audits/treatment_initialization.json"
initialize "${CONTROL_CONFIG}" "${CONTROL_INITIAL}" "${OUTPUT_ROOT}/audits/control_initialization.json"

GATE0="${OUTPUT_ROOT}/audits/zero_step_contract.json"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v20_slot_owned_replacement_contract \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_INITIAL}" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${TRAIN_LIST}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers 0 \
    --expected-iteration "${SOURCE_V19_ITERATION}" \
    --output-json "${GATE0}"
fi
GATE0_PASSED="$(${PYTHON} - "${GATE0}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${GATE0_PASSED}" != "1" ]]; then
  echo "V20 Gate 0 FAIL: cache and optimizer are closed."
  exit 0
fi

if [[ "${RUN_CACHE}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.cache_v20_slot_owned_replacement \
    --config "${TREATMENT_CONFIG}" \
    --checkpoint "${TREATMENT_INITIAL}" \
    --dataset-root "${DATA_ROOT}" \
    --split train \
    --list-path "${TRAIN_LIST}" \
    --output-dir "${CACHE_ROOT}" \
    --device "${DEVICE}" \
    --batch-size "${CACHE_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --shard-size "${CACHE_SHARD_SIZE}" \
    --seed "${SEED}"
fi
CACHE_MANIFEST="${CACHE_ROOT}/manifest.json"
if [[ ! -f "${CACHE_MANIFEST}" ]]; then
  echo "V20 exact cache manifest is missing." >&2
  exit 1
fi

train_arm() {
  local config="$1"
  local initial="$2"
  local output="$3"
  local mode="$4"
  local endpoint="$5"
  [[ -f "${endpoint}" ]] && return
  local resume_args=()
  local latest
  latest="$(find "${output}" -maxdepth 1 -type f -name 'iter_*.pt' | sort | tail -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    resume_args=(--resume "${latest}")
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train_v20_cached_replacement \
    --config "${config}" \
    --cache-manifest "${CACHE_MANIFEST}" \
    --base-checkpoint "${initial}" \
    --output-dir "${output}" \
    --device "${DEVICE}" \
    --context-mode "${mode}" \
    --steps "${TRAINING_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --checkpoint-interval 1000 \
    --seed "${SEED}" \
    "${resume_args[@]}" 2>&1 | tee -a "${output}/train.log"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_arm "${TREATMENT_CONFIG}" "${TREATMENT_INITIAL}" "${TREATMENT_TRAIN}" treatment "${TREATMENT_ENDPOINT}"
  train_arm "${CONTROL_CONFIG}" "${CONTROL_INITIAL}" "${CONTROL_TRAIN}" masked "${CONTROL_ENDPOINT}"
fi
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "V20 training complete; fixed endpoint evaluation remains paused."
  exit 0
fi
for endpoint in "${TREATMENT_ENDPOINT}" "${CONTROL_ENDPOINT}"; do
  if [[ ! -f "${endpoint}" ]]; then
    echo "Missing V20 fixed endpoint: ${endpoint}" >&2
    exit 1
  fi
done

official_report() {
  local config="$1"
  local endpoint="$2"
  local split="$3"
  local list_path="$4"
  local output="$5"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v20_slot_owned_replacement_official \
    --config "${config}" \
    --source-config "${V7_CONFIG}" \
    --checkpoint "${endpoint}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --device "${DEVICE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --output-json "${output}"
}
official_report "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" train "${HELDOUT_LIST}" "${OUTPUT_ROOT}/reports/heldout_treatment.json"
official_report "${TREATMENT_CONFIG}" "${TREATMENT_ENDPOINT}" val "${VAL_LIST}" "${OUTPUT_ROOT}/reports/validation_treatment.json"
official_report "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" train "${HELDOUT_LIST}" "${OUTPUT_ROOT}/reports/heldout_control.json"
official_report "${CONTROL_CONFIG}" "${CONTROL_ENDPOINT}" val "${VAL_LIST}" "${OUTPUT_ROOT}/reports/validation_control.json"

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v20_slot_owned_replacement_gate \
  --contract "${GATE0}" \
  --cache-manifest "${CACHE_MANIFEST}" \
  --heldout-treatment "${OUTPUT_ROOT}/reports/heldout_treatment.json" \
  --validation-treatment "${OUTPUT_ROOT}/reports/validation_treatment.json" \
  --heldout-control "${OUTPUT_ROOT}/reports/heldout_control.json" \
  --validation-control "${OUTPUT_ROOT}/reports/validation_control.json" \
  --output-json "${OUTPUT_ROOT}/v20_fixed_gate_summary.json"

echo "V20 one-edit version is complete. Stop here for user/Sol planning; no second edit, full validation, long training, threshold/NMS search or test was started."

