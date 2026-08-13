#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
SOURCE_ITERATION=225000
ENDPOINT_ITERATION=227000
TRAINING_STEPS=2000
CHECKPOINT_INTERVAL=500
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_CONTRACT="${RUN_CONTRACT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

V16_CONFIG="${V16_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v16_candidate_reranker_225k_to227k.yaml}"
SOURCE_V7_CHECKPOINT="${SOURCE_V7_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/v16_candidate_aligned_local_reranker_225k}"
LIST_ROOT="${LIST_ROOT:-${OUTPUT_ROOT}/lists}"
PREFLIGHT_SUMMARY="${PREFLIGHT_SUMMARY:-${OUTPUT_ROOT}/v16_candidate_group_preflight_summary.json}"
INITIAL_CHECKPOINT="${OUTPUT_ROOT}/initialization/iter_0225000.pt"
TRAIN_DIR="${OUTPUT_ROOT}/train"
ENDPOINT="${TRAIN_DIR}/iter_0227000.pt"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in \
  "${V16_CONFIG}" \
  "${SOURCE_V7_CHECKPOINT}" \
  "${PREFLIGHT_SUMMARY}" \
  "${DATA_ROOT}/list/train_gt.txt" \
  "${DATA_ROOT}/list/val.txt"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V16 artifact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_V7_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_V7_CHECKPOINT is not iteration 225000." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "V16 requires effective batch 16." >&2
  exit 1
fi
PREFLIGHT_PASSED="$(${PYTHON} - "${PREFLIGHT_SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${PREFLIGHT_PASSED}" != "1" ]]; then
  echo "V16 candidate-group preflight did not pass; Stage A is closed."
  exit 0
fi

mkdir -p \
  "${OUTPUT_ROOT}/audits" \
  "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/initialization" \
  "${TRAIN_DIR}" \
  "${LIST_ROOT}"

TRAIN_LIST="${LIST_ROOT}/train_clip512_image4096.txt"
HELDOUT_LIST="${LIST_ROOT}/heldout_clip_image256.txt"
VAL_LIST="${LIST_ROOT}/val_clip_balanced_image256.txt"
# Rebuild deterministically even when preflight lists already exist.  The list
# contents remain identical, while the provenance report records this arm's
# exact 2,000-step exposure contract instead of the list builder's old default.
"${PYTHON}" -u -m dynlaneseq_eg.tools.build_v11_bridge_lists \
  --train-list "${DATA_ROOT}/list/train_gt.txt" \
  --val-list "${DATA_ROOT}/list/val.txt" \
  --output-dir "${LIST_ROOT}" \
  --seed "${SEED}" \
  --train-clips 512 \
  --train-images 4096 \
  --seen-images 256 \
  --same-clip-unseen-images 256 \
  --heldout-clips 64 \
  --heldout-images 256 \
  --val-images 256 \
  --optimizer-steps "${TRAINING_STEPS}" \
  --effective-batch-size "$((BATCH_SIZE * GRAD_ACCUM))" \
  --experiment-name "V16 candidate-aligned hard reranker" \
  --output-json "${OUTPUT_ROOT}/audits/list_protocol.json"

HELDOUT_WRONG_LIST="${LIST_ROOT}/heldout_cross_clip_wrong_image256.txt"
VAL_WRONG_LIST="${LIST_ROOT}/val_cross_clip_wrong_image256.txt"
HELDOUT_CROSSCLIP="${OUTPUT_ROOT}/audits/heldout_cross_clip_derangement.json"
VAL_CROSSCLIP="${OUTPUT_ROOT}/audits/val_cross_clip_derangement.json"
if [[ ! -f "${HELDOUT_WRONG_LIST}" || ! -f "${HELDOUT_CROSSCLIP}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${HELDOUT_LIST}" \
    --output-list "${HELDOUT_WRONG_LIST}" \
    --output-json "${HELDOUT_CROSSCLIP}" \
    --seed "${SEED}"
fi
if [[ ! -f "${VAL_WRONG_LIST}" || ! -f "${VAL_CROSSCLIP}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.build_cross_clip_derangement \
    --input-list "${VAL_LIST}" \
    --output-list "${VAL_WRONG_LIST}" \
    --output-json "${VAL_CROSSCLIP}" \
    --seed "${SEED}"
fi

if [[ ! -f "${INITIAL_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.initialize_v16_candidate_reranker_checkpoint \
    --config "${V16_CONFIG}" \
    --source-checkpoint "${SOURCE_V7_CHECKPOINT}" \
    --seed "${SEED}" \
    --iteration "${SOURCE_ITERATION}" \
    --output-checkpoint "${INITIAL_CHECKPOINT}" \
    --output-json "${OUTPUT_ROOT}/audits/initialization.json"
fi

CONTRACT="${OUTPUT_ROOT}/audits/zero_step_contract.json"
if [[ "${RUN_CONTRACT}" == "1" ]]; then
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v16_candidate_reranker_contract \
    --config "${V16_CONFIG}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --list-path "${TRAIN_LIST}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers 0 \
    --start-iteration "${SOURCE_ITERATION}" \
    --cross-clip-report "${HELDOUT_CROSSCLIP}" \
    --cross-clip-report "${VAL_CROSSCLIP}" \
    --output-json "${CONTRACT}"
fi
if [[ ! -f "${CONTRACT}" ]]; then
  echo "V16 zero-step contract is missing." >&2
  exit 1
fi
CONTRACT_PASSED="$(${PYTHON} - "${CONTRACT}" <<'PY'
import json, sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text()).get("passed") is True else "0")
PY
)"
if [[ "${CONTRACT_PASSED}" != "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}/v16_completion.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "experiment": "V16 candidate-aligned hard reranker",
    "zero_step_contract_passed": False,
    "training_executed": False,
    "v16_complete": True,
    "long_training_authorized": False,
    "full_validation_authorized": False,
    "next_version_authorized": False,
    "test_set_used": False,
    "decision": "v16_gate0_fail_stop_for_review",
}, indent=2, sort_keys=True) + "\n")
PY
  echo "V16 Gate 0 FAIL: no optimizer step. V16 is complete; stop for review."
  exit 0
fi

train_endpoint() {
  if [[ -f "${ENDPOINT}" ]]; then
    if (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
      echo "Existing V16 endpoint has the wrong iteration." >&2
      exit 1
    fi
    echo "Reusing fixed V16 endpoint ${ENDPOINT}"
    return
  fi
  local latest=""
  local latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    local iteration
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < ENDPOINT_ITERATION )); then
      latest="${candidate}"
      latest_iteration="${iteration}"
    fi
  done
  local command=(
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train
    --config "${V16_CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${TRAIN_DIR}"
    --checkpoint-base "${INITIAL_CHECKPOINT}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
    --resume-safe-data true
    --train-list "${TRAIN_LIST}"
  )
  if [[ -n "${latest}" ]]; then
    command+=(
      --resume "${latest}"
      --max-iters "$((ENDPOINT_ITERATION - latest_iteration))"
    )
  else
    command+=(
      --init-from "${INITIAL_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAINING_STEPS}"
    )
  fi
  "${command[@]}" 2>&1 | tee -a "${TRAIN_DIR}/train.log"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_endpoint
fi
if [[ ! -f "${ENDPOINT}" ]]; then
  echo "Fixed V16 endpoint is missing: ${ENDPOINT}" >&2
  exit 1
fi
if (( $(checkpoint_iteration "${ENDPOINT}") != ENDPOINT_ITERATION )); then
  echo "V16 endpoint is not iteration 227000." >&2
  exit 1
fi

official_report() {
  local split="$1"
  local list_path="$2"
  local wrong_list="$3"
  local crossclip="$4"
  local output="$5"
  [[ -f "${output}" ]] && return
  "${PYTHON}" -u -m \
    dynlaneseq_eg.tools.audit_v16_candidate_reranker_official \
    --config "${V16_CONFIG}" \
    --checkpoint "${ENDPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --split "${split}" \
    --list-path "${list_path}" \
    --wrong-list-path "${wrong_list}" \
    --cross-clip-report "${crossclip}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --metric-workers "${METRIC_WORKERS}" \
    --output-json "${output}"
}

HELDOUT_REPORT="${OUTPUT_ROOT}/reports/heldout_candidate_reranker_official.json"
VAL_REPORT="${OUTPUT_ROOT}/reports/validation_candidate_reranker_official.json"
official_report train "${HELDOUT_LIST}" "${HELDOUT_WRONG_LIST}" \
  "${HELDOUT_CROSSCLIP}" "${HELDOUT_REPORT}"
official_report val "${VAL_LIST}" "${VAL_WRONG_LIST}" \
  "${VAL_CROSSCLIP}" "${VAL_REPORT}"

SUMMARY="${OUTPUT_ROOT}/v16_candidate_reranker_stage_a_summary.json"
"${PYTHON}" -u -m \
  dynlaneseq_eg.tools.summarize_v16_candidate_reranker_gate \
  --preflight-summary "${PREFLIGHT_SUMMARY}" \
  --contract "${CONTRACT}" \
  --heldout "${HELDOUT_REPORT}" \
  --validation "${VAL_REPORT}" \
  --output-json "${SUMMARY}"

cp "${SUMMARY}" "${OUTPUT_ROOT}/v16_completion.json"
echo "V16 Stage A is complete. No full validation, long training, test, or V17 was run."
