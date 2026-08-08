#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
SOURCE_ITERATION="${SOURCE_ITERATION:-25000}"
TARGET_ITERATION="${TARGET_ITERATION:-28000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v7_hard_slot_assignment_gate}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v7_hard_slot_assignment_gate}"

CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_hard_reference_gate_25k_to28k.yaml
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${OUTPUT_ROOT}/shared_slot_init_iter_0025000.pt}"
MEMORIZATION_SUMMARY="${MEMORIZATION_SUMMARY:-${OUTPUT_ROOT}/v7_hard_reference_memorization_26k_to27k_summary.json}"
TRAIN_DIR="${OUTPUT_ROOT}/generalization/hard_reference_fresh25k_to28k"
SUMMARY="${OUTPUT_ROOT}/v7_hard_reference_generalization_25k_to28k_summary.json"
GRADIENT_REPORT="${OUTPUT_ROOT}/v7_hard_reference_generalization_iter_0028000_gradient_contract.json"

if (( SOURCE_ITERATION != 25000 || TARGET_ITERATION != 28000 )); then
  echo "This causal gate is fixed to fresh iteration 25000 -> 28000." >&2
  exit 1
fi
if (( CHECKPOINT_INTERVAL != 500 )); then
  echo "This trajectory gate requires CHECKPOINT_INTERVAL=500." >&2
  exit 1
fi
if (( BATCH_SIZE != 4 || GRAD_ACCUM != 4 )); then
  echo "This paired-data contract requires physical batch 4 and accumulation 4." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "This gate requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi
for required in "${CONFIG}" "${INIT_CHECKPOINT}" "${MEMORIZATION_SUMMARY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing hard-reference generalization artefact: ${required}" >&2
    exit 1
  fi
done

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

if (( $(checkpoint_iteration "${INIT_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "INIT_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" "${MEMORIZATION_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
memory = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
checks = {
    "memorization_pass": memory.get("pass") is True,
    "hard_reference": selection["four_slot_refinement_reference_mode"] == "hard_st",
    "hard_slot_assignment": loss["four_slot_assignment_mode"] == "hard_min",
    "all_gt": loss["four_slot_target_mode"] == "all_gt",
    "frozen_detector": training["frozen_detector_eval"] is True,
    "effective_batch_16": (
        int(training["batch_size"])
        * int(training["gradient_accumulation_steps"])
        == 16
    ),
}
print({"hard_reference_generalization_contract": checks})
if not all(checks.values()):
    raise SystemExit("invalid hard-reference generalization contract")
PY

mkdir -p "${TRAIN_DIR}" "${CACHE_ROOT}"
FINAL_TAG="$(printf '%07d' "${TARGET_ITERATION}")"
FINAL_CHECKPOINT="${TRAIN_DIR}/iter_${FINAL_TAG}.pt"
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  latest_checkpoint=""
  latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${TRAIN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    candidate_iteration="$(checkpoint_iteration "${candidate}")"
    if (( candidate_iteration > latest_iteration && candidate_iteration < TARGET_ITERATION )); then
      latest_checkpoint="${candidate}"
      latest_iteration="${candidate_iteration}"
    fi
  done
  train_args=(
    --config "${CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${TRAIN_DIR}"
    --checkpoint-base "${INIT_CHECKPOINT}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
  )
  if [[ -n "${latest_checkpoint}" ]]; then
    echo "Resuming hard-reference generalization: ${latest_iteration} -> ${TARGET_ITERATION}"
    train_args+=(
      --resume "${latest_checkpoint}"
      --max-iters "$((TARGET_ITERATION - latest_iteration))"
    )
  else
    echo "Training fresh hard-reference head: ${SOURCE_ITERATION} -> ${TARGET_ITERATION}"
    train_args+=(
      --init-from "${INIT_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "$((TARGET_ITERATION - SOURCE_ITERATION))"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
    2>&1 | tee -a "${TRAIN_DIR}/train.log"
fi
if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Missing hard-reference final checkpoint: ${FINAL_CHECKPOINT}" >&2
  exit 1
fi

summary_args=()
for ((iteration=SOURCE_ITERATION+CHECKPOINT_INTERVAL; iteration<=TARGET_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
  tag="$(printf '%07d' "${iteration}")"
  checkpoint="${TRAIN_DIR}/iter_${tag}.pt"
  report="${OUTPUT_ROOT}/generalization_hard_iter_${tag}_uniform256.json"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing trajectory checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/generalization_hard_${tag}" \
      --max-batches 64 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --metric-workers "${METRIC_WORKERS}" \
      --sample-strategy uniform \
      --stage main \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --near-min-iou 0.30 \
      --line-width 30 \
      --min-valid-rows 5 \
      --hard-diversity-distances 20 \
      --mmr-sigmas 20 \
      --mmr-penalties 0.50 \
      --output-json "${report}"
  fi
  summary_args+=(--report "${iteration}=${report}")
done

if [[ ! -f "${GRADIENT_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_reference_gradient_contract \
    --config "${CONFIG}" \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 2 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype none \
    --output-json "${GRADIENT_REPORT}"
fi

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v7_hard_reference_generalization \
  "${summary_args[@]}" \
  --gradient-contract "${GRADIENT_REPORT}" \
  --output-json "${SUMMARY}"

echo "Hard-reference generalization gate complete: ${SUMMARY}"
