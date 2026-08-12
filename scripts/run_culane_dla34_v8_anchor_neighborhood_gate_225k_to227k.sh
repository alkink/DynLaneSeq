#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v8_anchor_neighborhood_gate_225k_to227k.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_joint_four_slot_resume_safe.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-225000}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
SEED="${SEED:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v8_anchor_neighborhood_gate_225k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v8_anchor_neighborhood_gate_225k}"
RUN_CONTRACT="${RUN_CONTRACT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_TRAJECTORY="${EVAL_TRAJECTORY:-0}"

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}

for required in "${CONFIG}" "${BASELINE_CONFIG}" "${SOURCE_CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V8 gate artefact: ${required}" >&2
    exit 1
  fi
done
if (( $(checkpoint_iteration "${SOURCE_CHECKPOINT}") != SOURCE_ITERATION )); then
  echo "SOURCE_CHECKPOINT is not iteration ${SOURCE_ITERATION}." >&2
  exit 1
fi
if (( TRAIN_STEPS < 500 || TRAIN_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "TRAIN_STEPS must be >=500 and divisible by CHECKPOINT_INTERVAL." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "The causal gate requires effective batch size 16." >&2
  exit 1
fi

"${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model

cfg = load_config(sys.argv[1])
cfg["model"]["pretrained_backbone"] = False
cfg["model"]["require_pretrained_backbone"] = False
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
training = cfg["training"]
model = build_model(cfg)
refiner = model.structured_query_head.set_selection_head.slot_refinement
checks = {
    "neighborhood_reference": selection["four_slot_refinement_reference_mode"] == "neighborhood_soft",
    "local4_48px": selection["four_slot_refinement_neighborhood_max_candidates"] == 4
        and float(selection["four_slot_refinement_neighborhood_max_mean_distance_px"]) == 48.0,
    "selection_frozen": float(loss["w_four_slot_selection"]) == 0.0,
    "geometry_only": float(loss["w_four_slot_geometry"]) == 1.0,
    "all_slots": loss["four_slot_geometry_match_all_slots"] is True,
    "only_refiner_trainable": training["trainable_parameter_prefixes"]
        == ["structured_query_head.set_selection_head.slot_refinement"],
    "frozen_detector_eval": training["frozen_detector_eval"] is True,
    "zero_mix": float(refiner.neighborhood_mix.detach()) == 0.0,
    "score_unchanged": cfg["postprocess"]["score_mode"] == "four_slot",
}
print({"v8_anchor_neighborhood_contract": checks})
if not all(checks.values()):
    raise SystemExit("invalid V8 anchor-neighborhood config")
PY

mkdir -p "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/audits" "${CACHE_ROOT}"
CONTRACT_REPORT="${OUTPUT_ROOT}/audits/v8_anchor_neighborhood_gradient_contract_225k.json"
if [[ "${RUN_CONTRACT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v8_anchor_neighborhood_contract \
    --config "${CONFIG}" \
    --baseline-config "${BASELINE_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --num-workers 0 \
    --output-json "${CONTRACT_REPORT}"
elif [[ ! -f "${CONTRACT_REPORT}" ]]; then
  echo "Missing V8 contract while RUN_CONTRACT=0: ${CONTRACT_REPORT}" >&2
  exit 1
fi

SOURCE_REPORT="${SOURCE_REPORT:-${OUTPUT_ROOT}/reports/source_iter_0225000_uniform256_fp32.json}"
if [[ "${RUN_EVAL}" == "1" && ! -f "${SOURCE_REPORT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
    --config "${CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_ROOT}/source" \
    --max-batches "${MAX_BATCHES}" \
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
    --amp-dtype none \
    --output-json "${SOURCE_REPORT}"
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}/anchor_neighborhood"
mkdir -p "${RUN_DIR}"
FINAL_CHECKPOINT="${RUN_DIR}/iter_${END_TAG}.pt"
if [[ "${RUN_TRAIN}" == "1" && ! -f "${FINAL_CHECKPOINT}" ]]; then
  latest_checkpoint=""
  latest_iteration="${SOURCE_ITERATION}"
  for candidate in "${RUN_DIR}"/iter_*.pt; do
    [[ -f "${candidate}" ]] || continue
    iteration="$(checkpoint_iteration "${candidate}")"
    if (( iteration > latest_iteration && iteration < END_ITERATION )); then
      latest_checkpoint="${candidate}"
      latest_iteration="${iteration}"
    fi
  done
  train_args=(
    --config "${CONFIG}"
    --dataset-root "${DATA_ROOT}"
    --device "${DEVICE}"
    --output-dir "${RUN_DIR}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --num-workers "${NUM_WORKERS}"
    --seg-aux-amp-dtype "${AMP_DTYPE}"
    --compile-model false
    --checkpoint-base "${SOURCE_CHECKPOINT}"
  )
  if [[ -n "${latest_checkpoint}" ]]; then
    train_args+=(
      --resume "${latest_checkpoint}"
      --max-iters "$((END_ITERATION - latest_iteration))"
    )
  else
    train_args+=(
      --init-from "${SOURCE_CHECKPOINT}"
      --init-iteration "${SOURCE_ITERATION}"
      --max-iters "${TRAIN_STEPS}"
    )
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
fi

report_args=()
eval_start="${END_ITERATION}"
if [[ "${EVAL_TRAJECTORY}" == "1" ]]; then
  eval_start=$((SOURCE_ITERATION + CHECKPOINT_INTERVAL))
fi
for ((iteration=eval_start; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
  tag="$(printf '%07d' "${iteration}")"
  checkpoint="${RUN_DIR}/iter_${tag}.pt"
  report="${OUTPUT_ROOT}/reports/iter_${tag}_uniform256_fp32.json"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing V8 trajectory checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  if [[ "${RUN_EVAL}" == "1" && ! -f "${report}" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/iter_${tag}" \
      --max-batches "${MAX_BATCHES}" \
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
      --amp-dtype none \
      --output-json "${report}"
  fi
  if [[ ! -f "${report}" ]]; then
    echo "Missing V8 report: ${report}" >&2
    exit 1
  fi
  report_args+=(--report "${iteration}=${report}")
done

SUMMARY="${OUTPUT_ROOT}/v8_anchor_neighborhood_gate_summary.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v8_anchor_neighborhood_gate \
  --source-report "${SOURCE_REPORT}" \
  --contract "${CONTRACT_REPORT}" \
  "${report_args[@]}" \
  --output-json "${SUMMARY}"

echo "V8 anchor-neighborhood gate complete: ${SUMMARY}"
echo "Long training remains closed regardless of this short-gate verdict."
