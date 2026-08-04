#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_cluster_soft_pointer.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v4_5_cluster_soft_pointer_gate_50k/seed_3407/cluster_soft_pointer/iter_0060000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-60000}"
Q1_CONFIG="${Q1_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_1_q1_quality_policy_split.yaml}"
Q2_CONFIG="${Q2_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_5_1_q2_quality_policy_listwise.yaml}"
ARMS="${ARMS:-q1_quality_policy_split q2_quality_policy_listwise}"
SEEDS="${SEEDS:-3407}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GRAD_AUDIT="${RUN_GRAD_AUDIT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TRAJECTORY_EVAL="${RUN_TRAJECTORY_EVAL:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-0.75}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_5_1_quality_policy_gate_60k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_5_1_quality_policy_gate_60k}"

config_for_arm() {
  case "$1" in
    q1_quality_policy_split) echo "${Q1_CONFIG}" ;;
    q2_quality_policy_listwise) echo "${Q2_CONFIG}" ;;
    *) echo "Unknown V4.5.1 arm: $1" >&2; return 1 ;;
  esac
}

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch
try:
    payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(sys.argv[1], map_location="cpu")
print(int(payload.get("iteration", -1)))
PY
}

for required in "${SOURCE_CONFIG}" "${SOURCE_CHECKPOINT}" "${Q1_CONFIG}" "${Q2_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V4.5.1 artefact: ${required}" >&2
    exit 1
  fi
done
if (( TRAIN_STEPS < 1 || CHECKPOINT_INTERVAL < 1 )); then
  echo "TRAIN_STEPS and CHECKPOINT_INTERVAL must be positive" >&2
  exit 1
fi
if (( TRAIN_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "TRAIN_STEPS must be divisible by CHECKPOINT_INTERVAL" >&2
  exit 1
fi
actual_source_iteration="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( actual_source_iteration != SOURCE_ITERATION )); then
  echo "Expected V4.5 source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
  exit 1
fi

"${PYTHON}" - "${Q1_CONFIG}" "${Q2_CONFIG}" "${CHECKPOINT_INTERVAL}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

q1 = load_config(sys.argv[1])
q2 = load_config(sys.argv[2])
interval = int(sys.argv[3])
checks = {}
for name, cfg in (("q1", q1), ("q2", q2)):
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    training = cfg["training"]
    checks[f"{name}_decoupled"] = selection.get("pointer_quality_policy_mode") == "decoupled"
    checks[f"{name}_bounded_scale"] = float(selection.get("pointer_quality_prior_max_scale", 0.0)) == 2.0
    checks[f"{name}_cluster_soft"] = selection.get("pointer_teacher_mode") == "cluster_soft_randomized"
    checks[f"{name}_max_quality"] = loss.get("pointer_unary_target_mode") == "max_quality"
    checks[f"{name}_quality_weight"] = abs(float(loss.get("pointer_quality_weight", -1.0)) - 0.10) < 1e-12
    checks[f"{name}_checkpoint_interval"] = int(training.get("checkpoint_interval", -1)) == interval
checks["q1_no_listwise"] = float(q1["loss"].get("pointer_cluster_listwise_weight", -1.0)) == 0.0
checks["q2_listwise_only_factor"] = abs(float(q2["loss"].get("pointer_cluster_listwise_weight", -1.0)) - 0.25) < 1e-12
print(checks)
if not all(checks.values()):
    raise SystemExit("V4.5.1 Q1/Q2 config contract failed")
PY

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON}" - "${OUTPUT_ROOT}" "${MIN_FREE_GB}" <<'PY'
import shutil
import sys
from pathlib import Path
path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(path).free
minimum = float(sys.argv[2]) * 1024 ** 3
print({"checkpoint_filesystem_free_gib": round(free / 1024 ** 3, 2)})
if free < minimum:
    raise SystemExit(
        f"only {free / 1024 ** 3:.2f} GiB free; V4.5.1 needs "
        f"at least {minimum / 1024 ** 3:.2f} GiB"
    )
PY
else
  mkdir -p "${OUTPUT_ROOT}"
  echo "[SKIP] disk gate (RUN_TRAIN=0)"
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
echo "V4.5.1 quality-policy gate"
echo "source: ${SOURCE_CHECKPOINT}"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "training arms per seed: ${ARMS}"

mkdir -p "${OUTPUT_ROOT}/audits"
baseline_audit="${OUTPUT_ROOT}/audits/source_v4_5_gradient_contract.json"
if [[ "${RUN_GRAD_AUDIT}" == "1" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_5_1_quality_policy_contract \
    --config "${SOURCE_CONFIG}" \
    --baseline-config "${SOURCE_CONFIG}" \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --source-iteration "${SOURCE_ITERATION}" \
    --dataset-root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --num-workers "${NUM_WORKERS}" \
    --amp-dtype "${AMP_DTYPE}" \
    --output-json "${baseline_audit}"
elif [[ ! -f "${baseline_audit}" ]]; then
  echo "Missing baseline V4.5 audit while RUN_GRAD_AUDIT=0: ${baseline_audit}" >&2
  exit 1
fi
gradient_args=()
for arm in ${ARMS}; do
  config="$(config_for_arm "${arm}")"
  audit="${OUTPUT_ROOT}/audits/${arm}_gradient_contract.json"
  if [[ "${RUN_GRAD_AUDIT}" == "1" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_5_1_quality_policy_contract \
      --config "${config}" \
      --baseline-config "${SOURCE_CONFIG}" \
      --checkpoint "${SOURCE_CHECKPOINT}" \
      --source-iteration "${SOURCE_ITERATION}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --batch-size 4 \
      --num-workers "${NUM_WORKERS}" \
      --amp-dtype "${AMP_DTYPE}" \
      --output-json "${audit}"
  elif [[ ! -f "${audit}" ]]; then
    echo "Missing V4.5.1 audit while RUN_GRAD_AUDIT=0: ${audit}" >&2
    exit 1
  fi
  gradient_args+=("${arm}=${audit}")
done

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  report_dir="${seed_root}/reports"
  mkdir -p "${report_dir}"
  source_report="${report_dir}/source_v4_5_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  if [[ "${RUN_EVAL}" == "1" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${SOURCE_CONFIG}" \
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
      --output-json "${source_report}"
  fi

  trajectory_args=()
  for arm in ${ARMS}; do
    config="$(config_for_arm "${arm}")"
    output_dir="${seed_root}/${arm}"
    mkdir -p "${output_dir}"
    final_checkpoint="${output_dir}/iter_${END_TAG}.pt"
    if [[ "${RUN_TRAIN}" == "1" ]]; then
      if [[ -f "${final_checkpoint}" ]] && (( $(checkpoint_iteration "${final_checkpoint}") == END_ITERATION )); then
        echo "[SKIP] existing seed=${seed} arm=${arm} checkpoint ${final_checkpoint}"
      else
        "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
          --config "${config}" \
          --dataset-root "${DATA_ROOT}" \
          --device "${DEVICE}" \
          --output-dir "${output_dir}" \
          --max-iters "${TRAIN_STEPS}" \
          --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
          --seed "${seed}" \
          --batch-size "${BATCH_SIZE}" \
          --grad-accum "${GRAD_ACCUM}" \
          --seg-aux-amp-dtype "${AMP_DTYPE}" \
          --init-from "${SOURCE_CHECKPOINT}" \
          --init-iteration "${SOURCE_ITERATION}" \
          --checkpoint-base "${SOURCE_CHECKPOINT}" \
          2>&1 | tee "${output_dir}/train.log"
      fi
      if [[ ! -f "${final_checkpoint}" ]] || (( $(checkpoint_iteration "${final_checkpoint}") != END_ITERATION )); then
        echo "V4.5.1 failed to produce ${final_checkpoint}" >&2
        exit 1
      fi
    fi

    if [[ "${RUN_EVAL}" == "1" ]]; then
      for ((iteration=SOURCE_ITERATION + CHECKPOINT_INTERVAL; iteration<=END_ITERATION; iteration+=CHECKPOINT_INTERVAL)); do
        tag="$(printf '%07d' "${iteration}")"
        checkpoint="${output_dir}/iter_${tag}.pt"
        if [[ ! -f "${checkpoint}" ]]; then
          echo "Missing V4.5.1 trajectory checkpoint: ${checkpoint}" >&2
          exit 1
        fi
        report="${report_dir}/${arm}_iter_${tag}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
        if (( iteration == END_ITERATION || RUN_TRAJECTORY_EVAL == 1 )); then
          "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
            --config "${config}" \
            --checkpoint "${checkpoint}" \
            --dataset-root "${DATA_ROOT}" \
            --split val \
            --device "${DEVICE}" \
            --cache-dir "${CACHE_ROOT}/seed_${seed}/${arm}_${tag}" \
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
            --output-json "${report}"
          trajectory_args+=("${arm}:${iteration}=${report}")
        fi
      done
    fi
  done

  if [[ "${RUN_EVAL}" == "1" ]]; then
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_5_1_quality_policy_gate \
      --source "${source_report}" \
      --baseline-gradient-audit "${baseline_audit}" \
      --gradient-audit "${gradient_args[@]}" \
      --trajectory "${trajectory_args[@]}" \
      --output-json "${report_dir}/v4_5_1_summary.json"
  fi
done

echo "V4.5.1 quality-policy gate completed under ${OUTPUT_ROOT}"
