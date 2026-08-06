#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-3407}"
ARMS="${ARMS:-a b}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_protected_ownership_gate}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v5_protected_ownership_gate}"

CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_a_protected_ownership_sidecar_25k.yaml}"
ASSIGNMENT_CONFIG="${ASSIGNMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_b_protected_ownership_assignment_25k.yaml}"

sample_count=$((EVAL_BATCH_SIZE * MAX_BATCHES))
for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    if [[ "${arm}" == "a" ]]; then
      config="${CONTROL_CONFIG}"
      arm_name="v5_a_sidecar"
    elif [[ "${arm}" == "b" ]]; then
      config="${ASSIGNMENT_CONFIG}"
      arm_name="v5_b_assignment"
    else
      echo "ARMS accepts only 'a' and/or 'b'; got ${arm}." >&2
      exit 1
    fi
    arm_dir="${OUTPUT_ROOT}/seed_${seed}/${arm_name}"
    report_dir="${OUTPUT_ROOT}/seed_${seed}/reports/${arm_name}"
    mkdir -p "${report_dir}"
    checkpoints=()

    for iteration in 5000 10000 15000 20000 25000; do
      tag="$(printf '%07d' "${iteration}")"
      checkpoint="${arm_dir}/iter_${tag}.pt"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing V5 trajectory checkpoint: ${checkpoint}" >&2
        exit 1
      fi
      checkpoints+=("${checkpoint}")
      oracle_report="${report_dir}/iter_${tag}_uniform${sample_count}.json"
      representative_report="${report_dir}/iter_${tag}_representatives_uniform${sample_count}.json"

      "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --dataset-root "${DATA_ROOT}" \
        --split val \
        --device "${DEVICE}" \
        --cache-dir "${CACHE_ROOT}/seed_${seed}/${arm_name}/iter_${tag}" \
        --reuse-cache \
        --max-batches "${MAX_BATCHES}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --official-iou-workers "${METRIC_WORKERS}" \
        --sample-strategy uniform \
        --top-k 4 \
        --iou-thresholds 0.50 0.75 \
        --operating-points 0.0:0.5 0.0:-1.0 \
        --fixed-points-only \
        --line-width 30 \
        --min-valid-rows 5 \
        --nms-distance-thresh-px 0 \
        --exact-postprocess \
        --output-json "${oracle_report}"

      "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v5_ownership_representatives \
        --oracle-report "${oracle_report}" \
        --representable-min 0.50 \
        --cluster-min 0.30 \
        --pair-margin 0.01 \
        --output-json "${representative_report}"
    done

    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability \
      --config "${config}" \
      --checkpoints "${checkpoints[@]}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --max-batches "${MAX_BATCHES}" \
      --num-workers "${NUM_WORKERS}" \
      --amp-dtype "${AMP_DTYPE}" \
      --line-width 30 \
      --min-valid-rows 5 \
      --top-k 4 \
      --score-mode exist \
      --quality-power 0.0 \
      --output-json "${report_dir}/ownership_stability_uniform${sample_count}.json"
  done

  if [[ " ${ARMS} " == *" a "* && " ${ARMS} " == *" b "* ]]; then
    control_reports=()
    assignment_reports=()
    control_representatives=()
    assignment_representatives=()
    for iteration in 5000 10000 15000 20000 25000; do
      tag="$(printf '%07d' "${iteration}")"
      control_reports+=(
        "${OUTPUT_ROOT}/seed_${seed}/reports/v5_a_sidecar/iter_${tag}_uniform${sample_count}.json"
      )
      assignment_reports+=(
        "${OUTPUT_ROOT}/seed_${seed}/reports/v5_b_assignment/iter_${tag}_uniform${sample_count}.json"
      )
      control_representatives+=(
        "${OUTPUT_ROOT}/seed_${seed}/reports/v5_a_sidecar/iter_${tag}_representatives_uniform${sample_count}.json"
      )
      assignment_representatives+=(
        "${OUTPUT_ROOT}/seed_${seed}/reports/v5_b_assignment/iter_${tag}_representatives_uniform${sample_count}.json"
      )
    done
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v5_protected_ownership_gate \
      --control-reports "${control_reports[@]}" \
      --assignment-reports "${assignment_reports[@]}" \
      --control-representatives "${control_representatives[@]}" \
      --assignment-representatives "${assignment_representatives[@]}" \
      --control-stability "${OUTPUT_ROOT}/seed_${seed}/reports/v5_a_sidecar/ownership_stability_uniform${sample_count}.json" \
      --assignment-stability "${OUTPUT_ROOT}/seed_${seed}/reports/v5_b_assignment/ownership_stability_uniform${sample_count}.json" \
      --output-json "${OUTPUT_ROOT}/seed_${seed}/v5_summary.json"
  fi
done

echo "V5 protected-ownership trajectory reports are ready under ${OUTPUT_ROOT}."
