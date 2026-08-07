#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-3407}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v5_1_shared_trunk_gate}"

TRUNK_CONFIG="${TRUNK_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_10k.yaml}"
CONTROL_CONFIG="${CONTROL_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_control_10k_to25k.yaml}"
ASSIGNMENT_CONFIG="${ASSIGNMENT_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v5_1_shared_trunk_assignment_10k_to25k.yaml}"

sample_count=$((EVAL_BATCH_SIZE * MAX_BATCHES))

audit_checkpoint() {
  local config="$1"
  local checkpoint="$2"
  local cache_dir="$3"
  local oracle_report="$4"
  local representative_report="$5"

  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_oracle_topk \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${DATA_ROOT}" \
    --split val \
    --device "${DEVICE}" \
    --cache-dir "${cache_dir}" \
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
}

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  report_root="${seed_root}/reports"
  trunk_checkpoint="${seed_root}/shared_trunk/iter_0010000.pt"
  trunk_report_dir="${report_root}/shared_trunk"
  mkdir -p "${trunk_report_dir}"
  if [[ ! -f "${trunk_checkpoint}" ]]; then
    echo "Missing shared trunk checkpoint: ${trunk_checkpoint}" >&2
    exit 1
  fi
  trunk_oracle="${trunk_report_dir}/iter_0010000_uniform${sample_count}.json"
  trunk_representative="${trunk_report_dir}/iter_0010000_representatives_uniform${sample_count}.json"
  audit_checkpoint \
    "${TRUNK_CONFIG}" \
    "${trunk_checkpoint}" \
    "${CACHE_ROOT}/seed_${seed}/shared_trunk/iter_0010000" \
    "${trunk_oracle}" \
    "${trunk_representative}"

  control_reports=()
  assignment_reports=()
  control_representatives=()
  assignment_representatives=()

  for arm in control assignment; do
    if [[ "${arm}" == "control" ]]; then
      config="${CONTROL_CONFIG}"
      arm_name="control_fork"
    else
      config="${ASSIGNMENT_CONFIG}"
      arm_name="assignment_fork"
    fi
    arm_dir="${seed_root}/${arm_name}"
    report_dir="${report_root}/${arm_name}"
    mkdir -p "${report_dir}"
    stability_checkpoints=("${trunk_checkpoint}")

    for iteration in 15000 20000 25000; do
      tag="$(printf '%07d' "${iteration}")"
      checkpoint="${arm_dir}/iter_${tag}.pt"
      if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing V5.1 fork checkpoint: ${checkpoint}" >&2
        exit 1
      fi
      oracle_report="${report_dir}/iter_${tag}_uniform${sample_count}.json"
      representative_report="${report_dir}/iter_${tag}_representatives_uniform${sample_count}.json"
      audit_checkpoint \
        "${config}" \
        "${checkpoint}" \
        "${CACHE_ROOT}/seed_${seed}/${arm_name}/iter_${tag}" \
        "${oracle_report}" \
        "${representative_report}"
      stability_checkpoints+=("${checkpoint}")
      if [[ "${arm}" == "control" ]]; then
        control_reports+=("${oracle_report}")
        control_representatives+=("${representative_report}")
      else
        assignment_reports+=("${oracle_report}")
        assignment_representatives+=("${representative_report}")
      fi
    done

    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_unified_selector_ownership_stability \
      --config "${config}" \
      --checkpoints "${stability_checkpoints[@]}" \
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

  "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v5_1_shared_trunk_gate \
    --trunk-report "${trunk_oracle}" \
    --trunk-representative "${trunk_representative}" \
    --trunk-contract "${seed_root}/shared_trunk_contract.json" \
    --control-reports "${control_reports[@]}" \
    --assignment-reports "${assignment_reports[@]}" \
    --control-representatives "${control_representatives[@]}" \
    --assignment-representatives "${assignment_representatives[@]}" \
    --control-stability "${report_root}/control_fork/ownership_stability_uniform${sample_count}.json" \
    --assignment-stability "${report_root}/assignment_fork/ownership_stability_uniform${sample_count}.json" \
    --output-json "${seed_root}/v5_1_shared_trunk_summary.json"
done

echo "V5.1 shared-trunk causal reports are ready under ${OUTPUT_ROOT}."
