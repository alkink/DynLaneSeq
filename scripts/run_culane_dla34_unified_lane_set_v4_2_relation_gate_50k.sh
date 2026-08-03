#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-50000}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
SEEDS="${SEEDS:-3407}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
TRAJECTORY_MAX_BATCHES="${TRAJECTORY_MAX_BATCHES:-16}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TRAJECTORY_EVAL="${RUN_TRAJECTORY_EVAL:-1}"
RUN_GRAD_AUDIT="${RUN_GRAD_AUDIT:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_2_relation_gate_50k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_2_relation_gate_50k}"

arm_names=(
  r0_generic_frozen
  r1_generic_semantic
  r2_relation_semantic
  r3_relation_setloss
)
arm_configs=(
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_2_score_r0_generic_frozen.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_2_score_r1_generic_semantic.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_2_score_r2_relation_semantic.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_2_score_r3_relation_setloss.yaml
)

if [[ ! -f "${SOURCE_CONFIG}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing V4 source config/checkpoint" >&2
  exit 1
fi
if (( TRAIN_STEPS < 500 || TRAIN_STEPS % 500 != 0 )); then
  echo "TRAIN_STEPS must be a positive multiple of 500" >&2
  exit 1
fi

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

actual_source_iteration="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( actual_source_iteration != SOURCE_ITERATION )); then
  echo "Expected source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}" >&2
  exit 1
fi

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
        f"refusing trajectory checkpoints: only {free / 1024 ** 3:.2f} GiB free, "
        f"need at least {minimum / 1024 ** 3:.2f} GiB"
    )
PY

"${PYTHON}" - "${arm_configs[@]}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

expected_interactions = (
    "transformer",
    "transformer",
    "relation_transformer",
    "relation_transformer",
)
for index, (path, expected) in enumerate(zip(sys.argv[1:], expected_interactions)):
    cfg = load_config(path)
    selection = cfg["model"]["structured_query"]["set_selection"]
    training = cfg["training"]
    loss = cfg["loss"]
    if selection["candidate_interaction"] != expected:
        raise SystemExit(f"candidate interaction mismatch in {path}")
    if not selection.get("detach_geometry_features", False):
        raise SystemExit(f"geometry detach disabled in {path}")
    if training.get("checkpoint_interval") != 500:
        raise SystemExit(f"500-step trajectory disabled in {path}")
    if not training.get("checkpoint_include_optimizer", False):
        raise SystemExit(f"optimizer trajectory disabled in {path}")
    prefixes = training.get("trainable_parameter_prefixes", [])
    if index == 0 and prefixes != ["structured_query_head.set_selection_head"]:
        raise SystemExit("R0 must keep semantic score adapters frozen")
    if index > 0 and not any("semantic_attention" in item for item in prefixes):
        raise SystemExit(f"semantic score adapters are not trainable in {path}")
    relation_weights = (
        float(loss.get("set_selection_coverage_weight", 0.0)),
        float(loss.get("set_selection_duplicate_weight", 0.0)),
        float(loss.get("set_selection_winner_weight", 0.0)),
        float(loss.get("set_selection_count_weight", 0.0)),
    )
    if (index < 3) != (relation_weights == (0.0, 0.0, 0.0, 0.0)):
        raise SystemExit(f"set-loss isolation mismatch in {path}: {relation_weights}")
print("[OK] V4.2 R0-R3 config contract")
PY

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"

echo "V4.2 frozen-geometry relation gate"
echo "source: ${SOURCE_CHECKPOINT}"
echo "source SHA256: $(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "seeds: ${SEEDS}"
echo "R0 longer-control; R1 +semantic; R2 +relations; R3 +set losses"

if [[ "${RUN_GRAD_AUDIT}" == "1" ]]; then
  gradient_dir="${OUTPUT_ROOT}/gradient_contract"
  mkdir -p "${gradient_dir}"
  for index in "${!arm_names[@]}"; do
    name="${arm_names[$index]}"
    config="${arm_configs[$index]}"
    echo "===== GRADIENT CONTRACT ${name} ====="
    "${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v4_2_gradient_contract \
      --config "${config}" \
      --checkpoint "${SOURCE_CHECKPOINT}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --batch-size 2 \
      --num-workers "${NUM_WORKERS}" \
      --amp-dtype "${AMP_DTYPE}" \
      --output-json "${gradient_dir}/${name}.json"
  done
fi

for seed in ${SEEDS}; do
  seed_root="${OUTPUT_ROOT}/seed_${seed}"
  result_dir="${seed_root}/reports"
  mkdir -p "${seed_root}" "${result_dir}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    for index in "${!arm_names[@]}"; do
      name="${arm_names[$index]}"
      config="${arm_configs[$index]}"
      output_dir="${seed_root}/${name}"
      checkpoint="${output_dir}/iter_${END_TAG}.pt"
      mkdir -p "${output_dir}"
      if [[ -f "${checkpoint}" ]]; then
        actual_iteration="$(checkpoint_iteration "${checkpoint}")"
        if (( actual_iteration == END_ITERATION )); then
          echo "[SKIP] seed=${seed} ${name}: ${checkpoint}"
          continue
        fi
      fi
      echo "===== TRAIN seed=${seed} ${name} ====="
      # An incomplete arm is intentionally replayed from the same 50k source;
      # partial DataLoader cursor state is not checkpointed and must not be
      # silently mixed into this causal comparison.
      "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
        --config "${config}" \
        --dataset-root "${DATA_ROOT}" \
        --device "${DEVICE}" \
        --output-dir "${output_dir}" \
        --max-iters "${TRAIN_STEPS}" \
        --seed "${seed}" \
        --batch-size "${BATCH_SIZE}" \
        --grad-accum "${GRAD_ACCUM}" \
        --seg-aux-amp-dtype "${AMP_DTYPE}" \
        --init-from "${SOURCE_CHECKPOINT}" \
        --init-iteration "${SOURCE_ITERATION}" \
        --checkpoint-base "${SOURCE_CHECKPOINT}" \
        2>&1 | tee "${output_dir}/train.log"
      actual_iteration="$(checkpoint_iteration "${checkpoint}")"
      if (( actual_iteration != END_ITERATION )); then
        echo "${name} failed to produce iteration ${END_ITERATION}" >&2
        exit 1
      fi
    done
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    report_names=(source_v4 "${arm_names[@]}")
    report_configs=("${SOURCE_CONFIG}" "${arm_configs[@]}")
    report_checkpoints=("${SOURCE_CHECKPOINT}")
    for name in "${arm_names[@]}"; do
      report_checkpoints+=("${seed_root}/${name}/iter_${END_TAG}.pt")
    done
    for index in "${!report_names[@]}"; do
      name="${report_names[$index]}"
      config="${report_configs[$index]}"
      checkpoint="${report_checkpoints[$index]}"
      report="${result_dir}/${name}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
      echo "===== FINAL EVAL seed=${seed} ${name} ====="
      "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --dataset-root "${DATA_ROOT}" \
        --split val \
        --device "${DEVICE}" \
        --cache-dir "${CACHE_ROOT}/seed_${seed}/${name}/final" \
        --max-batches "${MAX_BATCHES}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --metric-workers "${METRIC_WORKERS}" \
        --sample-strategy uniform \
        --amp-dtype "${AMP_DTYPE}" \
        --stage main \
        --top-k 4 \
        --iou-thresholds 0.50 0.75 \
        --near-min-iou 0.30 \
        --line-width 30 \
        --min-valid-rows 5 \
        --row-visibility-thresh 0 \
        --hard-diversity-distances 20 \
        --mmr-sigmas 20 \
        --mmr-penalties 0.50 \
        --output-json "${report}" \
        2>&1 | tee "${result_dir}/${name}.log"
    done

    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_2_relation_gate \
      --source "${result_dir}/source_v4_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
      --r0 "${result_dir}/r0_generic_frozen_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
      --r1 "${result_dir}/r1_generic_semantic_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
      --r2 "${result_dir}/r2_relation_semantic_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
      --r3 "${result_dir}/r3_relation_setloss_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
      --output-json "${result_dir}/summary.json"
  fi

  if [[ "${RUN_TRAJECTORY_EVAL}" == "1" ]]; then
    for index in "${!arm_names[@]}"; do
      name="${arm_names[$index]}"
      config="${arm_configs[$index]}"
      trajectory_dir="${result_dir}/trajectory/${name}"
      mkdir -p "${trajectory_dir}"
      for ((iteration=SOURCE_ITERATION+500; iteration<=END_ITERATION; iteration+=500)); do
        tag="$(printf '%07d' "${iteration}")"
        checkpoint="${seed_root}/${name}/iter_${tag}.pt"
        report="${trajectory_dir}/iter_${tag}.json"
        if [[ ! -f "${checkpoint}" ]]; then
          echo "Missing trajectory checkpoint: ${checkpoint}" >&2
          exit 1
        fi
        if [[ -f "${report}" ]]; then
          continue
        fi
        echo "===== TRAJECTORY seed=${seed} ${name} ${tag} ====="
        "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
          --config "${config}" \
          --checkpoint "${checkpoint}" \
          --dataset-root "${DATA_ROOT}" \
          --split val \
          --device "${DEVICE}" \
          --cache-dir "${CACHE_ROOT}/seed_${seed}/${name}/iter_${tag}" \
          --max-batches "${TRAJECTORY_MAX_BATCHES}" \
          --eval-batch-size "${EVAL_BATCH_SIZE}" \
          --num-workers "${NUM_WORKERS}" \
          --metric-workers "${METRIC_WORKERS}" \
          --sample-strategy uniform \
          --amp-dtype "${AMP_DTYPE}" \
          --stage main \
          --top-k 4 \
          --iou-thresholds 0.50 0.75 \
          --near-min-iou 0.30 \
          --line-width 30 \
          --min-valid-rows 5 \
          --row-visibility-thresh 0 \
          --hard-diversity-distances 20 \
          --mmr-sigmas 20 \
          --mmr-penalties 0.50 \
          --output-json "${report}"
      done
    done
    "${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_2_score_trajectory \
      --root "${result_dir}/trajectory" \
      --arms "${arm_names[@]}" \
      --output-json "${result_dir}/trajectory_summary.json"
  fi
done

echo "V4.2 relation gate completed under ${OUTPUT_ROOT}"
