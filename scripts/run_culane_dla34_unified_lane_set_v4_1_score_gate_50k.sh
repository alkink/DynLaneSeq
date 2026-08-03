#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CONFIG="${SOURCE_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-50000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-8}"
MAX_BATCHES="${MAX_BATCHES:-64}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_1_score_gate_50k}"
RESULT_DIR="${RESULT_DIR:-${OUTPUT_ROOT}/reports}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_1_score_gate_50k}"

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Missing V4 source config: ${SOURCE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Missing V4 source checkpoint: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if (( TRAIN_STEPS < 1 )); then
  echo "TRAIN_STEPS must be positive" >&2
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
  echo "Expected source iteration ${SOURCE_ITERATION}, found ${actual_source_iteration}: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

END_ITERATION=$((SOURCE_ITERATION + TRAIN_STEPS))
END_TAG="$(printf '%07d' "${END_ITERATION}")"
mkdir -p "${OUTPUT_ROOT}" "${RESULT_DIR}" "${CACHE_ROOT}"

arm_names=(
  a_mlp_shared
  b_mlp_unique
  c_set_shared
  d_set_unique
)
arm_configs=(
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_a_mlp_shared.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_b_mlp_unique.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_c_set_shared.yaml
  dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_1_score_d_set_unique.yaml
)

for config in "${arm_configs[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing V4.1 arm config: ${config}" >&2
    exit 1
  fi
done

"${PYTHON}" - "${arm_configs[@]}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

expected = (
    ("independent", True, 1.0, 0.0),
    ("independent", False, 2.0, 0.25),
    ("transformer", True, 1.0, 0.0),
    ("transformer", False, 2.0, 0.25),
)
for path, contract in zip(sys.argv[1:], expected):
    cfg = load_config(path)
    selection = cfg["model"]["structured_query"]["set_selection"]
    loss = cfg["loss"]
    actual = (
        selection["candidate_interaction"],
        bool(loss["set_selection_share_matcher_assignment"]),
        float(loss["set_selection_negative_weight"]),
        float(loss["set_selection_rank_weight"]),
    )
    if actual != contract:
        raise SystemExit(f"V4.1 arm contract mismatch in {path}: {actual} != {contract}")
    if not selection.get("detach_geometry_features", False):
        raise SystemExit(f"geometry detach is disabled in {path}")
    if cfg["training"].get("trainable_parameter_prefixes") != [
        "structured_query_head.set_selection_head"
    ]:
        raise SystemExit(f"unexpected trainable parameter scope in {path}")
    if cfg["training"].get("checkpoint_model_prefixes") != [
        "structured_query_head.set_selection_head"
    ]:
        raise SystemExit(f"unexpected checkpoint tensor scope in {path}")
    if cfg["training"].get("checkpoint_include_optimizer", True):
        raise SystemExit(f"diagnostic checkpoint unnecessarily stores optimizer in {path}")
    if cfg["postprocess"].get("score_mode") != "selection":
        raise SystemExit(f"selection deployment score is disabled in {path}")
print("[OK] V4.1 2x2 config contract")
PY

echo "V4.1 frozen-geometry score gate"
echo "source checkpoint: ${SOURCE_CHECKPOINT}"
echo "source SHA256: $(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
echo "logical iterations: ${SOURCE_ITERATION} -> ${END_ITERATION}"
echo "four arms use the same seed, dataset, augmentation config, batch, and accumulation"
echo "only structured_query_head.set_selection_head is trainable"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  for index in "${!arm_names[@]}"; do
    name="${arm_names[$index]}"
    config="${arm_configs[$index]}"
    output_dir="${OUTPUT_ROOT}/${name}"
    checkpoint="${output_dir}/iter_${END_TAG}.pt"
    mkdir -p "${output_dir}"
    if [[ -f "${checkpoint}" ]]; then
      actual_iteration="$(checkpoint_iteration "${checkpoint}")"
      if (( actual_iteration != END_ITERATION )); then
        echo "Arm checkpoint has wrong internal iteration: ${checkpoint}" >&2
        exit 1
      fi
      echo "[SKIP] ${name} already completed: ${checkpoint}"
      continue
    fi
    echo "===== TRAIN ${name} ====="
    echo "config: ${config}"
    echo "output: ${output_dir}"
    # Always restart an incomplete arm from the same 50k source. The generic
    # checkpoint does not store the DataLoader/augmentation cursor, so an
    # automatic partial resume would silently break the paired-data contract.
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train \
      --config "${config}" \
      --dataset-root "${DATA_ROOT}" \
      --device "${DEVICE}" \
      --output-dir "${output_dir}" \
      --max-iters "${TRAIN_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --seg-aux-amp-dtype "${AMP_DTYPE}" \
      --init-from "${SOURCE_CHECKPOINT}" \
      --init-iteration "${SOURCE_ITERATION}" \
      --checkpoint-base "${SOURCE_CHECKPOINT}" \
      2>&1 | tee "${output_dir}/train.log"
    actual_iteration="$(checkpoint_iteration "${checkpoint}")"
    if (( actual_iteration != END_ITERATION )); then
      echo "Arm ${name} did not produce the expected ${END_ITERATION} checkpoint" >&2
      exit 1
    fi
  done
fi

source_report="${RESULT_DIR}/source_v4_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
if [[ "${RUN_EVAL}" == "1" ]]; then
  report_names=(source_v4 "${arm_names[@]}")
  report_configs=("${SOURCE_CONFIG}" "${arm_configs[@]}")
  report_checkpoints=("${SOURCE_CHECKPOINT}")
  for name in "${arm_names[@]}"; do
    report_checkpoints+=("${OUTPUT_ROOT}/${name}/iter_${END_TAG}.pt")
  done

  for index in "${!report_names[@]}"; do
    name="${report_names[$index]}"
    config="${report_configs[$index]}"
    checkpoint="${report_checkpoints[$index]}"
    report="${RESULT_DIR}/${name}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint required for evaluation: ${checkpoint}" >&2
      exit 1
    fi
    echo "===== EVAL ${name} ====="
    "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --split val \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_ROOT}/${name}" \
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
      --row-visibility-thresh 0 \
      --nms-min-overlap-points 5 \
      --hard-diversity-distances 20 \
      --mmr-sigmas 20 \
      --mmr-penalties 0.50 \
      --output-json "${report}" \
      2>&1 | tee "${RESULT_DIR}/${name}.log"
  done
fi

for name in source_v4 "${arm_names[@]}"; do
  report="${RESULT_DIR}/${name}_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
  if [[ ! -f "${report}" ]]; then
    echo "Missing V4.1 report: ${report}" >&2
    exit 1
  fi
done

"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v4_1_score_gate \
  --source "${source_report}" \
  --a "${RESULT_DIR}/a_mlp_shared_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --b "${RESULT_DIR}/b_mlp_unique_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --c "${RESULT_DIR}/c_set_shared_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --d "${RESULT_DIR}/d_set_unique_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json" \
  --output-json "${RESULT_DIR}/summary.json"

echo "V4.1 gate completed: ${RESULT_DIR}/summary.json"
