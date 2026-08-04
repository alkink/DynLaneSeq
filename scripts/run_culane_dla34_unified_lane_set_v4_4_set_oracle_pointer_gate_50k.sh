#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

POINTER_CONFIG="${POINTER_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v4_4_set_oracle_pointer.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v4_4_set_oracle_pointer_gate_50k}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v4_4_set_oracle_pointer_gate_50k}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
SOURCE_ITERATION="${SOURCE_ITERATION:-50000}"
TRAIN_STEPS="${TRAIN_STEPS:-15000}"
SEEDS="${SEEDS:-3407}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_EVAL="${RUN_EVAL:-1}"

python_bin="${PYTHON:-python}"
"${python_bin}" - "${POINTER_CONFIG}" <<'PY'
import sys
from dynlaneseq_eg.config import load_config

cfg = load_config(sys.argv[1])
selection = cfg["model"]["structured_query"]["set_selection"]
loss = cfg["loss"]
checks = {
    "permutation_invariant_set_teacher": selection.get("pointer_teacher_mode")
    == "permutation_invariant_set",
    "unique_representative_unary": loss.get("pointer_unary_target_mode")
    == "unique_representative",
    "standard_bce": float(loss.get("set_selection_focal_beta", -1.0)) == 0.0,
    "geometry_detached": bool(selection.get("detach_geometry_features", False)),
}
print(checks)
if not all(checks.values()):
    raise SystemExit("V4.4 set-oracle pointer contract failed")
PY

POINTER_CONFIG="${POINTER_CONFIG}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CACHE_ROOT="${CACHE_ROOT}" \
ARM_NAME=set_oracle_pointer \
EXPERIMENT_LABEL="V4.4 permutation-invariant set-oracle pointer gate" \
  bash scripts/run_culane_dla34_unified_lane_set_v4_3_pointer_stop_gate_50k.sh

if [[ "${RUN_EVAL}" == "1" ]]; then
  end_iteration=$((SOURCE_ITERATION + TRAIN_STEPS))
  end_tag="$(printf '%07d' "${end_iteration}")"
  for seed in ${SEEDS}; do
    checkpoint="${OUTPUT_ROOT}/seed_${seed}/set_oracle_pointer/iter_${end_tag}.pt"
    report_dir="${OUTPUT_ROOT}/seed_${seed}/reports"
    alignment_report="${report_dir}/target_alignment_uniform$((EVAL_BATCH_SIZE * MAX_BATCHES)).json"
    "${python_bin}" -u -m dynlaneseq_eg.tools.analyze_v4_3_pointer_target_alignment \
      --config "${POINTER_CONFIG}" \
      --checkpoint "${checkpoint}" \
      --dataset-root "${DATA_ROOT}" \
      --cache-dir "${CACHE_ROOT}/seed_${seed}/pointer" \
      --split val \
      --max-batches "${MAX_BATCHES}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --sample-strategy uniform \
      --line-width 30 \
      --min-valid-rows 5 \
      --top-k 4 \
      --iou-thresholds 0.50 0.75 \
      --output-json "${alignment_report}"
    "${python_bin}" -u -m dynlaneseq_eg.tools.summarize_v4_4_set_oracle_gate \
      --metric-summary "${report_dir}/summary.json" \
      --target-alignment "${alignment_report}" \
      --output-json "${report_dir}/v4_4_summary.json"
  done
fi
