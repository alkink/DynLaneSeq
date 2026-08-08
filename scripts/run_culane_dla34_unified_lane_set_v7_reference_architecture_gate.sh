#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/diagnostics/unified_lane_set_v5_1_shared_trunk_gate/seed_3407/assignment_fork/iter_0025000.pt}"
SOURCE_ITERATION="${SOURCE_ITERATION:-25000}"
SEED="${SEED:-3407}"
MEMORIZE_STEPS="${MEMORIZE_STEPS:-1000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-12}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
RUN_GENERALIZATION="${RUN_GENERALIZATION:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostics/unified_lane_set_v7_hard_slot_assignment_gate}"
CACHE_ROOT="${CACHE_ROOT:-outputs/diagnostic_cache/unified_lane_set_v7_hard_slot_assignment_gate}"

HARD_CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_hard_reference_gate_25k_to28k.yaml
DIRECT_CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_direct_reference_gate_25k_to28k.yaml
HARD_MEMORY_CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_hard_reference_memorize64.yaml
DIRECT_MEMORY_CONFIG=dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v7_direct_reference_memorize64.yaml

for required in \
  "${SOURCE_CHECKPOINT}" \
  "${HARD_CONFIG}" \
  "${DIRECT_CONFIG}" \
  "${HARD_MEMORY_CONFIG}" \
  "${DIRECT_MEMORY_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing V7 reference-gate artefact: ${required}" >&2
    exit 1
  fi
done
if (( MEMORIZE_STEPS < 1 || TRAIN_STEPS < 1 || CHECKPOINT_INTERVAL < 1 )); then
  echo "MEMORIZE_STEPS, TRAIN_STEPS and CHECKPOINT_INTERVAL must be positive." >&2
  exit 1
fi
if (( MEMORIZE_STEPS % CHECKPOINT_INTERVAL != 0 || TRAIN_STEPS % CHECKPOINT_INTERVAL != 0 )); then
  echo "Both step counts must be divisible by CHECKPOINT_INTERVAL." >&2
  exit 1
fi
if (( BATCH_SIZE * GRAD_ACCUM != 16 )); then
  echo "The paired generalization gate requires effective batch 16." >&2
  exit 1
fi
if [[ "${AMP_DTYPE}" != "bfloat16" ]]; then
  echo "The V7 reference gate requires AMP_DTYPE=bfloat16." >&2
  exit 1
fi
if [[ "${RUN_GENERALIZATION}" != "0" && "${RUN_GENERALIZATION}" != "1" ]]; then
  echo "RUN_GENERALIZATION must be 0 or 1." >&2
  exit 1
fi

checkpoint_iteration() {
  "${PYTHON}" - "$1" <<'PY'
import sys
from dynlaneseq_eg.engine.checkpoint import _torch_load
print(int(_torch_load(sys.argv[1]).get("iteration", -1)))
PY
}
ACTUAL_SOURCE_ITERATION="$(checkpoint_iteration "${SOURCE_CHECKPOINT}")"
if (( ACTUAL_SOURCE_ITERATION != SOURCE_ITERATION )); then
  echo "Expected source iteration ${SOURCE_ITERATION}, found ${ACTUAL_SOURCE_ITERATION}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"
INIT_CHECKPOINT="${OUTPUT_ROOT}/shared_slot_init_iter_0025000.pt"
INIT_REPORT="${OUTPUT_ROOT}/shared_slot_init.json"
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  "${PYTHON}" -u -m dynlaneseq_eg.tools.materialize_v7_reference_gate_init \
    --config "${HARD_CONFIG}" \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --source-iteration "${SOURCE_ITERATION}" \
    --seed "${SEED}" \
    --output-checkpoint "${INIT_CHECKPOINT}" \
    --output-json "${INIT_REPORT}"
fi
if [[ ! -f "${INIT_REPORT}" ]]; then
  echo "Missing shared-initialization provenance: ${INIT_REPORT}" >&2
  exit 1
fi
"${PYTHON}" - "${INIT_REPORT}" "${SOURCE_CHECKPOINT}" "${SOURCE_ITERATION}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = Path(sys.argv[2]).resolve()
checks = {
    "source_path": Path(report["source_checkpoint"]).resolve() == source,
    "source_size": int(report["source_checkpoint_size"]) == source.stat().st_size,
    "source_iteration": int(report["source_iteration"]) == int(sys.argv[3]),
    "seed": int(report["seed"]) == int(sys.argv[4]),
}
print({"shared_initialization_provenance": checks})
if not all(checks.values()):
    raise SystemExit("stale V7 shared initialization; use a fresh OUTPUT_ROOT")
PY

# A deterministic, uniformly spaced fixed set avoids a single-video-prefix
# memorization artefact.  This is an output artefact, never a dataset mutation.
FIXED_LIST="${OUTPUT_ROOT}/train_uniform64.txt"
"${PYTHON}" - "${DATA_ROOT}/list/train_gt.txt" "${FIXED_LIST}" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
count = min(64, len(lines))
if count < 1:
    raise SystemExit("empty CULane training list")
indices = [round(index * (len(lines) - 1) / max(count - 1, 1)) for index in range(count)]
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("\n".join(lines[index] for index in indices) + "\n", encoding="utf-8")
print({"fixed_list": str(destination), "images": count, "first": indices[0], "last": indices[-1]})
PY

# Both arms must start from byte-identical trainable tensors.  The only
# behavioral difference is hard_st versus soft geometry reference forward.
"${PYTHON}" - "${HARD_CONFIG}" "${DIRECT_CONFIG}" "${INIT_CHECKPOINT}" <<'PY'
import hashlib
import sys
import torch
from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint
from dynlaneseq_eg.factory import build_model

def digest(config_path):
    cfg = load_config(config_path)
    cfg.setdefault("model", {})["pretrained_backbone"] = False
    cfg["model"]["require_pretrained_backbone"] = False
    model = build_model(cfg)
    load_checkpoint(sys.argv[3], model, strict=False)
    head = model.structured_query_head.set_selection_head
    state = head.state_dict()
    value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value.update(name.encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest(), sum(p.numel() for p in head.parameters())

hard = digest(sys.argv[1])
direct = digest(sys.argv[2])
print({"hard": hard, "direct": direct, "exact_shared_initialization": hard == direct})
if hard != direct:
    raise SystemExit("paired V7 heads do not share exact initialization")
PY

run_arm() {
  local arm="$1"
  local config="$2"
  local steps="$3"
  local output_dir="$4"
  local train_list="${5:-}"
  local accumulation="$6"
  local final_iteration=$((SOURCE_ITERATION + steps))
  local final_tag
  final_tag="$(printf '%07d' "${final_iteration}")"
  local final_checkpoint="${output_dir}/iter_${final_tag}.pt"
  mkdir -p "${output_dir}"
  if [[ ! -f "${final_checkpoint}" ]]; then
    local latest_checkpoint=""
    local latest_iteration="${SOURCE_ITERATION}"
    local candidate_iteration
    for candidate in "${output_dir}"/iter_*.pt; do
      [[ -f "${candidate}" ]] || continue
      candidate_iteration="$(checkpoint_iteration "${candidate}")"
      if (( candidate_iteration > latest_iteration && candidate_iteration < final_iteration )); then
        latest_checkpoint="${candidate}"
        latest_iteration="${candidate_iteration}"
      fi
    done
    local train_args=(
      --config "${config}"
      --dataset-root "${DATA_ROOT}"
      --device "${DEVICE}"
      --output-dir "${output_dir}"
      --checkpoint-base "${INIT_CHECKPOINT}"
      --checkpoint-interval "${CHECKPOINT_INTERVAL}"
      --seed "${SEED}"
      --batch-size "${BATCH_SIZE}"
      --grad-accum "${accumulation}"
      --num-workers "${NUM_WORKERS}"
      --seg-aux-amp-dtype "${AMP_DTYPE}"
      --compile-model false
    )
    if [[ -n "${train_list}" ]]; then
      train_args+=(--train-list "${train_list}")
    fi
    if [[ -n "${latest_checkpoint}" ]]; then
      echo "Resuming ${arm}: ${latest_iteration} -> ${final_iteration}"
      train_args+=(
        --resume "${latest_checkpoint}"
        --max-iters "$((final_iteration - latest_iteration))"
      )
    else
      echo "Training ${arm}: ${SOURCE_ITERATION} -> ${final_iteration}"
      train_args+=(
        --init-from "${INIT_CHECKPOINT}"
        --init-iteration "${SOURCE_ITERATION}"
        --max-iters "${steps}"
      )
    fi
    "${PYTHON}" -u -m dynlaneseq_eg.tools.train "${train_args[@]}" \
      2>&1 | tee -a "${output_dir}/train.log"
  fi
  if [[ ! -f "${final_checkpoint}" ]]; then
    echo "Missing ${arm} checkpoint: ${final_checkpoint}" >&2
    exit 1
  fi
  printf '%s\n' "${final_checkpoint}"
}

evaluate_report() {
  local config="$1"
  local checkpoint="$2"
  local report="$3"
  local cache="$4"
  local split="$5"
  local list_path="${6:-}"
  local max_batches="$7"
  if [[ -f "${report}" ]]; then
    return
  fi
  local args=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --dataset-root "${DATA_ROOT}"
    --split "${split}"
    --device "${DEVICE}"
    --cache-dir "${cache}"
    --max-batches "${max_batches}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --metric-workers "${METRIC_WORKERS}"
    --sample-strategy "$([[ "${split}" == train ]] && echo sequential || echo uniform)"
    --stage main
    --top-k 4
    --iou-thresholds 0.50 0.75
    --near-min-iou 0.30
    --line-width 30
    --min-valid-rows 5
    --hard-diversity-distances 20
    --mmr-sigmas 20
    --mmr-penalties 0.50
    --output-json "${report}"
  )
  if [[ -n "${list_path}" ]]; then
    args+=(--list-path "${list_path}")
  fi
  "${PYTHON}" -u -m dynlaneseq_eg.tools.analyze_v4_selection_coverage "${args[@]}"
}

MEMORY_HARD_DIR="${OUTPUT_ROOT}/memorize64/hard"
MEMORY_DIRECT_DIR="${OUTPUT_ROOT}/memorize64/direct"
run_arm memorize64_hard "${HARD_MEMORY_CONFIG}" "${MEMORIZE_STEPS}" "${MEMORY_HARD_DIR}" "${FIXED_LIST}" 1
run_arm memorize64_direct "${DIRECT_MEMORY_CONFIG}" "${MEMORIZE_STEPS}" "${MEMORY_DIRECT_DIR}" "${FIXED_LIST}" 1
MEMORY_TAG="$(printf '%07d' "$((SOURCE_ITERATION + MEMORIZE_STEPS))")"
MEMORY_HARD_CKPT="${MEMORY_HARD_DIR}/iter_${MEMORY_TAG}.pt"
MEMORY_DIRECT_CKPT="${MEMORY_DIRECT_DIR}/iter_${MEMORY_TAG}.pt"
MEMORY_HARD_REPORT="${OUTPUT_ROOT}/memorize64_hard_iter_${MEMORY_TAG}.json"
MEMORY_DIRECT_REPORT="${OUTPUT_ROOT}/memorize64_direct_iter_${MEMORY_TAG}.json"
MEMORY_SUMMARY="${OUTPUT_ROOT}/v7_hard_assignment_memorization_summary.json"
evaluate_report "${HARD_MEMORY_CONFIG}" "${MEMORY_HARD_CKPT}" "${MEMORY_HARD_REPORT}" "${CACHE_ROOT}/memorize_hard_${MEMORY_TAG}" train "${FIXED_LIST}" 0
evaluate_report "${DIRECT_MEMORY_CONFIG}" "${MEMORY_DIRECT_CKPT}" "${MEMORY_DIRECT_REPORT}" "${CACHE_ROOT}/memorize_direct_${MEMORY_TAG}" train "${FIXED_LIST}" 0

"${PYTHON}" - "${MEMORY_HARD_REPORT}" "${MEMORY_DIRECT_REPORT}" "${MEMORY_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

def passed(path):
    report = json.loads(Path(path).read_text())
    method = report["methods"].get("four_slot_refined") or report["methods"]["four_slot_global_unique"]
    slot = report["four_slot_diagnostics"]
    return {
        "f1_050": method["0.50"]["f1"],
        "cardinality_exact": slot["cardinality"]["exact_fraction"],
        "semantic_duplicate": slot["semantic_duplicate_cluster_fraction"],
        "close_pair_fraction_20px": method["0.50"]["selected_curve_diversity"]["close_pair_fraction_below_20px"],
    }

rows = {"hard": passed(sys.argv[1]), "direct": passed(sys.argv[2])}
thresholds = {
    "f1_050_min": 0.90,
    "cardinality_exact_min": 0.90,
    "semantic_duplicate_max": 0.02,
    "close_pair_fraction_20px_max": 0.02,
}
arm_pass = {
    name: (
        row["f1_050"] >= thresholds["f1_050_min"]
        and row["cardinality_exact"] >= thresholds["cardinality_exact_min"]
        and row["semantic_duplicate"] <= thresholds["semantic_duplicate_max"]
        and row["close_pair_fraction_20px"]
        <= thresholds["close_pair_fraction_20px_max"]
    )
    for name, row in rows.items()
}
payload = {
    "experiment": "V7 hard slot-to-GT assignment fixed-64 gate",
    "assignment_mode": "hard_min",
    "candidate_target_within_gt": "soft_cluster",
    "thresholds": thresholds,
    "arms": rows,
    "arm_pass": arm_pass,
    "all_pass": all(arm_pass.values()),
}
Path(sys.argv[3]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print({"memorization_gate": payload})
for name, did_pass in arm_pass.items():
    if not did_pass:
        raise SystemExit(f"{name} fixed-64 memorization gate failed")
PY

if [[ "${RUN_GENERALIZATION}" != "1" ]]; then
  echo "Fixed-64 hard-assignment gate passed; generalization remains paused."
  echo "summary: ${MEMORY_SUMMARY}"
  exit 0
fi

HARD_DIR="${OUTPUT_ROOT}/generalization/hard"
DIRECT_DIR="${OUTPUT_ROOT}/generalization/direct"
run_arm hard_reference "${HARD_CONFIG}" "${TRAIN_STEPS}" "${HARD_DIR}" "" "${GRAD_ACCUM}"
run_arm direct_reference "${DIRECT_CONFIG}" "${TRAIN_STEPS}" "${DIRECT_DIR}" "" "${GRAD_ACCUM}"
FINAL_TAG="$(printf '%07d' "$((SOURCE_ITERATION + TRAIN_STEPS))")"
HARD_FINAL="${HARD_DIR}/iter_${FINAL_TAG}.pt"
DIRECT_FINAL="${DIRECT_DIR}/iter_${FINAL_TAG}.pt"

hard_summary_args=()
direct_summary_args=()
for ((offset=CHECKPOINT_INTERVAL; offset<=TRAIN_STEPS; offset+=CHECKPOINT_INTERVAL)); do
  iteration=$((SOURCE_ITERATION + offset))
  tag="$(printf '%07d' "${iteration}")"
  hard_checkpoint="${HARD_DIR}/iter_${tag}.pt"
  direct_checkpoint="${DIRECT_DIR}/iter_${tag}.pt"
  hard_report="${OUTPUT_ROOT}/reports/hard_iter_${tag}_uniform256.json"
  direct_report="${OUTPUT_ROOT}/reports/direct_iter_${tag}_uniform256.json"
  evaluate_report "${HARD_CONFIG}" "${hard_checkpoint}" "${hard_report}" "${CACHE_ROOT}/hard_${tag}" val "" 64
  evaluate_report "${DIRECT_CONFIG}" "${direct_checkpoint}" "${direct_report}" "${CACHE_ROOT}/direct_${tag}" val "" 64
  hard_summary_args+=(--hard-report "${iteration}=${hard_report}")
  direct_summary_args+=(--direct-report "${iteration}=${direct_report}")
done

HARD_GRADIENT="${OUTPUT_ROOT}/hard_final_gradient_contract.json"
DIRECT_GRADIENT="${OUTPUT_ROOT}/direct_final_gradient_contract.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_reference_gradient_contract \
  --config "${HARD_CONFIG}" \
  --checkpoint "${HARD_FINAL}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch-size 2 \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype none \
  --output-json "${HARD_GRADIENT}"
"${PYTHON}" -u -m dynlaneseq_eg.tools.audit_v7_reference_gradient_contract \
  --config "${DIRECT_CONFIG}" \
  --checkpoint "${DIRECT_FINAL}" \
  --dataset-root "${DATA_ROOT}" \
  --device "${DEVICE}" \
  --batch-size 2 \
  --num-workers "${NUM_WORKERS}" \
  --amp-dtype none \
  --output-json "${DIRECT_GRADIENT}"

SUMMARY="${OUTPUT_ROOT}/v7_hard_assignment_reference_gate_summary.json"
"${PYTHON}" -u -m dynlaneseq_eg.tools.summarize_v7_reference_gate \
  --hard-memorization "${MEMORY_HARD_REPORT}" \
  --direct-memorization "${MEMORY_DIRECT_REPORT}" \
  "${hard_summary_args[@]}" \
  "${direct_summary_args[@]}" \
  --hard-gradient "${HARD_GRADIENT}" \
  --direct-gradient "${DIRECT_GRADIENT}" \
  --output-json "${SUMMARY}"

echo "V7 reference architecture gate complete: ${SUMMARY}"
