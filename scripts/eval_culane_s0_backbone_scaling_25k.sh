#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Evaluate ResNet18/ResNet101 backbone-scaling checkpoints at fixed intervals.
#
# Defaults are intentionally paper-safe:
#   - SPLIT=val, not test.
#   - score=0.30 and quality_power=0.50, matching the selected protocol.
#   - missing checkpoints are skipped so this can be re-run while training is ongoing.
#
# Examples:
#   bash scripts/eval_culane_s0_backbone_scaling_25k.sh
#   MODELS="res18" ITERS="25000 50000" bash scripts/eval_culane_s0_backbone_scaling_25k.sh
#   SPLIT=test CATEGORIES=--categories bash scripts/eval_culane_s0_backbone_scaling_25k.sh

SPLIT="${SPLIT:-val}"
MODELS="${MODELS:-res18 res101}"
ITERS="${ITERS:-25000 50000 75000 100000 125000 150000 175000 200000 225000 250000 275000}"

SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.5}"
CATEGORIES="${CATEGORIES:-}"
DEVICE="${DEVICE:-cuda}"

# If 1, do not re-run checkpoints whose metrics.txt already exists.
SKIP_DONE="${SKIP_DONE:-1}"

# If 1, missing checkpoints are treated as errors. Default is to skip them.
STRICT="${STRICT:-0}"

SUMMARY_DIR="${SUMMARY_DIR:-outputs/backbone_scaling_eval_summaries}"
SUMMARY_FILE="${SUMMARY_FILE:-${SUMMARY_DIR}/${SPLIT}_summary.tsv}"
mkdir -p "${SUMMARY_DIR}"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf "model\titer\tsplit\tscore_thresh\tquality_power\tf1_50\tf1_70\tmetrics\n" > "${SUMMARY_FILE}"
fi

score_tag="${SCORE_THRESH/./p}"
quality_tag="${QUALITY_POWER/./p}"
nms_tag="${NMS_DISTANCE_THRESH_PX/./p}"

f1_from_metrics() {
  local metrics_file="$1"
  local iou_label="$2"
  python - "$metrics_file" "$iou_label" <<'PY'
import re
import sys

path, label = sys.argv[1], sys.argv[2]
pattern = re.compile(rf"^IoU {re.escape(label)}: .*?F1=([0-9.]+)")
try:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line.strip())
            if m:
                print(m.group(1))
                sys.exit(0)
except FileNotFoundError:
    pass
print("")
PY
}

eval_one() {
  local model="$1"
  local iter="$2"
  local out_dir=""
  local eval_script=""

  case "${model}" in
    res18)
      out_dir="outputs/culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep"
      eval_script="scripts/eval_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_${SPLIT}.sh"
      ;;
    res101)
      out_dir="outputs/culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep"
      eval_script="scripts/eval_culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_${SPLIT}.sh"
      ;;
    *)
      echo "Unknown model '${model}'. Use MODELS=\"res18 res101\"." >&2
      return 2
      ;;
  esac

  if [[ ! -x "${eval_script}" ]]; then
    echo "Missing eval script: ${eval_script}" >&2
    return 2
  fi

  local iter_tag
  iter_tag="$(printf "%07d" "${iter}")"
  local ckpt="${out_dir}/iter_${iter_tag}.pt"
  local pred_dir="${out_dir}/${SPLIT}_eval_iter_${iter_tag}_thr${score_tag}_q${quality_tag}_nms${nms_tag}"
  local metrics_file="${pred_dir}/metrics.txt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] ${model} iter_${iter_tag}: checkpoint not found: ${ckpt}"
    if [[ "${STRICT}" == "1" ]]; then
      return 1
    fi
    return 0
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${metrics_file}" ]]; then
    echo "[done] ${model} iter_${iter_tag}: ${metrics_file}"
  else
    echo "============================================================"
    echo "model: ${model}"
    echo "split: ${SPLIT}"
    echo "checkpoint: ${ckpt}"
    echo "score_thresh: ${SCORE_THRESH}"
    echo "quality_power: ${QUALITY_POWER}"
    echo "iou_thresholds: ${IOU_THRESHOLDS}"
    echo "============================================================"
    CKPT="${ckpt}" \
    DEVICE="${DEVICE}" \
    SCORE_THRESH="${SCORE_THRESH}" \
    QUALITY_POWER="${QUALITY_POWER}" \
    TOP_K="${TOP_K}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX}" \
    NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS}" \
    IOU_THRESHOLDS="${IOU_THRESHOLDS}" \
    CATEGORIES="${CATEGORIES}" \
    bash "${eval_script}"
  fi

  if [[ -f "${metrics_file}" ]]; then
    local f1_50 f1_70
    f1_50="$(f1_from_metrics "${metrics_file}" "0.50")"
    f1_70="$(f1_from_metrics "${metrics_file}" "0.70")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${model}" "${iter_tag}" "${SPLIT}" "${SCORE_THRESH}" "${QUALITY_POWER}" "${f1_50}" "${f1_70}" "${metrics_file}" \
      >> "${SUMMARY_FILE}"
  fi
}

for model in ${MODELS}; do
  for iter in ${ITERS}; do
    eval_one "${model}" "${iter}"
  done
done

echo "summary: ${SUMMARY_FILE}"
