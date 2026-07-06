#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-val}"
EVAL_LIST="${EVAL_LIST:-dataset/list/val.txt}"
CACHE_DIR="${CACHE_DIR:-outputs/diagnostic_cache/paper_val_sweep}"
OUT_ROOT="${OUT_ROOT:-outputs/paper_val_sweeps}"
STAGE="${STAGE:-main}"
TOP_K="${TOP_K:-4}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
SCORE_THRESHOLDS="${SCORE_THRESHOLDS:-0.30 0.35 0.40 0.45 0.50 0.55 0.60}"
QUALITY_POWERS="${QUALITY_POWERS:-0.25 0.50 0.75}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95}"
SELECTION_IOU="${SELECTION_IOU:-0.50}"
TIE_BREAK_IOU="${TIE_BREAK_IOU:-0.70}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-0}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-0}"
ONLY_CASES="${ONLY_CASES:-}"

STRUCTURED_CONFIG="${STRUCTURED_CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"
UNSTRUCTURED_CONFIG="${UNSTRUCTURED_CONFIG:-dynlaneseq_eg/configs/culane_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml}"

STRUCTURED_DIR="${STRUCTURED_DIR:-outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep}"
UNSTRUCTURED_DIR="${UNSTRUCTURED_DIR:-outputs/culane_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep}"

mkdir -p "${OUT_ROOT}"

run_case() {
  local name="$1"
  local config="$2"
  local ckpt="$3"
  local out_json="${OUT_ROOT}/${name}_val_sweep.json"
  local out_txt="${OUT_ROOT}/${name}_val_sweep.txt"

  if [[ -n "${ONLY_CASES}" && " ${ONLY_CASES} " != *" ${name} "* ]]; then
    echo "skip case by ONLY_CASES: ${name}" >&2
    return 0
  fi

  if [[ ! -f "${ckpt}" ]]; then
    if [[ "${REQUIRE_ALL}" == "1" ]]; then
      echo "missing checkpoint: ${ckpt}" >&2
      exit 1
    fi
    echo "skip missing checkpoint: ${ckpt}" >&2
    return 0
  fi

  echo "============================================================"
  echo "case: ${name}"
  echo "config: ${config}"
  echo "checkpoint: ${ckpt}"
  echo "output: ${out_json}"
  echo "============================================================"

  "${PYTHON_BIN}" -u -m dynlaneseq_eg.tools.sweep_cached_culane_thresholds \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --split "${SPLIT}" \
    --list-path "${EVAL_LIST}" \
    --device "${DEVICE}" \
    --cache-dir "${CACHE_DIR}" \
    --reuse-cache \
    --max-batches "${MAX_BATCHES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --cache-num-workers "${CACHE_NUM_WORKERS}" \
    --stage "${STAGE}" \
    --top-k "${TOP_K}" \
    --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
    --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
    --score-thresholds ${SCORE_THRESHOLDS} \
    --quality-powers ${QUALITY_POWERS} \
    --iou-thresholds ${IOU_THRESHOLDS} \
    --selection-iou "${SELECTION_IOU}" \
    --tie-break-iou "${TIE_BREAK_IOU}" \
    --output-json "${out_json}" \
    --output-txt "${out_txt}"
}

run_case "structured_iter_0225000" \
  "${STRUCTURED_CONFIG}" \
  "${STRUCTURED_DIR}/iter_0225000.pt"

run_case "unstructured_iter_0225000" \
  "${UNSTRUCTURED_CONFIG}" \
  "${UNSTRUCTURED_DIR}/iter_0225000.pt"

run_case "unstructured_iter_0250000" \
  "${UNSTRUCTURED_CONFIG}" \
  "${UNSTRUCTURED_DIR}/iter_0250000.pt"

run_case "unstructured_iter_0200000" \
  "${UNSTRUCTURED_CONFIG}" \
  "${UNSTRUCTURED_DIR}/iter_0200000.pt"

run_case "unstructured_iter_0175000" \
  "${UNSTRUCTURED_CONFIG}" \
  "${UNSTRUCTURED_DIR}/iter_0175000.pt"
