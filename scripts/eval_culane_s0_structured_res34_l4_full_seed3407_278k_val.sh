#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_full_seed3407_278k.yaml}"
CKPT="${CKPT:-/home/alki/projects/DynLaneSeq/outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_full_seed3407_278k/iter_0225000.pt}"
DEVICE="${DEVICE:-cuda}"
SCORE_THRESH="${SCORE_THRESH:-0.30}"
QUALITY_POWER="${QUALITY_POWER:-0.50}"
TOP_K="${TOP_K:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:--1}"
EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:--1}"
METRIC_WORKERS="${METRIC_WORKERS:-0}"
METRIC_CHUNKSIZE="${METRIC_CHUNKSIZE:-64}"
AMP_DTYPE="${AMP_DTYPE:-none}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
REUSE_PREDICTIONS="${REUSE_PREDICTIONS:-0}"
NMS_DISTANCE_THRESH_PX="${NMS_DISTANCE_THRESH_PX:-20.0}"
NMS_MIN_OVERLAP_POINTS="${NMS_MIN_OVERLAP_POINTS:-5}"
IOU_THRESHOLDS="${IOU_THRESHOLDS:-0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

read -r CKPT_ITER CKPT_INTRA CKPT_SEED < <(
  python -c 'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu"); c=p.get("cfg", {}); s=c.get("model", {}).get("structured_query", {}); print(int(p.get("iteration", 0)), s.get("use_intra_attention", True), c.get("seed"))' "${CKPT}"
)
if [[ "${CKPT_INTRA}" != "True" || "${CKPT_SEED}" != "3407" ]]; then
  echo "Refusing checkpoint: iteration=${CKPT_ITER}, use_intra=${CKPT_INTRA}, seed=${CKPT_SEED}; expected full seed3407." >&2
  exit 1
fi

read -r -a IOU_ARGS <<< "${IOU_THRESHOLDS}"
CKPT_TAG="$(basename "${CKPT%.pt}")"
CKPT_DIR="$(dirname "${CKPT}")"
SCORE_TAG="${SCORE_THRESH/./p}"
QUALITY_TAG="${QUALITY_POWER/./p}"
NMS_TAG="${NMS_DISTANCE_THRESH_PX/./p}"
MODE_TAG=""
if [[ "${EVAL_BATCH_SIZE}" != "8" ]]; then
  MODE_TAG+="_b${EVAL_BATCH_SIZE}"
fi
if [[ "${AMP_DTYPE}" != "none" ]]; then
  MODE_TAG+="_amp${AMP_DTYPE}"
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  MODE_TAG+="_compile"
fi
PRED_DIR="${PRED_DIR:-${CKPT_DIR}/val_eval_${CKPT_TAG}_thr${SCORE_TAG}_q${QUALITY_TAG}_nms${NMS_TAG}${MODE_TAG}}"
LOG_FILE="${LOG_FILE:-${PRED_DIR}/eval.log}"
RESULT_TXT="${RESULT_TXT:-${PRED_DIR}/metrics.txt}"
RESULT_JSON="${RESULT_JSON:-${PRED_DIR}/metrics.json}"

mkdir -p "${PRED_DIR}"

EXTRA_ARGS=(--no-pretrained-init --amp-dtype "${AMP_DTYPE}")
if [[ "${COMPILE_MODEL}" == "1" ]]; then
  EXTRA_ARGS+=(--compile-model)
fi
if [[ "${REUSE_PREDICTIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-write)
fi

echo "checkpoint iteration: ${CKPT_ITER}"
echo "use_intra_attention: ${CKPT_INTRA}"
echo "seed: ${CKPT_SEED}"
echo "selection split: val"

python -m dynlaneseq_eg.tools.evaluate_culane \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --split val \
  --device "${DEVICE}" \
  --score-thresh "${SCORE_THRESH}" \
  --quality-score-power "${QUALITY_POWER}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --eval-num-workers "${EVAL_NUM_WORKERS}" \
  --eval-prefetch-factor "${EVAL_PREFETCH_FACTOR}" \
  --metric-workers "${METRIC_WORKERS}" \
  --metric-chunksize "${METRIC_CHUNKSIZE}" \
  --top-k "${TOP_K}" \
  --nms-distance-thresh-px "${NMS_DISTANCE_THRESH_PX}" \
  --nms-min-overlap-points "${NMS_MIN_OVERLAP_POINTS}" \
  --iou-thresholds "${IOU_ARGS[@]}" \
  --pred-dir "${PRED_DIR}" \
  --output-txt "${RESULT_TXT}" \
  --output-json "${RESULT_JSON}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
