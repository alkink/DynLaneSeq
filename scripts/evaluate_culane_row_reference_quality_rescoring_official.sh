#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
CONFIG="${CONFIG:-dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_from55k_evidence1e5_to75k.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/diagnostics/dla34_rowref_from55k_evidence1e5_to75k/iter_0065000.pt}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-outputs/diagnostics/row_reference_quality_rescoring_65k.pt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/diagnostics/row_reference_quality_rescoring_65k_uniform256_official.json}"

for path in "$CONFIG" "$CHECKPOINT" "$PROBE_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python -u -m dynlaneseq_eg.tools.evaluate_quality_rescoring_official \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --probe-checkpoint "$PROBE_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-max-batches "$EVAL_MAX_BATCHES" \
  --num-workers "$NUM_WORKERS" \
  --sample-strategy uniform \
  --amp-dtype "$AMP_DTYPE" \
  --line-width 30 \
  --top-k 4 \
  --quality-power 0.5 \
  --nms-distance 20 \
  --nms-min-overlap-points 5 \
  --min-valid-rows 5 \
  --output-json "$OUTPUT_JSON"

echo
echo "official quality-rescoring report: $OUTPUT_JSON"
