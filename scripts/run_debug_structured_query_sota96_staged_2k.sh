#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
STAGES="${STAGES:-s0 s0_continue s1 s2 s3}"

run_stage() {
  local stage="$1"
  case "${stage}" in
    s0)
      python -m dynlaneseq_eg.tools.train \
        --config dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k.yaml \
        --device "${DEVICE}"
      ;;
    s0_continue)
      python -m dynlaneseq_eg.tools.train \
        --config dynlaneseq_eg/configs/debug/culane_s0_structured_query_sota96_2k_continue_12k.yaml \
        --device "${DEVICE}" \
        --init-from outputs/debug_s0_structured_query_sota96_2k/last.pt
      ;;
    s1)
      python -m dynlaneseq_eg.tools.train \
        --config dynlaneseq_eg/configs/debug/culane_s1_residual_structured_query_sota96_2k_init_structured.yaml \
        --device "${DEVICE}" \
        --init-from outputs/debug_s0_structured_query_sota96_2k_continue_12k/last.pt
      ;;
    s2)
      python -m dynlaneseq_eg.tools.train \
        --config dynlaneseq_eg/configs/debug/culane_s2_residual_structured_query_sota96_2k_from_s1.yaml \
        --device "${DEVICE}" \
        --init-from outputs/debug_s1_residual_structured_query_sota96_2k_init_structured/last.pt
      ;;
    s3)
      python -m dynlaneseq_eg.tools.train \
        --config dynlaneseq_eg/configs/debug/culane_s3_active_corridor_qualitycal_structured_query_sota96_2k_from_s2.yaml \
        --device "${DEVICE}" \
        --init-from outputs/debug_s2_residual_structured_query_sota96_2k_from_s1/last.pt
      ;;
    *)
      echo "Unknown stage '${stage}'. Use one of: s0 s0_continue s1 s2 s3" >&2
      exit 1
      ;;
  esac
}

for stage in ${STAGES}; do
  echo
  echo "== running SOTA96 stage: ${stage} =="
  run_stage "${stage}"
done
