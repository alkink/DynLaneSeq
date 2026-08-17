#!/usr/bin/env bash
set -euo pipefail

project=/workspace/DynLaneSeq_v25
python=/venv/clrernet/bin/python
root=${project}/outputs/diagnostics/v25_s1_oof_immutable_selector
folds=${root}/folds
manifest=${folds}/oof_fold_manifest.json

cd "${project}"
export PYTHONPATH="${project}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DYNLANESEQ_DLA34_WEIGHTS=/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth

mkdir -p "${root}"
if [[ ! -f "${manifest}" ]]; then
  "${python}" -m dynlaneseq_eg.tools.build_v25_s1_oof_folds \
    --dataset-root /workspace/CULane \
    --output-dir "${folds}" \
    --folds 2 \
    --seed 3407
fi

for fold in 0 1; do
  output=${root}/fold_${fold}/g0_train
  endpoint=${output}/v25_g0_endpoint.pt
  resume=${output}/resume_latest.pt
  if [[ -f "${endpoint}" ]]; then
    echo "V25-S1 fold ${fold} G0 endpoint already exists"
    continue
  fi
  args=(
    -m dynlaneseq_eg.tools.train_v25_image_mediated_lane_objects
    --config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml
    --dataset-root /workspace/CULane
    --output-dir "${output}"
    --device cuda
    --num-workers 2
    --mode gate
    --log-interval 25
    --resume-interval 500
    --oof-fold-manifest "${manifest}"
    --oof-fold-index "${fold}"
    --train-list "${folds}/fold_${fold}_train.txt"
  )
  if [[ -f "${resume}" ]]; then
    args+=(--resume "${resume}")
  fi
  "${python}" "${args[@]}"
done

# The remaining cache/selector/evaluation stages are appended by the same
# branch before the first fold finishes. Until then this marker makes a stale
# first revision fail closed instead of silently claiming completion.
if ! "${python}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("dynlaneseq_eg.tools.run_v25_s1_selector_pipeline") else 1)'; then
  echo "V25-S1 selector pipeline module is not installed yet; refusing partial success" >&2
  exit 75
fi

exec "${python}" -m dynlaneseq_eg.tools.run_v25_s1_selector_pipeline \
  --dataset-root /workspace/CULane \
  --output-dir "${root}" \
  --fold-manifest "${manifest}" \
  --v7-config dynlaneseq_eg/configs/culane_v23_ordered_slot_cost_volume_gate.yaml \
  --v7-checkpoint /workspace/DynLaneSeq/outputs/diagnostics/unified_lane_set_v7_resume_safe_data_gate_125k/seed_3407/resume_safe/iter_0225000.pt \
  --g0-config dynlaneseq_eg/configs/culane_v25_g0_direct_four_lane_objects_1ep.yaml \
  --full-g0-checkpoint /workspace/DynLaneSeq_v25/outputs/diagnostics/v25_g0_direct_four_lane_objects_1ep/v25_g0_endpoint.pt \
  --wrong-image-list /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.txt \
  --wrong-image-report /workspace/DynLaneSeq/outputs/diagnostics/v22_lane_field_stage_a_official/controls/official_val_cross_clip_wrong.json
