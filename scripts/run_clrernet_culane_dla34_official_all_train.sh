#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/workspace/.venv-clrernet-official/bin/python}"
DYNLANESEQ_ROOT="${DYNLANESEQ_ROOT:-/workspace/DynLaneSeq}"
CLRERNET_ROOT="${CLRERNET_ROOT:-/workspace/CLRerNet_official}"
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}"
WORK_DIR="${WORK_DIR:-/workspace/CLRerNet_runs/culane_dla34_v030_all_official_train}"
CONFIG="${CONFIG:-${DYNLANESEQ_ROOT}/external_baselines/clrernet/clrernet_culane_dla34_official_all_train.py}"
NMS_PATCH="${NMS_PATCH:-${DYNLANESEQ_ROOT}/external_baselines/clrernet/torch211_cuda128_nms_compat.patch}"
MMCV_SHIM="${MMCV_SHIM:-${DYNLANESEQ_ROOT}/external_baselines/clrernet/mmcv_ext_import_shim.py}"
DLA_PRETRAINED="${DLA_PRETRAINED:-/root/.cache/torch/hub/checkpoints/dla34-ba72cf86.pth}"
CONTRACT="${WORK_DIR}/official_protocol_contract.json"

mkdir -p "${WORK_DIR}"
export PYTHONPATH="${CLRERNET_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:/venv/clrernet/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON}" \
  "${DYNLANESEQ_ROOT}/external_baselines/clrernet/audit_official_protocol.py" \
  --config "${CONFIG}" \
  --clrernet-root "${CLRERNET_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --nms-patch "${NMS_PATCH}" \
  --mmcv-shim "${MMCV_SHIM}" \
  --dla-pretrained "${DLA_PRETRAINED}" \
  --output "${CONTRACT}"

cd "${CLRERNET_ROOT}"
exec "${PYTHON}" tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher none
