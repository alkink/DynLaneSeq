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
CONTRACT="${WORK_DIR}/official_protocol_contract.json"

mkdir -p "${WORK_DIR}"
export PYTHONPATH="${CLRERNET_ROOT}:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON}" \
  "${DYNLANESEQ_ROOT}/external_baselines/clrernet/audit_official_protocol.py" \
  --config "${CONFIG}" \
  --clrernet-root "${CLRERNET_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --nms-patch "${NMS_PATCH}" \
  --mmcv-shim "${MMCV_SHIM}" \
  --output "${CONTRACT}"

cd "${CLRERNET_ROOT}"
exec "${PYTHON}" tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher none
