#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_NAME="$(
  nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2>/dev/null \
    | tr -d '\r'
)"
if [[ "${GPU_NAME}" != *"RTX 5070 Ti"* ]]; then
  echo "Warning: this launcher is tuned for RTX 5070 Ti; detected: ${GPU_NAME:-unknown}" >&2
fi

PYTHON="${PYTHON:-python}"
"${PYTHON}" -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
capability = torch.cuda.get_device_capability()
required = f"sm_{capability[0]}{capability[1]}"
arches = torch.cuda.get_arch_list()
print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "compute_capability": capability,
    "compiled_cuda_arches": arches,
    "native_arch_available": required in arches,
})
if required not in arches:
    raise SystemExit(
        f"PyTorch lacks native {required} kernels; install a CUDA build that supports this GPU."
    )
'

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_lanerownet_5070ti}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# This is the validated, memory-safe setting measured at 20.65 img/s.  Keep
# the exact same micro-batch split on both GPUs so hardware does not introduce
# an additional BatchNorm/gradient-accumulation variable.
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
bash scripts/run_culane_dla34_row_reference_full_278k.sh
