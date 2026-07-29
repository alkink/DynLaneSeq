#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_NAME="$(
  nvidia-smi --id=0 --query-gpu=name --format=csv,noheader 2>/dev/null \
    | tr -d '\r'
)"
if [[ "${GPU_NAME}" != *"RTX 5090"* ]]; then
  echo "Warning: this launcher is tuned for RTX 5090; detected: ${GPU_NAME:-unknown}" >&2
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

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_lanerownet_5090}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Deliberately retain batch 4 x accumulation 4.  RTX 5090 has enough memory
# for larger micro-batches, but changing that split also changes auxiliary
# BatchNorm statistics.  The identical setting is the cleanest one-run
# scientific comparison and should still exploit the faster GPU.
DATA_ROOT="${DATA_ROOT:-/workspace/CULane}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACCUM="${GRAD_ACCUM:-4}" \
bash scripts/run_culane_dla34_row_reference_full_278k.sh
