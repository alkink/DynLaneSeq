# LaneRowNet Training Performance Audit

## Scope

This audit targets the CULane DLA-34 balanced-detail-neck experiment at
1600x640, 32 lane instances, 160 row states, 800 localization bins, and four
structured decoder layers.  Measurements below use real CULane batches rather
than synthetic input.

## Audited machine

- GPU: NVIDIA GeForce RTX 3090, 24 GB
- CPU: AMD Ryzen 5 5600 (6 cores / 12 threads)
- PyTorch: 2.1.0+cu121
- cuDNN: 8.9.0
- Native CUDA architecture: `sm_86`
- Main autocast: FP16
- Segmentation auxiliary branch: BF16
- Channels-last: enabled

## Root causes

1. The structured decoder is the dominant cost.  Each of four layers performs
   row-local cross-attention over 160 rows, 32 lane states, and 400 horizontal
   evidence tokens.  Backward is consequently much more expensive than the
   neck.
2. PyTorch 2.1's default SDPA heuristic selects a slower kernel for the
   row-local attention shape on RTX 3090.  Flash-preferred SDPA is faster.
3. The criterion performed repeated CUDA-to-CPU synchronizations inside
   per-lane loops (`Tensor.item()` / `Tensor.any()`), serializing the stream.
4. The structured S0 training path still constructed the legacy holistic
   positional memory even though neither structured predictions nor losses
   consumed it.
5. Expected-X decoding and DFL independently normalized the same 800-bin
   distribution.
6. Stopped evaluation jobs can retain worker processes, host RAM, and a CUDA
   context.  `Ctrl-Z` suspends a job; it does not terminate it.
7. An RTX 50-series GPU used with PyTorch 2.1 + CUDA 12.1 has no native
   Blackwell (`sm_120`) kernels.  GPU model comparisons are invalid unless the
   software stacks are compatible.

## Measured phase breakdown after eager-path fixes

Batch size 4, Flash-preferred SDPA, real CULane data:

| Phase | Time per micro-step | Share |
|---|---:|---:|
| Data wait | 0.49 ms | 0.1% |
| Host-to-device | 6.02 ms | 1.5% |
| Forward | 112.32 ms | 28.3% |
| Matcher | 13.98 ms | 3.5% |
| Criterion | 29.97 ms | 7.6% |
| Backward | 227.86 ms | 57.5% |
| Optimizer | 5.94 ms | 1.5% |
| Total | 396.60 ms | 100% |

This rules out the dataloader, CPU, disk, and the new neck as the primary
bottleneck.

Under the same eager, Flash-preferred, synchronized diagnostic, the original
SimpleFPN takes 342.19 ms/micro-step and BalancedDetailFPN takes 396.60 ms.
The richer neck therefore adds about 15.9% training time.  This cost is real,
but it cannot explain the previously observed multi-fold GPU difference.

## End-to-end measurements

| Runtime | Micro-batch | Accumulation | Effective batch | Throughput | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| Eager + default SDPA | 8 | 2 | 16 | 9.9 img/s | 19.14 GB |
| Eager + Flash-preferred | 8 | 2 | 16 | 10.6 img/s | 19.13 GB |
| Compile-default + Flash-preferred | 8 | 2 | 16 | 13.8 img/s | 16.97 GB |
| Compile-default + default SDPA | 8 | 2 | 16 | **14.4 img/s** | **15.06 GB** |

The final row preserves the paper experiment's original 8x2 schedule.  The
first compiled step has a one-time compilation pause and must not be included
in steady-state throughput.

## Attention microbenchmark

For the actual row-local attention shape
`batch_rows=640, queries=32, keys=400, dim=256, heads=8`:

| PyTorch 2.1 backend | Forward + backward |
|---|---:|
| Default heuristic | 14.193 ms |
| Flash | **10.066 ms** |
| Memory-efficient | 13.583 ms |
| Math | 20.087 ms |

Run `python scripts/benchmark_attention_backends.py` on each new GPU/software
stack before forcing a backend.  Modern PyTorch versions may choose a better
default than PyTorch 2.1.

## Recommended commands

RTX 3090:

```bash
DATA_ROOT=/home/alki/projects/CULane \
bash scripts/run_culane_s0_structured_query_dla34_balanced_detail_fpn_fast.sh
```

Resume:

```bash
DATA_ROOT=/home/alki/projects/CULane \
RESUME=outputs/culane_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_balanced_detail_fpn256_l4_dfl_50ep/last.pt \
bash scripts/resume_culane_s0_structured_query_dla34_balanced_detail_fpn_fast.sh
```

For RTX 5070/5080/5090, first use PyTorch 2.7+ with a CUDA 12.8 wheel (or a
newer official Blackwell-capable wheel).  The training entry point prints the
GPU capability and wheel architecture list and warns on a native-architecture
mismatch.  Benchmark the default and Flash-preferred attention paths on that
stack rather than assuming the RTX 3090 result transfers unchanged.
