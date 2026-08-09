# V7 training performance root-cause audit

## Scope

This audit compares commit `3801799` with an optimization-only implementation
of the same V7 objective.  The training configuration, parameterization,
effective batch size, losses, matcher weights, learning rates, scheduler,
postprocess, and checkpoint contract are unchanged.

The reference protocol is:

```text
GPU                 RTX 3090 24 GiB
input               1600 x 640
physical batch      4
gradient accumulation 4
effective batch     16
AMP                 bfloat16
channels_last       true
workers             8
```

## Root cause

V7 is neither input-pipeline-bound nor primarily limited by available VRAM.
It is a launch-heavy eager training graph with avoidable synchronization and
repeated tensor preparation around an already expensive multi-head objective.

One optimizer step contains four microbatches.  Every microbatch executes:

- DLA-34 and FPN on a large image;
- four row-reference decoder layers over 32 candidates and 72 rows;
- four lane-state layers;
- four protected ownership layers;
- proposal encoding, four-slot decoding, and bounded slot refinement;
- final plus three intermediate Hungarian assignments;
- final plus three intermediate point/range/LineIoU/DFL objectives.

Most individual Transformer and loss tensors are small.  Before the changes,
one profiled optimizer step issued about 70k CUDA kernel launches.  A faster GPU
shortens the kernels but does not proportionally shorten Python dispatch,
allocator bookkeeping, CPU matching, or stream synchronization.  This is why
a high-compute/high-VRAM GPU can remain underutilized and scale much less than
its nominal FLOP ratio suggests.  More VRAM alone cannot fix this regime.

The concrete avoidable costs were:

1. The same projected P2 tensor was converted from BF16 BHWC to FP32 NCHW
   independently in all four reference layers.
2. Final and three auxiliary matcher costs were constructed with repeated
   small per-layer/per-image graphs and CPU/GPU index transport.
3. Point, range, LineIoU, and DFL each independently gathered the same matched
   lanes.
4. Four-slot target construction and permutation evaluation used many small
   candidate/GT paths instead of a padded batch tensor.
5. Structured routing ran an image loop and performed synchronization-prone
   validity handling.
6. Fixed row grids, sample indices, position encodings, and route combinations
   were rebuilt every forward.
7. Audit-only bounded-delta reductions ran during every training forward.

After the changes, the profiler snapshot contains about 36.9k launches, a
reduction of roughly 48%.  A dispatch-level audit reports no repository-owned
`_local_scalar_dense` or `nonzero` call on a CUDA tensor; the remaining calls
inside the matcher operate on the already transferred CPU cost batch.

## F1-neutral implementation changes

### Matcher and geometry losses

- `HungarianMatcherS0.compute_cost_many` constructs all compatible decoder
  layer costs in one padded GPU graph.  Mixed-size auxiliary query groups are
  grouped by tensor shape, so older hybrid contracts remain supported.
- The cost tensors cross to CPU once, the exact same SciPy Hungarian solve is
  used, and all integer pairs return to the GPU in one packed transfer.
- Range-aware row-strip IoU has an exact batched implementation.
- `_MatchedLaneBatch` gathers each layer's matched prediction/GT geometry once;
  point, range, LineIoU, and DFL reuse it.
- The `all_gt <= 4` four-slot target path is vectorized and does not run an
  unnecessary Hungarian solve.
- Four-slot hard/marginal permutation costs are evaluated over cached path
  tensors rather than scalar Python loops.

### Model forward

- Fixed row/index/sample tensors and sine position encodings are cached by
  device, dtype, and shape.
- The P2 FP32 NCHW grid-sample input is materialized once.  A custom autograd
  alias casts each consumer gradient back to the source dtype independently,
  preserving the historical AMP accumulation contract.
- Real-route combinations are registered once as a non-persistent buffer.
- Collision diagnostics and structured Sinkhorn routing are batched.
- Bounded-delta max/mean reductions are produced only for `inference_only`
  contract audits, where they are consumed.

No approximation, reduced layer count, disabled deep supervision, smaller
image, altered batch, NMS, threshold, or changed precision mode is used.

## Measured result

Stable 15-step measurements on the same local GPU:

| Stage | clean `3801799` | optimized | change |
|---|---:|---:|---:|
| Forward + final matching | 0.9620 s | 0.6682 s | -30.5% |
| Loss + auxiliary matching | 0.4323 s | 0.1466 s | -66.1% |
| Backward | 1.6652 s | 1.4103 s | -15.3% |
| Total optimizer step | 3.1153 s | 2.2909 s | -26.5% |
| Throughput | 5.14 img/s | 6.98 img/s | +35.8% |
| Peak allocated memory | 12.10 GiB | 11.43 GiB | -0.67 GiB |

Data wait is approximately 0.002 seconds per optimizer step, confirming that
worker count/storage is not the bottleneck.

An instrumented three-step run reports the following largest forward/loss
regions (inclusive CUDA time per optimizer step):

| Region | CUDA time |
|---|---:|
| DLA backbone | 88.8 ms |
| intermediate losses | 66.0 ms |
| four reference layers | 57.8-59.3 ms each |
| image-conditioned initial reference | 55.6 ms |
| row feature construction | 52.8 ms |
| FPN | 42.3 ms |
| slot refinement | 37.4 ms |
| four-slot geometry loss | 33.7 ms |
| DFL (20 calls/step) | 24.3 ms |
| four-slot selection loss | 21.0 ms |
| LineIoU (20 calls/step) | 20.7 ms |
| all decoder-layer matching | 18.3 ms |

The remaining time is now predominantly real architecture/backward work, not
one hidden data or synchronization defect.  Further large gains would require
kernel fusion/compilation, fewer supervised layers, a different attention
operator, or a changed physical batch.  Those options are not F1-neutral and
are deliberately excluded from this patch.

## Numerical and regression safety

The clean and optimized repositories were initialized with the same seed and
ran the same saved augmented image with the same dropout RNG state:

```text
selected deploy/geometry outputs: 12 / 12 bit-identical
scalar loss terms:                 68 / 68 bit-identical
Hungarian assignments:             exact
gradient global cosine:             0.999998853
gradient relative L2 difference:    0.001515
```

The tiny gradient difference is floating-point reduction ordering from batching
mathematically identical sums.  It does not change the loss function.  Exact
bitwise checkpoint trajectories would require retaining the original scalar
reduction order and therefore giving up much of the speedup.

The complete CPU regression suite passes:

```text
520 passed
```

`test_v7_performance_parity.py` additionally checks cached constants, position
encoding, shared P2 conversion and BF16 gradient behavior, batched Sinkhorn,
batched row-strip IoU, vectorized slot paths, cached route decoding, and batched
matcher assignments against their original implementations.

