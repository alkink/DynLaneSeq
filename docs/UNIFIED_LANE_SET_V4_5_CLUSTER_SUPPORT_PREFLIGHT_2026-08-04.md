# Unified Lane Set V4.5 cluster-support preflight

This diagnostic is the mandatory zero-training gate before implementing the
V4.5 teacher-only cluster-soft pointer objective. It preserves the frozen V4
geometry and measures the candidate-to-GT quality tensor `Q[GT, candidate]`
without reducing it to one hard Hungarian candidate identity.

The audit answers four contract questions:

1. How many GT lanes are jointly representable at strict IoU `> 0.50` under a
   one-to-one, maximum-four-candidate assignment?
2. Does ordinary maximum-IoU Hungarian matching lose match cardinality compared
   with a lexicographic cardinality-first assignment?
3. For each `(support_min, quality_delta, temperature)` setting, how soft and
   how high-quality is the representative target distribution?
4. Do two GT clusters share candidate support, making independent categorical
   teacher sampling unsafe?
5. Which row-strip representability threshold best retains the official CULane
   IoU `0.50` candidate oracle without teaching avoidable false positives?

The runner scans row-strip representability thresholds from `0.00` through
`0.50`, ties each cluster support floor to the scanned threshold, and evaluates
the resulting ideal teacher set with official raster IoU at `0.50` and `0.75`.
This calibration is necessary because a numeric `0.50` in the differentiable
row-strip surrogate is not assumed to equal an official raster IoU of `0.50`.

Run:

```bash
DATA_ROOT=/workspace/CULane \
bash scripts/analyze_culane_dla34_unified_lane_set_v4_5_cluster_support.sh
```

Default output:

```text
outputs/diagnostics/unified_lane_set_v4_5_cluster_support_preflight.json
```

This command does not train. It reuses its candidate cache when present;
otherwise it performs one frozen V4-50k inference pass over the uniform-256
diagnostic subset. Official raster IoU is computed on CPU and stored back into
the same cache.

Do not launch V4.5 training merely because the report contains a mechanically
eligible row. First inspect joint-support coverage, sub-threshold target mass,
and cross-GT support collision. If cross-GT overlap is non-zero, the discrete
teacher state must be sampled with a collision-safe one-to-one procedure rather
than independent per-cluster categorical samples.
