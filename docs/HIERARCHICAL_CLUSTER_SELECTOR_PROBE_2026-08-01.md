# Frozen Hierarchical Cluster-Selector Probe

## Question

The 10k decomposition found two separable losses: ranking the correct physical
lane clusters and choosing the most accurate curve inside a selected cluster.
This probe tests whether the existing frozen 10k descriptors already contain
enough information to learn both tasks when their supervision is decoupled.

## Frozen contract

The detector, FPN, row-reference decoder, curve geometry, 32 candidates, and
20-pixel NMS partition are frozen. The NMS partition is used only as a stable
diagnostic grouping; this is not a proposed final NMS-dependent architecture.

Two independent models are trained:

1. `cluster existence`: pool candidate descriptors inside each frozen cluster,
   assign GT lanes one-to-one to clusters in official raster-IoU space, and
   rank clusters;
2. `representative quality`: regress every candidate's best official raster
   IoU and apply pairwise ranking only among candidates in the same cluster.

Unmatched candidates are not automatically assigned a false quality of zero.
Their physical IoU remains the representative target.

## Independent arms

- learned clusters with source representatives isolate cluster ranking;
- source clusters with learned representatives isolate quality ranking;
- learned clusters with learned representatives test the combined hierarchy.

The cluster arm must gain at least 5 recall points at IoU 0.50. The
representative arm must gain at least 3 points at IoU 0.70. The combined arm
must pass both thresholds. A positive combined arm alone is insufficient.

## Interpretation

- all three pass: existing 10k descriptors are sufficient and the current
  single-score supervision/head contract is the bottleneck;
- only cluster passes: lane-level existence/ranking is learnable but the
  descriptors do not expose precise representative quality;
- only representative passes: physical IoU quality is learnable but semantic
  lane-cluster selection is not;
- neither passes: 10k descriptors are immature or missing evidence; repeat on
  a later compatible checkpoint before changing the full architecture.

## Command

```bash
bash scripts/probe_culane_dla34_hierarchical_cluster_selector_10k.sh
```

Output:

```text
outputs/diagnostics/dla34_hierarchical_cluster_selector_10k/
  hierarchical_cluster_selector.json
  hierarchical_cluster_selector.pt
```
