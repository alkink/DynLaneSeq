# Bottom-aware proposal clustering confirmatory contract

Date: 2026-08-13  
Scope: V7 225k, training-free diagnostic only

## Question

Can the 32 frozen V7 proposal curves be grouped into physical lanes using
only their geometry, with special emphasis on the lower road, and can this
grouping improve the deployed same-count four-lane output without training?

This audit is not V16 and does not modify a checkpoint. It performs no
optimizer step, backward pass, model training, threshold/NMS search, full
validation, or test evaluation.

## GT-free pair geometry

For every valid proposal pair, use only predicted x/range geometry:

- common visible-row count and overlap fraction;
- full-overlap x gap with weight `0.10 + 0.90*y^3`;
- lower-road median and q90 x gap;
- top and bottom endpoint x gaps;
- predicted visible-range start/end gaps;
- lower-minus-upper divergence;
- a ridge quadratic fit evaluated only inside the observed common range.

The lower-road guards are deliberate: two different physical lanes may
converge near the horizon while being widely separated near the vehicle.
Upper-only overlap is never enough to merge a pair.

## Deterministic clusters

Use complete-link agglomeration, not connected components. Two clusters merge
only when every cross-pair passes the fixed policy. This prevents an ambiguous
middle proposal from chaining two physical lanes together.

Three policies are reported, but the confirmatory primary is fixed before
results:

```text
policy:       perspective_balanced_48
prototype:    medoid
selection:    slot_cluster_mass_unique
```

The medoid is an actual proposal curve and therefore cannot create a
row-wise Frankenstein average. Coordinate median and arithmetic mean remain
diagnostic ablations only.

## Cluster-level slot selection

For each V7 writer-valid active slot, normalize the real proposal logits and
sum probability inside every GT-free geometry cluster:

```text
cluster_mass[slot, cluster] = sum(route_probability[slot, proposal])
```

Assign active slots to distinct clusters with a deterministic linear
assignment maximizing total log cluster mass. Emit the selected cluster
medoids. The prediction count must equal the exact current V7 writer count on
every image.

This tests the concrete hypothesis that V7 often knows the correct physical
lane distributionally but loses because probability is split across several
near-duplicate proposal IDs.

## GT use

GT is forbidden in clustering, prototype construction, cluster scoring, slot
assignment, and emitted geometry. GT is used only after predictions are fixed
for:

- pair purity/recall and fragmentation diagnostics;
- official CULane IoU/F1;
- same-count cluster-prototype and all-32 proposal oracles.

## Domains

- Calibration: fixed uniform 256 images from the training split.
- Confirmation: fixed uniform 256 images from validation.
- Test remains closed.

No policy or prototype may be selected from validation after results. The
other fixed combinations are explicitly post-hoc diagnostics and cannot turn
a failed primary into PASS.

## Predeclared gates

Structural gates on both domains:

```text
pair merge precision                    >= 0.98
same-GT pair recall                     >= 0.50
catastrophic multi-GT cluster fraction  <= 0.02
candidates in multi-member clusters     >= 0.50
medoid same-count oracle TP retention   >= 0.95 on validation at .50 and .75
```

Deployable primary gates:

```text
calibration delta TP@.50 >= 0
calibration delta TP@.75 >= 0
validation  delta TP@.50 >= +3
validation  delta TP@.75 >= 0
exact per-image prediction-count parity
```

A deterministic paired bootstrap is reported but is not used as a post-hoc
gate.

## Terminal condition

The audit always stops after calibration and validation summaries. PASS or
FAIL authorizes neither a new model version nor training. The result must be
reviewed with the user before any subsequent action.
