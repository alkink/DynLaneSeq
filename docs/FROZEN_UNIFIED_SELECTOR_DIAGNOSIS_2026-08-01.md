# Frozen Unified-Selector Diagnosis

## What is already measured

The 10k unified-selection run does **not** fail because the row-reference
decoder has stopped generating useful geometry.  On the fixed uniform subset,
the candidate pool retains very high raw recall and improves strict-IoU raw
geometry.  The failure occurs when four candidates are selected:

- raw candidate recall remains high;
- learned Top-4 recall is much lower;
- geometry NMS recovers a large part of the loss;
- the oracle gap is larger than for the control.

This localizes the current failure to proposal ownership/ranking/diversity.  It
does not yet distinguish two mechanisms:

1. the unique positive query assigned to a GT lane moves while geometry and
   the selector are jointly optimized; or
2. the current scalar set scorer cannot express diverse four-lane coverage
   even with stationary supervision.

## Decisive short experiment

`scripts/diagnose_culane_dla34_unified_selector_frozen_10k.sh` performs two
bounded diagnostics.

### 1. Ownership stability

The same uniformly sampled validation images are evaluated at 2.5k, 5k, 7.5k,
and 10k.  A one-to-one official-raster-IoU assignment records which persistent
query owns each GT lane.  The report measures owner retention, positive-query
set Jaccard, and whether owner changes happen between near-equivalent duplicate
curves.

### 2. Frozen selector capacity

The 10k detector, curve geometry, row states, FPN, and curve-aligned P2 evidence
are frozen. Their exact selector descriptors are cached. Three cheap selector
arms are trained:

- `continued_matcher`: trained 10k initialization with the exact existing
  matcher target held stationary;
- `fresh_matcher`: fresh initialization with that same stationary target;
- `fresh_official`: fresh initialization with unique official raster-IoU
  targets.

This third arm distinguishes an incapable scalar selector from a selector whose
training target is simply misaligned with the official metric.

The validation gate is threshold-free raw Top-4 recall at IoU 0.50 and 0.70.
NMS Top-4 and oracle Top-4 are reported as diagnostics, but cannot make the
frozen-selector gate positive.

## How the result changes the design

| Frozen selector | Ownership | Interpretation | Next design |
|---|---|---|---|
| positive | unstable | Moving one-to-one targets are primary | Stable/canonical owner teacher or selector warmup before joint training |
| positive | stable | Selector capacity exists, joint optimization fails | Isolate selector warmup and audit selector/decoder gradients |
| negative | unstable | Churn exists but is not the only limitation | Explicit selection without replacement / coverage slots |
| negative | stable | Scalar scorer/features are insufficient | Explicit selection without replacement / coverage slots |

## What this experiment cannot claim

It cannot guarantee 80+ CULane F1.  It decides which selection architecture is
justified before another full training run.  The external claim that
curve-aligned sampling itself must break 80 is already contradicted by the
current results: curve-aligned sampling improves the available geometry, but it
does not automatically prevent duplicate proposals or guarantee correct Top-4
coverage.
