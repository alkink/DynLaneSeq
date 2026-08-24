# V38 Direct-Primary 50K Training-Free Autopsy Contract

## Question

V38 failed the matched 50K gate. This diagnostic separates three causes without
training a new model:

1. the four direct curves do not contain enough official-raster geometry;
2. sufficient curves exist but the fixed existence policy suppresses them;
3. both geometry support and existence/set conversion fail.

## Frozen inputs

- V38 `iter_0050000.pt` only;
- official CULane validation list, all 9,675 rows;
- score threshold `0.50`;
- top four, no NMS;
- 30-pixel official raster width;
- no checkpoint or threshold selection;
- no test split.

## Policies

- `deployed`: the exact existing V38 existence decision;
- `forced_all_valid`: emit every valid one of the four direct curves;
- `cardinality_oracle`: non-deployable maximum-cardinality official-raster
  matching over those same four curves.

The cached deployed TP/FP/FN must match the completed V38 official report
exactly. A mismatch aborts the audit.

## Diagnostics

- best-of-four official support at IoU 0.50 and 0.75;
- support-to-deployed conversion and recoverable missed GT lanes;
- existence-score Spearman and AUC against each curve's best official IoU;
- fixed-row range coverage and x-error in four vertical bands;
- deployed and forced-active crossing/duplicate rates.

## Decision rule

For each threshold, compare V38 deployed TP, V38 best-of-four support TP, and
exact V7 50K TP.

- support at least V7: existence/set-conversion limited;
- at least 60% of the V7 TP deficit remains even under support oracle: direct
  geometry-capacity limited;
- otherwise: mixed geometry and conversion failure.

This diagnostic cannot authorize test use or a new model by itself. It only
selects the next causal training gate.
