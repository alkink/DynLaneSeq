# V16 candidate-aligned local reranker contract

## Question

Can the deployed V7 slots recover their correct representative proposal when
geometry is used only to define a variable-size competition set and a learned,
image-conditioned scorer chooses one complete proposal curve?

V16 does not average, median-pool, splice, or fit a polynomial as the emitted
lane. Those operations failed because proposal populations can be strongly
one-sided around the correct lane.

## Preflight (no training)

1. Freeze the V7 225k checkpoint, active slots, writer-valid count, scores, and
   routed proposal IDs.
2. Treat each routed proposal as an anchor.
3. Measure proposal-to-anchor distance over common visible rows with lower-road
   weighting `0.10 + 0.90 y^3`.
4. Assign each valid proposal to its nearest anchor (a curve-level Voronoi
   partition).
5. The primary group keeps candidates inside
   `clamp(0.60 * nearest-anchor separation, 72 px, 256 px)` at 1600 px width.
6. Group sizes remain variable. The anchor is retained, but no remote proposal
   is inserted to reach a fixed K.
7. GT is used only after grouping to measure target coverage and the official
   one-proposal-per-slot oracle.

The preflight must pass independently on clip-disjoint held-out train images
and validation images:

- target-support member coverage >= 0.90;
- a proposal within 0.02 official IoU of the global best is present >= 0.80;
- fixed-V7-assignment local-oracle closure >= 0.65 at IoU .50;
- fixed-V7-assignment local-oracle closure >= 0.70 at IoU .75;
- exact anchor retention and zero cross-group duplicates.

If either domain fails, V16 stops before any optimizer step.

## Stage A (only after preflight PASS)

Each real candidate is evaluated as a coherent lane:

```text
V7 slot state + anchor geometry
                 |
candidate proposal row token + candidate geometry
                 |
P2 sampled along that candidate's entire curve
                 |
shared vertical row encoder
                 |
one scalar candidate-quality score
```

The direct target is candidate quality/ranking for the GT fixed to the slot by
the frozen V7 assignment. Inference uses hard argmax over the variable group.
Coordinates are never averaged. Stage A retains exact V7 deployment output and
evaluates the reranked proposal reference as a counterfactual sidecar, so a
failed association learner cannot masquerade as a geometry improvement.

Activity, count, scores, proposal detector, backbone, and the existing V7
refiner remain frozen. A fresh geometry refiner is a later version and is not
authorized by this contract.

## Stop condition

V16 ends after its fixed short Stage A endpoint and held-out/validation report.
No full validation, long continuation, threshold search, test evaluation, or
automatic V17 is allowed. The result must be reviewed before another version.
