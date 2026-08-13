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

The scorer has hidden width 128, three candidate-aligned P2 samples per row at
`[-24, 0, 24]` px, and dilated row blocks `[1, 2, 4]`. It produces one scalar
for every member of the variable group. Its only objective is:

1. Smooth-L1 calibration of `sigmoid(score)` to detached official-surrogate
   proposal/GT quality;
2. pairwise logistic ordering only for quality gaps of at least `0.02`;
3. hard best-member cross entropy only when best versus second-best quality
   differs by at least `0.02`.

Groups whose best member is below IoU `0.50` do not supervise the scorer. This
prevents an unrepresentable slot/GT pair from teaching an arbitrary proposal
identity. Near-ties are not forced into a false exact-ID target.

Activity, count, scores, proposal detector, backbone, and the existing V7
refiner remain frozen. A fresh geometry refiner is a later version and is not
authorized by this contract.

### Immutable training protocol

- source: V7 iteration 225000;
- train subset: 4096 images from 512 clips;
- held-out: 256 images from 64 train-disjoint clips;
- validation diagnostic: fixed uniform/balanced 256 images;
- optimizer: AdamW, LR `1e-4`, weight decay `1e-4`, constant schedule;
- exactly 2000 optimizer steps, physical batch 4, accumulation 4;
- augmentation and dropout disabled;
- fixed endpoint only; intermediate 500-step checkpoints are debug artefacts;
- FP32 evaluation, threshold 0, Top-4, NMS 0, test closed.

Before training, Gate 0 requires bit-exact V7 public tensors and writer files,
disjoint variable-size groups, retained anchors, exact whole-proposal gathers,
zero selected duplicates, a target-free forward, and nonzero gradients in the
visual, proposal, row, and score subpaths. Both wrong-image controls must be
deterministic cross-clip derangements with zero same-image and zero same-clip
pairs.

### Fixed endpoint decision gate

The same checks must pass independently on held-out-256 and validation-256:

- reranked raw proposal gains at least 3 TP at IoU .50 and 6 TP at IoU .75
  over the raw V7 routed proposal;
- F1 strictly improves at both thresholds;
- prediction/activity/count/score remain exact V7;
- correct P2 is non-inferior to cross-clip-wrong and zero-content P2 at both
  thresholds;
- cross-clip P2 changes at least 1% of hard decisions and changes candidate
  scores by more than `1e-4` on average;
- selected duplicates remain zero; no averaging, padding, or fixed K occurs.

These are association-stage diagnostics, not a claim that raw proposal output
is the final deployable V16 lane. Passing authorizes review, not another run.

## Stop condition

V16 ends after its fixed short Stage A endpoint and held-out/validation report.
No full validation, long continuation, threshold search, test evaluation, or
automatic V17 is allowed. The result must be reviewed before another version.
