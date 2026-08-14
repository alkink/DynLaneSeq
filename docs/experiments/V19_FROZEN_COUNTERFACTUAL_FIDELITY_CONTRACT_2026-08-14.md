# V19 — Frozen Counterfactual Proposal Fidelity

## Decision

V18 is closed. V19 tests one isolated hypothesis:

> Can a candidate-independent, whole-curve quality encoder rank the final
> frozen-V7 result of every slot/proposal alternative well enough to improve
> exact four-slot routing on two unseen domains?

No V7 parameter or running buffer may change. V19 changes only which existing
proposal is routed; V7 activity, count, bounded refinement, writer rules and
the 32-proposal population remain exact.

## Graph

```text
exact V7, frozen + eval
  slot state                         [B,4,256]
  route logits                       [B,4,32]
  proposal row memory                [B,32,160,256]
  proposal x/range                   [B,32,160] / [B,32,2]
  P2/P4/P5 image tensors             frozen
          |
          | run the frozen V7 bounded refiner for all 4*32 alternatives
          v
  counterfactual final x/range       [B,4,32,160] / [B,4,32,2]
          |
          | proposal memory + signed image samples + geometry
          | NO candidate/inter attention
          | one intra/vertical layer along 160 rows
          v
  p50, p75, expected IoU             [B,4,32]
          |
          | additive neutral-at-zero calibration of immutable V7 logits
          v
  existing exact unique four-slot route
          v
  frozen V7 bounded refiner + exact V7 activity/writer
```

The final quality layer is exactly zero initialized. At iteration 225000 all
candidates receive the same neutral fidelity; public V7 tensors and serialized
lane files are bit-exact.

## Training targets

For each frozen counterfactual lane, the online training target is the maximum
detached range-aware row-strip IoU to any GT lane. The loss contains:

```text
BCE(p50, IoU >= .50)
+ 0.5 BCE(p75, IoU >= .75)
+ SmoothL1(expected_iou, IoU)
+ same-GT threshold-weighted pairwise ranking
```

The online target is a fast CULane-aligned surrogate, not a falsely labelled
copy of the exact raster evaluator. Exact official raster IoU is used for the
fixed endpoint gate. The evaluator also reports surrogate-vs-raster Pearson,
Spearman and threshold agreement; both unseen domains must pass the
predeclared target-audit thresholds.

Exact raster targets were not placed in the optimizer loop because production
augmentation changes every counterfactual population per visit and rasterizing
128 lanes against every GT at 1600x640 would dominate training. This deviation
from the external recommendation is explicit and falsifiable.

## Fixed run

```text
source checkpoint   V7 iteration 225000
train list          fixed 8192 images / 640 clips
optimizer steps     8000
effective batch     16
optimizer           AdamW, LR 1e-4, constant
augmentation        inherited V7 production augmentation
trainable params    only counterfactual_fidelity (1,456,643)
detector mode       frozen + eval
checkpoint choice   fixed 233000 endpoint only
held-out            fixed clip-disjoint 256
validation          fixed 256
full validation     closed
test                closed
threshold / NMS     fixed 0 / 0
```

## Gate 0

Passed locally on 2026-08-14:

- 940 mature source tensors exact;
- all frozen buffers exact after forward/backward audits;
- route/activity/x/range/scores and writer bytes exact V7;
- only 49 V19 parameter tensors trainable;
- quality output receives gradient at exact neutral initialization;
- after an unsaved `1e-4` output perturbation, intra, visual and
  proposal/slot/geometry paths all receive finite nonzero gradients;
- 128 counterfactual alternatives are finite;
- no candidate/inter module exists.

## Fixed endpoint PASS

Every condition must pass independently on held-out-256 and validation-256:

```text
proposal tensors and proposal oracle exact source
prediction count exact source
writer-invalid count non-increasing

TP@.50 >= source +5
TP@.75 >= source +5
F1@.50/.75 non-regression
improved images > worsened images at both thresholds
source-correct lane loss <1% at both thresholds

same-GT pair rank accuracy >=70%
threshold-crossing pair rank accuracy >=70%
official fidelity Pearson and ordinal Spearman >=0.30
selected official counterfactual regret < source regret

training-target audit:
  surrogate Pearson/Spearman >=0.85
  threshold agreement .50 >=0.75
  threshold agreement .75 >=0.85
```

Any failure stops V19. No inter, refinement, longer training, full validation,
test evaluation or checkpoint/threshold selection is authorized afterward
without a new joint decision.
