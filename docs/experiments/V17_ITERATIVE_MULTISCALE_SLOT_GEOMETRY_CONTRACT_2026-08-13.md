# V17 iterative multi-scale continuous slot geometry — immutable causal gate

## Decision under test

V7's 32-proposal population contains strong lane curves, but V8--V16.1 did
not learn a deployable mapping from frozen image/proposal state to one exact
proposal ID. V13 and V15 already tested one-shot, parity-anchored slot-owned
residual geometry and did not transfer. V17 is not a larger V15 or another
proposal reranker.

V17 tests one missing mechanism:

```text
exact deployed V7 lane geometry
  -> read P2/P3/P4 at the current curve
  -> read all 32 proposal rows as feature memory
  -> predict a small signed row displacement
  -> move the curve
  -> re-read image and proposal evidence at the new curve
  -> repeat three times
```

Every stage is supervised under one immutable V7 slot-to-GT assignment. No
proposal ID, proposal coordinate average, cluster prototype, or selected
proposal curve may own final geometry.

## Why this is not V11, V13, or V15

| Earlier graph | Measured limitation | V17 change |
|---|---|---|
| V11 | soft proposal barycenter replaced V7 geometry; P2 was late | exact V7 anchor; proposal coordinates are context only |
| V13 | frozen V12 state plus one wide output jump | jointly trained image consumer; three bounded re-centered updates |
| V15 | one full-width P2 pass and one global wide residual | P2/P3/P4 evidence is re-sampled after every update |
| V16/V16.1 | whole-proposal hard replacement | no proposal selection in the geometry path |

Increasing attention heads, widening one stencil, changing a loss weight, or
extending an earlier checkpoint is outside this contract.

## Frozen sources

```text
S7       [B,4,256]       V7 four-slot state
X7       [B,4,R]         exact deployed V7 refined x
rho7     [B,4,2]         exact deployed V7 range
A7       [B,4]           exact V7 activity
score7   [B,4]           exact V7 score
I7       [B,4]           exact V7 public proposal provenance

P        [B,32,R,256]    frozen proposal row memory
Xp       [B,32,R]        frozen proposal coordinates (relative features only)
rhop     [B,32,2]        frozen proposal ranges
validp   [B,32]          frozen proposal validity

F2       P2              frozen V7 projected evidence
F3       P3              frozen FPN evidence plus a fresh V17-only projection
F4       P4              exact frozen V7 projection plus fresh V17 adapters
F5       P5              exact frozen V7 semantic path (not read by V17)
```

The V7 detector, proposal coordinate generator, selector/router, refiner,
activity, score, backbone and FPN are frozen. A fresh P3 1x1 projection is
part of the V17 trainable image adapter. P4/P5 remain bit-exact V7 because
the legacy selector consumes them; V17's own P4 key/value adapters are fresh
and trainable inside the new decoder.

## Fixed assignment

One detached, score-independent assignment is built from source geometry:

```text
M7 = Hungarian(1 - row-strip-IoU(X7, rho7, GT))
```

Only source-active and source-writer-valid slots participate. V17 geometry,
scores, intermediate stages and final outputs cannot change this assignment.
The same `M7` supervises all three stages.

## Iterative decoder

Initial state and geometry:

```text
H0[s,r] = slot_projection(S7[s]) + slot_id[s] + row_position[r]
           + geometry_embedding(X7[s,r], rho7[s])
X0 = X7
rho0 = rho7
```

For stage `l in {0,1,2}`:

1. Four slots exchange same-row, geometry-aware messages.
2. P2/P3/P4 are sampled at identical physical signed offsets around `Xl`.
   Scale content is fused per offset. These offset logits are directly
   supervised against `GT_x - Xl`.
3. All 32 proposal row tokens are queried as memory. Compatibility contains
   signed proposal-to-current-x displacement, slope difference, range and
   visibility. Proposal coordinates affect context only.
4. Per-lane vertical interaction enforces whole-curve coherence.
5. A bounded signed distribution emits x and small range updates.

```text
X(l+1)   = clamp(Xl + delta_x_l)
rho(l+1) = clamp/sort(rho_l + delta_rho_l)
```

The next stage samples around `X(l+1)`, not the original V7 center. Each
stage has at most +/-64 px x support, allowing +/-192 px total movement
without one discontinuous jump.

Forbidden equations:

```text
Xfinal = Xp[argmax proposal]
Xfinal = sum_n attention[n] * Xp[n]
Xfinal = cluster mean/median/medoid
```

## Trainable graph

```text
structured_query_head.set_selection_head.iterative_slot_geometry.*
encoder.ms_proj.p3.*
```

The old P2 projection is frozen; V17 has a fresh P2 adapter internally.

## Initialization and objective

Every stage x/range head has symmetric offsets and zero weights. Step zero is
therefore bit-exact V7 for x, range, activity, score, indices, count and
writer files.

At every stage, under the same `M7`:

```text
Lstage = 1.0 * Lvisual_signed_DFL
       + 5.0 * Lpoint
       + 1.0 * Lrange
       + 2.0 * Lrow_IoU
       + 1.0 * Lrow_DFL
```

Stage weights are fixed at `[0.25, 0.50, 1.00]`. There is no activity,
proposal-ID, route, cluster, cardinality, threshold or score loss.

Direct visual DFL trains image queries/adapters at step zero. Exact zero
residual delays ordinary geometry gradients into the deep fusion trunk until
the output head moves; Gate 0 audits both zero step and one fixed diagnostic
head update. No gradient-only straight-through geometry term is allowed.

## Gradient contract

At zero step:

```text
visual DFL -> P2/P3/P4 adapters, stage query, slot state       > 0
geometry   -> corresponding x/range output heads               > 0
all losses -> V7/backbone/FPN/proposal coordinates/activity      = 0
```

After one unsaved diagnostic output-head update:

```text
geometry -> image adapters, proposal context, stage trunks and heads > 0
```

## Short causal run

```text
source checkpoint       V7 iteration 225000
train                    fixed 4096 images / 512 clips
held-out                 fixed clip-disjoint 256 images
validation               fixed 256 images
optimizer steps          5000
physical batch           4
gradient accumulation    4
effective batch          16
optimizer                AdamW, LR 1e-4, weight decay 1e-4
scheduler                constant
dropout                  0
augmentation             exactly disabled
training/evaluation      BF16 / FP32
seed                     3407
checkpoint selection     none; fixed endpoint only
test                     closed
threshold / Top-K / NMS  0 / 4 / 0
```

Intermediate checkpoints are debug evidence only.

## Mandatory endpoint interventions

Both unseen domains are evaluated with exact V7 activity/count under:

```text
correct image evidence
deterministic cross-clip wrong P2/P3/P4 evidence
zero image content
no proposal-row context
stage 1 output
stage 2 output
stage 3 output
```

The cross-clip map must contain zero same-image and zero same-clip pairs.

## Gate 0

Training remains closed unless all checks pass:

```text
bit-exact public tensors and byte-exact writer parity
fixed M7 deterministic and score/output independent
no hard proposal ID or proposal coordinate mixture in V17 output
all stage losses finite
all image/proposal attention rows finite and normalized
zero-step and post-diagnostic-update gradient matrices satisfy the contract
all frozen/upstream/activity gradients exactly zero
expanded config and source/initialization/code SHA manifest written
```

## Fixed endpoint PASS

Every primary condition must pass independently on held-out-256 and
validation-256:

```text
prediction count exact V7
semantic duplicate <= 2%
writer-invalid increase = 0
TP@.50 >= V7 refined +3
TP@.75 >= V7 refined +6
F1@.50 strictly improves
F1@.75 non-regression
correct image >= cross-clip wrong image +5 TP @.50
correct image >= cross-clip wrong image +8 TP @.75
correct image >= zero image +3 TP at both thresholds
improved-image count > worsened-image count
proposal oracle unchanged
```

Failure on either domain stops V17. Full validation, test, long training,
threshold/NMS search, LR/loss/offset sweep, or V18 is not authorized.

## Terminal rule

At V17 completion, stop and package at most 15 copied artifacts for review
with the user and Sol Pro. Do not begin another version automatically.
