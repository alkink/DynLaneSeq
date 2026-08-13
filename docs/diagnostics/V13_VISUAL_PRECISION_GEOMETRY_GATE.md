# V13 visual-precision geometry gate

## Decision

V13 is a new final-geometry owner, not a continuation of the V8--V11
proposal-routing fixes.  A hard proposal ID never produces its output curve.
The exact V7 output is retained only as the zero-step anchor and as the frozen
activity/score source during the causal bridge.

Long training and the CULane test set remain closed.  The only authorized run
is a 2,000-step, 4,096-image bridge followed by two disjoint 256-image official
IoU evaluations.

## Evidence that selects this architecture

The earlier experiments isolate three facts:

1. The frozen 32-proposal population contains substantial geometry capacity,
   but V7 maps a slot to the target proposal support unreliably.
2. Opening geometry gradients in the pooled global router can memorize a
   fixed set but does not improve held-out association.
3. V12's full-width P2 path has real, image-specific and held-out visual
   localization signal.  Its direct curve is too imprecise, while converting
   that state back into one hard or global soft proposal identity removes the
   advantage.

Consequently, neither another route loss nor an anchor-local mixture addresses
the measured failure.  V13 keeps the visual lane-object state continuous and
uses proposal rows as feature evidence instead of asking one proposal ID to
own the final lane.

## Tensor graph

Frozen inputs:

```text
V12 visual state       [B,4,R,256]
V12 visual x           [B,4,R]
V7 deployed x/range    [B,4,R] / [B,4,2]
32 proposal row tokens [B,32,R,256]
32 proposal x/range    [B,32,R] / [B,32,2]
P2 row features        [B,R,X,256]
```

Trainable V13 graph:

```text
V12 visual slot-row state
        |
        +--> row-wise proposal-token attention A[B,4,R,32]
        |        (feature context; no final proposal identity)
        |
        +--> fine P2 samples around both visual-x and V7-x
        |        offsets = [-128 ... +128] px
        |
        +--> per-row four-slot attention
        +--> per-slot vertical transformer
        |
        +--> row/slot-conditioned candidate gate
        +--> full-image x distribution [-1600 ... +1600] px
        +--> full-range start/end distribution [-1 ... +1]
        |
        `--> final slot-owned x/range
```

The proposal attention is `[B,4,R,32]`, rather than V11's lane-global
`[B,4,32]`.  Its values are proposal row tokens.  Proposal coordinates form a
candidate/context signal, but no `argmax` or hard route index is accepted by
the V13 forward interface.

## Initialization and gradient contract

The candidate gate, x head and range head start at zero.  Symmetric offset
supports therefore give exact V7 x/range at iteration 227,000.  A zero-value
straight-through candidate term exposes the visual/proposal/slot-row paths to
geometry gradients on the first step without changing forward values.

Required at zero step:

```text
V12/V7 inherited tensors                    bit exact
V13 final x/range                           exact V7
activity, score, slot indices               exact V7
hard proposal-ID input to V13               absent
geometry -> row-wise proposal context       nonzero
geometry -> local P2 consumer               nonzero
geometry -> cross-slot/vertical trunk       nonzero
geometry -> x/range heads                   nonzero
geometry -> V12/V7/backbone/proposal source zero
```

## Objective

Only `w_four_slot_visual_precision = 1` is active.  The stable V7-anchor
Hungarian assignment supervises final point, range, line-IoU and DFL losses.
V12 and all legacy objectives are zero, preventing a hidden route or activity
objective from deciding the result.

## Generalization protocol

Training uses 4,096 images from 512 train clips for 2,000 optimizer steps.
Evaluation uses:

- 256 images from disjoint held-out train clips;
- 256 images from validation;
- no CULane test images.

Each domain is evaluated with the official raster-IoU path under:

```text
V7 anchor
V13 final
wrong-image P2 -> V13
zero P2 -> V13
V13 x + V7 range
V7 x + V13 range
```

The activity mask is identical for all policies.  Both normal writer-valid
metrics and same-neural-active metrics must improve, so a PASS cannot be
manufactured by changing prediction count.

## Predeclared PASS

Both held-out and validation must independently satisfy:

- at least `+5 TP` at IoU .50 and at least `+5 TP` at IoU .75;
- F1 non-regression at both thresholds;
- the same conditions in the count-controlled neural-active evaluation;
- correct P2 must beat wrong-image P2 by at least 5 TP at both thresholds;
- correct P2 must beat zero P2 by at least 5 TP at .50;
- neural-active prediction count must be exactly unchanged;
- writer-valid count drift must remain within 0.05 lane/image.

PASS authorizes one predeclared full-validation checkpoint evaluation, not long
training.  FAIL stops this V13 family; it does not authorize LR, threshold,
loss-weight, checkpoint or offset sweeps.

## Interpretation

- PASS means the continuous visual slot state can be converted into precise,
  deployable geometry without a proposal-ID bottleneck.
- Strong training improvement but held-out failure means the new consumer is
  still memorizing and the frozen visual representation is insufficient.
- Correct and wrong P2 producing similar official geometry means the decoder
  is falling back to V7/proposal priors rather than using image evidence.
- Strict-IoU-only gains in the x/range factorial localize whether final x or
  visible range is responsible; they do not by themselves authorize a sweep.
