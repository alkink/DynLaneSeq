# V14 corrected visual-first Stage A/B contract

Date: 2026-08-13
Status: predeclared before any V14 optimizer step

## Question being tested

V11 showed that proposal row memory contains localization information, but its
global proposal association was computed before P2 image evidence.  V11 also
gave the P2 consumer no direct association target and relied heavily on the
frozen V7 route prior.  V14 tests one precise causal chain:

```text
correct image evidence
  -> visual row localization
  -> global proposal-cluster association
  -> parity-anchored slot-owned geometry
```

V14 is one version with two conditional stages.  Stage B is not an independent
experiment and is impossible to invoke through the main runner unless the
fixed Stage-A summary passes every gate in both held-out domains.

## Frozen source and data protocol

- Source: exact V7 iteration 225000.
- Training list: 4096 images from 512 train clips.
- Held-out domain: 256 images from disjoint train clips.
- Validation domain: fixed 256-image validation list.
- Wrong-image P2 is a deterministic full-list derangement with zero same-image
  and zero same-clip partners. Batch rolling is forbidden.
- Seed 3407; physical batch 4; gradient accumulation 4; effective batch 16.
- AdamW, LR 1e-4, weight decay 1e-4, constant scheduler.
- Dropout and every augmentation path are explicitly disabled.
- BF16 training and FP32 evaluation.
- Threshold 0, Top-K 4, NMS 0, quality power 0.
- Test split, threshold search, NMS search and checkpoint selection are closed.

## Fixed training assignment

One score-independent assignment is built from exact, detached V7 deployed
geometry and writer-valid source slots:

```text
M7 = Hungarian(1 - row-strip-IoU(V7 x/range, GT))
```

V14 predictions cannot change this assignment. Unmatched source slots target
their private dustbin. A matched GT whose frozen proposal pool has no member
at IoU >= 0.50 is excluded from proposal association loss but retains its
visual row target. Inactive/non-writer-valid V7 slots cannot acquire a GT.

## Stage A: visual-first association identifiability

### Forward graph

Inputs are detached V7/P2/proposal tensors:

```text
V7 slot state / x / range / activity
P2 row grid [B,R,X,D]
proposal row memory [B,32,R,D] + proposal x/range
```

Each slot-row first attends over full-width P2. X position is added to the key
only, never to the value. The resulting visual context and expected-x state
pass through vertical interaction. Only after this visual read are all 32
proposal row memories ranked.

There is no V7 route logit term (`beta = 0`) and no hard proposal ID input.
The association query uses a detached U0 plus trainable visual context, so the
association loss cannot learn a proposal/anchor-only shortcut through U0.

Transport is a joint Sinkhorn polytope over 32 real proposal columns and four
private dustbins:

- every slot row sums to one;
- every real proposal column has capacity at most one;
- slot `s` alone can use private dustbin `s`.

### Targets and losses

Visual attention receives an interpolated two-bin DFL target at each valid GT
row. Proposal targets use frozen proposal/GT row-strip IoU, representable floor
0.50, near-best delta 0.10 and temperature 0.03. All slot targets are projected
through the same private-dustbin transport before KL is computed.

```text
L_A = 1.0 * L_visual_DFL + 1.0 * KL(A_target || A_predicted)
```

The public deployment output throughout Stage A is bit-exact V7: x, range,
activity, score, indices and writer files cannot change.

### Stage-A Gate 0

Before any optimizer step:

- V7 model state and public output tensors are bit exact;
- writer prediction files are byte exact;
- augmentation is off at the actual factory-consumed config path;
- cross-clip maps have zero same-image and zero same-clip pairs;
- forward signature contains no target/GT/assignment input;
- prediction and target transports satisfy row and column constraints at 1e-6;
- `L_visual -> P2 + slot-row`, but not proposal path;
- `L_assoc -> P2 + slot-row + proposal transport`;
- `L_assoc -> detached U0 bypass` is exactly zero;
- all upstream V7/backbone/FPN/proposal parameters are frozen.

### Stage-A endpoint and gate

Exactly 2000 steps, endpoint iteration 227000. Intermediate 500-step
checkpoints are diagnostics only and cannot be selected.

All checks below must pass independently on both held-out-256 and val-256:

- deployment parity exact;
- representable target-support mass >= exact V7 baseline + 0.05;
- hard target-ID Top-1 >= exact V7 baseline + 5 percentage points;
- correct-P2 support mass >= cross-clip-wrong-P2 + 0.05;
- correct visual DFL <= 0.95 * wrong-P2 visual DFL;
- correct visual DFL <= 0.95 * position-only visual DFL;
- zero-content and position-only association cannot preserve the gain.

The endpoint audit also records correct, cross-clip wrong, zero-content,
position-only, zero-content+zero-position, x-reversed and row-reversed P2;
exact tensor shapes; V7/V14 logit magnitudes; rank distributions; 32-batch
component gradients, clipping and AdamW predicted update/parameter; and the
fixed 500-step trajectory.

Any Stage-A failure ends V14. Stage B, full validation and long training stay
closed.

## Stage B: parity-anchored geometry (conditional)

Stage B is initialized only from the fixed Stage-A endpoint. The entire Stage-A
association module is frozen. A fresh geometry consumer reads its visual state,
joint proposal transport and proposal row memory, but final geometry is owned
by a residual around exact V7:

```text
x_final     = x_V7     + delta_x
range_final = range_V7 + delta_range
activity / score / public indices = exact V7
```

Symmetric full-image x offsets and full range offsets have zero-initialized
heads, giving exact source parity at step zero. The fixed M7 assignment remains
the only supervision identity.

```text
L_B = 5 L_point + 1 L_range + 2 L_lineIoU + 1 L_DFL
```

Only the fresh Stage-B geometry subtree is trainable. It runs for exactly 2000
steps (227000 -> 229000) on the same list and optimizer contract.

### Stage-B gates

Gate 0 requires exact public V7 parity, byte-exact writer output, positive
geometry gradient, exact frozen V7/Stage-A state and a pure Stage-B trainable
set.

Both held-out-256 and val-256 must independently satisfy:

- TP@0.50 >= V7 + 3;
- F1@0.50 strictly improves;
- F1@0.75 does not regress;
- neural activity count is exactly V7;
- writer-invalid count does not increase;
- correct P2 beats cross-clip wrong P2 in TP at both 0.50 and 0.75;
- semantic duplicate fraction <= 2%;
- frozen proposal oracle is identical.

Only then is one fixed full-validation endpoint evaluated. Full validation
passes at F1@0.50 >= V7 + 0.50 point and F1@0.75 non-regression.

## Terminal rule

V14 always terminates after its applicable gate. Even a full-validation PASS
does not authorize long continuation, test evaluation or V15. All code,
resolved configs, checkpoint/list/report SHA256 values, endpoint reports and
the immutable decision are packaged into one Sol Pro review directory. The
next action is chosen only after that review.
