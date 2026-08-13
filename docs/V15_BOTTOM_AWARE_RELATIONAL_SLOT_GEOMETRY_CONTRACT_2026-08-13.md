# V15 bottom-aware relational slot geometry — immutable short-gate contract

## Decision being tested

V14 showed that a four-slot state can learn row-local P2 localization, but
forcing that state to predict a globally unique proposal-ID distribution did
not generalize.  The subsequent V7 proposal-clustering audit showed that curve
geometry—especially lower-road separation—cleanly distinguishes reliably
labelled physical lanes, while hard clusters and cluster prototypes destroy
strict-IoU capacity.

V15 tests one substantial architecture, not a threshold or optimizer patch:

```text
frozen V7 lane slot and exact deployed geometry
        +
full-width, row-aligned P2 visual state
        +
all-32 proposal row memory connected by a continuous bottom-aware graph
        ↓
persistent four-slot row states
        ↓
slot-owned residual x/range geometry
```

The graph is context only.  It never merges proposals, emits a prototype,
chooses a public proposal ID, or directly owns final coordinates.  There is no
proposal-ID KL/CE objective.  Original proposal members remain separate.

## Frozen input tensors

```text
S7       [B,4,D]       V7 slot state
X7       [B,4,R]       exact deployed V7 refined x
rho7     [B,4,2]       exact deployed V7 range
A7       [B,4]         exact V7 activity
score7   [B,4]         exact V7 deployment score

P        [B,32,R,D]    frozen proposal row memory
Xp       [B,32,R]      frozen proposal curves
rhop     [B,32,2]      frozen proposal ranges
validp   [B,32]        frozen candidate validity
F2       [B,R,X,D]     frozen P2 row grid
```

The V7 proposal detector, backbone/FPN, selector, legacy route, refiner,
activity and score paths remain frozen.

## Fixed assignment

Training uses one detached assignment per image:

```text
M7 = Hungarian(1 - official-row-strip-IoU(X7, rho7, GT))
```

Only source-active, source-writer-valid V7 slots participate.  Prediction
scores and V15 outputs cannot alter the assignment.  No target tensor enters
inference.

## Visual-first state

For slot `s` and row `r`:

```text
U0[s,r] = Ws(LN(S7[s])) + slot[s] + row[r] + geometry(X7[s,r], rho7[s])
```

One full-width P2 distribution is formed before proposal memory is consumed:

```text
Lvis[s,r,x] = q(U0[s,r]) dot (k(F2[r,x]) + position_key[x]) / sqrt(D)
Pvis        = softmax_x(Lvis)
Cvis        = sum_x Pvis * v(F2)
Xvis        = sum_x Pvis * x
Uvis        = VerticalEncoder(U0 + Cvis + position_state(Xvis))
```

Position is never added to P2 values.  `Lvis` receives a direct two-bin row
DFL target under `M7`.

## Continuous perspective-aware proposal graph

For every proposal pair `(i,j)`, V15 computes detached continuous edge
features from common visible rows.  The feature vector contains:

```text
common-row fraction
global mean and q90 |Xi-Xj|
perspective-weighted mean and q90, weight = 0.10 + 0.90*y^3
lower-road mean and q90
bottom-three-row endpoint distance
max(0, lower mean - upper mean)
range start/end/length differences
```

All pixel distances are normalized by image width.  Invalid/no-overlap pairs
are masked; self edges remain valid.  A fixed monotonic geometric log-prior and
a learned edge MLP produce row-stochastic graph weights:

```text
G[i,j] = softmax_j(f_edge(edge[i,j]) + fixed_bottom_aware_prior[i,j])
```

Graph message passing preserves node identities:

```text
Pgraph[i,r] = Pvalue[i,r] + Wgraph * sum_j G[i,j] * Pvalue[j,r]
```

There is no complete-link clustering and no medoid/mean/median replacement.

## Slot-to-memory context without proposal-ID supervision

`Uvis` queries all graph-refined proposal nodes.  Compatibility combines
content similarity with a continuous visual-to-proposal geometric bias derived
from `Xvis`, with the same lower-road weighting.  Per-slot softmax is used;
there is no hard route, Sinkhorn uniqueness requirement, or dustbin in the
geometry-producing path.

```text
Asn          = softmax_n(content(Uvis,Pgraph) + visual_curve_bias(Xvis,Xp))
Cprop[s,r]   = sum_n Asn[s,n] * Pgraph[n,r]
H[s,r]       = FusionVertical(Uvis[s,r] + Cprop[s,r])
```

`Asn` is diagnostic/contextual.  Final geometry does not equal a weighted
coordinate average and exact proposal identity is not a training target.

## Slot-owned output and exact initialization

```text
X15   = X7   + symmetric_expectation(delta_x_logits(H))
rho15 = rho7 + symmetric_expectation(delta_range_logits(H))
```

Both output heads are initialized to exactly uniform logits, hence zero
residual.  At step zero, public x/range/activity/score/indices and writer files
must be bit-exact V7.  Activity and score remain exact V7 throughout the short
gate.

## Objective

No proposal-ID, graph-cluster, activity, legacy, detector, or backbone loss is
enabled.  Under fixed `M7`:

```text
L15 = 1.0 * Lvisual_DFL
    + 5.0 * Lpoint
    + 1.0 * Lrange
    + 2.0 * Lline_IoU
    + 1.0 * Lrow_DFL
```

## Gradient contract

| loss | visual P2 consumer | proposal graph/key/value | slot fusion trunk | x/range heads | V7/upstream | activity/score |
|---|---:|---:|---:|---:|---:|---:|
| visual DFL | >0 | 0 | visual state only | 0 | 0 | 0 |
| final geometry, zero step | visual-key path >0; value path may be 0 | may be 0 by exact zero-head design | may be 0 | >0 | 0 | 0 |
| final geometry, after first fixed update | >0 | >0 | >0 | >0 | 0 | 0 |

Exact V7 parity and a fully live zero-step graph are incompatible without a
gradient-only straight-through term.  V15 deliberately avoids that confound:
the zero-initialized output heads update first, and Gate 0 performs a second
backward after one fixed diagnostic head update to prove that geometry then
reaches the visual-value, proposal-graph and fusion paths.  Endpoint audits
must also report pre/post-clip norms and predicted Adam update per group.

## Data and optimizer protocol

```text
source checkpoint: V7 iter 225000
train: fixed clip-disjoint 4096 images / 512 clips
held-out: fixed disjoint 256 images
validation diagnostic: fixed 256 images
steps: 3000
physical batch: 4
gradient accumulation: 4
effective batch: 16
AdamW LR: 1e-4, constant
weight decay: 1e-4
dropout: 0
augmentation: explicitly disabled
training: BF16
evaluation: FP32
seed: 3407
checkpoint selection: none; fixed endpoint only
test: closed
threshold: 0
Top-K: 4
NMS: 0
```

## Mandatory interventions

At the fixed endpoint, both unseen domains are replayed with:

```text
correct P2
deterministic cross-clip wrong P2
zero P2 content with position keys retained
correct proposal graph
identity graph (no inter-proposal messages)
geometry-shuffled graph
zero proposal row-token content with geometry retained
no proposal context (visual/anchor slot trunk only)
```

These are causal diagnostics, not checkpoint-selection arms.

## Gates

### Gate 0 — before optimizer step

```text
exact V7 public tensor and writer parity
fixed M7 deterministic and score-independent
finite graph features/weights; graph rows sum to one
invalid edges receive zero probability
no hard clustering/prototype/GT inference dependency
zero-step and post-one-diagnostic-update gradient-access matrices exact
all upstream/activity/score gradients exactly zero
expanded config, code and checkpoint SHA manifest written
```

### Fixed endpoint gate — must pass on held-out-256 and validation-256

```text
V15 correct-input TP@.50 >= V7 +3
V15 correct-input F1@.50 > V7
V15 correct-input F1@.75 >= V7
prediction count exactly V7
semantic duplicate <= 2%
writer-invalid increase = 0

correct P2 TP@.50 >= cross-clip-wrong P2 +2
correct graph TP@.50 >= identity/shuffled graph best +2
correct graph final geometry >= proposal-context-free final geometry
```

The gate fails if either domain fails any primary metric, if correct graph/P2
has no causal advantage, or if the model gains only on the training subset.
No intermediate checkpoint may replace the fixed endpoint.

## Terminal rule

At V15 completion, stop.  Do not start long training, full validation, V16,
threshold search, or test evaluation until the user reviews the result.
