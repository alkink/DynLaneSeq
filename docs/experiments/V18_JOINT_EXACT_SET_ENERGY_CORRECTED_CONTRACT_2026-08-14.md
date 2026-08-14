# V18 Joint Exact Set-Energy — Corrected Causal Contract

Date: 2026-08-14

## Decision

V17 is closed. V18 is not a longer V17 run and not another candidate-local
reranker. It keeps the mature V7 proposal population, but trains the deployed
four-lane decision at the level where the measured failure occurs: the complete
physical lane set.

The only authorized experiment is a paired treatment/control short gate from
the exact V7 225k source. Long training, full validation, test, threshold search
and NMS search remain closed.

## Measured problem

The 32-proposal population has strong GT-conditioned capacity, while V7 often
allocates a slot to the wrong global proposal cluster. Exact proposal-ID
classification is also ill-posed because several proposal IDs can represent the
same physical lane. V8–V17 showed that local averaging, late frozen reranking,
proposal-independent posterior replay and repeated forced residual correction
do not transfer reliably to unseen images.

V18 therefore separates four operations:

1. preserve the strong 32-proposal geometry objective;
2. learn live image/proposal association features;
3. score every injective four-proposal set exactly with unary and pair energy;
4. retain V7's bounded anchor and offer one explicit KEEP/REFINE alternative.

## Important corrections to the supplied V18 proposal

The supplied reference was not safe to run unchanged.

### 1. Pair branch initialization

Zeroing both a global pair scale and the pair output head would make the pair
representation receive no first-step gradient. The implementation has no global
zero scale. Only the final pair output is zero initialized. At step zero the
output head receives gradient; after its first update the pair representation
receives gradient.

### 2. Finite invalid-set masking

Invalid sets use a finite `-1e9` sentinel. The listwise loss indexes only valid
sets. This prevents the undefined `0 * -inf` path.

### 3. Training unit is an unordered physical set

Deployment searches all ordered assignments:

```text
32 P 4 = 863,040
```

Training marginalizes the 24 slot permutations for each physical four-set:

```text
C(32,4) = 35,960 physical sets
24 permutations per physical set
```

This avoids inventing a semantic meaning for proposal IDs or arbitrary slot
order. The existing V7 unary prior initializes the deployment permutation.

### 4. Images with fewer than four GT lanes

For `G < 4`, target reward matches the available GT lanes injectively and
marginalizes unused set members. It does not label arbitrary extra proposals as
positive or duplicate. Activity supervision decides which slots are emitted.

### 5. Honest zero-initialization gradient contract

Exact V7 parity and immediate upstream set gradient cannot both hold when the
last unary/pair heads are zero. Gate 0 therefore has two explicit phases:

- phase 1, saved initialization: output-head gradient is nonzero, association
  and shared-image gradients are exactly zero;
- phase 2, unsaved `1e-4` head perturbation: treatment association/shared-image
  gradients must become nonzero, while the detach-control stays exactly zero
  with identical forward values.

The perturbation is restored and never saved.

### 6. Route gradient ownership

Only the unordered physical-set listwise loss trains unary/pair route energy.
Hard geometry cannot claim a route gradient. The set margin is detached before
the KEEP/REFINE policy, and the mature V7 route/anchor/activity tensors are
detached before the new geometry alternative.

### 7. Proposal-coordinate protection

V18 samples at detached proposal x/range coordinates. Geometry and set losses
may train live proposal-row/image representation, but they cannot backpropagate
through proposal coordinate/range outputs. The existing proposal losses retain
ownership of proposal geometry.

### 8. Shared-gradient protection covers the complete V18 objective

On shared backbone/FPN parameters, the gradient protected against the mature
proposal objective is the complete V18 objective—not only listwise routing.
When the dot product is negative, the conflicting V18 component is projected
off the proposal gradient. Private V18 parameters are never projected.

## Forward graph

```text
live DLA/FPN P2/P3/P4
live 32 proposal row states + detached proposal x/range sampling coordinates
              |
              v
candidate association states [B,32,256]
              |
              +--> residual unary [B,4,32]
              +--> semantic pair energy [B,6,32,32]

V7 unary prior + residual unary + activity-weighted pair energy
              |
              v
all 35,960 physical sets x 24 slot permutations
              |
              +--> logsumexp permutation marginal for training
              +--> exact ordered argmax for deployment
              |
              v
existing V7 bounded refiner = anchor X0/range0
              |
              v
one new proposal/image-context alternative X1/range1
              |
              v
categorical KEEP exact X0 or REFINE exact X1
```

No proposal-coordinate average owns final geometry. No straight-through
coordinate barycenter is used.

## Objective

The physical-set reward is the best injective match between members of a set
and valid GT lanes. Each lane reward is threshold aware:

```text
sigmoid((IoU - .50) / tau50)
+ 0.5 * sigmoid((IoU - .75) / tau75)
+ 0.1 * IoU
```

Near-best physical sets form a soft target. Other V18 terms supervise one-shot
geometry, range, visual localization, uncertainty, activity, KEEP/REFINE regret,
soft F1 at `.50/.75`, and asymmetric anchor non-degradation. Existing proposal
existence/point/range/line-IoU/row-DFL/segmentation/centerline losses stay active.

## Paired causal control

Treatment and control have identical:

- initialization and parameter count;
- forward values;
- data order and augmentation;
- losses, optimizer, LR and step horizon;
- geometry, policy and activity gradients.

The only difference is:

```text
treatment: physical-set loss -> live association/image representation
control:   physical-set loss -X-> association/image representation
```

## Gate 0

Required before any optimizer step:

- source/init shared state exact;
- same-forward V7 route, activity, score, x and range exact;
- writer bytes exact;
- 35,960 / 24 / 863,040 enumeration counts exact;
- no repeated proposal ID in deployment;
- inference forward target-free;
- all set energies, losses and gradients finite;
- saved-zero and unsaved-perturbation gradient phases pass;
- proposal-coordinate-head V18 gradient exactly zero;
- treatment/control forward exact;
- all old V11–V17 and inherited auxiliary losses explicitly zero;
- complete-V18/proposal gradient projection enabled.

Local real-checkpoint result: PASS. The authoritative remote Gate 0 must repeat
on the target CUDA environment before training.

## Short paired gate

```text
source:             exact V7 iteration 225000
train:              fixed 8192 images / 1024 clips
held-out:           fixed clip-disjoint 256
validation:         fixed uniform 256
optimizer steps:    8000
physical batch:     4
gradient accum:     4
effective batch:    16
seed:               3407
augmentation:       V7 production augmentation, fixed
endpoint selection: none
threshold:          0
Top-K:              4
NMS:                0
test/full val:      closed
```

Both domains must independently satisfy the predeclared official gate, and
treatment must beat the paired detach-control. If either domain or the causal
comparison fails, V18 stops. No checkpoint selection, continuation or metric
tuning is authorized.

## Stop rule

When the V18 fixed endpoint and its causal audit are complete, stop. Do not
design or start V19. Review the V18 artifacts and the external Sol Pro analysis
with the user first.
