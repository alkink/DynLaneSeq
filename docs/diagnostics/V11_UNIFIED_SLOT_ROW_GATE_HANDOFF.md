# V11 Unified Slot-Row Decoder — implementation and audit handoff

Date: 2026-08-12

Branch: `experiment_v11_unified_slot_row_decoder_gate`

Source model: V7 resume-safe `iter_0225000.pt`
Status: implementation complete; long training is **not authorized**.

## 1. Executive verdict

V11 is not another detach toggle, local-neighborhood adapter, LR sweep, or
P2-only replacement. It changes ownership of final lane geometry:

```text
V7/V8/V9:
    one hard proposal ID -> final/reference geometry

V10:
    cold P2-only slot -> final geometry

V11:
    all 32 proposal row memories
          + four-way structured soft set attention
          + full-width row-aligned P2 evidence
          + persistent 4 x R slot-row state
          -> slot-owned final x/range
```

The old hard ID remains only for diagnostic provenance and writer indexing. It
is not gathered into the V11 geometry equation, does not define a local
neighborhood, and does not bound the final lane.

This implementation is motivated by the following measured chain:

1. The 32-proposal pool has large diagnostic capacity.
2. GT-conditioned target-hard routing recovers almost all of the target-soft
   reference gain; therefore proposal capacity and hard-ID representability
   are not the main limitations.
3. Most recoverable route errors are `OUTSIDE_SUPPORT`, not wrong-member errors
   inside the correct local cluster.
4. V8's anchor-local mixture solved the wrong subproblem and failed full val.
5. V8.1/V9 geometry-to-pooled-router detach removal memorized fixed-64 but did
   not generalize to uniform-256.
6. V10 discarded the strong proposal memory and asked a cold P2-only branch to
   rediscover lanes; fixed-64 failed.

The remaining hypothesis is therefore not “one missing gradient edge.” It is:

> The four output objects need a row-level state that jointly consumes the
> strong proposal population and image evidence, and owns final geometry under
> one coherent set assignment.

V11 tests that hypothesis. It does **not** prove it before the fixed-64 and
held-out gates run.

## 2. Important errata: what the final implementation does not do

An early local draft still used the V7 hard-routed/refined curve as the base and
added a full-width residual. That was not faithful to slot-owned geometry:
despite wide offsets, the hard proposal still owned initialization. That draft
was discarded before commit.

An early draft also used zero-forward straight-through bridge projections to
push first-step geometry gradient into the row trunk. Those bridges were also
discarded. Final V11 uses tiny, real `1e-5` output-head initialization, so all
gradient paths are ordinary derivatives of the deployed forward graph.

Final V11 therefore has:

```text
legacy V7 refiner in final graph             false
hard proposal gather in final geometry       false
anchor-local proposal support                false
single global mix scalar                     false
gradient-only ST bridge in V11                false
P2-only cold geometry branch                  false
all-proposal row memory                       true
full-width P2 evidence                        true
cross-slot structured proposal competition   true
slot-owned x/range                            true
post-geometry activity                        true
one final assignment for all V11 targets      true
```

## 3. Exact tensor graph

Symbols:

```text
B = batch
N = 32 proposals
S = 4 final slots
R = lane rows
D = 256 proposal/P2 feature dimension
H = 256 fresh V11 hidden dimension
X = full P2 horizontal bins
```

Frozen V7/proposal inputs:

```text
proposal row memory  P     [B,N,R,D]
proposal x           Xp    [B,N,R]
proposal range       Rp    [B,N,2]
legacy slot state    Sold  [B,S,D]
legacy route logits  Zold  [B,S,N]
legacy activity      aold  [B,S]
P2 row grid          Fp2   [B,R,X,D]
```

All these source tensors are detached in the first controlled gate. The
proposal detector, backbone, FPN, old selector, and old activity head remain
frozen. They supply stable evidence; they do not receive V11 gradients.

### 3.1 Fresh persistent slot-row seed

```text
Useed[s,r] = Wslot(norm(Sold[s]))
             + learned_slot_token[s]
             + Wrow(position_basis(r))

Useed: [B,S,R,H]
```

The old slot state is context, not a trainable owner. All transforms after the
detach are new and trainable.

### 3.2 Global proposal-row attention over all 32 candidates

For each proposal, the full visible row sequence is retained:

```text
Kp[n,r] = Wkey(norm(P[n,r]))
Vp[n,r] = Wvalue(norm(P[n,r]))
```

The fresh slot-row state produces a row-aware score, averaged only over valid
proposal rows:

```text
zlearned[s,n] = mean_r <Wquery(Useed[s,r]), Kp[n,r]> / sqrt(H)
z[s,n]        = stopgrad(Zold[s,n]) + zlearned[s,n]
```

`Wquery` starts at std `1e-3`, so initialization is close to the measured V7
predicted-soft distribution instead of a random re-ranking. It is an unbounded
learnable logit residual, not a scalar mixture gate.

The four slots are coupled by rectangular Sinkhorn:

```text
A = StructuredUniqueMarginal(z)   [B,S,N]
sum_n A[s,n] = 1
sum_s A[s,n] <= 1
```

This is the new cross-slot interaction. It prevents independent slots from
placing full mass on the same proposal while keeping the path differentiable.
There is no hard proposal ID in this operator.

Soft proposal memory creates a coherent coarse lane object:

```text
X0[s,r] = sum_n A[s,n] Xp[n,r]
R0[s]   = sum_n A[s,n] Rp[n]
C[s,r]  = sum_n A[s,n] Vp[n,r]
```

### 3.3 Full-width image evidence

V11 does not crop around the old hard route. It attends over every horizontal
P2 bin for every slot row:

```text
Kf[r,x], Vf[r,x] = projections(Fp2[r,x] + x_position[x])

E[s,r] = sum_x softmax_x(
              <Wq(U[s,r]), Kf[r,x]>
              + broad_position_prior(X0[s,r], x)
          ) Vf[r,x]
```

The positional term is a broad prior, not a visibility mask or local support.
Every x bin remains reachable. Two full-width visual reads surround the
vertical row interaction.

### 3.4 Persistent same-lane vertical interaction

```text
U [B,S,R,H]
  -> two-layer vertical Transformer independently per slot
  -> row FFN
  -> Ufinal [B,S,R,H]
```

The proposal context, visual evidence, row position, and lane-wide vertical
state coexist before final output.

### 3.5 Slot-owned final geometry

The initial geometry is `X0/R0`, not V7's hard-routed/refined curve.

```text
px = softmax(delta_head(Ufinal))
dx = sum_k px[k] offset_x[k]
Xfinal = clip(X0 + dx, 0, input_width-1)
```

The x support spans the whole 800-pixel frame:

```text
[-800,-600,-400,-300,-200,-128,-64,-32,0,
   32,  64, 128, 200, 300, 400, 600,800]
```

Range is also slot-owned:

```text
Rfinal = sorted_clip(R0 + dR, 0, 1)
dR support = [-1,-.5,-.25,-.1,0,.1,.25,.5,1]
```

The x/range heads use real `1e-5` initialization. On the local real-checkpoint
CPU contract batch, the initial maximum residual was approximately:

```text
x       0.0303 px
range   6.94e-5
```

### 3.6 Post-geometry activity and deployment score

```text
afinal[s] = stopgrad(aold[s]) + Wactivity(mean_r Ufinal[s,r])
active[s] = afinal[s] >= 0
score[s]  = sigmoid(afinal[s])
```

Selected-route probability is not multiplied into the public score. The new
activity head reads the same final lane-object state that produced geometry.
Its real weight initialization is `1e-7`, small enough to preserve initial
activity decisions while keeping the activity-to-state derivative nonzero.
Activity loss cannot directly update x/range heads, while activity gradient
does reach the shared post-geometry state.

The old unique proposal index is retained as public provenance for the writer,
but changing that integer does not change `Xfinal` or `Rfinal`.

## 4. One assignment and one V11 training contract

V7 had independent selection and geometry assignment contracts. V11 disables
both old final-slot losses:

```yaml
w_four_slot_selection: 0
w_four_slot_geometry: 0
w_four_slot_unified: 1
```

One Hungarian assignment is built from detached **final V11 geometry only**:

```text
cost[s,g] = 1 - row_strip_IoU(Xfinal[s], Rfinal[s], GT[g])
```

Activity confidence is absent from matching. Therefore activity cannot choose
its own positive target.

The same slot-to-GT match supervises all of the following:

1. post-geometry active/no-lane BCE;
2. global proposal-attention CE to the matched GT's near-best proposal cluster;
3. final point loss;
4. final range loss;
5. final LineIoU loss;
6. final row DFL;
7. low-weight auxiliary geometry on `X0/R0`.

There is no independent second assignment inside V11.

Loss weights:

```text
activity       1.0
attention      1.0
point          5.0
range          1.0
LineIoU        2.0
DFL            1.0
coarse aux     0.25 x (point + range + LineIoU)
```

The soft proposal target is intentionally a memory-retrieval target rather
than a deployment confidence. Final geometry and activity remain the primary
deployed objects.

## 5. Gradient contract

Only this prefix is trainable:

```text
structured_query_head.set_selection_head.unified_slot_decoder
```

Fresh trainable size:

```text
2,317,313 parameters
64 parameter tensors
initialization SHA256:
1cfabc99f5e26b5da9937478389fef97699944481769d99b983dfad13b726b36
```

Required topology:

```text
final geometry loss
  -> x/range heads
  -> vertical slot-row trunk
  -> full-width P2 projections
  -> global proposal row keys/values/query
  -> structured soft attention A

proposal-attention loss
  -> proposal row projections
  -> slot-row state used to form A

activity loss
  -> post-geometry activity head
  -> shared final slot-row state

geometry loss  X-> activity-head parameters
activity loss  X-> x/range output-head parameters
all V11 loss   X-> old selector/proposal detector/FPN/backbone
```

The local one-real-batch initialization audit passed this topology. It is only
a smoke result, not a population result. The remote runner requires 16 batches
before it allows training.

## 6. Local verification completed

```text
targeted V7/V9/V10/V11 regression tests      22 passed
entire repository test suite                 562 passed
Python compilation                           passed
shell syntax                                 passed
git diff whitespace audit                    passed
real V7 225k initialization                  passed
real one-batch CPU gradient contract         passed
real one-step CUDA bf16 train/checkpoint     passed
compact base+delta checkpoint semantics      passed
```

The one-step CUDA smoke produced a delta checkpoint with optimizer state. It
changed the global proposal query, P2 key, vertical trunk, x head, and activity
head, confirming that the optimizer reaches every intended fresh subsystem.
It does not provide an F1 claim.

## 7. Predeclared experiment sequence

### Stage A — 16-batch initialization/gradient contract

No optimizer step. Training stops if any required edge is zero, a forbidden
edge is nonzero, attention is non-finite/non-normalized, Sinkhorn column
capacity is violated, discrete activity/index warm start changes, or loss is
non-finite.

### Stage B — fixed-64 memorization gate

Train only the 2.317M fresh V11 parameters for 3,000 steps. Frozen proposal
oracle counts must remain invariant.

Primary pass gates:

```text
same-count proposal-oracle gap closure @.50 >= 80%
same-count proposal-oracle gap closure @.75 >= 70%
cardinality exact                         >= 95%
semantic duplicate                        <= 2%
active count drift                         <= .05 lane/image
final surrogate quality not below coarse base
F1@.50 and F1@.75 not below source
```

Failure stops the branch. Gates must not be relaxed after seeing the result.

### Stage C — uniform-256 held-out generalization gate

Run only after fixed-64 passes, and restart independently from the V11 225k
initialization (not from the memorized fixed-64 endpoint).

Primary treatment-vs-source gates:

```text
target-support mass gain   >= 5 percentage points
TP gain @.50               >= 3
TP gain @.75               >= 6
F1@.50 non-regression
F1@.75 non-regression
proposal oracle unchanged
duplicate/count constraints preserved
```

### Stage D — full validation

Only if Stage C passes. Long training is still unauthorized until exact full
validation confirms the primary metric. Test split must not be used for model
or checkpoint selection.

## 8. Honest unresolved risks

1. **Soft barycenter risk.** Correct target-soft was strong diagnostically, but
   learned attention can remain diffuse and create cross-cluster averages.
   Attention CE, auxiliary coarse geometry, structured column capacity, and
   the final slot decoder mitigate this; they do not guarantee success.
2. **Frozen evidence risk.** Full proposal row memory is richer than the old
   819D descriptor, but its producer is still frozen. If lane-cluster identity
   is not recoverable from these rows plus frozen P2, V11 can memorize and fail
   generalization again.
3. **Legacy prior risk.** `Zold` is only an additive prior and can be overcome
   by an unbounded learned residual, but it may slow escape from a wrong
   cluster. Its small learned-query initialization is deliberate; removing the
   prior would be a separate causal arm, not an unreported change.
4. **Assignment non-stationarity.** Matching is based on current final geometry.
   This is DETR-like and coherent, but a fresh head may swap slot identities
   early. Fixed-64 trajectory and cardinality/duplicate audits must reveal it.
5. **Structured-vs-unstructured soft policy.** The old target-soft diagnostic
   used a GT-conditioned policy and was not deployable. V11 uses structured
   cross-slot Sinkhorn for a deployable set. This is principled but remains an
   empirical choice that must be evaluated.
6. **225k adaptation risk.** A mature frozen representation may be harder to
   repurpose than end-to-end training from scratch. The current gate tests
   whether a cheap adaptation is viable before authorizing a costly run.

There is no honest numerical promise of 81+ at this stage. The oracle proves
headroom, not learnability. V11 succeeds only if it closes a meaningful part of
that gap on unseen validation images.

## 9. Files to inspect

```text
dynlaneseq_eg/modeling/four_slot_selection.py
dynlaneseq_eg/modeling/structured_queries.py
dynlaneseq_eg/losses/loss_s0.py
dynlaneseq_eg/factory.py
dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_unified_slot_row_225k_to228k.yaml
dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v11_unified_slot_row_memorize64.yaml
dynlaneseq_eg/tests/test_v11_unified_slot_row_decoder.py
dynlaneseq_eg/tools/initialize_v11_unified_slot_row_checkpoint.py
dynlaneseq_eg/tools/audit_v11_unified_slot_row_contract.py
dynlaneseq_eg/tools/audit_v11_unified_checkpoint_state.py
dynlaneseq_eg/tools/summarize_v11_unified_slot_row_gate.py
scripts/run_culane_dla34_v11_unified_slot_row_gate_225k_to228k.sh
```

## 10. Requested GPT-5.6 Sol Pro audit

Please audit the actual files above, not only this narrative. Be adversarial
and do not assume V11 is correct because tests pass.

Answer these questions in order:

1. Reconstruct the exact forward tensor graph and verify whether any hard
   proposal ID or V7 refiner output still causally owns final x/range.
2. Reconstruct the gradient graph for final geometry, proposal attention, and
   activity. Identify every intended and unintended zero/nonzero edge.
3. Verify that a single final-geometry Hungarian assignment is reused by all
   V11 targets and that activity cannot influence matching.
4. Check whether structured Sinkhorn attention plus soft cluster CE is
   mathematically compatible with the auxiliary/final geometry objectives, or
   whether it creates an unidentifiable/conflicting optimum.
5. Check whether the old route-logit prior can practically trap the new
   unbounded residual despite the graph being theoretically escapable.
6. Check for tensor-shape, visibility-mask, range, DFL-support, AMP, clamp,
   writer-index, checkpoint-chain, or resume bugs that tests may miss.
7. Decide whether the fixed-64 and uniform-256 gates are sufficiently strict
   and causally interpretable. Do not weaken them after observing data.
8. If there is a concrete graph defect, propose one corrected architecture,
   not a list of speculative micro-tweaks. State exact tensor equations and
   gradient ownership.
9. If the implementation is sound, state what fixed-64 result would falsify
   the whole V11 hypothesis and what held-out result would justify full val.
10. Give a blunt probability assessment for reaching 80 and 81 only after
    separating oracle capacity from deployable learnability.

Required response labels:

```text
[CODE] directly verified from source/config
[MEASUREMENT] supported by supplied JSON/results
[INFERENCE] logically derived
[HYPOTHESIS] not yet measured
```

Do not authorize long training from fixed-64 memorization alone.
