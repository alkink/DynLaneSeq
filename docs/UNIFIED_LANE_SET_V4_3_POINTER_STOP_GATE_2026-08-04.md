# Unified Lane Set V4.3: Sequential Pointer + STOP Gate

## Why this branch exists

V4 solved V3's catastrophic geometry collapse.  At 50k the frozen pool still
contains enough curves for roughly 95% recall at IoU 0.50 and 82% at IoU 0.75.
V4.1/V4.2 nevertheless selected many copies of the same lane because they
assigned 32 scalar scores in parallel and only afterwards applied Top-K.

The decisive counterfactual was MMR: without changing a curve or scalar score,
conditioning each next pick on the already selected curves raised diagnostic
F1 by roughly 20--27 points.  V4.2's relation-biased parallel Transformer did
not internalize that behavior.  The missing operation was conditional state:

```text
select candidate A
       |
       v
mark A unavailable and expose relation(candidate_i, A)
       |
       v
recompute the next decision
```

V4.3 implements that operation directly.

## Why the next gate is pointer, not another parallel-scalar rescue

The V4.2 R3 set objective was a real improvement: diagnostic F1@0.50 rose by
about five points and duplicate FP fraction fell from roughly 66% to 47%.
It nevertheless left an external-MMR gain of about 19.75 F1 points.  This is
well beyond the predeclared five-point boundary at which a static scalar
selector should be considered unable to internalize selected-set state.

A cluster-presence / cluster-winner loss and a one-shot asymmetric dominance
head remain useful publication ablations.  They are not the primary rescue
because they still produce all 32 deployment scores simultaneously and have
no explicit record of which lane has already been emitted.  V4.3 incorporates
their useful ingredients in the stateful formulation:

```text
cluster presence       -> one unique GT representative at each pointer step
cluster winner quality -> detached max-IoU unary quality supervision
asymmetric dominance   -> suppression only after a better candidate is chosen
variable cardinality   -> explicit STOP target
```

This does not assert that V4.3 will pass.  It makes the higher-information
structural hypothesis the next falsifiable gate.  If its residual MMR gap is
still large, the failure belongs to the pointer target/state design rather
than to the already-validated V4 geometry.

## Architecture

```text
V4 frozen detector
|
+-- final lane curves [B, 32, 160] ---------------- detach --+
+-- predicted ranges [B, 32, 2] ------------------- detach --+
+-- row states / confidence ----------------------- detach --+
+-- curve-aligned P2 evidence --------------------- detach --+
+-- P4/P5 score-semantic view --------------------- safe -----+
                                                            |
                                                            v
                              candidate descriptor projection
                                                            |
                                                            v
                         relation-aware candidate encoder
                                                            |
                          +---------------------------------+
                          | unary localization-quality logit|
                          +---------------------------------+
                                                            |
                                                            v
step 1: pointer over 32 candidates + STOP
       |
       +-- candidate selected -> update GRU state
       |                         mask selected candidate
       |                         aggregate explicit relation tensor
       |
       `-- STOP selected ------> emit no more lanes
                                                            |
                                                            v
step 2 -> step 3 -> step 4, each conditioned on prior choices
```

The explicit relation tensor has shape `[B, 32, 32, 6]` and contains:

1. mean curve distance;
2. top-weighted curve distance;
3. bottom-weighted curve distance;
4. visible-row overlap;
5. range IoU;
6. soft strip similarity.

The pointer contains a positive learnable strip-similarity penalty from the
first update.  This is the structural equivalent of the successful MMR fact,
while the learned relation MLP can adjust it using the other five channels.

## Training target

The pointer target never uses the current deployment score.

```text
detached candidate curves x GT lanes
        |
        v
range-aware 30 px row-strip IoU matrix
        |
        v
score-independent Hungarian unique representatives
        |
        v
sort GT lanes left-to-right
        |
        v
[candidate, candidate, ..., STOP, IGNORE, ...]
```

The lightweight pointer decoder is rerun with those teacher choices after the
geometry matcher is known.  This produces a deterministic autoregressive
supervision path without running the backbone or geometry decoder twice.

The loss has two terms:

```text
L_pointer = CE(candidate-or-STOP at every active step)
          + 0.5 * QFL(unary score, max detached localization IoU)
```

The first term learns unique coverage and variable cardinality.  The second
term chooses the best localized representative inside a duplicate cluster,
which is needed for IoU 0.75.

## Gradient contract

Allowed gradients:

```text
set_selection_head.*
final score-only P4/P5 semantic adapters
decision_norm
```

Forbidden gradients:

```text
backbone / FPN / P2
lane-state geometry blocks
row-reference decoder
bounded-delta heads
range head
```

All curve, range, row-state, and P2 evidence inputs cross a detach boundary.
The run script refuses to train before a real-batch gradient audit passes.

## Deployment contract

Inference no longer means `score -> threshold -> forced Top-4`.

```text
pointer candidate -> emit lane
pointer candidate -> emit lane
pointer STOP      -> finish image
```

There is no NMS, score multiplication, or forced filling of unused slots.  The
official writer consumes `selection_pointer_indices` directly.  `Top-K=4` is
only a hard maximum and `score_thresh=0.0` avoids adding a second cardinality
rule on top of STOP.

## Experiment size

This gate has **one training arm per seed**, not four:

```text
V4 50k frozen geometry -> V4.3 pointer/STOP scorer for 15k optimizer steps
```

The default is one seed (`3407`).  Only one compact final delta checkpoint is
written, so periodic checkpoints cannot exhaust the filesystem.

## Command

```bash
cd /workspace/DynLaneSeq

git fetch origin refs/heads/experiment_unified_lane_set_v4_3_pointer_stop_gate
git switch -C experiment_unified_lane_set_v4_3_pointer_stop_gate FETCH_HEAD

DATA_ROOT=/workspace/CULane \
SOURCE_CHECKPOINT=outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt \
SEEDS="3407" \
TRAIN_STEPS=15000 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
AMP_DTYPE=bfloat16 \
bash scripts/run_culane_dla34_unified_lane_set_v4_3_pointer_stop_gate_50k.sh
```

Primary result:

```text
outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/
  seed_3407/reports/summary.json
```

Final compact checkpoint:

```text
outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/
  seed_3407/pointer_stop/iter_0065000.pt
```

## Predeclared diagnostic gate

The uniform-256 gate requires all of the following:

```text
gradient isolation                    PASS
All-32 oracle @0.50/@0.75             unchanged within 0.5 point
pointer direct recall @0.50           >= 70%
pointer direct recall @0.75           >= 60%
duplicate fraction among FP @0.50     < 25%
mean emitted lanes                    < 3.95 (STOP is actually used)
```

A pass authorizes full validation.  It does not yet authorize a 278k run or a
benchmark claim.  A failure means the pointer contract, not V4 geometry, must
be redesigned before more long training.

## Full test command after a validation pass

```bash
DATA_ROOT=/workspace/CULane \
CKPT=outputs/diagnostics/unified_lane_set_v4_3_pointer_stop_gate_50k/seed_3407/pointer_stop/iter_0065000.pt \
EVAL_BATCH_SIZE=8 \
AMP_DTYPE=none \
bash scripts/eval_culane_dla34_unified_lane_set_v4_3_pointer_full_test.sh
```
