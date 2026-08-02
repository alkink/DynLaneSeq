# Unified Lane-Set Decoder v3

**Date:** 2026-08-02

**Preserved control:** `experiment_coherent_lane_state_25k` (`33a7dec`)

**Implementation branch:** `experiment_unified_lane_set_decoder_v3_25k`

## 1. Why this is an architecture correction

The 25k coherent model produced the following NMS-free CULane test result at
`score=0.30`, `quality_power=0`, and `Top-K=4`:

| Metric | Coherent 25k | Historical DLA-34 25k |
|---|---:|---:|
| TP | 71,181 | 72,752 |
| FP | 39,343 | 13,488 |
| FN | 33,705 | 32,134 |
| Precision | 64.40 | 84.36 |
| Recall | 67.87 | 69.36 |
| F1@0.50 | 66.09 | 76.13 |

The true-positive and false-negative changes are modest compared with the
additional **25,855 false positives**. This is not primarily an inability to
draw lanes. It is a failure to convert a duplicate-prone curve population into
a unique lane set without NMS.

The referenced NMS=20 test file was not present in the local worktree when
this implementation was prepared. It must be copied with the other remote
outputs before its exact TP/FP counts can be added here.

## 2. Broken computation graph

The coherent 25k candidate improved the training contract but left its lane
state as a read-only branch:

```text
instance + ordered rows
        |
        v
P2 row-reference decoder -----------> row distributions ---> x geometry
        |
        +-- rows --> persistent lane state --> existence/range
                         X
                         | no edge back to row geometry
                         | no interaction with other lane states
```

This has three consequences.

1. The state that predicts foreground does not causally own the curve.
2. A lane-level score cannot compare its complete curve with another
   candidate's complete curve; same-row attention is not an adequate
   whole-lane duplicate decision.
3. Removing `quality` and NMS asks an independent existence MLP to infer
   existence, localization quality, uniqueness, and cardinality from a state
   that cannot control or compare the corresponding geometry.

The old four-group model did not solve this problem. It deliberately trained
approximately four copies per ground-truth lane and then used quality scoring
and NMS to clean them. Its components were limited but mutually consistent.
The coherent model removed that cleanup before it had built a DETR-like set
decoder.

## 3. New end-to-end pipeline

The new block uses one persistent state `q_i` for candidate `i` and one
ordered row state `r_i,j` for each image row `j`:

```text
                         P4/P5 coarse context
                                |
                                v
learned lane q_i --> lane-set self-attention -----------+
       |                                                |
       | fixed q_i -> rows residual                     |
       v                                                |
ordered row states r_i,j                                |
       |                                                |
       +-> sample P2 around previous curve reference    |
       +-> same-row candidate interaction               |
       +-> same-lane vertical interaction               |
       |                                                |
       +-> x distribution and next detached reference   |
       |                                                |
       +-> rows-to-same-lane cross-attention -> q_i ----+
                                                       |
                       score-only semantic decision view
                                                       |
                                      one foreground logit
```

The actual order inside every decoder block is:

1. **Lane-set competition.** All 32 deployable lane states exchange complete
   candidate information through self-attention.
2. **Lane-to-row ownership.** The updated lane state is projected and added to
   every ordered row belonging to that lane. This edge has no learned scalar
   gate and therefore cannot collapse to zero as an optimizer shortcut.
3. **P2 curve refinement.** The existing curve-aligned row-reference decoder
   samples seven local horizontal offsets from high-resolution P2, performs
   same-row candidate interaction, and performs vertical same-lane attention.
4. **Row-to-lane collection.** The same lane state cross-attends only to its
   own updated rows. Existence now observes the evidence that generated the
   curve.
5. **Score-only semantics.** P4 and P5 are independently pooled and attended.
   Their fused context creates a temporary decision view of the same lane
   state. It does not persist into the next geometry block and cannot blur the
   next row reference.
6. **Shared prediction.** Row states produce x distributions; the base lane
   state produces visible range; the semantic decision view produces exactly
   one foreground logit.

## 4. Explicit gradient contract

| Loss | Required gradient route | Intentionally blocked route |
|---|---|---|
| Point / DFL / LineIoU | row head -> row state -> lane-to-row -> lane-set state | none inside P2 geometry |
| Range | range head -> base lane-set state | P4/P5 decision residual -> range |
| Foreground focal | one score -> decision view -> base lane state -> rows-to-lane -> P2 evidence | second quality/selector score |
| Foreground focal | one score -> decision view -> P4/P5 projection/backbone | P4/P5 residual -> next x refinement |
| Cardinality | foreground probabilities -> same one score | predicted geometry -> count target |
| Score margin | assigned/unmatched logits -> same one score | second ranking head |

Unit and smoke tests explicitly verify non-zero gradients on:

- lane-set self-attention;
- lane-to-row projection;
- rows-to-lane attention;
- P2 features from foreground loss;
- P4 semantic attention from foreground loss.

They also verify that a pure geometry loss produces no P4/P5 gradient.

## 5. One scoring contract

The model no longer has two unconstrained existence logits. It predicts one
scalar `s_i`. For compatibility with existing matching and evaluation code,
it exposes `[s_i, 0]`; therefore:

```text
softmax([s_i, 0])[lane] == sigmoid(s_i)
```

The same probability is used for:

- bounded Hungarian object cost;
- focal foreground/background supervision;
- cardinality regularization;
- positive-vs-hard-negative ranking;
- validation thresholding;
- direct Top-K inference.

`quality`, a detached selector, score multiplication, and lane NMS are all
disabled in the v3 config.

The coherent control contains 61.421M parameters (9.585M in its structured
head). V3 contains 69.061M parameters (17.225M in the head). The 7.640M
increase is primarily the four causal set-state blocks and their pooled
semantic attention. It is about 30.6MB of FP32 weights; its semantic memories
are pooled to 10x25 before attention, so the activation increase is much
smaller than attending to full P4/P5 maps.

## 6. Why the two added set losses exist

Focal loss supervises each candidate independently. With 32 candidates and at
most a few lanes, it does not directly control total emitted foreground mass
or the ordering of an assigned lane relative to a near-duplicate unmatched
candidate.

- `L_cardinality = SmoothL1(sum_i sigmoid(s_i), number_of_GT_lanes)` controls
  the image-level lane count and directly penalizes cross/no-lane false
  positives.
- `L_margin` compares every assigned score with the eight hardest unmatched
  scores using `softplus(margin - s_pos + s_neg)`. It explicitly trains the
  direct Top-K decision surface without a second head.

The configured weights are deliberately secondary to the existing focal
loss: `0.10` for cardinality and `0.25` for ranking.

## 7. Assignment and deep supervision

The v3 gate retains strict global one-to-one Hungarian matching, bounded
`-p_lane` cost, `lambda_obj=0.5`, and fresh matching at each returned decoder
layer. Fresh layer-local assignment is the CondLSTR/MapTR/DETR convention: an
early decoder layer is supervised against the geometry it currently emits,
not forced to inherit a potentially unrelated final-layer permutation.

Reference coordinates are still detached between decoder blocks. This blocks
uncontrolled higher-order gradients through repeated sampling while the
explicit lane-state residual provides a stable differentiable path across
blocks.

## 8. Predeclared 25k decision

The first run is a matched from-scratch **25,000-iteration architecture gate**,
not a claim that 25k is the final optimum. The full 278k cosine horizon,
warmup, seed, effective batch size, backbone, P2 row-reference geometry, and
localization losses remain unchanged.

The design is successful only if it shows all of the following on paired
validation data:

1. NMS-free precision and F1 improve materially over coherent 25k;
2. direct score Top-K closes the gap to oracle Top-K;
3. the F1 gain obtained by adding NMS shrinks rather than grows;
4. IoU@0.75 geometry does not collapse;
5. foreground probability mass follows GT lane count, especially on cross and
   no-line images.

Threshold tuning alone is not considered success. If NMS=20 still repairs a
large precision loss, the set contract remains incomplete even when one
threshold produces an attractive F1.

## 9. Commands

Architecture audit only:

```bash
AUDIT_ONLY=1 \
bash scripts/run_culane_dla34_unified_lane_set_v3_fromscratch_25k.sh
```

Train/resume to 25k:

```bash
DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 GRAD_ACCUM=4 AUTO_RESUME=1 \
bash scripts/run_culane_dla34_unified_lane_set_v3_fromscratch_25k.sh
```

One-inference validation sweep plus a cached NMS=20 dependency audit:

```bash
DATA_ROOT=/workspace/CULane \
bash scripts/evaluate_culane_dla34_unified_lane_set_v3_25k_val.sh
```

Use `best_nmsfree.score_threshold` from the resulting `summary.json` for the
single official test. NMS-free full test after the gate:

```bash
DATA_ROOT=/workspace/CULane SCORE_THRESH=0.30 \
bash scripts/eval_culane_dla34_unified_lane_set_v3_full_test.sh
```
