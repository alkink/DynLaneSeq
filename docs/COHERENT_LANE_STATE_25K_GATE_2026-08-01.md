# Coherent Primary Lane-State 25k Gate

## Decision

The next experiment stops the sequence of post-hoc quality, selector, and
threshold patches. It tests one architectural contract:

> One primary query identity must own its row geometry, persistent lane state,
> Hungarian target, foreground score, and visible range from training through
> deployment.

This is a 25,000-iteration validation gate, not a claim that the final CULane
ceiling has already been solved.

## Why this experiment exists

The completed diagnostics established the following:

- the row-reference decoder can generate a high-recall 32-curve candidate
  population;
- the useful candidates do not map to stable query owners across training;
- a frozen scalar selector, hierarchical selector, hard selector, and
  multi-scale frozen evidence readout do not generalize the candidate oracle
  advantage;
- five thousand frozen-readout steps strengthen train fit but make validation
  worse, so the failure is not explained by a 1k under-trained probe;
- CondLSTR and MapTR do not delegate geometry and set detection to unrelated
  identities. They train the same query/vector state with one-to-one matching,
  unmatched-background classification, layer-local supervision, and direct
  NMS-free scores.

The strongest remaining diagnosis is therefore the interface among decoder,
assignment, supervision, and inference—not a proven lack of FPN evidence.

## Candidate contract

The candidate uses:

1. **32 deployable primary queries only.** There are no train-only one-to-many
   groups in this first gate. This avoids mixing geometry augmentation with the
   identity question.
2. **Strict global one-to-one Hungarian assignment.** The bounded `-p`
   foreground cost has weight `0.5`.
3. **Independent matching at every decoder layer.** Intermediate layers do
   not inherit the final layer's permutation.
4. **A persistent lane state per query.** At each decoder block it attends
   only to the ordered row states of the same query, then carries that state to
   the next block. Existence and visible range are predicted from this state.
5. **Direct foreground supervision.** Every unmatched primary query is
   background. The separate quality and set-selection losses are disabled.
6. **Detached iterative row references.** A block's predicted curve is used as
   the next block's sampling reference after `detach`, while that block still
   receives its own geometry loss.
7. **NMS-free direct Top-4 deployment.** Ranking uses existence probability
   alone. Quality multiplication and the second selector are absent.

The persistent lane state reads row states but is not fed back as a new
geometry residual in this gate. This keeps the causal question narrow: can a
stable, image-conditioned lane identity learn the deployable foreground score
without adding another mechanism that can perturb good curves?

## Matched control

The control is the existing primary-only, row-reference,
`matcher.lambda_obj=0.5` from-scratch 25k run:

```text
dynlaneseq_eg/configs/
  culane_s0_structured_query_dla34_slots32_b4x4_1600x640_bins800_
  fpn256_l4_dfl_rowref_r15_deepsup_obj0p5_50ep.yaml
```

Both arms use DLA-34, 1600x640 input, 160 rows, 800 DFL bins, P2 evidence,
four decoder blocks, radius 15, deep supervision, seed 3407, effective batch
16, a 1k warmup, and the full 278k cosine scheduler horizon. The candidate is
stopped at 25k; the learning-rate schedule is not compressed to 25k.

## Predeclared evaluation

The gate uses a fixed uniform 256-image CULane validation subset and exact
30-pixel official-raster IoU. The test split is never read; score calibration
is restricted to the predeclared finite grid below.

At IoU 0.50 and 0.75 it reports:

- all-32 candidate capacity;
- oracle Top-4 capacity;
- raw direct-existence Top-4 recall;
- hypothetical 20-pixel NMS gain over raw Top-4;
- the oracle-to-direct-Top-4 gap; and
- training-owner retention over the 5k, 10k, 15k, 20k, and 25k checkpoints.

It also evaluates the direct-existence, NMS-free set on the fixed score grid
`[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`. The best validation
threshold is selected by F1@0.50, with F1@0.75, precision, and then the higher
threshold as tie-breakers. This cached pass performs no additional model
inference.

The predeclared verdict is:

- **continue** when best NMS-free F1@0.50 improves by at least 0.5 point,
  F1@0.75 loses no more than 0.5 point, mean direct Top-4 recall improves by at
  least 0.5 point, neither capacity threshold loses more than 0.5 point, and
  recoverable-query owner retention is at least 0.75;
- **longer gate warranted** when NMS-free F1@0.50 is non-negative, direct
  Top-4 improves by at least 0.25 point, capacity is safe, and the NMS or
  oracle gap measurably shrinks;
- **stop** when both mean candidate capacity and direct Top-4 fall by at least
  1 point;
- otherwise **mixed/inconclusive**.

## Commands

Train and diagnose in one run:

```bash
DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
MAX_BATCHES=64 \
bash scripts/run_and_evaluate_culane_dla34_coherent_lane_state_25k.sh
```

Contract audit without training:

```bash
AUDIT_ONLY=1 \
bash scripts/run_culane_dla34_coherent_lane_state_fromscratch_25k.sh
```

Primary diagnostic output:

```text
outputs/diagnostics/coherent_lane_state_25k_gate/summary.json
```

## Scope limits

A positive 25k result licenses longer training; it does not guarantee 80+ F1.
A negative result rejects this exact persistent-state/detached-reference/direct
score contract. It does not prove that every decoder or FPN redesign must
fail. Full validation and then a single frozen full-test setting are required
before any benchmark claim.
