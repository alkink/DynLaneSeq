# Unified Lane-Set V3: 25k–75k Collapse and the Matched Rescue Gate

Date: 2026-08-02

This record starts after the 25k Unified V3 checkpoint. It separates measured
facts from hypotheses and defines the next causal experiment. Frozen subset
audits are diagnostics, not CULane benchmark results.

## 1. What is proven

The 25k checkpoint contains useful lane geometry:

| Frozen uniform validation metric | 25k | 50k | 75k |
|---|---:|---:|---:|
| All-32 recall at IoU 0.50 | 95.29% | 0% | 0% |
| All-32 recall at IoU 0.75 | 75.86% | 0% | 0% |
| Matched mean best official IoU | 0.776 | 0.112 | 0.084 |
| Native assigned-row MAE | 7.39 px | 165.18 px | 249.34 px |

Lowering the score threshold, Oracle Top-4 selection, using all 32 candidates,
and row-logit temperature values from 0.5 to 16 do not recover 50k or 75k.
Therefore this is not a threshold, NMS, quality-score, Top-K, or softmax
temperature failure. The candidate geometry itself has disappeared.

The checkpoint chain is internally consistent:

- model, optimizer, and scheduler states are present;
- optimizer steps and scheduler phase agree with 25k, 50k, and 75k;
- the expanded model/config contract is unchanged;
- no NaN/Inf parameter corruption was found;
- learned instance tokens retain high effective rank;
- reference anchors retain candidate spread.

This rules out a simple wrong-checkpoint, random restart, scheduler restart,
parameter-level query collapse, or anchor-template collapse explanation.

## 2. Where the failure lives

Frozen module swaps establish direction of causality:

```text
healthy 25k encoder + failed 50k decoder  -> 0 recall
failed 50k encoder  + healthy 25k decoder -> partial recall
```

Thus the failed decoder is sufficient to kill geometry. Encoder/FPN drift adds
damage but is not the primary 50k cause.

The decoder failure is distributed rather than one bad scalar:

```text
failed lane-state core + healthy remainder -> almost zero recall
failed row readout    + healthy remainder  -> zero recall
healthy lane-state core alone in failed net -> no rescue
healthy row readout alone in failed net     -> no rescue
```

The lane-state path and shared row readout have co-adapted into an incompatible
coordinate representation. Calling this only a `row_norm` bug would be wrong.

## 3. The abnormal scale trajectory

The same `row_norm`/`row_x` readout exists in the older, healthy row-reference
decoder. Its checkpoint trajectory provides a direct control:

| Tensor L2 norm | Old row-reference 25k | Old 75k | V3 25k | V3 50k | V3 75k |
|---|---:|---:|---:|---:|---:|
| `row_norm.weight` | 18.76 | 23.08 | 25.07 | 63.30 | 75.89 |
| `row_norm.bias` | 0.59 | 1.27 | 1.33 | 32.02 | 43.09 |
| `row_x.weight` | 29.25 | 47.04 | 46.62 | 209.64 | 253.33 |
| `row_x.bias` | 1.07 | 1.33 | 1.61 | 11.35 | 13.04 |

The old model uses the same 2e-4 evidence LR, so weight growth by itself is not
automatically a bug. The magnitude and speed in V3, however, are architecture-
specific runaway behavior. Temperature failure also shows that the problem is
not merely a common logit scale: feature directions and x-bin ordering have
become wrong.

## 4. Best current mechanism (not yet final proof)

The evidence supports this chain:

```text
four persistent lane-state blocks + shared readout
                 |
                 v
independent per-layer Hungarian identities and foreground losses
                 +---- score enters matching
                 +---- cardinality/margin act on score mass/ranking
                 |
                 v
lane-state/readout co-adaptation at one 2e-4 evidence LR
                 |
                 v
row_norm and row_x scale/direction run away
                 |
                 v
predicted references move into wrong local corridors
                 |
                 v
later local sampling sees background instead of the lane
                 |
                 v
geometry becomes self-reinforcingly wrong
```

Only the first and last portions are directly measured. The initiating cause
could be the score/assignment contract, the optimizer scale of the new decoder,
or both. Persistent query-ID and global-acquisition changes are not licensed
yet: the current decoder already performs image-conditioned full-row initial
reference acquisition, and parameter-level identity spread did not collapse.

## 5. Why the next experiment has three arms

All arms start from the exact same 25k checkpoint, including AdamW moments and
the 278k cosine phase. Each runs exactly 5,000 optimizer steps on the same seed.

### Control

No model, matcher, loss, optimizer, or schedule change. This establishes whether
collapse is already visible by 30k in a freshly replayed continuation.

### Contract rescue

Architecture and optimizer remain exact. The intervention is:

```yaml
matcher.lambda_obj: 0.0
loss.w_intermediate_exist: 0.0
loss.w_cardinality: 0.0
loss.w_score_margin: 0.0
```

Final foreground supervision remains active. Intermediate point, range,
LineIoU, and DFL supervision remain active with independent geometry-only
Hungarian assignments. This tests whether score/assignment feedback initiates
the geometry drift.

### Scale rescue

The complete original objective remains exact. Only these base learning rates
change from 2e-4 to 5e-5:

```text
structured_query_head.lane_state_layers.*
structured_query_head.row_norm.*
structured_query_head.row_x.*
```

Per-parameter AdamW moments are remapped into the new groups. A newly built
scheduler is explicitly aligned to iteration 25,000, so this arm is neither an
optimizer restart nor a warmup restart.

## 6. Decision rule

```text
control healthy at 30k
  -> no winner yet; extend all arms to 35k/40k

control fails, contract survives
  -> factor matcher / intermediate-score / count-margin effects

control fails, scale survives
  -> extend scale arm and tune only isolated decoder LRs

control fails, both survive
  -> extend both separately; compare strict-IoU and norm slopes

all fail
  -> try the combined arm once; then test structural acquisition/identity changes
```

No from-scratch 278k model should be launched until one arm preserves geometry
beyond the point where the exact control begins to fail.

## 7. Reproduction entry point

```bash
DATA_ROOT=/workspace/CULane \
SOURCE_CHECKPOINT=outputs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k/iter_0025000.pt \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
bash scripts/run_culane_dla34_unified_lane_set_v3_rescue_25k_to30k.sh
```

The final diagnostic is written to:

```text
outputs/diagnostics/unified_lane_set_v3_rescue_25k_to30k/summary.json
```
