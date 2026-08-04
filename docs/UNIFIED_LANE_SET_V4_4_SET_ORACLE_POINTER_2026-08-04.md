# Unified Lane-Set V4.4: Set-Oracle Pointer Acquisition

## Evidence that selects this intervention

The V4.3 full-validation audit showed that representative ordering accounts
for 95.95% of the recoverable IoU@0.75 gap.  The subsequent uniform-256
target-alignment audit ruled out the geometry target itself:

| Measurement | Result |
| --- | ---: |
| Row-strip / official pairwise Pearson | 0.9396 |
| Row-strip / official best-candidate agreement | 94.60% |
| Training-target TP / official oracle TP @0.75 | **716 / 716** |
| Pointer TP / training-target TP @0.75 | **512 / 716** |
| Pointer / teacher target-set Jaccard | 0.181 |

The rollout audit then localized the failure before exposure:

| Measurement | Result |
| --- | ---: |
| First-step fixed leftmost target | 20.25% |
| First-step any valid target representative | 21.10% |
| Ordering-only difference | 0.84 points |
| Failed unordered trajectories beginning at step 1 | **80.85%** |
| First-step target median rank | 3 |
| First-step mean target margin | -0.528 |

Therefore neither a new IoU surrogate nor scheduled sampling alone is the
correct next action.  V4.3 fails to acquire a unique high-quality
representative before it has consumed any model-generated history.

## V4.3 gradient conflict

V4.3 uses two incompatible candidate contracts:

```text
continuous unary target:
    every high-IoU duplicate -> high score

fixed pointer sequence:
    one Hungarian representative -> positive at one exact step
    every other representative and duplicate -> competing class
```

At the first step, the candidate logit is

```text
unary quality + pointer content + zero relation history
```

so relation-aware suppression cannot repair an incorrect first decision.

## V4.4 training graph

Geometry and visual evidence remain frozen and detached exactly as in V4.3:

```text
V4 bounded-delta geometry
        |
        +-- final curves / ranges / row states / P2-P5 evidence
        |
        X  stop-gradient
        |
        v
candidate encoder + pointer
```

Only the target contract changes.

### 1. Unique representative unary supervision

The already validated score-independent Hungarian assignment produces a
binary mask over the unique representative set.  Only those candidates retain
their continuous IoU target; unmatched duplicates receive target zero.
Positive and negative candidate losses are balanced per image, preventing 28
background slots from overwhelming roughly 3--4 representatives.

### 2. Permutation-invariant remaining-set loss

At pointer step `t`, let `T_t` be the unselected Hungarian representative set.
Instead of supervising one arbitrary left-to-right class, V4.4 minimizes

```text
L_t = -log sum(i in T_t) p(i | state_t)
```

When `T_t` is empty, STOP is the only valid class.  The model's highest-logit
member of `T_t` is used as the dynamic teacher state and removed.  This keeps
training on a valid trajectory while aligning the state order with the
model's own preference.

Inference is unchanged: greedy pointer, learned STOP, at most four lanes, no
threshold, no NMS and no quality-score multiplication.

## Controlled gate

Exactly one scorer is trained from the frozen V4 50k geometry checkpoint:

```bash
DATA_ROOT=/workspace/CULane \
SOURCE_CHECKPOINT=outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt \
SEEDS="3407" \
TRAIN_STEPS=15000 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
AMP_DTYPE=bfloat16 \
bash scripts/run_culane_dla34_unified_lane_set_v4_4_set_oracle_pointer_gate_50k.sh
```

The runner writes one compact final checkpoint, evaluates uniform-256, and
reruns target/rollout alignment from the generated cache.

Primary outputs:

```text
outputs/diagnostics/unified_lane_set_v4_4_set_oracle_pointer_gate_50k/
  seed_3407/set_oracle_pointer/iter_0065000.pt
  seed_3407/reports/summary.json
  seed_3407/reports/target_alignment_uniform256.json
  seed_3407/reports/v4_4_summary.json
```

## Gate

Geometry oracle must remain bitwise/tolerance-equivalent to V4.  In addition:

| Selection measurement | Required direction |
| --- | ---: |
| Direct recall @0.50 | at least 75%, preferably above V4.3's 79.66% |
| Direct recall @0.75 | at least 60%, preferably above 65% |
| First-step any target representative | materially above V4.3's 21.10% |
| Pointer/target set Jaccard | materially above V4.3's 0.181 |
| Exact target-set rate | materially above V4.3's 8.20% |
| Variable cardinality / STOP | retained |

Only a passing uniform gate authorizes full validation.  Test remains sealed
until the validation contract is selected.
