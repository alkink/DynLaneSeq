# CondLSTR--MapTR--LaneRowNet Contract Audit

## Scope

This audit asks a larger question than whether one threshold, activation, or
loss weight is wrong:

> Does the current LaneRowNet training contract turn its strong curve
> candidates into a stable, uniquely scored lane set?

The comparison uses the checked-in CondLSTR and MapTR implementations under
`lanemodels/`, the historical grouped LaneRowNet configuration, the current
row-reference/unified-selector configuration, and the completed frozen
multi-scale evidence probe.  It distinguishes candidate geometry from set
detection; a model may be good at the former and fail at the latter.

## The common contract in CondLSTR and MapTR

CondLSTR and MapTR implement different tasks, but their detection contracts
share four properties.

| Property | CondLSTR | MapTR |
|---|---|---|
| Candidate identity | One persistent lane query | One vector instance plus ordered point queries |
| Assignment | Strict one-to-one Hungarian | Strict one-to-one vector assignment |
| Negative supervision | Every unmatched query is background | Every unmatched vector is background |
| Score/geometry coupling | Object score and query-conditioned dense curve come from the same query | Vector score and points come from the same decoder state |
| Decoder supervision | A fresh match and all losses at every returned decoder layer | A fresh match and class/point/direction losses at every decoder layer |
| Inference | Object score threshold; no lane NMS | Direct score Top-K; NMS-free coder |

### CondLSTR

CondLSTR does not ask a late scoring module to infer which anonymous curve is
useful.  Each query generates object logits, visible range, and the dynamic
parameters of a query-conditioned dense mask/regression map.  The curve and
its object score therefore share the same identity and image response.  Its
matcher uses bounded `-p` object cost plus row geometry, and its loss assigns
the background class to every unmatched query.  When intermediate decoder
outputs are returned, matching and the full loss are recomputed independently
for each layer.

Relevant implementation sites:

- `lanemodels/CondLSTR/.../cond_lstr_2d.py`: `_forward`, lines 238--260;
- `lanemodels/CondLSTR/.../matcher.py`: bounded object and row costs, lines 35--132;
- `lanemodels/CondLSTR/.../loss.py`: unmatched background, lines 37--47, and
  per-layer matching, lines 307--337;
- `lanemodels/CondLSTR/.../postprocess.py`: direct object-score filtering,
  lines 41--64.

### MapTR

MapTR explicitly combines a vector-instance embedding with ordered point
embeddings.  Every decoder layer predicts both the vector score and its point
geometry.  Every layer receives its own assignment and classification loss;
all unmatched vectors are background.  Reference points are updated inside
the decoder, then detached before becoming the next layer's sampling
reference.  This preserves iterative refinement without allowing a later
sampling operation to rewrite the earlier reference through an uncontrolled
gradient path.

Relevant implementation sites:

- `lanemodels/MapTR/.../maptr_head.py`: instance-plus-point queries, lines
  182--188 and 223--228; coupled class/point heads, lines 270--313;
- `lanemodels/MapTR/.../maptr_head.py`: unmatched background, lines 380--404,
  and losses on every decoder layer, lines 651--699;
- `lanemodels/MapTR/.../decoder.py`: iterative reference update and detach,
  lines 44--81;
- `lanemodels/MapTR/.../nms_free_coder.py`: score Top-K decoding, lines
  80--100.

## Historical LaneRowNet: limited, but internally coherent

The historical model used 32 candidates partitioned into four isolated groups
of eight.  Each ground-truth lane was deliberately matched once in every
group, so the training target created approximately four positive duplicates.
Existence and quality were trained directly, and inference multiplied those
scores, applied a threshold, ran lane NMS, and retained four lanes.

This was not a clean NMS-free set detector.  It had a clear ceiling and heavy
NMS dependence.  It was nevertheless internally coherent:

1. training intentionally produced duplicate hypotheses;
2. direct lane scores were trained on those hypotheses; and
3. inference explicitly removed the duplicates.

That coherence explains why the old model could be well calibrated around
`score=0.30`, `quality_power=0.50` even though its absolute ceiling remained.

## Current row-reference/unified-selector model: the contract break

The row-reference decoder improved candidate geometry.  The completed 10k
diagnostics show a validation candidate-oracle recall of **93.91% at IoU 0.50**
and **81.38% at IoU 0.70**.  The P2/FPN path and row decoder therefore do not
lack the raw capacity to draw useful curves.

However, the current configuration combines several incompatible choices:

1. The primary and auxiliary paths still produce a rich, duplicate-prone,
   permutation-symmetric candidate population.
2. Direct existence and quality supervision are disabled (`w_exist=0`,
   `w_quality=0`).
3. A separate set selector must convert those moving hypotheses into four
   unique lanes (`w_set_selection=1`).
4. Geometry features used by its explicit curve-evidence path are detached.
5. Intermediate supervision reuses the final-layer assignment instead of
   establishing a fresh layer-local identity contract.
6. NMS is removed even though the generator has not yet learned stable unique
   ownership.

This is the central incompatibility.  The decoder is being trained primarily
as a **curve proposal generator**, while inference asks a different module to
behave as the **set detector**.  CondLSTR and MapTR make the same query state
responsible for both jobs.

Two secondary design risks reinforce this failure:

- Lane-level decisions are produced late by pooling all image-row states.
  Many of those rows are outside the visible lane, so an unmasked mean can
  dilute foreground evidence.  CondLSTR preserves a persistent lane query;
  MapTR pools a fixed set of actual vector points rather than mostly invalid
  image rows.
- The current row-reference path assigns
  `reference_x = pred_x_rows` without a layer-boundary detach.  MapTR's
  explicit detach makes each refinement stage optimize its residual around a
  fixed previous reference.  This is not proven to be the main failure, but it
  is an unjustified difference that should be isolated before a final design.

## Evidence, not conjecture

The diagnosis is layered rather than absolute:

- **Proven sound enough:** frozen candidate geometry.  The oracle has over
  93% recall at IoU 0.50.
- **Proven unstable:** unique ownership.  Across the 2.5k-to-10k trajectory,
  only roughly one quarter of recoverable lanes retain the same owner query.
- **Proven weak in the present model:** deployable selection.  At 10k, raw
  selection remains far below the candidate oracle and NMS recovers a large
  part of the gap.
- **Not yet proven:** whether the first frozen evidence readout failed because
  the representation is insufficient or because 1,000 optimization steps
  were too short.

The 1k equal-capacity multi-scale probe is negative: correct P2 strips,
correct P3/P4 context, and their joint use do not reliably beat wrong-image or
zero-image controls on train and validation.  It cannot yet justify adding a
larger FPN or a curve sampler to the full model.  Its 1k duration is the last
material alternative explanation.

## Controlled 5k convergence test

`scripts/probe_culane_dla34_multiscale_evidence_convergence_5k.sh` repeats the
same frozen experiment from a clean initialization for 5,000 optimizer steps.
It deliberately does not resume the 1k probe because that checkpoint contains
no optimizer state; resetting AdamW would make a continuation scientifically
ambiguous.

Interpret the result before starting a full training run:

1. **Correct evidence separates on train and validation.**  The 1k probe was
   under-trained.  Joint evidence acquisition remains plausible, and only the
   evidence arm that beats all counterfactual controls should enter a full
   model.
2. **Correct evidence separates on train but not validation.**  The readout
   memorizes the cache.  More selector capacity or training is not the answer;
   the decoder must learn stable identities jointly.
3. **Correct evidence does not separate even on train.**  The present frozen
   interface is insufficient.  Stop post-hoc selector, quality, and threshold
   patches and redesign the primary query contract.

## Recommended redesign if the 5k gate is negative

The smallest coherent redesign is not a larger FPN.  It is a new primary
detection contract:

1. keep one deployable 32-query primary set with strict one-to-one matching;
2. keep any one-to-many groups train-only as geometry auxiliaries;
3. maintain one persistent lane token per primary query, updated from its own
   row states with range-aware attention rather than an unmasked mean;
4. predict a single foreground logit directly from that same lane token;
5. label every unmatched primary query as background at every decoder layer;
6. rematch independently at each decoder layer;
7. detach updated row references at decoder-layer boundaries;
8. use direct score Top-K without quality multiplication or NMS as the primary
   deployment gate.

This retains LaneRowNet's explicit lane--row representation, which the oracle
and geometry probes support, while replacing the incompatible late set
selection contract.  It is a substantial module-interface correction, not a
claim that one scalar loss adjustment will break the benchmark ceiling.

## Honest conclusion

The current evidence does **not** show that the entire decoder is bad, and it
does **not** identify FPN capacity as the main bottleneck.  It shows that the
decoder is a strong curve generator but has not been made into a stable unique
set detector.  The largest present problem is the assignment--identity--score
contract between decoder, supervision, and inference.  The 5k frozen test is
the final cheap check before committing compute to that larger redesign.
