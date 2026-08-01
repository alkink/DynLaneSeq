# LaneRowNet Row-Reference Redesign: History, Diagnosis, and Unified Selection Contract

**Date:** 2026-08-01  
**Status:** implemented short-gate candidate and experiment record  
**Preserved implementation:** `experiment_rowref_hybrid_primary_aux`  
**Redesign implementation:** `experiment_rowref_unified_selection_v2`

This document records what replaced the historical LaneRowNet training
contract, what each intervention established, why the current row-reference
model is internally inconsistent, and the exact short experiment required
before another full CULane run. It complements
`docs/DLA34_CEILING_DIAGNOSIS_2026-07-27.md`; it does not overwrite the older
diagnostic history.

The central conclusion is:

> The row-reference decoder has demonstrated useful proposal geometry, but it
> is trained and deployed through a scoring/assignment contract inherited from
> the old grouped one-to-many model. The resulting existence and quality heads
> mostly learn assignment identity rather than final geometric usefulness.

## 0. Old-to-new architecture map

| Part | Historical paper-era model | Current redesigned gate | Evidence/status |
|---|---|---|---|
| Backbone | ResNet/DLA encoder | DLA-34 | unchanged and considered sound |
| Neck/evidence | fused FPN P2 | fused FPN P2 | retained after P2-vs-C2 probes |
| Candidate topology | 4 isolated groups x 8 | 32 global primary + 3 train-only groups x 8 | primary is the deployable set; auxiliaries are geometry-only |
| Initial row geometry | learned instance + row states, global row evidence | image-conditioned row reference | row-reference gave a positive proposal-recall signal |
| Decoder evidence | full-row cross-attention | seven local samples around the evolving curve | coarse-to-fine acquisition retained |
| Decoder interactions | same-row group attention + same-lane vertical attention | same interactions; primary group spans all 32 deployable candidates | axis-factorized core retained |
| Lane pooling | mean + raw max + instance residual | mean + instance residual | removes unmasked `amax` shortcut in the new arm only |
| Primary matching | grouped one-to-many with confidence term | global Hungarian on range-aware 30-pixel row-strip IoU | confidence is absent from the new assignment |
| Intermediate matching | independently rematched at each layer | final ownership reused at every intermediate layer | prevents within-step identity permutations |
| Localization losses | point, range, LineIoU, DFL, deep supervision | same geometric losses, radius 15 | retained |
| Auxiliary classification | existence and quality on every group | none | avoids conflicting positive priors in shared deployable heads |
| Final score | existence x quality^power | one set-aware `selection_logit` | legacy product is not consumed by new inference |
| Score evidence | pooled lane query | range-masked states, distribution confidence, final curve, exact-curve P2, and all other candidates | directly observes the final decision surface |
| Score target | matcher membership duplicated in existence and quality | selected target `0.5 + 0.5*IoU`, unmatched zero | same final matcher pairs are reused |
| Post-processing | threshold + lane NMS + Top-4 | unified threshold + Top-4; NMS disabled | low NMS dependence is an explicit gate |

The old code path remains the default whenever the new config keys are absent.
The redesign is opt-in and lives on its own branch/config, so historical
checkpoints and paper experiments remain reproducible.

## 1. Historical coherent model

The paper-era structured model used the following flow:

```text
P2 visual evidence
  -> 32 lane candidates arranged as four isolated groups of eight
  -> four structured decoder blocks
  -> row distributions, range, existence, and quality
  -> score = P(lane) * sigmoid(quality)^0.50
  -> threshold 0.30
  -> lane NMS
  -> Top-4 lanes
```

Its important training properties were:

- `matcher.assignment = grouped_one_to_many`;
- every GT lane was matched once inside each of four groups;
- a typical four-lane image therefore produced roughly 16 positive candidates
  among 32 outputs;
- unmatched candidates received existence target `background` and quality
  target `0`;
- four intentionally duplicated predictions were removed by NMS;
- the existence-quality product and the `quality=0.50, threshold=0.30`
  operating point were calibrated to this repeated-positive distribution.

This contract was not elegant and was NMS-dependent, but its components agreed
with one another. The four groups, the positive prior, the quality targets, and
the deployment NMS all described the same task.

## 2. What was changed after the historical model

### 2.1 Radius-15 and deep supervision

The line-IoU radius was corrected from `7.5` to `15.0` for the 30-pixel CULane
protocol. Three intermediate decoder outputs were supervised with weights
`[1, 2, 3]`, normalized and multiplied by `lambda_intermediate=0.5`.
Smoothness supervision was disabled.

This corrected the geometric scale and improved early decoder learning, but it
did not change the final existence-quality contract.

### 2.2 Bounded matcher confidence

The matcher object term was changed from `-log(p_lane)` to bounded `-p_lane`.
The mature-checkpoint counterfactual showed that this is sound cleanup but not
the ceiling solution. Later experiments reduced `lambda_obj` to `0.5` and also
tested removing confidence from matching.

### 2.3 Backbone/FPN investigation

Raw DLA C2 looked more locally distinct than fused P2 under cosine distance,
but equal-capacity residual probes showed that fused P2 was more predictive of
the required lane correction. Raw-C2 bypasses, larger generic FPN changes, and
feature-only refinement hypotheses did not receive sufficient causal support.

The current evidence does **not** identify the backbone or FPN as the primary
failure boundary.

### 2.4 Train-many/infer-one

The four historical groups were shown to be near replicas. Keeping only one
group at inference could retain competitive validation F1 and remove most NMS
dependence, but it did not create additional lane hypotheses.

This established that group replication was a deployment inefficiency, not a
coverage solution.

### 2.5 Row-reference acquisition

The global row search was replaced by an explicit coarse-to-fine curve
reference:

```text
image-conditioned initial x reference
  -> sample seven P2 offsets around each lane-row reference
  -> update row state
  -> predict a new row distribution
  -> use expected x as the next decoder block's reference
```

The final distribution also receives a Gaussian prior centered on the current
reference. This intervention produced a large early raw-proposal signal. In
the matched 10k diagnostic, row-reference proposal recall substantially
exceeded the global-row control. The acquisition mechanism is therefore a
proven positive component and should be retained.

### 2.6 Global one-to-one primary queries

The deployable candidate set changed from four isolated groups to one global
32-query Hungarian set:

- `num_groups = 1`;
- `matcher.assignment = hungarian`;
- approximately one positive query per GT lane;
- only about 10--12% of the 32 primary candidates are positive on a typical
  image.

This is a fundamentally different class prior and ownership problem from the
historical four-group model.

### 2.7 Learning-rate and cooldown experiments

The row-reference model learned quickly, peaked early, and degraded under the
original long schedule. Evidence-specific learning rates, LR floors,
continuations from 50k/55k/65k, and selective cooldowns were tested. These
experiments improved individual checkpoints, but none repaired the selection
contract. The early plateau must not be interpreted as proof that the decoder
has reached its geometric ceiling; a downstream ranking failure can conceal
useful proposals while training continues.

### 2.8 Matcher ownership experiments

Counterfactual assignment analysis showed that reducing or removing the
object-confidence term selected better geometry. A from-scratch
`lambda_obj=0.5` candidate produced a modest F1 trade-off, not a complete
solution. This established that confidence lock-in contributes to ownership
errors, but matcher weighting alone cannot make the score head understand
final geometry.

### 2.9 Late/frozen quality and set-selection probes

The following were tested on frozen or mature row-reference checkpoints:

- all-proposal quality targets;
- unique-Hungarian quality targets;
- official-raster set-selection targets;
- a learned set-aware selector;
- query-only, geometry-summary, and balanced scoring heads;
- curve-aligned visual verification;
- official-set rescoring and threshold-free ranking analyses.

Most gates were negative or too small. This does **not** show that set-aware
selection is unnecessary. It shows that a small head bolted onto a frozen
representation cannot undo an incompatible assignment and scoring history.

### 2.10 Hybrid primary 32 + auxiliary 3x8 model

The latest experiment retained all 32 global one-to-one primary candidates and
added three isolated train-only groups of eight. The auxiliary groups share the
decoder and prediction heads but are removed at inference.

The intention was sensible: obtain more positive geometry gradients without
reducing the deployable set. The implementation, however, also sent auxiliary
existence and quality losses through the deployed shared heads. Consequently:

- primary queries train the shared score heads with roughly 10--12% positives;
- auxiliary grouped queries train the same heads with roughly 40--50%
  positives;
- the shared heads see different interaction topology and different label
  priors;
- inference contains only the sparse-primary population.

The hybrid model improved raw proposal capacity, but its deployed calibration
and ranking became less coherent.

## 3. The exact current scoring problem

### 3.1 Quality is not a pure geometry target

The current quality loss starts with zero for every candidate and writes an IoU
target only at matcher-selected indices:

```python
target_quality = torch.zeros_like(quality_logits)
target_quality[matched_prediction] = matched_row_strip_iou
```

A high-IoU duplicate that loses the one-to-one Hungarian tie receives quality
target `0`. Therefore quality means:

```text
matcher membership x matched row-strip IoU
```

rather than physical lane quality.

Existence uses the same assignment membership. The deployed score then
multiplies two strongly redundant signals:

```text
score = existence * quality^q
```

Lowering `q` can weaken the second penalty but cannot correct the ordering.

### 3.2 Threshold drift is a symptom, not the complete diagnosis

Changing radius, assignment, and positive prior can legitimately move the
numeric score scale; therefore the fact that the historical
`q=0.50, threshold=0.30` point no longer works is not sufficient by itself.

Threshold-free diagnostics make the stronger case. On the same uniform
validation sample at IoU 0.50:

| Selection strategy | Row-reference recall |
|---|---:|
| All valid 32 candidates | 94.83% |
| Oracle Top-4 | 94.83% |
| Existence Top-4 | 64.94% |
| Quality Top-4 | 64.37% |
| Existence x quality^0.25 Top-4 | 64.37% |
| Existence x quality^0.50 Top-4 | 64.60% |

At IoU 0.75, the candidate oracle reaches `73.56%` while deployed ranking is
approximately `49--50%`. Changing quality power barely changes which proposals
are selected.

This is a ranking/selection failure, not merely scalar miscalibration.

### 3.3 NMS is rescuing the wrong ranking

At IoU 0.50, NMS adds only about `0.8` recall points to the historical/base
ranking, but roughly `12.6` points to the row-reference candidate. NMS is no
longer just removing intentionally repeated group outputs; it is compensating
for a score head that fills Top-4 with redundant candidates.

### 3.4 Good hidden candidates were positively assigned

The 25k assignment trace found that, among useful candidates hidden by the
deployed score:

- `93.55%` at IoU 0.50 were final-layer matcher positives;
- `96.24%` at IoU 0.75 were final-layer matcher positives;
- none of the IoU-0.50 hidden candidates were never matched.

The decoder found them and the matcher supervised them. The final score still
failed to expose them. The diagnostic classification is
`score_learning_failure_after_positive_assignment`.

### 3.5 Assignment and official-compatible selection disagree

On 221 GT lanes at 70k:

- the training matcher and official-compatible unique selection disagreed on
  96 lanes (`43.44%`);
- official-compatible selection won 71 disagreements;
- the configured matcher won none;
- mean assigned official IoU rose from `0.7418` to `0.7836`;
- selection recovered 10 IoU-0.50 hits and 21 IoU-0.70 hits with no reverse
  losses.

The corresponding shared row-state gradients were negative on `62.5%` of the
sampled micro-batches. Separate matcher and selector objectives therefore push
shared states in conflicting directions.

### 3.6 The score head does not directly observe the final decision

The legacy lane query is pooled as:

```text
mean(all row states) + max(all row states) + instance residual
```

The final row logits are subsequently modified by a reference-centered bias.
The quality head does not directly consume:

- the final x curve;
- row-distribution entropy or peak confidence;
- reference-to-final correction magnitude;
- predicted visible range masking;
- final curve-aligned P2 evidence;
- the other candidates that it must suppress as duplicates.

Raw `amax` may also select an irrelevant row outside the lane's visible range.
This is a strongly suspected interface weakness, although it is not by itself
proven as the sole root cause.

## 4. What is currently considered sound

The following components have positive evidence and remain in the redesign:

1. DLA-34 backbone wiring and pretrained initialization.
2. Fused P2 as the main evidence surface.
3. Explicit lane-row states.
4. Same-row inter-candidate and same-lane vertical interactions.
5. Row distributions and DFL localization.
6. Radius-15 LineIoU scale.
7. Row-reference coarse-to-fine P2 acquisition.
8. Final one-to-one deployable candidate set.

The failure boundary is downstream of proposal acquisition and upstream of
final Top-4 selection.

## 5. Unified-selection redesign

The new contract is:

```text
P2
  -> image-derived row references
  -> four reference-guided row blocks
  -> final x distributions + predicted range
  -> visibility/range-masked row-state and final P2 pooling
  -> set-aware interaction over all 32 primary candidates
  -> one unified selection logit
  -> threshold + Top-4 (no mandatory lane NMS)
```

### 5.1 One official-compatible assignment

The final primary matcher uses a range-aware row-strip IoU surrogate with
30-pixel line width. It does not include the current prediction score. The same
matched pairs supervise:

- point geometry;
- range;
- LineIoU;
- DFL;
- final unified selection.

Intermediate primary layers reuse the final assignment instead of rematching
an independent ownership identity at every block.

### 5.2 One deployed score

The final model does not deploy `existence * quality^q`. It predicts one
permutation-equivariant `selection_logit`. A unique matcher-selected proposal
receives target `0.5 + 0.5 * range_aware_IoU`; unmatched proposals receive
zero. The positive floor prevents an all-background collapse before
from-scratch geometry has nonzero overlap, while the continuous term raises
well-localized selections toward one. A quality-focal term and an in-set
pairwise ranking term train this score.

There is no `quality_power` in the redesigned deployment path.

### 5.3 Set-aware exact-decision features

The selector observes:

- range-masked row states;
- confidence-weighted row states;
- final curve-aligned P2 evidence;
- final x samples;
- row-distribution confidence;
- predicted range and range length;
- slope and curvature summaries;
- reference-to-final correction statistics;
- every other primary candidate through a small set transformer.

It does not start as a residual correction to the legacy
existence-quality product.

### 5.4 Geometry-only auxiliary queries

The three train-only 8-query groups remain available as a coverage/geometry
regularizer, but receive only:

- point loss;
- range loss;
- LineIoU loss;
- row DFL;
- geometry-only deep supervision.

They do not train the deployed unified selector and do not send existence or
quality labels through shared score heads.

### 5.5 Geometry-only deep supervision

Intermediate decoder outputs receive geometry losses only. Selection is a
final-set decision and is supervised only at the final primary output.

## 6. Predeclared 10k causal gate

The redesign must pass a short from-scratch gate before any 278k run.
Checkpoints are evaluated at `2.5k`, `5k`, `7.5k`, and `10k` on the same
uniformly sampled validation images.

Required signals:

1. raw 32-candidate recall may fall by at most 2 points relative to the
   row-reference control;
2. no-NMS unified-score Top-4 recall must improve by at least 8 points at IoU
   0.50 and 5 points at strict IoU;
3. the oracle-to-ranked Top-4 gap must shrink by at least one third;
4. NMS may add at most 3 recall points at IoU 0.50;
5. the result must not depend on any quality-power exponent;
6. full validation is allowed only after these fixed-subset gates pass.

This short run is a falsification experiment. Passing it supports a full
schedule; it is not itself a benchmark result.

### 6.1 Implemented files

- shared matcher/selector geometry:
  `dynlaneseq_eg/losses/range_aware_iou.py`;
- matcher cost and stable ownership:
  `dynlaneseq_eg/losses/matcher_s0.py` and
  `dynlaneseq_eg/engine/train_one_epoch.py`;
- unified target/loss:
  `dynlaneseq_eg/losses/loss_s0.py`;
- final-curve P2 sampling and the standalone set scorer:
  `dynlaneseq_eg/modeling/structured_queries.py`;
- preserved matched control:
  `dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_hybrid_control_gate_10k.yaml`;
- redesigned candidate:
  `dynlaneseq_eg/configs/culane_s0_structured_query_dla34_rowref_unified_selection_gate_10k.yaml`;
- one-command train/evaluate entry point:
  `scripts/run_and_evaluate_culane_dla34_rowref_unified_selection_gate_10k.sh`;
- threshold-free trajectory evaluator:
  `scripts/evaluate_culane_dla34_rowref_unified_selection_gate_10k.sh`.

The candidate retains legacy existence and quality modules only so historical
checkpoints and output dictionaries remain loadable. Their loss weights are
zero and deployment uses only `selection_logits`.

## 7. Decision rules

- **Raw proposals fall, selector improves:** selection is over-regularizing
  shared geometry; detach or reduce selector-to-decoder gradients.
- **Raw proposals remain strong, selector does not improve:** the set head
  still lacks separable final evidence; inspect curve-evidence and ownership
  features before full training.
- **Selector improves only with NMS:** uniqueness learning failed; do not hide
  the failure with post-processing.
- **Selector improves without NMS and the oracle gap shrinks:** proceed to a
  full schedule with validation-selected stopping and a schedule appropriate
  to the observed early learning rate.

## 8. Claims that are not yet justified

- The redesign is not guaranteed to exceed 80 F1.
- The backbone/FPN is not proven perfect; it is only not the current primary
  failure boundary.
- Raw `amax` pooling and missing final visual verification are plausible
  contributors, not independently proven root causes.
- A different numeric score threshold in the unified model is not itself a
  failure. Stable ranking, a broad validation optimum, and low NMS dependence
  are the relevant calibration criteria.

## 9. Primary diagnostic artifacts

The conclusions above are backed by these local artifacts when available:

- `outputs/diagnostics/paired_threshold_free_ranking.json`
- `outputs/diagnostics/paired_query_ownership.json`
- `outputs/diagnostics/rowref_object0p5_fromscratch_25k_assignment_score_trace.json`
- `outputs/diagnostics/dla34_joint_score_recovery_fromscratch_25k_uniform256.json`
- `outputs/diagnostics/dla34_joint_score_recovery_75k_uniform256.json`
- `outputs/diagnostics/dla34_joint_selection_assignment_gradient_conflict_70k_uniform64.json`
- `outputs/diagnostics/dla34_matcher_cost_counterfactual_70k_uniform64.json`
- `outputs/diagnostics/row_reference_quality_rescoring_65k_uniform256.json`
- `outputs/diagnostics/row_reference_official_set_selection_65k_uniform256.json`
- `outputs/diagnostics/dla34_rowref_object0p5_official_set_selection_75k_uniform256.json`
- `outputs/diagnostics/dla34_rowref_object0p5_curve_visual_verification_75k_uniform256.json`
- hybrid full-validation summaries under `outputs/diagnostics/summary.json`

Never infer a benchmark claim from the 64/256-image diagnostic subsets. Their
purpose is to locate the failing interface and reject unsupported full runs.
