# V34 Temporal Candidate Observability Results

Date: 2026-08-24

## Decision

The predeclared temporal observability gate failed in every fold and at both
official IoU thresholds.

This was not a weak null result. Adjacent-frame deployed V7 lanes systematically
supported the currently selected wrong proposal more strongly than the
oracle-good proposal. At the same time, a diagnostic adjacent-frame GT check
strongly supported the oracle-good proposal. The physical lane is temporally
consistent; V7's wrong geometric belief is also temporally persistent.

Therefore, temporal aggregation of frozen V7 proposals, route scores, or deployed
lanes must not be developed as the next rescue mechanism.

## What was tested

- Frozen mature V7 checkpoint at iteration 225,000.
- CULane validation only; test remained closed.
- 54 validation clips split deterministically into two disjoint 27-clip folds.
- 512 target frames per fold, each with listed previous and following frames at
  the 30-frame CULane interval.
- 3,072 unique target/context images cached once.
- 980 unique route-recoverable good-versus-current-wrong candidate pairs.
- GT was used only to form diagnostic pairs. It did not enter the primary
  temporal score.
- Primary score: target proposal warped with forward/backward-checked DIS optical
  flow, then aligned to the active refined V7 lanes in both adjacent frames.
- Controls: previous-only, following-only, no-flow identity, all-32 adjacent
  proposal bank, and unrelated same-fold wrong-clip context.

The exact protocol and thresholds were committed before the full result in
`V34_TEMPORAL_OBSERVABILITY_CONTRACT_20260824.md`.

## Primary frozen-V7 temporal result

Higher scores are supposed to identify the oracle-good proposal. An AUC of 0.50
is chance; values below 0.50 mean that the score prefers the current wrong
proposal.

| Fold | IoU | Eligible pairs | Scored pairs | Good-over-wrong accuracy | AUC | Clip-bootstrap AUC 95% CI | Wrong-clip AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | .50 | 229 | 197 | 0.348 | 0.333 | 0.266–0.408 | 0.520 |
| A | .75 | 423 | 369 | 0.280 | 0.319 | 0.267–0.370 | 0.495 |
| B | .50 | 195 | 175 | 0.251 | 0.269 | 0.206–0.338 | 0.489 |
| B | .75 | 366 | 344 | 0.273 | 0.313 | 0.259–0.374 | 0.495 |

Every primary confidence interval is entirely below 0.50. The result replicates
across two disjoint clip populations and both official thresholds.

The unrelated wrong-clip control is approximately chance. Therefore, the inverse
signal is not explained by a generic candidate-position shortcut. The correct
adjacent sequence specifically reinforces V7's current wrong mode.

## Does the adjacent proposal bank contain the answer?

| Fold | IoU | Adjacent all-32 bank AUC | Mean good support | Mean wrong support |
| --- | ---: | ---: | ---: | ---: |
| A | .50 | 0.444 | 0.553 | 0.584 |
| A | .75 | 0.459 | 0.654 | 0.675 |
| B | .50 | 0.434 | 0.585 | 0.619 |
| B | .75 | 0.472 | 0.678 | 0.689 |

The adjacent all-32 banks are much less harmful than the deployed adjacent lanes,
but they still do not identify the good proposal. Both geometric modes are
available, and the wrong mode remains at least as well supported.

This is the temporal version of the existing support-belief split:

1. Correct geometry is present.
2. The bank does not express which geometry deserves belief.
3. Final V7 selection consistently commits to the wrong physical structure.

## Optical-flow and label sanity check

After the primary gate had already failed, a diagnostic-only adjacent-GT check
was added without changing the gate. It asks whether the same warped target
proposal aligns with the annotated lanes in adjacent frames.

| Fold | IoU | Adjacent-GT pair accuracy | Adjacent-GT AUC | Clip-bootstrap AUC 95% CI |
| --- | ---: | ---: | ---: | ---: |
| A | .50 | 0.870 | 0.853 | 0.785–0.901 |
| A | .75 | 0.789 | 0.739 | 0.699–0.776 |
| B | .50 | 0.869 | 0.863 | 0.801–0.912 |
| B | .75 | 0.817 | 0.763 | 0.705–0.822 |

This rules out the most important implementation-level alternative explanation:
the flow proxy is not simply mapping the good proposal to the wrong location.
The oracle-good proposal remains geometrically consistent with adjacent
annotations. V7 predictions, not scene geometry, reverse the preference.

The no-flow adjacent-GT AUC is similarly strong (0.837, 0.762, 0.837, 0.766).
The listed frames are often close enough in image coordinates that optical flow
is not essential to the diagnostic conclusion.

## What the result means

### Supported by evidence

1. **The correct proposal is not merely arbitrary one-frame geometry.** It aligns
   with the annotated lane in adjacent frames.
2. **V7's wrong belief is temporally stable.** The same wrong physical structure
   attracts the model in previous and following frames.
3. **Frozen temporal aggregation would amplify the error.** Positive consistency
   rewards the wrong candidate in roughly 65–75% of scored recoverable pairs.
4. **The all-32 bank does not solve observability.** Adjacent support remains
   ambiguous even before the final route decision.
5. **The core issue is deeper than a detached scalar head.** The detector's learned
   visual semantics repeatedly prefer the wrong but stable structure.

### Not proved

1. This does not prove that a new end-to-end video model trained on raw frames
   cannot learn the correct lane.
2. The adjacent-GT score is diagnostic and unavailable at deployment; it is not a
   temporal solution.
3. Inverting temporal consistency is not a justified deploy policy. It would mean
   preferring unstable proposals and has not passed a protection audit on current
   true positives.
4. This result does not prove that every remaining error has the same cause.

## Relation to earlier experiments

- Frozen rerankers and pair-corridor probes failed because the existing
  representation did not expose reliable fine winner evidence.
- V28/V29 already gave a separate trainable image-belief tower a route-gradient
  path, but it failed on OOF images.
- V30/V31 temporarily improved early training, then lost the gain by 50K.
- V33 private-head gradient surgery repaired gradient conflict but did not improve
  official F1.
- V34 now shows that simply adding adjacent frozen V7 evidence is worse than
  insufficient: it reinforces the same wrong mode.

Together, these results close the following family as the main direction:

```text
frozen V7 selector / reranker
pair-corridor or mode-member scorer on frozen V7 features
AGF-style shallow score FPN over detached V7 representation
temporal voting or positive consistency over frozen V7 predictions
```

## Recommended next experiment

Do not build a temporal selector and do not revisit SPLIT/AGF with another scoring
head. The next experiment must change the representation that creates geometric
belief, not only read or aggregate that belief.

The lowest-risk next gate is a **raw-image semantic falsification audit**:

1. Form the same good/current-wrong proposal pairs.
2. Sample raw RGB ribbons and broad road context, not V7 features or scores.
3. Include target, previous, and following images, but do not include adjacent V7
   predictions.
4. Train and evaluate with support-model OOF proposal populations and clip-disjoint
   directions.
5. Compare a target-only raw-image arm with a three-frame raw-image arm.

This answers the remaining causal question:

> Is the correct annotation semantics learnable from raw visual evidence, or is
> the oracle-good proposal only identifiable because GT reveals the convention?

If both raw-image arms fail OOF, proposal selection should be closed completely.
The project should then move to a new generator/detector objective rather than
another V7 rescue module. If the three-frame raw-image arm passes while the
single-frame arm fails, only then is an end-to-end temporal detector justified.

## Reproducibility

- Branch: `codex/v34-temporal-observability-20260824`
- Gate implementation commit: `565e946`
- Target-only raster optimization: `68eb92e`
- Adjacent-GT sanity control: `559e2a2`
- Raw gate result: `outputs/diagnostics/v34_temporal_candidate_observability/v34_temporal_observability_gate_v1.json`
- GT-sanity result: `outputs/diagnostics/v34_temporal_candidate_observability/v34_temporal_observability_with_gt_sanity.json`
- Test set used: no
- Training performed: no
- Checkpoint/threshold selection performed: no

