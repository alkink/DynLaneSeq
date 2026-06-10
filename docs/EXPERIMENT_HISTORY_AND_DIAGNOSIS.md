# DynLaneSeq Experiment History and Diagnosis

This document summarizes the main experiments, design turns, failures, and current open questions in the DynLaneSeq CULane work. It is intentionally blunt. Its purpose is to prevent repeating old experiments and to make the next architectural decision clearer.

## 1. Core Pipeline Mental Model

DynLaneSeq is a staged lane detector:

```text
image -> backbone/FPN -> queries/slots -> S0 -> S1 -> S2 -> S3
```

### Backbone/FPN

The backbone and FPN produce the visual feature map. This is the model's visual evidence.

### Queries / Slots

The model uses a fixed number of lane candidates, usually `num_slots=20`. Each slot is a possible lane. The slots are not direct ground-truth assignments. Assignment is done during training by Hungarian matching.

### S0

S0 is the coarse lane proposal stage. It predicts:

- whether each slot is a lane (`exist_logits`)
- row-wise x coordinates (`pred_x_rows` / `row_x_logits`)
- valid vertical range (`range_norm`)
- optional quality/logits depending on the variant

S0 is the first gatekeeper. If S0 gives a bad existence score or poor coarse geometry, later stages have to work against that.

### S1

S1 is a row-wise refinement stage. In residual mode, it starts from S0's coarse output and predicts row-level residual corrections. It is not a query-assignment module. It helps the row geometry become more sequence-consistent.

### S2

S2 samples visual evidence along the coarse/predicted lane geometry and uses that evidence to refine row coordinates. It is the first stage that strongly ties lane geometry to local image features.

### S3

S3 adds final decision / quality calibration behavior. In the current best version, S3 uses Active Corridor and QualityCal. This stage has been the biggest practical source of improvement because it reduces false positives and calibrates final scores.

## 2. Full-Train Milestones

Important full CULane milestones reported during the project:

```text
S0 full train:                  ~72.83 F1
S1 residual full train:         ~73.43 F1
S2 residual full train:         ~73.87 F1
S3 Active Corridor + QualityCal: 77.10 F1
```

The large jump came from S3, not S1/S2.

Current best full test result:

```text
IoU 0.50:
TP=74716 FP=14216 FN=30170
P=0.8401 R=0.7124 F1=0.7710

score_thresh=0.50
quality_score_power=0.5
```

Category breakdown for the 77.10 model:

```text
normal:  F1=0.9250
crowd:   F1=0.7541
hlight:  F1=0.7063
shadow:  F1=0.7018
noline:  F1=0.5017
arrow:   F1=0.8782
curve:   F1=0.5556
cross:   FP=1353
night:   F1=0.7249
```

The largest high-impact categories are normal, crowd, and night. Curve is weak but small in dataset weight. Noline is also weak, but its ceiling in common CULane models is not very high.

## 3. Important 2K Debug Baselines

2K debug is not a perfect predictor of full train, but it is useful for killing bad ideas early.

Known 2K references:

```text
S0 strong 2k first run:          ~0.3617 F1
S0 strong 2k continue/fine-tune: ~0.5038 F1
S2 old max:                     ~0.58 F1 range
S3 Active Corridor v1:           ~0.6019 F1
S3 Active Corridor + QualityCal: ~0.6337 - 0.6380 F1 range
```

The stable S3 2K reference is roughly:

```text
S3 Active Corridor + QualityCal: ~0.6380 F1
```

For new S3 variants, a meaningful 2K signal should be around:

```text
0.645+ F1
```

Small differences like `0.6380 -> 0.6401` are not strong enough to justify full train by themselves.

## 4. Active Corridor and QualityCal

### Active Corridor v1

Active Corridor lets S2 search lateral offsets around the coarse lane:

```text
coarse_x -> sample offsets -> soft-argmax delta -> refined_x
```

This improved recall but created duplicate/false-positive pressure.

Example:

```text
S3 Active Corridor v1:
TP=3668 FP=2553 FN=2299
P=0.5896 R=0.6147 F1=0.6019
```

It found more lanes but created too many false positives.

### QualityCal

QualityCal was added to calibrate the final score using row evidence and offset statistics.

Key result:

```text
S3 Active Corridor + QualityCal:
TP=3486 FP=1549 FN=2481
P=0.6924 R=0.5842 F1=0.6337
```

This was a real improvement. It reduced false positives heavily while keeping enough true positives.

Full train with QualityCal produced the current best:

```text
F1=77.10
```

### q=0.0 vs q=0.5

Turning quality scoring off gave worse full-test behavior:

```text
q=0.0:
F1=74.79
cross FP much higher
curve improved relative to q=0.5
```

This showed that QualityCal helps overall precision, especially cross/false positives, but it may suppress some true curved lanes.

## 5. Failed or Weak S3-Side Variants

### LineIoU Matcher + Mean/Max QualityCal

This looked promising but did not beat the stable QualityCal baseline.

No-exist calibration variant:

```text
F1=0.6268
```

Exist calibration variant:

```text
best around F1=0.6185
```

Conclusion: the idea was not completely absurd, but it was not better than the stable S3 baseline.

### Cascade Matching

Cascade/final-stage matching was tested.

2K result:

```text
thr=0.55:
TP=3507 FP=1684 FN=2460
F1=0.6286
```

This underperformed the stable S3 baseline. The likely issue was matching instability and changed assignment dynamics.

Conclusion: stop using cascade matching for now.

### Curve-Aware QualityCal

Curve-aware stats gave a tiny change but not a real gain:

```text
Curve-aware best: ~0.6363
```

Conclusion: it may recover some true positives but also lets more false positives through.

### Wide64 Active Corridor

Wider offset search also did not help:

```text
Wide64 best: ~0.6353
```

Conclusion: wider search gives some extra true positives but also increases false positives. It is a trade-off, not a breakthrough.

### Centerline Aux

Centerline auxiliary supervision was tested:

```text
best around F1=0.6380
```

Conclusion: it did not break the S3 2K ceiling.

## 6. Dynamic Evidence V1

V1 dynamic evidence used slot-conditioned reference points:

```text
q1 -> reference points -> grid_sample features -> zero-init adapter -> q enhanced
```

### S3 V1

S3 with V1 dynamic evidence:

```text
best around F1=0.6380
```

It changed the TP/FP trade-off but did not improve the ceiling.

### S0 V1

S0 V1 first 12K was weak:

```text
best around F1=0.3685
```

S0 V1 continue 12K:

```text
F1=0.5073
```

Old S0 continue baseline:

```text
F1=0.5038
```

Conclusion: V1 was only a tiny improvement over the old S0 baseline. Not enough.

## 7. Geometry Evidence V2 / V2.1

V2 changed the evidence mechanism:

```text
q1 -> draft lane geometry -> sample evidence along draft_x_rows -> zero-init adapter -> shared S0 heads
```

V2.1 improved sampling:

```text
local window offsets: [-16, -8, 0, 8, 16]
local_reduce: max
row pooling: mean + max
```

### S0 Geometry V2.1

This was the clearest S0-side improvement:

```text
Old S0 continue:
TP=3077 FP=3171 FN=2890
F1=0.5038

S0 Geometry V2.1 scratch -> continue:
TP=3075 FP=2553 FN=2892
F1=0.5304
```

Interpretation:

- true positives stayed almost the same
- false positives dropped a lot
- recall did not explode
- S0 became cleaner, not dramatically more recall-positive

This is useful but not a 79+ breakthrough by itself.

### Direct S3 Geometry

S3 Active Corridor + QualityCal with geometry evidence added directly from the old S2 baseline:

```text
thr=0.55:
TP=3538 FP=1549 FN=2429
F1=0.6401
```

Stable S3 baseline:

```text
~0.6380
```

Conclusion: tiny gain only. Not a full-train candidate.

### Staged S0 -> S1 -> S2 -> S3 Geometry

After adding geometry draft supervision, staged results:

```text
S1 geometry:
TP=3300 FP=3041 FN=2667
F1=0.5362

S2 geometry:
TP=3491 FP=2710 FN=2476
F1=0.5738

S3 geometry from S2 geometry:
TP=3507 FP=1605 FN=2460
F1=0.6331
```

Conclusion:

- S1 no longer collapses after draft supervision, but improvement is small.
- S2 improves over S1, but it does not clearly beat old S2 behavior.
- S3 staged geometry underperforms the stable S3 baseline.

The geometry evidence branch does not currently look like the path to 79+.

## 8. Why S1/S2 Now Look Suspicious

The user's concern is valid:

```text
S0 full: 72.83
S1 full: 73.43
S2 full: 73.87
S3 full: 77.10
```

The marginal gains of S1 and S2 are small:

```text
S0 -> S1: +0.60 F1
S1 -> S2: +0.44 F1
S2 -> S3: +3.23 F1
```

This does not prove S1/S2 are useless, but it does prove that the largest value came from S3.

Possible interpretations:

1. S1/S2 are weak but still help prepare features for S3.
2. S1/S2 are mostly redundant with S3 Active Corridor.
3. S3 could possibly work better if trained directly from a strong S0.
4. S1/S2 may need redesign because their role is not strong enough.

The open experiment that must be run:

```text
Direct S0 -> S3 Active Corridor + QualityCal
```

This answers:

```text
Did S1/S2 really help, or did S3 do almost all the work?
```

If direct S0 -> S3 beats the current S0->S1->S2->S3 pipeline, then S1/S2 are not justified in their current form.

If direct S0 -> S3 is worse, then S1/S2 still provide useful intermediate geometry/evidence, even if their standalone F1 gains look small.

## 9. Why S1/S2 May Still Matter Despite Small Standalone Gains

Standalone F1 is not the only measure. A stage can be useful if it improves the representation used by the next stage.

S1 may help:

- enforce row-wise lane consistency
- reduce jagged lane predictions
- give S2 a better coarse geometry to sample from

S2 may help:

- convert geometry into image-grounded evidence
- give S3 `row_hidden` and local evidence features
- prepare the quality calibrator

But the numbers show the current S1/S2 versions are not strong enough to be assumed necessary. They must be ablated directly.

## 10. Current Honest Diagnosis

The current best model is strong because of S3 QualityCal and Active Corridor, not because S1/S2 delivered large gains.

The project may be missing one of these:

1. A stronger proposal/query generation mechanism.
2. A better stage design where S1/S2 provide clearly useful evidence to S3.
3. A direct S0->S3 path that skips weak intermediate refiners.
4. A stronger classifier/resurrector after evidence sampling, not only coordinate refinement.

The current geometry evidence branch does not solve the main recall problem.

## 11. Most Important Open Questions

### Question A: Is S1/S2 necessary?

Run:

```text
S0 strong -> S3 Active Corridor + QualityCal
```

Compare against:

```text
S0 -> S1 -> S2 -> S3 = 77.10 full
```

and 2K:

```text
stable S3 2K ~= 0.6380
```

### Question B: Is S3 mostly a scoring module or a real geometry module?

Compare:

```text
q=0.0
q=0.5
different thresholds
```

If q=0.5 wins by FP reduction but recall stays limited, then S3 is mostly score calibration, not a recall engine.

### Question C: Can S2 update existence, not just geometry?

A major possible missing piece:

```text
S2 samples strong local evidence but does not fully resurrect exist scores.
```

If S2 sees strong lane pixels, it may need to update `exist_logits`, not only `row_x_logits`.

### Question D: Is 20 slots enough?

Probably enough for normal 4-lane scenes, but maybe not enough for hard CULane scenes. However, increasing slots without changing class imbalance and NMS behavior is dangerous.

Do not jump to 100 slots until direct S0->S3 and S2-resurrector questions are answered.

## 12. Recommended Next Experiments

### Experiment 1: Direct S0 -> S3 Active Corridor + QualityCal

This is the most important next ablation.

Goal:

```text
Test whether S1/S2 are actually needed.
```

2K version first:

```text
init from S0 strong/geometry checkpoint
train S3 Active Corridor + QualityCal
no S1/S2 staged init
```

Interpretation:

- If direct S0->S3 is better than staged S3, S1/S2 are likely hurting or redundant.
- If worse, S1/S2 still provide useful preparation.

### Experiment 2: S2/S3 existence resurrection

Let evidence-aware stages update existence score:

```text
final_exist_logits = coarse_exist_logits + delta_exist_from_evidence
```

This should be tested carefully because earlier decision-head attempts were weak, but the current Active Corridor evidence is stronger than older row_hidden-only signals.

### Experiment 3: Simpler S1 ablation

S1 may be too weak or poorly aligned. Test:

```text
S0 -> S2
S0 -> S3
S0 -> S1 -> S3
S0 -> S2 -> S3
```

This identifies which intermediate stage is actually useful.

### Experiment 4: Full category-driven tuning

The 77.10 model loses important points in:

- crowd
- night
- shadow
- curve

But total F1 is most affected by normal/crowd/night because of dataset weight.

## 13. What Not To Do Right Now

Do not full-train every small 2K improvement.

Do not add:

- 100 slots
- focal loss
- dynamic query top-k
- cascade matching
- LaneAF/GANet-style bottom-up heads

until the direct S0->S3 ablation is answered.

## 14. Current Decision

The immediate next rational step is:

```text
Build and run direct S0 -> S3 Active Corridor + QualityCal.
```

This is the cleanest way to answer the user's main question:

```text
Are S1 and S2 actually useful, or did S3 do almost everything?
```

Until that result is known, it is premature to redesign the whole architecture.

