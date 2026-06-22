# DynLaneSeq Current IGSR Experiment and Decision Plan

## 1. Document Purpose

This document records exactly what is currently being tested, why it is being tested, what has already failed, and what result is required before starting a full end-to-end training run.

The current work is **not** another full DynLaneSeq stage and is **not** a 50-epoch end-to-end experiment. It is a controlled test of one missing capability:

> Can a small image-grounded geometry head use visual evidence around an S0 lane proposal to select the correct row-wise lateral displacement?

The answer to this question determines whether the final architecture should retain an Active Corridor style image-grounded refiner.

## 2. Current Architectural Diagnosis

### 2.1 S0 status

S0 is not being replaced. It is currently the strongest and most reliable part of the project.

S0 responsibilities:

- extract shared image features;
- initialize structured lane queries;
- produce coarse row-wise lane geometry;
- produce existence, range, and quality estimates.

Current initialization checkpoint:

```text
outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt
```

The 175k checkpoint is used because it is more recall-balanced than the later precision-oriented checkpoints.

### 2.2 Old S1 status

The old residual S1 produced only marginal gains after the strong 50-epoch S0 and sometimes degraded it during joint training.

Decision:

- do not retain old S1 as a separately trained production stage;
- potentially reuse only useful row-sequence ideas inside a future unified refiner.

### 2.3 CSR-S2 status

The frozen CSR-S2 did not provide a meaningful gain over the strong S0/S1 chain. Diagnostics showed rescued proposals but also substantial geometry and score losses.

Decision:

- do not use CSR-S2 in the main architecture;
- do not spend another full training run tuning the existing CSR configuration.

### 2.4 S3 status

S3 contains two conceptually different parts:

1. Active Corridor image evidence and geometry refinement.
2. Quality/existence calibration and final filtering.

The historical S3 produced the largest staged F1 increase. However, later diagnostics showed that its quality score can increase precision by aggressively reducing recall. The useful concept is image-grounded resampling, not necessarily the existing S3 training recipe.

Decision:

- retain and repair the image-grounded geometry concept;
- postpone selection/quality redesign until geometry refinement is proven;
- do not preserve S3 as an independently frozen final stage in the target architecture.

## 3. Intended Final Architecture

The intended final model remains modular in code but will eventually be trained end to end:

```text
Image
  -> Shared Encoder
  -> S0 Structured Query Initialization
  -> Image-Grounded Structured Geometry Refiner
  -> Learned Selection / Existence Score
  -> Optional Final Quality Calibration
  -> Final Lanes
```

The image-grounded refiner is intended to replace the functional role of the old S1 and CSR-S2 while reusing the useful visual-resampling idea from S3.

The final training strategy is **not**:

```text
train S0 -> freeze -> train S1 -> freeze -> train S2 -> freeze -> train S3
```

After every component passes its isolated gate, the final strategy should be:

```text
initialize from strong S0
-> briefly warm up new heads if necessary
-> jointly optimize S0 + refiner + selection with stage-specific learning rates
```

We are not at that stage yet.

## 4. Why the 2k Experiments Were Run

The 175k S0 model was trained on the full 88,880-image CULane training set. During the 2k experiments, S0 was frozen. The full S0 was **not** fine-tuned on 2k images.

Only the new Active Corridor geometry head was trained on `train_2k.txt` and evaluated on the disjoint `test_2k.txt` list.

The 2k experiments were screening tests designed to:

- catch implementation and gradient-flow errors cheaply;
- compare architectures with the same S0, seed, data order, and optimizer steps;
- verify that image evidence changes the prediction;
- measure offset accuracy, geometry MAE, rescued/killed lanes, and F1;
- reject obviously ineffective designs before spending days on full training.

The 2k experiment is not a reliable final-performance estimator. With batch size 4 and 12k iterations, the model sees 48k image presentations, which means the same 2k training scenes are repeated approximately 24 times. This creates a scene-diversity and overfitting limitation.

Therefore:

- a large failure on 2k is meaningful;
- a small gain below roughly 0.3 F1 is inconclusive;
- a 2k gain must be supported by mechanistic geometry metrics;
- 2k results must not be reported as publication validation results.

## 5. Diagnostics That Identified the Current Bottleneck

### 5.1 Geometry-authority bug

Previously, Active Corridor predicted a refined coordinate, but the row decoder independently decoded the final geometry from residual logits. The decoder erased the small Active Corridor gain.

This was corrected with config-gated authoritative geometry:

```text
final pred_x_rows = active corridor refined_x
```

After the fix, Active Corridor MAE and final MAE became identical. This confirms that the geometry-authority diagnosis was correct.

### 5.2 Existing offset scorer failure

The independent MLP offset scorer mostly learned the center-offset prior instead of reading the lateral image profile.

Representative 2k diagnostics:

```text
coarse MAE:                 12.5248 px
learned authoritative MAE: 12.4482 px
oracle discrete MAE:        6.2813 px
offset-target correlation:  0.1182
offset direction accuracy:  57.67%
```

The representational capacity existed, but the learned scorer used almost none of it.

### 5.3 GT-oracle discrete-offset test

An exact CULane raster-IoU oracle was implemented. For every grouped matched proposal and valid row, it selected only the closest offset from the existing set:

```text
[-32, -24, -16, -8, 0, 8, 16, 24, 32] px
```

It did not use unrestricted GT coordinates. It preserved the existing scores, ranges, NMS, and Top-K behavior.

Results:

| Geometry | F1 | Raw R@0.5 | Raw R@0.7 |
|---|---:|---:|---:|
| S0 coarse | 75.97 | 76.24 | 61.77 |
| Learned authoritative offset | 76.18 | 76.37 | 62.21 |
| GT-oracle discrete offset | **87.07** | **89.12** | **84.48** |

Approximately 9.94% of supervised rows were outside the corridor, but the discrete oracle still reached 87.07 F1.

Interpretation:

- S0 proposals are not the dominant limitation for this experiment;
- the current corridor width and discrete offset representation have enough theoretical capacity;
- the main failure is learning the correct offset from image evidence;
- 87.07 is an oracle ceiling, not an expected real score;
- the oracle uses GT at every valid row and must never be presented as model performance.

Oracle output:

```text
outputs/active_corridor_diagnostics/oracle_discrete_offset_2k.json
```

## 6. Lateral Profile Scorer Experiment

The independent per-offset MLP was replaced with a profile scorer that:

- observes all nine lateral offset features jointly;
- computes features relative to the center sample;
- applies Conv1D interaction along the lateral offset axis;
- injects query and row information through FiLM-style conditioning;
- retains authoritative final geometry;
- starts as an identity update through zero initialization.

2k comparison:

| Metric | Independent authoritative MLP | Lateral profile scorer |
|---|---:|---:|
| Best q=0 F1 | **76.18** | 76.09 |
| Final MAE | 12.4482 | **12.4327** |
| Improved-lane rate | 54.97% | **56.04%** |
| Offset-target correlation | 0.1182 | **0.1626** |
| Offset Top-1 accuracy | 57.63% | **59.81%** |

Interpretation:

- the profile scorer uses image evidence somewhat better;
- its mechanistic improvements are small;
- it did not improve F1 over the independent authoritative MLP;
- it did not pass the 2k architecture gate;
- the result does not justify a 50-epoch or end-to-end run.

## 7. Current Experiment: Full-Data Short Frozen-Head Control

### 7.1 Question being tested

The current experiment asks:

> Did the lateral profile scorer fail because it saw only 2,000 unique training scenes repeatedly, or because the scorer/loss itself is inadequate?

### 7.2 What is trainable

Trainable:

- lateral profile Active Corridor scorer only.

Frozen:

- backbone;
- FPN;
- structured query head;
- S0 coarse geometry and scores;
- row decoder;
- bridge/adapter paths;
- quality calibrator.

The final geometry remains the authoritative Active Corridor geometry.

### 7.3 Training budget

```text
Training images:       88,880
Batch size:            4
Iterations:            25,000
Image presentations:   100,000
Approximate epochs:    1.125
Checkpoint interval:   5,000 iterations
S0 initialization:     iter_0175000.pt
```

This is a full-data diversity test, not a full convergence run.

### 7.4 Config and output

Config:

```text
dynlaneseq_eg/configs/culane_igsr_lateral_profile_frozen_25k.yaml
```

Output directory:

```text
outputs/culane_igsr_lateral_profile_frozen_25k
```

## 8. Commands to Run

### 8.1 Start training

```bash
bash scripts/run_culane_igsr_lateral_profile_frozen_25k.sh
```

Default initialization:

```text
outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt
```

### 8.2 Evaluate the final checkpoint on the same 2k diagnostic split

```bash
bash scripts/eval_culane_igsr_lateral_profile_frozen_25k_2k.sh
```

### 8.3 Evaluate an intermediate checkpoint

```bash
CKPT_NAME=iter_0015000.pt \
bash scripts/eval_culane_igsr_lateral_profile_frozen_25k_2k.sh
```

Useful checkpoints:

```text
iter_0005000.pt
iter_0010000.pt
iter_0015000.pt
iter_0020000.pt
last.pt / iter_0025000.pt
```

## 9. What Must Be Measured

Primary fair comparison uses `quality_score_power=0.0` because this experiment tests geometry, not quality filtering.

Report:

- precision, recall, and F1 at score thresholds 0.40, 0.45, 0.50, and 0.55;
- coarse, active, and final MAE;
- improved-lane and worsened-lane rates;
- offset-target correlation;
- offset direction and Top-1 accuracy;
- image-evidence intervention sensitivity;
- duplicate-safe raw R@0.5 and R@0.7;
- rescued TP, geometry-killed TP, score-killed TP, and FP changes.

Do not select a checkpoint using the official CULane test set for publication. `test_2k.txt` is only the existing engineering gate. A successful design must later be selected on official validation and evaluated once on official test.

## 10. Decision Rules

### 10.1 Pass

Proceed toward the unified refiner only if the full-data short run shows all of the following:

- a clear F1 improvement beyond the approximate 0.3-point noise band over the strong q=0 S0 baseline;
- a meaningful raw R@0.7 increase, not only threshold movement;
- improved offset-target correlation and direction accuracy;
- normal image evidence consistently outperforms shuffled/reversed evidence;
- rescued TPs exceed geometry/score/NMS-killed TPs by a useful margin;
- no hidden recall collapse compensated by precision.

If it passes:

1. retain S0 initialization and authoritative geometry;
2. add a learned IoU-aware selection score as a separate controlled experiment;
3. validate geometry and selection jointly;
4. only then prepare the full end-to-end training configuration.

### 10.2 Fail

Treat the current profile scorer as insufficient if:

- F1 remains within noise;
- raw R@0.7 remains nearly unchanged;
- MAE improvement remains a tiny fraction of the oracle gap;
- offset prediction remains dominated by the center prior;
- shuffled evidence performs almost as well as normal evidence.

If it fails:

- do not run 50 epochs;
- do not add selection/quality to hide the geometry failure;
- inspect or redesign the feature representation and offset objective;
- consider stronger cross-offset contrast, multi-scale evidence, continuous displacement, or coarse-to-fine resampling as separate ablations;
- retain S0 as the baseline.

## 11. What This Experiment Does Not Prove

Even a successful 25k result would not prove:

- 80 F1 will be reached;
- the final end-to-end model will remain stable;
- the oracle score is learnable;
- the profile scorer is a publishable novelty by itself;
- S0 proposal coverage is sufficient for every difficult CULane category.

It only determines whether the image-grounded geometry component is strong enough to deserve integration into the final model.

## 12. Questions for External Review

An external reviewer should specifically assess:

1. Is the GT-oracle discrete-offset experiment a valid upper-bound diagnostic given grouped one-to-many matching?
2. Is 25k iterations at batch size 4 sufficient for a frozen 110k-parameter geometry head to test full-data diversity?
3. Does the lateral profile scorer have enough cross-offset comparison capacity?
4. Is the current hard offset CE plus pixel Smooth-L1 objective likely to preserve center-prior collapse?
5. Should the next scorer predict a continuous offset, an interpolated soft offset distribution, or a coarse-to-fine displacement?
6. Are the proposed pass/fail gates strict enough to prevent another unjustified full run?

## 13. Full-Data Lateral-Profile Result

The 25k full-data frozen-head control rejected the original lateral-profile scorer.

| Metric | 2k-trained profile | Full-data 25k profile |
|---|---:|---:|
| Best q=0 F1 | 76.09 | 76.07 |
| Final MAE | 12.4327 | 12.4257 |
| Offset-target correlation | 0.1626 | 0.1843 |
| Offset Top-1 accuracy | 59.81% | 59.34% |
| Direction accuracy | 57.38% | 56.95% |

The full-data head reduced MAE by only 0.0991 px relative to the 12.5248 px coarse geometry. The discrete oracle reduces it by 6.2435 px. Therefore, the learned head captured only about 1.6% of the available oracle MAE improvement.

Conclusion:

- training-scene diversity was not the main bottleneck;
- the original single-stage profile representation and hard-bin objective were inadequate;
- checkpoint selection and longer training were rejected;
- the original lateral-profile scorer must not be used for end-to-end training.

## 14. Replacement Experiment: Coarse-to-Fine IGSR v2

The replacement scorer changes the evidence representation and objective rather than extending the failed training schedule.

Architecture:

- coarse search over `[-32, -24, -16, -8, 0, 8, 16, 24, 32]` px;
- fine resampling around the predicted coarse position over `[-8, -4, -2, 0, 2, 4, 8]` px;
- joint row-lateral 2D depthwise blocks instead of independent row classification;
- P2 appearance features plus frozen S0 segmentation and centerline logits as explicit dense priors;
- continuous expected displacement from both stages;
- authoritative final geometry;
- no row decoder or quality calibration in this isolated geometry experiment.

Loss changes:

- Gaussian soft offset targets instead of hard nearest-bin CE alone;
- separate coarse and fine regression/classification losses;
- increased weight for large non-center corrections;
- row-to-row displacement tangent supervision;
- normal point and LineIoU losses on the authoritative final geometry.

Isolation guarantee:

```text
Total trainable parameters: 240,530
Active Corridor parameters: 240,530
All S0, bridge, decoder, quality, and calibration parameters: frozen
```

The old "frozen head" setup accidentally left bridge/adapter parameters trainable through the optimizer grouping. IGSR v2 explicitly freezes every parameter except `active_corridor.*` and skips the unused row decoder.

Config:

```text
dynlaneseq_eg/configs/culane_igsr_coarse_to_fine_frozen_25k.yaml
```

Training:

```bash
bash scripts/run_culane_igsr_coarse_to_fine_frozen_25k.sh
```

Evaluation of the final checkpoint only:

```bash
bash scripts/eval_culane_igsr_coarse_to_fine_frozen_25k_2k.sh
```

This remains a 25k full-data geometry gate. It is not an end-to-end experiment and does not test selection/quality calibration.

## 15. Superseded Pre-v2 Summary

The original lateral-profile scorer has been rejected; the project is now testing whether a strictly isolated coarse-to-fine, dense-prior, row-continuous geometry head can convert a meaningful fraction of the 87.07-F1 oracle geometry potential before any end-to-end training is authorized.

## 16. Coarse-to-Fine IGSR v2 Result

The coarse-to-fine v2 experiment also failed the geometry gate.

| Metric | Original full-data profile | Coarse-to-fine v2 |
|---|---:|---:|
| Best q=0 F1 | 76.07 | 76.14 |
| Best-per-GT final MAE | 12.4257 | 12.3976 |
| Offset-target correlation | 0.1843 | 0.2064 |
| Direction accuracy | 56.95% | 57.27% |
| Offset-reverse delta change | 1.5547 px | 1.2660 px |
| Lane-roll delta change | 1.1524 px | 1.8043 px |

Relative to the approximately 75.97 S0 q=0 baseline, v2 gained only about 0.17 F1. Its MAE improvement over S0 was only 0.1272 px. The model became slightly more correlated with the target but did not become strongly sensitive to the left/right ordering of local evidence.

Interpretation:

- coarse-to-fine resolution was not the missing solution;
- query/row conditioning and dense priors remained possible shortcuts;
- the v2 scorer did not justify end-to-end integration;
- longer training or checkpoint selection was rejected.

## 17. Pixel-Only Causal Control

The next experiment deliberately removes the main shortcut paths.

Pixel-only scorer inputs:

- normalized P2 lateral image features;
- feature differences relative to the center sample;
- a fixed normalized offset coordinate.

Explicitly excluded:

- structured query conditioning;
- row embeddings;
- FiLM conditioning;
- segmentation priors;
- centerline priors;
- learned per-offset embeddings;
- coarse-to-fine resampling;
- row decoder, bridge, quality, and calibration updates.

Training-only center perturbation:

```text
center_train = coarse_x + smooth_random_jitter
jitter range = +/-24 px
jitter knots = 5
center_eval = coarse_x
```

This prevents the head from minimizing the average loss by always predicting zero displacement. Jitter is disabled automatically in evaluation.

Loss:

- expected continuous displacement regression;
- pixel-space Huber with an 8 px normalizer instead of 32 px;
- low-weight Gaussian soft CE retained for distribution identifiability;
- row-tangent supervision;
- moderate magnitude weighting.

Compute budget is matched by image presentations:

```text
micro batch:                  4
gradient accumulation:       4
effective batch:            16
optimizer steps:          6250
image presentations:    100000
approximate epochs:        1.125
trainable parameters:      69066
```

Training:

```bash
bash scripts/run_culane_igsr_pixel_only_frozen_100kimg.sh
```

Evaluation:

```bash
bash scripts/eval_culane_igsr_pixel_only_frozen_100kimg_2k.sh
```

The experiment is successful only if local-evidence sensitivity and geometry improve together. A large intervention response without lower MAE is not success.

## 18. Latest One-Sentence Summary

Both conditioned IGSR scorers failed; the project is now running a strictly pixel-only, jitter-balanced causal control to determine whether frozen P2 features contain enough local lane evidence for learned lateral correction at all.

## 19. Pixel-Only Result and Jitter Execution Correction

Pixel-only evaluation produced:

```text
best F1:                    76.00
best-per-GT final MAE:      12.3991 px
offset-target correlation:  0.2012
offset-reverse delta:         4.0405 px
normal better than reverse:  67.56%
```

The result proved that removing query/row/FiLM shortcuts increased causal sensitivity to lateral image evidence. It did not produce a meaningful F1 or MAE gain.

During implementation of the next evidence-tower experiment, a train/eval-mode bug was found:

```text
freeze_non_active_modules -> parent model kept in eval mode
old jitter condition       -> checked parent model.training
actual result              -> center jitter remained disabled
```

Therefore, the completed pixel-only run was a valid pure pixel-only experiment but **not** a pixel-only-plus-jitter experiment. The intervention improvement came from removing shortcut inputs, not from jitter.

The code now checks `active_corridor.training`, so train-only jitter works even while the frozen parent stays in evaluation mode. A real CULane batch verified nonzero jitter and valid gradients.

## 20. Corrected Experiment Order

### Phase A: Pixel-only plus mixed jitter

Run this first to isolate the effect of balanced perturbation:

```bash
bash scripts/run_culane_igsr_pixel_only_mixed_jitter_frozen_100kimg.sh
```

Evaluate:

```bash
bash scripts/eval_culane_igsr_pixel_only_mixed_jitter_frozen_100kimg_2k.sh
```

Settings:

```text
jitter maximum:       24 px
jitter probability:      0.5 per lane
jitter knots:             5
effective batch:         16
image presentations: 100000
```

Half the lanes retain the deployment-like S0 center while half receive smooth perturbations. This trains evidence sensitivity without removing identity preservation from the training distribution.

### Phase B: Trainable lane evidence tower

Only run Phase B if corrected mixed jitter still fails:

```bash
bash scripts/run_culane_igsr_evidence_tower_frozen_100kimg.sh
```

The tower contains:

- trainable P2 evidence projection;
- dense segmentation and centerline supervision;
- pixel-only signed offset prediction;
- confidence-only no-harm gate;
- mixed train-only jitter;
- no query, row, or FiLM signed-offset shortcut.

Evaluation:

```bash
bash scripts/eval_culane_igsr_evidence_tower_frozen_100kimg_2k.sh
```

This order prevents a jitter fix and evidence-representation change from being conflated.

## 21. Latest One-Sentence Summary

The pure pixel-only run increased genuine evidence sensitivity but not geometry quality; because its configured jitter was accidentally inactive, the next mandatory control is corrected mixed jitter, after which the already implemented dense-supervised evidence tower is the final frozen-S0 geometry test.

## 22. Corrected Mixed-Jitter Result

The corrected pixel-only mixed-jitter experiment failed and was worse than both S0 and the no-jitter pixel-only control.

```text
best F1:                    75.80
coarse MAE:                 12.5248 px
final MAE:                  12.5373 px
offset-target correlation:   0.1052
direction accuracy:          53.99%
improved-lane rate:          47.70%
offset-reverse delta:         4.1030 px
```

Error-bin behavior:

| Coarse-error bin | Coarse MAE | Mixed-jitter MAE | Result |
|---|---:|---:|---|
| 0-4 px | 1.4649 | 1.6836 | substantially worse |
| 4-8 px | 5.6944 | 5.7003 | neutral/slightly worse |
| 8-16 px | 11.2805 | 11.2231 | tiny gain |
| 16-32 px | 22.5761 | 22.2428 | small gain |
| 32+ px | 79.9784 | 79.1525 | insufficient gain |

Interpretation:

- jitter increased response magnitude but not response correctness;
- the train-time perturbation distribution damaged the deployment-dominant 0-4 px rows;
- evidence sensitivity alone is not a geometry-quality metric;
- mixed jitter is rejected and must not be carried into the next frozen experiment.

## 23. Final Frozen-S0 Representation Gate

The next evidence-tower run has been corrected to isolate the visual representation change:

```text
center jitter:       disabled
confidence gate:     disabled
query/row/FiLM:      absent from signed offset path
trainable component: P2 lane evidence tower + dense heads + pixel-only scorer
dense supervision:  segmentation + centerline
```

Run:

```bash
bash scripts/run_culane_igsr_evidence_tower_frozen_100kimg.sh
```

Evaluate:

```bash
bash scripts/eval_culane_igsr_evidence_tower_frozen_100kimg_2k.sh
```

Only if this clean evidence-tower experiment improves geometry should a confidence gate be enabled as a separate ablation.

## 24. Latest One-Sentence Summary

Pure pixel evidence reacts causally but does not localize accurately, and mixed jitter makes it worse; the remaining frozen-S0 test is whether dense-supervised adaptation of P2 itself can create a lane-center representation that the same pixel-only scorer can use.

## 25. Frozen Evidence-Tower Result

The final frozen-S0 representation experiment failed its geometry gate:

```text
best F1:                    76.04
coarse MAE:                 12.5248 px
final MAE:                  12.4110 px
offset-target correlation:   0.1719
direction accuracy:          56.74%
improved-lane rate:          55.27%
offset-reverse delta:         2.0938 px
```

This closes the current frozen-S0 local-correction program. It does not prove that local sampling is physically incapable: the discrete corridor oracle remains much stronger. It proves that the tested frozen representation and offset heads do not learn enough of that available correction.

## 26. Controlled Joint 25k Gate

The next experiment is a falsification test for controlled joint optimization, not a blind 50-epoch run.

Initialization:

```text
outputs/culane_igsr_evidence_tower_frozen_100kimg/last.pt
```

This checkpoint already supplies the head-only warm-up. The joint graph then uses:

```text
evidence tower/refiner LR: 1e-4
structured head/FPN LR:    1e-5
backbone LR:               1e-6
backbone BatchNorm:        frozen
gradient clipping:         per optimizer group, max norm 1.0
scheduler horizon:         278k optimizer steps
joint gate stop:           25k optimizer steps
```

The full coarse anchor includes existence, point, range, smoothness, LineIoU, quality, original S0 segmentation, and original S0 centerline losses. Tower dense supervision is kept separate. Jitter, confidence gating, and quality-powered postprocessing remain disabled.

Before training, verify real-batch gradient flow:

```bash
bash scripts/probe_culane_igsr_joint_controlled_gradients.sh
```

Run the gate:

```bash
bash scripts/run_culane_igsr_joint_controlled_25k.sh
```

Evaluate internal coarse and final outputs on validation with identical postprocessing:

```bash
bash scripts/eval_culane_igsr_joint_controlled_gate.sh
```

The run passes only if:

1. Internal coarse F1 drops by no more than 0.2 points.
2. Final F1 exceeds the same checkpoint's internal coarse F1 by at least 0.3 points.
3. Final R@0.7 improves.
4. Final geometry MAE improves materially.

Only after passing all gates may training continue with the same optimizer and scheduler state:

```bash
bash scripts/continue_culane_igsr_joint_controlled_to_50ep.sh
```

No 80 F1 claim is attached to this experiment. Its purpose is to determine whether joint visual and structured optimization creates a real refinement gain without destroying S0.
