# Image-Grounded Structured Refinement: Diagnosis and Implementation Plan

## 1. Document status

This document is the implementation contract for the next DynLaneSeq architecture only if the diagnostic gates in Phase 0 pass. It is not permission to immediately add another large module.

The required order is:

1. Diagnose proposal, geometry, ranking, and post-processing separately.
2. Decide whether the dominant bottleneck is proposal coverage, geometry, ranking, or gradient isolation.
3. Implement the smallest image-grounded refiner that addresses the measured bottleneck.
4. Prove each claimed contribution through isolated ablations.

No Mamba, sheaf, diffusion, dynamic kernel, or additional novelty mechanism belongs in the first implementation. The current problem is not a shortage of architectural names. It is an unmeasured coupling problem between image evidence, geometry, existence, and ranking.

## 2. Honest interpretation of the current evidence

The current evidence supports the following diagnosis:

- A weak/short-trained S0 benefited from later S1/S2 training partly because those stages provided additional optimization and corrected an unfinished base model.
- Once S0 reached roughly 76.5 F1, frozen S1 saturated around 76.6 and unfrozen S1 often degraded the base model.
- The current S1 mostly refines latent row representations. It does not collect decisive new image evidence for missed or low-score lanes.
- The frozen CSR-S2 configuration explicitly disables existence, range, segmentation, and centerline losses. It can polish matched geometry but cannot recover a lane rejected by S0.
- S3 produces the largest useful gain, primarily by removing false positives. It also removes true positives, so its score is not a sufficiently accurate ranking statistic.
- Staged/frozen training worsens these limitations because final evidence and ranking losses cannot fully reshape proposal generation. It is not, by itself, the root cause.
- Simply putting the unchanged S1, S2, and S3 implementations in one end-to-end graph is not expected to create a 2-point gain. Their responsibilities overlap and their losses are not currently organized around the remaining error.

The core architecture is therefore not considered failed. S0 and S3 have demonstrated real value. The current S1/S2 decomposition is the weak part.

## 3. Important correction about Oracle Top-K

Oracle Top-K is critical, but its interpretation must be precise.

If GT-IoU Oracle Top-K recall is high while model-ranked Top-K recall is low, this proves that good candidates exist and ranking is a major bottleneck. It does not, by itself, prove that S1, S2, or S3 caused the ranking failure. The stage-transition analysis is required to identify where the candidates were demoted, distorted, suppressed, or removed.

Oracle metrics are upper bounds and must never be reported as deployable performance. Ground truth is used only to select or rank candidates during diagnosis.

## 4. Available checkpoint inventory

### 4.1 Strong S0 checkpoints available locally

- `outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt`
- `outputs/culane_s0_structured_query_res34_b16_50ep/iter_0225000.pt`

Use 175k as the recall-balanced base and 225k as the best-F1/precision-oriented base. Both must be analyzed. Selecting only one can hide whether the next stage merely inherits a checkpoint-specific precision/recall bias.

### 4.2 S1 checkpoints available locally

Unfrozen S1:

- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep/iter_0075000.pt`
- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep/iter_0125000.pt`
- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep/iter_0175000.pt`

Frozen S1:

- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep_frozen/iter_0025000.pt`
- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep_frozen/iter_0050000.pt`
- `outputs/culane_s1_residual_structured_query_res34_b16_from_s0_50ep_frozen/iter_0075000.pt`

### 4.3 CSR-S2 checkpoints available locally

- `outputs/culane_s2_residual_structured_query_csr_res34_b16_from_s1_frozen_50ep/iter_0025000.pt`
- `outputs/culane_s2_residual_structured_query_csr_res34_b16_from_s1_frozen_50ep/iter_0075000.pt`
- `outputs/culane_s2_residual_structured_query_csr_res34_b16_from_s1_frozen_50ep/iter_0100000.pt`

### 4.4 Full S3 checkpoint expected on the remote server

- `outputs/culane_s3_active_corridor_qualitycal_structured_query_res34_b16_from_s2/last.pt`

This is the model that produced approximately 77.88 F1. It was not found in the current local checkpoint inventory and must be retained on or copied from the remote server before the complete stage analysis.

## 5. Phase 0: mandatory diagnosis before architecture work

### 5.1 Evaluation split discipline

The official test split must not be repeatedly used to choose architecture, threshold, quality power, or checkpoint. Use a fixed validation split for all diagnosis and ablation decisions.

Requirements:

- Use one immutable image list for every compared checkpoint.
- Record the list path and a hash of its contents in every output JSON.
- Use identical resize, cut-height, row sampling, line width, and minimum-valid-row settings.
- Cache raw model outputs once per checkpoint. Recompute ranking and post-processing metrics from the cache rather than rerunning inference for every threshold.
- Use the official test split only after architecture and post-processing choices are frozen.

For 2k experiments, the training subset and diagnostic validation subset must be disjoint. A model trained and selected on the same 2k images does not provide a fair architecture comparison.

### 5.2 Raw prediction cache contract

For each image and each model stage, save the following before thresholding, Top-K, and NMS:

- image identifier
- stage name: `coarse`, `final`, and any internal iteration name
- `pred_x_rows [N, R]`
- `row_x_logits [N, R, K]` when available
- `range_norm [N, 2]`
- `row_visibility_logits [N, R]` when available
- lane existence probability and raw existence margin
- quality probability and raw quality logit
- the exact final score used by post-processing
- query/slot index and structured group index
- active-corridor offset distribution and expected offset when available
- valid predicted row mask
- NMS suppression parent, if the proposal is removed by NMS

Store floating-point arrays in a compact tensor file and store metadata/metrics in JSON. Do not write only final CULane text predictions; those files discard the information needed to explain ranking failures.

### 5.3 Diagnostic A: stage-transition accounting

For every GT lane, match predictions independently at each stage using the same CULane-style line IoU implementation and line width. Also retain slot identity where the architecture preserves it.

Count these transitions:

- `rescued_tp`: IoU below 0.5 at the previous stage and at least 0.5 at the next stage
- `killed_tp_geometry`: IoU at least 0.5 before refinement and below 0.5 after refinement
- `kept_tp`: IoU at least 0.5 at both stages
- `never_found`: no candidate reaches 0.5 at either stage
- `fp_removed_score`: unmatched prediction removed by score filtering
- `fp_removed_nms`: unmatched prediction removed by NMS
- `tp_removed_score`: GT-matchable prediction removed by score filtering
- `tp_removed_nms`: GT-matchable prediction removed by NMS
- `fp_created_geometry`: a previously matchable slot is moved away and becomes unmatched, or a bad slot crosses the score threshold
- `duplicate_removed`: a second prediction for an already matched GT is removed

Report both counts and rates. Overall F1 alone is insufficient because a stage can remove many false positives while simultaneously killing the exact difficult true positives needed to reach 80.

Required transition pairs:

- S0 coarse to S0 final, when both exist
- S1 model-internal coarse to final
- S2 model-internal coarse to final
- S3 model-internal coarse to final
- externally, best S0 checkpoint to selected S1/S2/S3 checkpoint on the same image list

### 5.4 Diagnostic B: Oracle Top-K decomposition

Implement five distinct proposal-recall modes. They must not be conflated.

1. `all_raw`: every valid proposal, no score threshold, no Top-K, no NMS.
2. `model_topk`: Top-K using the actual model score, before NMS.
3. `model_topk_nms`: exact deployed score, threshold, Top-K, and NMS.
4. `oracle_topk`: choose at most K proposals by maximum-weight bipartite matching between proposals and GT lanes, using IoU as edge weight. A proposal can serve at most one GT.
5. `oracle_topk_nms`: assign oracle IoU as the diagnostic score and then apply the exact production NMS implementation. This measures whether NMS itself destroys otherwise recoverable lanes.

Run K in `{4, 6, 8}` for diagnosis, even if production remains K=4. Report R@0.3, R@0.5, R@0.7, mean best IoU, median best IoU, and the distribution of the rank of the first correct proposal.

Interpretation:

- High `all_raw`, high `oracle_topk`, low `model_topk`: ranking is the dominant failure.
- High `all_raw`, low `oracle_topk` at K=4 but high at K=8: candidate duplication/capacity allocation is the problem.
- High `oracle_topk`, low `oracle_topk_nms`: NMS geometry or distance threshold is destructive.
- Low `all_raw`: the model does not produce a usable candidate; ranking cannot repair it.
- High R@0.5 but low R@0.7: geometry precision remains the problem.

The current `analyze_proposal_recall.py` provides useful score-independent recall, but it is not enough for this experiment. It does not provide bipartite Oracle Top-K, transition accounting, or exact NMS attribution.

### 5.5 Diagnostic C: score and quality decomposition

For every proposal, compute its maximum GT IoU and whether it is selected by matching. Evaluate ranking using:

- existence score only
- quality score only
- the current `exist * quality^power` score
- raw existence margin plus quality logit in logit space
- an oracle IoU score

Report:

- Spearman correlation between score and max IoU
- AUROC and average precision for `max IoU >= 0.5`
- recall among the highest-scoring 4 proposals
- expected calibration error for existence
- calibration curve for quality versus observed IoU
- score distributions for TP, duplicate, and background proposals
- number of GT-matchable proposals below each threshold

The central question is not whether quality can increase precision. It already does. The question is whether quality ranks correct and incorrect candidates in the right order without systematically assigning low values to difficult true lanes.

### 5.6 Diagnostic D: frozen versus joint gradient control

Use the same S0 checkpoint, data order, seed, batch size, augmentations, optimizer steps, and new-head initialization.

Run these controls:

1. S0 continued alone.
2. S0 plus Active Corridor, geometry loss only, S0 frozen.
3. S0 plus Active Corridor, geometry loss only, structured head and FPN trainable.
4. S0 plus Active Corridor and selection/quality head, S0 frozen.
5. S0 plus Active Corridor and selection/quality head, structured head and FPN trainable.
6. Optional final control with the backbone trainable at one-tenth or one-twentieth of the refiner LR.

Do not compare a 12k joint run against a staged chain that received 36k additional updates and call it a unification result. Include `S0 continued alone` so additional optimization is not mistaken for architectural benefit.

Measure gradient norms for:

- backbone
- FPN/non-backbone encoder
- structured query head
- evidence sampler/adapter
- row refiner
- instance mixer
- geometry head
- selection/existence head
- quality head

Also report the fraction of parameters with nonzero gradients. A connected graph is not enough; gradients can be numerically dominated by the backbone or blocked by detach settings.

### 5.7 Phase 0 decision gates

Proceed to the image-grounded refiner only if at least one of these is true:

- Oracle Top-4 recall is materially higher than model-ranked Top-4 recall.
- Active Corridor produces a meaningful number of rescued TPs but the existing score head suppresses them.
- Joint evidence training clearly improves proposal ranking or stage-transition balance over its frozen control.
- S0 has adequate raw proposal recall but later stages cannot preserve it.

If `all_raw` proposal recall is low, stop. The next work belongs in S0 proposal generation, not refinement.

If geometry does not rescue lanes and R@0.7 does not improve, do not add CSR/IGT complexity. The evidence sampler or feature resolution must be fixed first.

If ranking is already close to oracle, do not add another score head. The remaining problem is proposal coverage or geometry.

## 6. Target architecture

### 6.1 Working name

`ImageGroundedStructuredRefiner` (IGSR)

The name is descriptive, not a novelty claim. The publishable contribution must come from measured behavior and the structured coupling, not the acronym.

### 6.2 Architectural objective

Replace the current sequence of mostly independent S1/S2 refiners with one repeated, image-grounded structured refinement block that jointly updates:

- row geometry
- lane existence/selection score
- localization quality
- row visibility/range evidence

S0 remains responsible for structured proposal initialization. S3 is reduced to optional final calibration after the unified refiner proves that its geometry and selection score are useful.

The intended path is:

```text
Image
  -> Shared ResNet34 + FPN features
  -> S0 structured initialization: q_ins, q_geo, coarse geometry, coarse scores
  -> IGSR iteration 1: sample image evidence, refine geometry and selection
  -> IGSR iteration 2: resample at the new geometry, refine again
  -> optional lightweight final calibration
  -> threshold, NMS, Top-K
```

This is not the old S0 -> frozen S1 -> frozen S2 -> S3 training chain. It is one computation graph with repeated image-grounded refinement and deep supervision.

### 6.3 Non-negotiable design principles

- Every refinement iteration must resample the image feature maps along the current lane geometry.
- There must be a single authoritative final geometry representation.
- Existence/selection must be allowed to increase for a weak but visually supported lane, not only decrease.
- The final training objective must reach the structured query head and selected encoder layers.
- `q_ins [B,N,C]` and `q_geo [B,N,R,C]` must remain separate; never flatten `N x R` into an undifferentiated token set.
- Lateral evidence offsets must remain distinguishable until after the model has inferred left/right displacement.
- Every iteration must expose its own outputs for deep supervision and diagnosis.
- Identity initialization is required so a new refiner initially reproduces S0 instead of destroying it.

## 7. IGSR input and output contract

### 7.1 Inputs

- multi-scale feature maps, initially P2 and P3; add P4 only after ablation
- `q_ins [B,N,C]`
- `q_geo [B,N,R,C]`
- current `pred_x_rows [B,N,R]`
- current `row_x_logits [B,N,R,K]` when available
- current `exist_logits [B,N,2]`
- current `quality_logits [B,N]`
- current `range_norm [B,N,2]`
- row validity/visibility estimate
- normalized row coordinates and normalized current x coordinates

### 7.2 Outputs per iteration

- updated `q_ins`
- updated `q_geo`
- updated `pred_x_rows`
- updated `row_x_logits` or an explicit compatibility representation
- updated `exist_logits` or `selection_logits`
- updated `quality_logits`
- updated `range_norm` or row visibility
- active offset logits and expected displacement
- evidence confidence and entropy diagnostics

### 7.3 Final model output

Maintain compatibility with the existing evaluator:

- `outputs["coarse"]`: unmodified S0 output
- `outputs["iterations"]`: ordered list of IGSR iteration outputs
- `outputs["final"]`: last IGSR output
- `outputs["evidence"]`: sampler and score diagnostics

Do not silently overload `coarse`, `stage2`, or `final` with inconsistent semantics.

## 8. IGSR internal design

### 8.1 Multi-scale curve-aligned sampler

Sample P2 and P3 at every current lane row using `grid_sample`.

Initial lateral offsets:

```text
[-24, -16, -8, -4, 0, 4, 8, 16, 24] pixels
```

Requirements:

- Preserve shape `[B,N,R,O,C]` until offset inference.
- Do not mean-pool the offset dimension before the model predicts displacement.
- Include an out-of-bounds mask and sampled-feature validity mask.
- Normalize x and y exactly according to `grid_sample` conventions and test both image borders.
- Respect `cut_height`, resized input dimensions, and FPN stride.
- Allow gradients to sampled feature values in all joint phases.
- Initially detach sampling coordinates only during head warm-up. Remove the detach in joint refinement so later losses can improve earlier geometry.
- Keep multi-scale fusion explicit with learned per-row scale gates. Do not concatenate every feature level blindly and hide memory growth.

The sampler must be tested with synthetic vertical and diagonal feature maps where the expected sampled values are analytically known.

### 8.2 Lateral evidence encoder

For every row and offset, fuse:

- sampled image feature
- offset embedding
- row embedding
- current normalized x coordinate
- `q_geo` row token
- broadcast `q_ins`
- current row confidence and visibility

Use a small normalized MLP or 1x1 projection. The final offset scorer produces `offset_logits [B,N,R,O]`.

Supervise offset logits with the GT displacement from the current geometry to the matched GT row. Use both:

- classification to the nearest offset bin
- smooth-L1 regression on expected displacement

Rows outside GT visibility must not contribute. Track the fraction of targets clamped by the lateral search radius. A high clamp ratio means the sampler cannot recover the error and the window must be changed; it is not a loss-weight problem.

### 8.3 Single geometry authority

The implementation must avoid two unrelated geometry predictions fighting each other.

Preferred first implementation:

1. Compute evidence displacement from offset probabilities.
2. Compute a bounded sequence residual from refined row features.
3. Update coordinates directly:

```text
x_next = clamp(x_current + delta_evidence + delta_sequence, 0, input_w - 1)
```

4. Expose `pred_x_rows = x_next` as the authoritative geometry used by matcher, point loss, LineIoU, quality target, and writer.

For compatibility, `row_x_logits` may remain as an auxiliary distribution head, but it must be tied to `x_next` through an expectation-consistency loss. The evaluator must never decode a different geometry from logits than the geometry used to train quality.

An alternative logit-shifting implementation is permitted only after it is unit-tested and memory-profiled. Do not create a dense `[B,N,R,O,K]` tensor; it is unnecessarily expensive at batch 16.

### 8.4 Intra-lane sequence refiner

Process each lane independently along its 72 ordered rows.

First version:

- LayerNorm
- depthwise Conv1D over rows
- dilations `[1, 2, 4]`
- gated pointwise MLP
- residual connection
- explicit valid-row masking

This reuses the useful inductive bias of CSR without preserving CSR as a separate training stage. Do not add Mamba in v1. If the convolutional refiner fails while evidence displacement is useful, then a sequence-model ablation becomes justified.

### 8.5 Geometry-to-instance feedback

Pool row features using valid-range-aware mean and max pooling. Include summary statistics:

- mean and max offset confidence
- offset entropy
- mean and max absolute displacement
- visible-row fraction
- geometry change from the previous iteration
- row-logit confidence
- optional curvature magnitude

Fuse this summary with `q_ins`. The instance token must represent the actual current lane geometry and sampled image evidence, not only the original S0 embedding.

### 8.6 Inter-lane structured mixer

Operate only on `[B,N,C]` instance tokens.

V1 uses noncompetitive gated context:

- structured-group mean and max
- scene mean and max
- lane token
- geometry/evidence summary
- FiLM-style scale and bias gates

Do not flatten `[N,R]`. Do not add a large full self-attention block in the first implementation. The goal is to test whether image-grounded structured coupling works, not to reopen the IGT architecture search.

If Oracle analysis shows duplicate allocation or cross-lane confusion, a sparse relation mixer can be added later using geometry-based neighbors. That is a separate ablation.

### 8.7 Instance-to-geometry injection

Inject the updated instance state into every row using gated residual modulation:

```text
q_geo_next = q_geo + gate(q_ins_next) * row_update + bias(q_ins_next)
```

Initialize the final projection and residual gates near zero. The first forward pass must stay close to the S0 geometry and score distribution.

### 8.8 Existence resurrection and final selection score

The current quality-power rule is too blunt. IGSR must learn a score optimized for final selection.

Use:

- base S0 existence margin
- pooled image evidence
- offset confidence/entropy
- geometry change statistics
- updated `q_ins`
- visible-row fraction

Predict a bounded residual `delta_selection`. The final selection logit is:

```text
selection_logit_next = selection_logit_current + scale * delta_selection
```

The residual must be able to be positive or negative. This is the resurrection path: a low-score candidate with coherent image evidence can move upward.

Recommended supervision:

- quality-aware focal/varifocal target based on detached max line-IoU or matched line-IoU
- pairwise ranking loss between matchable positives and high-scoring hard negatives in the same image
- explicit monitoring of low-score positives that are rescued above the operating threshold

Avoid a global listwise softmax over all lanes in v1. Sample hard pairs without forcing every valid lane to share a single probability budget.

Keep binary existence as an auxiliary diagnostic if needed, but post-processing should consume one clearly defined `selection_score`. Do not multiply two poorly calibrated probabilities with an externally tuned quality power and call that learned ranking.

### 8.9 Quality prediction

Quality estimates localization quality of the final geometry, not generic lane existence.

- Target: detached line-IoU of `pred_x_rows` against the assigned GT over valid rows.
- Use only final geometry to build the target.
- Train matched positives with a soft regression/calibration loss.
- Train unmatched proposals toward zero only when the assignment policy marks them as background.
- Report quality calibration separately from selection ranking.

Quality may be used as an input to the learned selection head. It should not automatically be raised to a manually chosen inference power.

### 8.10 Visibility and range

Predict row visibility or range when diagnostics show that invalid-row handling contributes to FP/FN errors.

- Supervise only from transformed GT masks.
- Use visibility to mask row losses and quality pooling.
- Do not let an early, inaccurate visibility head erase rows from geometry training.
- During warm-up use GT validity for loss masking and predicted visibility only as a feature.
- Introduce predicted visibility into inference after its calibration is verified.

## 9. Matching and identity stability

The model uses grouped one-to-many assignment. Refinement must not cause uncontrolled slot identity switching.

V1 policy:

- Compute matching from S0/coarse predictions.
- Reuse the same assignments for all IGSR iterations.
- Deeply supervise each iteration against those assignments.
- Track how often final oracle assignment disagrees with the coarse assignment.

If match disagreement is high and final geometry is visibly better, test cascade rematching as a separate ablation. Do not enable it by default because changing targets at every iteration can destabilize score learning and make stage-transition accounting ambiguous.

Duplicates across structured groups require explicit reporting. They are not automatically background if the grouped matcher intentionally creates multiple positives. Ranking targets and NMS must respect the group semantics.

## 10. Loss design

### 10.1 Coarse S0 deep supervision

Keep S0 losses active during joint training:

- existence
- point regression
- range
- LineIoU
- segmentation
- centerline
- coarse quality, if retained

Use a coarse loss coefficient initially around `0.25-0.5`. This prevents final losses from improving the refiner while destroying proposal initialization.

### 10.2 Per-iteration geometry losses

For each IGSR iteration:

- point Smooth-L1 on valid rows
- LineIoU
- tangent consistency
- low-weight curvature consistency
- bounded residual regularization
- active-offset CE
- active-offset regression
- optional row-logit expectation consistency

Tangent and curvature losses are regularizers, not primary objectives. Previous IGT results showed that more geometry loss can improve raw proposal IoU without improving final F1. Their weights must remain subordinate to point and LineIoU losses.

### 10.3 Selection losses

- quality-aware focal/varifocal selection loss
- hard-negative pairwise ranking loss
- optional binary existence auxiliary loss

Hard negatives should be selected from high-scoring unmatched proposals, especially those surviving NMS. Do not use category labels such as `night` or `noline` to define the loss. The mechanism must be general.

### 10.4 Quality loss

- soft target from final line-IoU
- calibration/regression loss on matched predictions
- background target for valid unmatched predictions

### 10.5 Deep-supervision weights

For two iterations, begin with:

```text
coarse: 0.35
iteration 1: 0.50
iteration 2/final: 1.00
```

These are starting values, not truths. Log the unweighted and weighted gradient contribution of each loss family before tuning.

### 10.6 Gradient conflict monitoring

Measure cosine similarity between gradient vectors from:

- geometry losses
- selection/existence losses
- quality loss
- dense auxiliary losses

At minimum, compute this on the shared structured head and FPN for sampled batches. If selection repeatedly opposes geometry, blindly adjusting scalar weights is not enough. Consider stop-gradient boundaries or staged unfreezing only after measuring the conflict.

Do not add GradNorm, PCGrad, or another optimizer method before a conflict is demonstrated.

## 11. Augmentation and image processing

### 11.1 Geometric transforms

All geometric transforms must update lane coordinates and segmentation/centerline targets consistently.

Retain:

- horizontal flip
- affine translation
- small rotation
- scale perturbation

For each transform, test lane coordinates before and after transformation using synthetic lanes. A visually transformed image with stale coordinates silently destroys evidence sampling supervision.

### 11.2 Photometric transforms

Use general visibility perturbations, not official-category-specific hacks:

- brightness/contrast/color jitter
- hue/saturation jitter
- gamma jitter
- moderate low-light simulation
- Gaussian blur
- motion blur
- random shadow
- limited image occlusion

Constraints:

- Do not stack all severe transforms independently at high probability.
- Add a severity budget so one image does not simultaneously receive extreme low light, blur, shadow, and occlusion.
- Preserve a substantial clean-image probability.
- Record the sampled augmentation parameters in debug metadata.
- Verify image range, dtype, normalization, and channel order after every transform path.

### 11.3 Occlusion semantics

CULane annotations generally represent the lane through partial occlusion. Image-only occlusion can therefore train continuation, but it must be controlled.

- Restrict occluder size and road-region placement.
- Do not cover nearly the entire valid lane.
- Preserve GT labels.
- Log the fraction of visible GT pixels hidden by synthetic occlusion when possible.

### 11.4 Normalization and feature resolution

- Keep normalization consistent with the pretrained ResNet34 backbone.
- Confirm that cut-height is applied before coordinate normalization.
- Verify P2/P3 stride and sampler coordinates with unit tests.
- Do not add sharpening, CLAHE, or handcrafted night preprocessing unless an ablation proves a general gain. Such processing can shift the pretrained backbone distribution.

## 12. Training strategy

### 12.1 Initialization

Initialize from both strong S0 candidates in debug experiments:

- 175k recall-balanced checkpoint
- 225k best-F1 checkpoint

Do not initialize from a poisoned S1/S2 checkpoint for the primary experiment. The purpose is to test whether direct image-grounded refinement adds value over strong S0.

### 12.2 Phase A: identity and head warm-up

- Freeze backbone, FPN, and structured S0 head.
- Train sampler adapters, row/instance refiner, displacement head, selection head, and quality head.
- Detach sampling coordinates only in this phase.
- Use a short warm-up, approximately 3k debug steps or 3-5k full-data steps.
- Require output geometry and score histograms to remain close to S0 at initialization.

### 12.3 Phase B: structured joint training

- Unfreeze structured query head and FPN/non-backbone encoder.
- Keep backbone frozen initially.
- Remove coordinate detaches between refinement iterations.
- Keep coarse deep supervision.
- Train for a measured budget such as 25k-50k full-data steps before deciding on longer training.

### 12.4 Phase C: optional low-LR backbone adaptation

Only run this if Phase B improves validation metrics and stage transitions.

- Backbone LR: roughly 0.05-0.1 times refiner LR.
- Structured/FPN LR: roughly 0.25-0.5 times refiner LR.
- Refiner heads: base refiner LR.
- Use a fresh scheduler state only when intentionally starting a new phase; preserve model weights.
- Compare against a same-step Phase B continuation so the gain is not merely extra training.

### 12.5 Optimizer and scheduler

Initial full-data ranges:

```text
refiner/evidence/selection heads: 4e-5 to 8e-5
structured query head and FPN:    1e-5 to 3e-5
backbone:                         1e-6 to 5e-6
weight decay:                     1e-4
warm-up:                          500-1000 steps
```

Use cosine decay only after the actual phase length is fixed. Do not configure a 278k cosine schedule for an experiment that will be judged at 25k and then interpret the early LR behavior as final convergence.

Checkpoint every 10k or 25k full-data steps. Select checkpoints on validation, not test.

## 13. Memory and runtime constraints

Batch 16 is desirable but not mandatory for proving the mechanism.

Memory rules:

- Sample only selected FPN levels in v1.
- Avoid materializing dense offset-by-x-bin tensors.
- Use chunked sampling over proposals if necessary.
- Keep AMP enabled but compute softmax entropy, quality targets, and IoU in float32.
- Use gradient checkpointing only after profiling; it can hide an unnecessarily large tensor.
- Report peak allocated memory and images/second for S0 and each IGSR iteration.

Provide batch-8 fallback with gradient accumulation only if effective batch size materially affects optimization. Gradient accumulation does not reduce activation memory per sample and should not be treated as a sampler-memory fix.

## 14. Implementation file plan

No files in this section should be created until Phase 0 authorizes implementation.

### 14.1 New model files

- `dynlaneseq_eg/modeling/image_grounded_structured_refiner.py`
  - IGSR block
  - evidence encoder
  - intra-lane refiner
  - instance mixer
  - geometry, selection, quality, and visibility heads

- `dynlaneseq_eg/modeling/evidence/multiscale_lane_sampler.py`
  - coordinate conversion
  - multi-scale curve sampling
  - out-of-bounds masks
  - optional chunking

### 14.2 Model integration

- Prefer a new top-level model such as `DynLaneSeqIGSR` rather than silently changing historical S1/S2/S3 behavior.
- Reuse `DynLaneSeqEncoder` and the structured query head.
- Preserve old model classes for reproducible ablations.
- Add config gating for iteration count, feature levels, offsets, detach policy, and optional final calibrator.

### 14.3 Loss files

- Add a dedicated criterion or a clean extension that exposes every loss separately.
- Do not bury selection and offset losses inside `loss_total` without logging.
- Reuse the existing CULane-style line IoU implementation to avoid metric/target mismatch.

### 14.4 Diagnostic tools

- `dynlaneseq_eg/tools/cache_stage_predictions.py`
- `dynlaneseq_eg/tools/analyze_stage_transitions.py`
- `dynlaneseq_eg/tools/analyze_oracle_ranking.py`
- `dynlaneseq_eg/evaluation/stage_diagnostics.py`

The diagnostic tools come before the new model implementation.

### 14.5 Configs

- one 2k diagnostic config with a disjoint validation list
- one full batch-16 config
- one batch-8 memory fallback
- explicit ablation configs for geometry-only, selection-only, one iteration, two iterations, frozen, and joint training

Avoid inheritance chains so deep that the effective loss or detach settings become unclear. Every publication experiment must dump its resolved config.

## 15. Test plan

### 15.1 Sampler tests

- exact samples on synthetic constant, vertical-gradient, and diagonal-gradient feature maps
- border behavior at x=0 and x=input_w-1
- correct P2/P3 coordinate scaling
- correct invalid mask outside image
- nonzero gradient to sampled feature maps
- nonzero gradient to coordinates when detach is disabled

### 15.2 Shape and compatibility tests

- batch sizes 1, 2, 8
- proposal counts 64 and diagnostic 96
- 72 rows and configured x-bin count
- one and two refinement iterations
- evaluator-compatible final output
- no NaN with an image containing no GT lanes
- no NaN with a lane having the minimum valid row count

### 15.3 Identity initialization tests

Before training:

- mean absolute geometry change must be near zero
- selection delta must be near zero
- quality delta must be near zero
- S0 and IGSR initial predictions must match within a defined tolerance

### 15.4 Gradient tests

For final loss:

- refiner heads receive nonzero gradients
- structured query head receives gradients in joint phase
- FPN receives gradients in joint phase
- backbone receives no gradients while frozen and small nonzero gradients when enabled
- no accidental detach through `pred_x_rows`, sampler coordinates, or selection base

### 15.5 Loss tests

- all-invalid row masks produce finite zero contribution
- offset target clamping is measured correctly
- quality target follows final, not coarse, geometry
- hard-negative ranking does not select matched positives as negatives
- grouped one-to-many matches are handled consistently
- per-iteration loss weights affect only the intended stage

### 15.6 Augmentation tests

- geometric transforms update image, lane rows, segmentation, and centerline consistently
- image-only transforms do not alter lane coordinates
- motion blur supports every configured kernel size
- output image dimensions, dtype, and value range remain valid
- fixed random seed reproduces augmentation parameters

### 15.7 Numerical and performance tests

- AMP forward/backward finite
- float32 diagnostic path agrees with AMP within tolerance
- peak memory measured at batch 8 and 16
- no unbounded prediction-cache memory growth

## 16. Ablation matrix

Run in this order:

1. Strong S0 continued, no new module.
2. S0 + image sampler + displacement head only.
3. Add intra-lane row refiner.
4. Add instance mixer.
5. Add learned selection/existence residual.
6. Add quality prediction, but do not use manual quality power.
7. One iteration versus two iterations.
8. Frozen versus structured-head/FPN joint training.
9. Optional low-LR backbone adaptation.
10. Optional final calibrator.

Every row reports:

- F1, precision, recall
- TP, FP, FN
- all-raw and Oracle Top-K recall
- model Top-K recall
- rescued and killed TPs
- FP removed and created
- R@0.5 and R@0.7 proposal recall
- score-IoU correlation
- memory and throughput

Do not run several additions together and infer which one worked.

## 17. Success and stop criteria

### 17.1 Diagnostic success

The diagnosis is successful when every lost GT lane can be assigned to one of these causes:

- no usable proposal
- usable proposal but poor geometry
- usable proposal ranked below Top-K/threshold
- usable proposal removed by NMS
- usable proposal destroyed by a refinement stage

Unknown/unattributed loss should be below a small reported fraction.

### 17.2 Debug architecture gate

On a disjoint 2k validation split, IGSR must beat both:

- S0 continued for the same number of steps
- the strongest existing refiner for the same number of steps

A gain below roughly 0.3 F1 on one seed is inconclusive. Run multiple seeds before interpreting changes in that range.

More importantly, require a mechanistic gain:

- model-ranked Top-K moves toward Oracle Top-K
- rescued TP exceeds killed TP
- selection score correlates better with IoU
- active evidence improves R@0.7 or recovers low-score positives

### 17.3 Full-data gate

The first full run is justified only after the debug gate passes.

Minimum useful result over the strong S0 base:

- clear validation gain beyond checkpoint noise
- no large recall collapse hidden by precision
- measurable closure of the model-versus-oracle ranking gap

If image-grounded geometry helps but selection does not, stop and repair ranking. If selection helps precision but kills true positives like the old S3, stop and repair its targets/calibration. Do not compensate with threshold sweeps on test.

## 18. Expected outcomes and risks

### 18.1 Realistic upside

The strongest plausible gain comes from retaining S3-like FP removal while recovering true positives that S3 currently suppresses. This is more realistic than expecting another row smoother to create a 2-point jump.

### 18.2 Main risks

- S0 raw proposal recall may already be too low; then refinement cannot reach 80.
- Evidence sampling around a poor center may miss the true lane; monitor clamp ratio.
- Selection and geometry losses may conflict in shared features.
- Grouped one-to-many training may create duplicate score targets inconsistent with Top-4/NMS.
- Two refinement iterations may increase compute without adding independent evidence.
- Synthetic visibility augmentations may damage calibration if too severe.
- A joint backbone may forget the strong S0 representation if LR is too high.

### 18.3 What this plan deliberately does not promise

It does not promise 80 F1. It creates an experiment where failure has a clear interpretation. If raw proposal coverage and Oracle Top-K are insufficient, the correct conclusion will be that S0 proposal generation must change. If Oracle is strong and learned ranking cannot close the gap, the score objective or candidate representation is inadequate. Both outcomes are more valuable than another unexplained 50-epoch run.

## 19. Immediate next action

Do not implement IGSR yet.

First implement and run the three diagnostic tools in Section 14.4 on:

1. S0 175k
2. S0 225k
3. frozen S1 50k or 75k
4. CSR-S2 75k
5. the remote full S3 checkpoint that produced 77.88

The first deliverable is one comparison table containing stage transitions, Oracle Top-K, exact post-processing attribution, and score calibration. That table decides whether IGSR is authorized and which of its heads are actually necessary.
