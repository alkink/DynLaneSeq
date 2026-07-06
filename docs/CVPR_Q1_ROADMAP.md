# Structured Row-Token Lane Decoder: CVPR / Q1 Roadmap

Son güncelleme: 2026-07-06

Bu doküman, mevcut structured S0 modelini CVPR veya Q1 journal seviyesinde savunulabilir bir makaleye çevirmek için yapılacak işleri sıralı ve operasyonel şekilde tanımlar.

Ana ilke: modeli yeni modüllerle şişirmek değil; mevcut fikrin gerçekten çalıştığını temiz, adil, validation-seçilmiş ve tekrar edilebilir deneylerle kanıtlamak.

## 0. Mevcut durum

### 0.1 Final structured model

```text
S0 structured row-token decoder
Backbone: ResNet34
Input: 1600x640
Slots: 32
Rows: 160
x_bins: 800
FPN channels: 256
Structured decoder layers: 4
DFL: enabled, final weight 0.5
EMA: no
TTA: no
Crossgate: no
```

Frozen output/config:

```text
Output:
outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep

Config:
dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml

Manifest:
docs/result_manifest.md
```

### 0.2 Current frozen CULane result

Protocol was selected on validation, not on test:

```text
score_thresh = 0.30
quality_power = 0.50
NMS distance = 20 px
top_k = 4
checkpoint = iter_0225000.pt
```

Final structured test result:

| Model | Iter | q | Score thr. | F1@0.50 | F1@0.70 | mF1 |
|---|---:|---:|---:|---:|---:|---:|
| Structured row-token S0 | 225k | 0.50 | 0.30 | 79.98 | 67.91 | 54.97 |

mF1 is averaged over IoU thresholds `0.50:0.05:0.95`.

### 0.3 Validation-selected threshold protocol

Compact validation grid:

```text
score_thresh  = 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
quality_power = 0.25, 0.50, 0.75
selection     = primary Val F1@0.50, tie-break mF1, then F1@0.70
```

Validation sweep artifacts:

```text
outputs/paper_val_sweeps/structured_iter_0225000_val_sweep.json
outputs/paper_val_sweeps/unstructured_iter_0175000_val_sweep.json
outputs/paper_val_sweeps/unstructured_iter_0200000_val_sweep.json
outputs/paper_val_sweeps/unstructured_iter_0225000_val_sweep.json
outputs/paper_val_sweeps/unstructured_iter_0250000_val_sweep.json
```

Validation summary:

| Model/checkpoint | q | Score thr. | Val F1@0.50 | Val F1@0.70 | Val mF1 |
|---|---:|---:|---:|---:|---:|
| Structured 225k | 0.50 | 0.30 | 82.29 | 68.47 | 55.49 |
| Holistic unstructured 175k | 0.50 | 0.30 | 80.60 | 65.00 | 51.58 |
| Holistic unstructured 200k | 0.50 | 0.30 | 80.70 | 66.61 | 53.22 |
| Holistic unstructured 225k | 0.50 | 0.30 | 80.69 | 66.42 | 53.19 |
| Holistic unstructured 250k | 0.50 | 0.30 | 80.70 | 66.32 | 53.21 |

Important selection note:

- `q=0.50 / score=0.30` is validation-selected for both structured and audited unstructured checkpoints.
- The unstructured 200k, 225k, and 250k validation F1 values differ by less than `0.02` absolute F1 points.
- The main paired unstructured baseline should use 225k because it matches the structured 225k training budget while having practically identical validation performance to nearby unstructured checkpoints.
- This paired selection must be described as validation/training-budget based, not test-score based.

## 1. Paper claim

The paper should defend this claim:

```text
Whole-lane queries are structurally under-specified for 2D lane geometry.
We decompose each lane into a global instance token and ordered row-level geometry tokens.
This factorization gives a stronger inductive bias for row-wise lane localization without handcrafted line anchors.
```

Turkish shorthand:

```text
Bir şeridin tüm geometrisini tek bir global query vektörüne sıkıştırmak zayıf bir temsil.
Biz lane kimliğini instance token ile, lane geometrisini ise sıralı row token dizisiyle ayırıyoruz.
Bu yapı, el yapımı line anchor kullanmadan row-wise geometriyi daha iyi öğreniyor.
```

The paper must not be framed as “we added DFL and tuned thresholds.” DFL is a localization supervision component. The main contribution is the structured instance-to-row decoder.

Recommended title direction:

```text
Structured Row-Token Decoding for 2D Lane Detection
Instance-to-Row Token Decoding for Line-Anchor-Free Lane Detection
Structured Instance-to-Row Decoder for Lane Geometry Modeling
```

## 2. Do not do

Do not spend more time on:

- Orthogonal evidence / verifier modules.
- Crossgate / gated cross-attention as main line.
- New geometry refiners.
- Large cascade systems.
- Test-set threshold hunting.
- “Scoring-only” patching.
- New modules that make the method hard to explain.

The next phase is:

```text
core ablation, fair comparison, tight metrics, generalization, writing
```

## 3. Completed / frozen items

| Item | Status | Notes |
|---|---|---|
| Non-crossgate structured branch/result freeze | Done | See `docs/result_manifest.md` |
| Final structured CULane result | Done | 79.98 F1@0.50, 67.91 F1@0.70, 54.97 mF1 |
| Validation threshold grid | Done | Compact grid, no test-tuned threshold selection |
| Structured 225k val sweep | Done | `q=0.50`, `score=0.30` selected |
| Same-condition holistic unstructured setup | Done | Config/script exists |
| Unstructured val sweeps 175k/200k/225k/250k | Done | 225k selected for paired comparison |
| Result manifest | Done | `docs/result_manifest.md` |
| ResNet18 backbone config/scripts | Done | Ready for remote training |
| ResNet101 backbone config/scripts | Done | Separate branch; ready for optional/parallel remote training |
| Structured no-DFL | Postponed | Keep in TODO; not the next run |
| Structured L2/L6 depth ablation | Postponed | Keep in TODO; run after backbone pass |

## 4. Immediate next step

The immediate next experiment is now backbone scaling with ResNet18.

Reason:

- The paired structured/unstructured protocol and validation threshold policy are already documented.
- No-DFL and L2/L6 remain useful, but they are not blocking the next remote run.
- ResNet18 gives a fast lower-capacity scaling point and helps answer whether the structured row-token design remains effective when the backbone is weaker.

### Next priority order

1. Finalize the paired structured vs holistic test table.
2. Add tight metrics for the paired holistic baseline: F1@0.70 and mF1.
3. Run ResNet18 structured backbone scaling.
4. Optionally run ResNet101 structured backbone scaling in parallel if compute is available.
5. Then run DLA34 if the implementation path is stable.
6. Later: structured no-DFL.
7. Later: decoder depth ablation L2 / L4 / L6.
8. Later: FPN128 vs FPN256 or 1024x384 vs 1600x640 fairness check.

## 5. Phase 1 — Paired structured vs holistic baseline

This is the most important internal ablation.

### 5.1 Main comparison

Use:

```text
Structured:   iter_0225000, q=0.50, score=0.30
Unstructured: iter_0225000, q=0.50, score=0.30
```

Why 225k for unstructured:

- It matches structured training budget.
- Its validation F1 is practically identical to 200k/250k unstructured.
- It avoids test-set-based checkpoint selection.

Required table:

| Model | Iter | q | Score thr. | F1@0.50 | F1@0.70 | mF1 | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Holistic query baseline | 225k | 0.50 | 0.30 | ... | ... | ... | ... | ... |
| Structured row-token decoder | 225k | 0.50 | 0.30 | 79.98 | 67.91 | 54.97 | 86.81 | 74.14 |

Required command for holistic tight metrics if not already done:

```bash
CKPT=outputs/culane_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
SCORE_THRESH=0.30 \
QUALITY_POWER=0.50 \
IOU_THRESHOLDS="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95" \
CATEGORIES=--categories \
bash scripts/eval_culane_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_test.sh
```

Acceptance criterion:

- Structured should beat holistic by a meaningful margin in F1@0.50.
- More importantly, structured should beat holistic in F1@0.70 and/or mF1. If the tight-metric gain is larger than the loose F1 gain, the paper claim becomes much stronger.

### 5.2 How to write this comparison

Do not write:

```text
Our baseline is weak and our method improves by 4-5 F1.
```

Write:

```text
We construct a strong high-resolution holistic query baseline under the same backbone, resolution, FPN, DFL, training schedule, and validation-selected post-processing protocol. Despite this strong baseline, explicit instance-to-row token factorization improves both F1@0.50 and tight localization metrics.
```

This framing is honest and reviewer-safe.

## 6. Phase 2 — Contribution isolation ablations

These remain important for the paper, but they are postponed until after the first backbone-scaling pass.

Current decision:

```text
Do not implement no-DFL now.
Do not implement L2/L6 now.
Proceed with ResNet18 first.
```

### 6.1 Structured no-DFL

Status: postponed.

Purpose:

```text
Separate row-token decoder contribution from distributional localization supervision.
```

Configuration:

```text
Same as final structured model
w_row_dfl = 0.0
Keep all else fixed
```

Minimum run:

- Ideally full 278k schedule.
- If time is limited, evaluate 175k/225k but state schedule clearly.

Required outputs:

```text
Val sweep using the same compact grid
Test F1@0.50 / F1@0.70 / mF1
```

Acceptance criterion:

- If no-DFL is close but lower, DFL is auxiliary.
- If no-DFL collapses, paper must admit DFL is a major component.

### 6.2 Decoder depth: L2 / L4 / L6

Status: postponed.

Purpose:

```text
Show that structured decoder depth matters and that L4 is a sensible operating point.
```

Run:

```text
Structured L2
Structured L4 final
Structured L6
```

Keep fixed:

```text
ResNet34, 1600x640, slots32, rows160, x_bins800, FPN256, DFL on
```

Required table:

| Layers | F1@0.50 | F1@0.70 | mF1 | Params | FPS |
|---:|---:|---:|---:|---:|---:|
| 2 | ... | ... | ... | ... | ... |
| 4 | 79.98 | 67.91 | 54.97 | ... | ... |
| 6 | ... | ... | ... | ... | ... |

Expected:

- L2 lower.
- L4 strong.
- L6 may marginally improve or plateau.

If L6 gives significant gain, use L6 as “accuracy” variant and L4 as “balanced” variant. If L6 gives only marginal gain, keep L4 final.

### 6.3 Instance/row factorization ablation

Minimum practical version:

```text
Holistic query baseline vs structured row-token decoder
```

Optional stronger versions:

```text
Instance token only
Shared/unordered row tokens
Row positional/order removed
```

These are scientifically useful but not all are mandatory if compute is tight.

## 7. Phase 3 — Fairness / capacity controls

These answer reviewer objections:

```text
Maybe the gain is just resolution, FPN width, or bin count.
```

### 7.1 FPN128 vs FPN256

Priority: high.

Run:

```text
Structured FPN128
Structured FPN256 final
```

If possible, also:

```text
Holistic FPN128
Holistic FPN256
```

Acceptance criterion:

- Structured should remain better than holistic at the same FPN width.

### 7.2 Resolution / bin fairness

Priority: high, but after the ResNet18 backbone pass and core ablations.

Run:

```text
1024x384 / rows96 / bins512 / FPN256 / L4 / DFL
1600x640 / rows160 / bins800 / FPN256 / L4 / DFL
```

If possible, include the older:

```text
800x288 or 1024x384 S0 structured result
```

Report:

```text
pixel per bin
rows
x_bins
F1@0.50
F1@0.70
mF1
FPS
```

### 7.3 Slot count

Priority: medium.

Run:

```text
slots20
slots32 final
slots64 optional
```

Reason:

- CondLSTR uses fewer lane queries.
- More slots can improve recall but worsen FP/scoring.
- This helps explain why slots32 was selected.

## 8. Phase 4 — Backbone scaling

Backbone scaling is now active.

The first run is ResNet18 with the same structured head and the same final high-resolution recipe.

Implemented files:

```text
dynlaneseq_eg/configs/culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml
scripts/run_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
scripts/eval_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_val.sh
scripts/eval_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_test.sh
```

ResNet101 parallel branch files:

```text
dynlaneseq_eg/configs/culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml
scripts/run_culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
scripts/eval_culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_val.sh
scripts/eval_culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_test.sh
```

Run command:

```bash
bash scripts/run_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
```

ResNet101 run command:

```bash
bash scripts/run_culane_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
```

Validation checkpoints:

```text
25k, 50k, 75k, 100k, 125k, 175k, 225k
```

Default paper threshold for this family remains:

```text
score=0.30, quality_power=0.50
```

### 8.1 Recommended order

Minimum paper package:

```text
ResNet18
ResNet34
```

Recommended if compute allows:

```text
DLA34
```

Optional:

```text
ResNet101
```

### 8.2 Why not DLA34/ResNet101 immediately?

Because backbone scaling answers:

```text
Does the method scale with feature quality?
```

It does not answer:

```text
Is the structured row-token representation the reason for the gain?
```

For CVPR/Q1, the second question is more important.

### 8.3 Proposed backbone table

| Backbone | Holistic | Structured | Gain | F1@0.70 gain | Params | FLOPs |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | ... | ... | ... | ... | ... | ... |
| ResNet34 | 78.xx | 79.98 | ... | ... | ... | ... |
| DLA34 | ... | ... | ... | ... | ... | ... |
| ResNet101 | optional | optional | optional | optional | optional | optional |

Decision:

- Run ResNet18 first.
- Run DLA34 next if ResNet18 is stable and the code path is clean.
- Run ResNet101 only if compute is available and the paper needs an “accuracy scaling” row.

## 9. Phase 5 — External datasets

Needed for strong Q1 and very helpful for CVPR.

Recommended order:

1. TuSimple
2. CurveLanes
3. LLAMAS

### 9.1 TuSimple

Purpose:

```text
Low-cost external dataset sanity check.
```

Need:

- Loader/evaluator.
- Structured vs holistic same recipe if possible.

Acceptance:

- Structured should not lose the gain entirely.

### 9.2 CurveLanes

Purpose:

```text
Stress-test ordered row tokens on curved geometry.
```

This dataset is especially useful because our strongest category gain is expected around curve/local geometry.

Acceptance:

- Structured should improve curve-heavy localization, ideally visible in tight metrics.

### 9.3 LLAMAS

Purpose:

```text
Large-scale robustness check.
```

More engineering overhead; do after TuSimple/CurveLanes unless loader is already easy.

## 10. Phase 6 — Public comparison table

Compare against:

- SCNN / RESA
- UFLD / UFLDv2
- LaneATT
- CondLaneNet
- CLRNet
- CondLSTR
- LaneFormer
- BézierLaneNet / BSNet
- Lane2Seq

The table must include:

```text
Backbone
Input resolution
F1@0.50
Precision
Recall
F1@0.70 or F1@0.75 if available
EMA/TTA status if known
FPS/Params/FLOPs if available
```

Important wording:

```text
Competitive ResNet34 result with a new structured decoder.
```

Do not write:

```text
We beat SOTA everywhere.
```

We do not beat CondLSTR’s headline ResNet34 result yet, so the contribution must be framed as representation/decoder design plus strong ablations, not pure leaderboard domination.

## 11. Phase 7 — EMA and TTA

EMA/TTA are final-score tools, not core-contribution proof.

### 11.1 EMA

Do later.

Need:

- EMA state in training loop.
- Eval option for EMA.

Report:

```text
Main table: no EMA/no TTA
Appendix or separate row: +EMA
```

### 11.2 Horizontal flip TTA

Do later.

Need:

- Flip image.
- Run model.
- Mirror x coordinates back.
- Merge predictions.
- NMS/top-k.

Report separately. Do not make TTA the main ablation number.

## 12. Phase 8 — Diagnostics and figures

Required figures:

1. Architecture overview: instance token + ordered row tokens.
2. Token construction explanation: why instance token and row token are added.
3. Structured vs holistic success/failure examples.
4. Category delta plot: structured minus holistic.
5. Row-wise x distribution heatmap.
6. Optional attention visualization.
7. Cross/no-line hallucination examples.

Required analysis:

```text
curve
night
shadow
crowd
no-line
cross false positives
```

The paper should openly state:

```text
The structured prior improves completeness/local geometry but can still hallucinate lane-like structure in lane-absent scenes.
```

## 13. Writing plan

### 13.1 Introduction

Problem:

```text
Holistic lane queries compress long lane geometry into a single vector.
This is weak for curved, occluded, or locally ambiguous lane markings.
```

Solution:

```text
Factorize lane representation into instance identity and ordered row-wise geometry.
```

### 13.2 Related work

Sections:

- segmentation-based lane detection
- anchor-based lane detection
- row-wise classification
- transformer/query-based lane detection
- hierarchical query/vectorized map analogies

Critical distinctions:

- CondLaneNet: row-wise output exists, but no decoder-state ordered row tokens.
- CondLSTR: lane query generates dynamic kernels; geometry is not explicit row-token sequence.
- LaneFormer: row/column attention is encoder-side, not lane-specific row decoder.
- UFLD: row-wise classification exists, but no instance-conditioned structured decoder.
- MapTR/LATR-like works: hierarchical query analogy exists, but not this 2D perspective row-token formulation.

### 13.3 Method

Sections:

1. overview
2. instance token
3. ordered row tokens
4. instance-to-row token construction
5. row-local cross-attention
6. inter-instance attention
7. intra-lane vertical attention
8. row-wise x distribution and DFL
9. existence/range/quality heads
10. training losses
11. inference and post-processing

### 13.4 Experiments

Order:

1. datasets and metrics
2. implementation details
3. main CULane comparison
4. paired structured vs holistic ablation
5. contribution ablations
6. external datasets
7. speed/complexity
8. diagnostics/failure analysis

## 14. Updated time plan

### Week 1 — Finish paired baseline package

- Holistic 225k test with F1@0.70/mF1 if not already available.
- Category-wise structured vs holistic delta table.
- Update paper tables.

Output:

```text
docs/result_manifest.md
docs/paired_baseline_table.md
```

### Week 2 — Backbone scaling pass

- ResNet18 structured run.
- Evaluate fixed-threshold val/test at the same protocol.
- If stable, prepare DLA34 next.

Output:

```text
docs/backbone_scaling.md
```

### Week 3 — Core ablation

- Structured no-DFL.
- L2/L4/L6 depth.
- Optional: unordered/shared row token.

Output:

```text
docs/ablation_core.md
```

### Week 4 — Fairness controls

- FPN128 vs FPN256.
- 1024x384 vs 1600x640.
- slots20 vs slots32 if time allows.

Output:

```text
docs/ablation_fairness.md
```

### Week 5 — External dataset

- TuSimple first.
- CurveLanes second.
- LLAMAS optional/third.

Output:

```text
docs/cross_dataset_results.md
```

### Week 6 — Paper draft

- Full method section.
- Main tables.
- Figures.
- Failure analysis.
- Related work positioning.

Output:

```text
docs/structured_row_token_ieee_paper_draft.tex
```

## 15. Go / no-go criteria

### CVPR submission is realistic if:

- Structured beats same-condition holistic baseline under paired protocol.
- Gain appears in F1@0.70 and/or mF1, not only F1@0.50.
- No-DFL/depth/fairness ablations support the representation claim.
- At least one external dataset supports the trend.
- Writing clearly distinguishes this from CondLaneNet, CondLSTR, UFLD, and MapTR/LATR-like hierarchical queries.

### Q1 journal is realistic if:

- CULane results remain strong and clean.
- Same-condition ablations are complete.
- External dataset result exists.
- Speed/params/FLOPs are reported.
- Failure analysis is honest.

### No-go / reframe if:

- Holistic baseline closes the gap in tight metrics.
- No-DFL/depth ablations show the gain is mostly DFL or capacity.
- External datasets show no structured gain.
- Cross/no-line FP is too severe and cannot be explained.

## 16. Concrete next actions

Do these next, in order:

1. Evaluate holistic unstructured 225k with `IOU_THRESHOLDS="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95"` and `CATEGORIES=--categories`.
2. Create `docs/paired_baseline_table.md` with structured vs holistic: F1@0.50, F1@0.70, mF1, P/R, categories.
3. Push and run ResNet18 structured scaling:

```bash
bash scripts/run_culane_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
```

4. Evaluate ResNet18 at fixed checkpoints using the ResNet18 val script.
5. Optionally run/evaluate ResNet101 in parallel if the remote machine has enough memory.
6. If ResNet18/ResNet101 scaling is stable, prepare DLA34.
7. Later: prepare structured no-DFL config/script.
8. Later: prepare structured L2 and L6 configs/scripts.

Backbone recommendation:

- Current action: run ResNet18 first.
- Optional parallel action: run ResNet101 if compute is free; monitor OOM risk.
- If the goal is stronger final score / closer public baseline comparison: run DLA34 next.
- Do not let ResNet101 block DLA34 and core ablations; it is expensive and does not directly prove the main idea.
