# DLA-34 Ceiling Diagnosis (2026-07-27)

This note separates verified observations from architectural hypotheses. The
decoder-layer measurements below use a short, fixed subset of 64 CULane
validation images (195 valid GT lanes) and a row-wise line-IoU proxy. They are
diagnostics, not official benchmark results.

> **Sampling correction.** The original 64-image subset used in Sections 1--16
> was the first sequential slice of the ordered CULane validation list. Those
> frames come from one video sequence and are therefore correlated. The
> intervention results remain useful for falsifying the narrow mechanisms they
> directly test, but their exact recall percentages must not be generalized to
> the full validation distribution. Sections 17 onward repeat the principal
> audits with uniformly spaced images spanning indices `0--9674` of the complete
> validation list. These uniform diagnostics supersede the earlier subset for
> cross-backbone rates, query utilization, decoder-stage recall, and loss
> analysis.

## 1. Decoder-layer progression

Each cell reports raw proposal recall as `R@0.50 / R@0.70`; scores, thresholding,
Top-K, and NMS are intentionally excluded.

| Checkpoint | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| ResNet-34, 225k | 0.0 / 0.0 | 27.2 / 0.5 | 81.5 / 61.0 | 81.5 / 67.2 |
| DLA-34, 225k | 0.0 / 0.0 | 65.1 / 30.8 | 77.9 / 62.1 | 80.5 / 64.1 |

The DLA backbone is already much stronger at L2, but the proposal-recall
advantage narrows sharply by the final decoder output. For ResNet-34, L4 adds no
proposal recall at IoU 0.50 over L3 and mainly sharpens existing lanes at IoU
0.70. For DLA-34, L4 adds only 2.6 recall points at IoU 0.50 and 2.0 points at
IoU 0.70.

The normalized DFL entropy of DLA-34 falls to `0.647` at L3 but rises to `0.695`
at L4. The final block makes the distribution less sharp on average, although
the later GT-aligned audit below shows that it still improves mean geometry.
Entropy alone must therefore not be read as proof that L4 destroys lanes.

## 2. Four training groups are almost exact replicas

At L4, using all 32 proposals instead of only group 0's eight proposals adds
only `0.5` recall points at IoU 0.50 for both backbones in the short diagnostic:

| Checkpoint | All 32 | Group 0 only |
|---|---:|---:|
| ResNet-34, 225k | 81.5 / 67.2 | 81.0 / 66.2 |
| DLA-34, 225k | 80.5 / 64.1 | 80.0 / 64.1 |

The historical full ResNet-34 validation cache confirms that one group can be
deployed without lane NMS. With a quality-only score and Top-K 4, group 3
reaches `0.823872` F1 at threshold `0.20`, compared with `0.822821` for the
historical four-group, existence-quality, NMS-based pipeline. This validates
train-many/infer-one as a cleaner deployment contract, but it does not create
additional proposal coverage.

## 3. Bounded matcher cost is cleanup, not the ceiling fix

Replacing `-log(p_lane)` with bounded `-p_lane` changes:

- `0.51%` of assignments for ResNet-34 at 25k;
- `2.56%` for DLA-34 at 25k;
- `0.00%` for both backbones at 225k on the diagnostic subset.

At 225k the mean assigned line IoU is exactly unchanged. The bounded cost is
kept for better early-training scaling, but a large final F1 gain must not be
attributed to it without new evidence.

## 4. Raw C2 separability does not translate into a useful lane residual

A separate 50-image frozen-feature probe samples offsets around matched lanes.
The mean cosine distance between the center feature and features at roughly
16-pixel offsets is:

| Backbone | Raw C2 | Fused P2 |
|---|---:|---:|
| ResNet-34 | 0.163--0.175 | 0.152--0.162 |
| ResNet-101 | 0.110--0.115 | 0.111--0.126 |
| DLA-34 | 0.261--0.271 | 0.037--0.041 |

These values are not accuracy metrics, but the collapse is specific and large:
the generic additive top-down FPN strongly homogenizes DLA's raw high-resolution
detail. This observation originally motivated a raw-C2 bypass, but cosine
distance alone cannot establish whether the distinction encodes the direction
of the lane residual rather than texture or high-frequency noise.

We therefore ran an equal-capacity task probe on frozen ResNet-34 and DLA-34
features. For the same matched lane-row anchors, the probe observes nine samples
at offsets from `-32` to `+32` pixels and predicts the GT residual. Raw C2 and
fused P2 use the same `4608`-dimensional input and the same `304,842`-parameter
probe. Results below are means over three probe seeds; `gain` is the reduction
in row-coordinate MAE relative to leaving the decoder anchor unchanged.

| Backbone | Layer | Source | Balanced acc. | Direction acc. | MAE gain |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | L2 | raw C2 | 27.66 | 57.02 | +1.69 px |
| ResNet-34 | L2 | fused P2 | **43.66** | **73.91** | **+4.97 px** |
| ResNet-34 | L3 | raw C2 | 21.42 | 48.34 | +0.19 px |
| ResNet-34 | L3 | fused P2 | **24.21** | **55.81** | **+0.49 px** |
| DLA-34 | L2 | raw C2 | 21.69 | 50.80 | +0.34 px |
| DLA-34 | L2 | fused P2 | **36.26** | **60.30** | **+1.40 px** |
| DLA-34 | L3 | raw C2 | 18.91 | 46.18 | -0.01 px |
| DLA-34 | L3 | fused P2 | **26.87** | **51.37** | **+0.08 px** |

The result is unchanged on large residuals. For DLA L2 rows with errors above
16 pixels, raw C2 gives `56.8%` direction accuracy and `+0.61` px MAE gain,
whereas fused P2 gives `70.3%` and `+4.69` px. ResNet-34 P2 is stronger still
at `87.6%` and `+12.37` px.

This falsifies the strong version of the raw-detail hypothesis: DLA C2 is more
locally distinct under cosine distance, but that distinction is not more useful
for predicting the lane correction. A full raw-C2 local-sampling architecture
is therefore not justified by the current evidence.

## 5. Actual decoder transitions do not show a geometry collapse

We then measured the real decoder updates against GT while keeping lane identity
fixed by the final group-0 assignment. The table reports row MAE before and
after the transition, update-direction accuracy for rows more than four pixels
away, the fraction of already-correct rows pushed outside four pixels, and the
lane-level proposal recall transition:

| Backbone | Transition | Row MAE | Direction | Harm from 4 px | Lane R@0.50 |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | L2 -> L3 | 21.80 -> 14.22 | 89.0% | 34.7% | 26.7 -> 81.0 |
| DLA-34 | L2 -> L3 | **11.63 -> 8.74** | 80.7% | **19.8%** | **65.1 -> 77.9** |
| ResNet-34 | L3 -> L4 | 14.22 -> 12.35 | 72.6% | 11.1% | 81.0 -> 81.0 |
| DLA-34 | L3 -> L4 | **8.74 -> 7.98** | 70.6% | **10.9%** | 77.9 -> 80.0 |

DLA retains better matched row geometry through the final block. It does not
simply destroy its strong L2 lanes. The important difference is lane discovery:

- ResNet-34 retains 51/52 L2 hits and recovers 107/143 L2 misses at L3
  (`74.8%` miss recovery).
- DLA-34 retains 121/127 L2 hits but recovers only 31/68 misses
  (`45.6%` miss recovery).

DLA finds many easy lanes one block earlier, but the remaining hard lane
instances are less likely to be recovered. At L4 the two models differ by only
two group-0 hits at IoU 0.50 (158 versus 156 of 195 GT lanes). At IoU 0.70,
ResNet-34 gains 15 and loses 2 lanes from L3 to L4, while DLA gains 9 and loses
3. Thus the final gap is a lane-level coverage/sharpening difference, not a
large row-coordinate failure on already matched lanes.

## 6. NMS dependence is real but is not the backbone ceiling

A range-aware candidate audit on the same 64 images gives:

| Diagnostic recall | ResNet-34 | DLA-34 |
|---|---:|---:|
| All valid proposals, IoU 0.50 | 81.54 | 80.00 |
| Model + threshold + NMS, IoU 0.50 | 80.51 | 80.00 |
| All valid proposals, IoU 0.70 | 64.62 | 61.54 |
| Model + threshold + NMS, IoU 0.70 | 62.56 | 61.03 |
| Global model Top-4 **without NMS**, IoU 0.50 | 29.74 | 27.69 |

The no-NMS Top-4 collapse confirms that the four one-to-many groups are ranked
as near-duplicate copies. NMS restores almost all available recall. Therefore
train-many/infer-one is a worthwhile deployment cleanup, but it cannot create
the missing lane hypotheses and should not be presented as the ceiling fix.

The learned range also costs strict-IoU recall: comparing the x-only group-0
diagnostic with range-aware proposals changes R@0.70 from `65.64` to `64.62`
for ResNet-34 and from `64.10` to `61.54` for DLA. Range prediction is a
secondary DLA-specific weakness, but it does not explain IoU-0.50 saturation.

## 7. Current conclusion

The evidence rejects four simple explanations:

1. **More training groups will find more lanes.** They are replicas.
2. **Changing only `-log(p)` to `-p` will break the ceiling.** Mature
   assignments are unchanged.
3. **The DLA backbone is wired incorrectly or produces worse geometry.** Its
   L2 proposal recall and GT-aligned row geometry are substantially better.
4. **Bypassing P2 with raw C2 will expose the missing correction signal.**
   Under an equal-capacity task probe, fused P2 is consistently more predictive.

The most defensible current diagnosis is now narrower:

- DLA already produces substantially better intermediate geometry at L2, so
  its backbone and the early backbone-to-decoder interface are functional;
- its matched row geometry remains better than ResNet-34 through L4;
- fused P2 is more task-predictive than raw C2, so a raw-detail bypass is not
  the supported next move;
- the remaining weakness is recovery of hard/missing lane instances plus a
  smaller strict-IoU range penalty.

This is an architectural ceiling rather than evidence of a simple backbone or
FPN wiring bug. The decoder is effective at refining a lane that a query has
already acquired, but has no explicit dense discovery mechanism that redirects
unused/duplicate hypotheses toward a still-missing lane. The four training
groups repeat the same solution instead of increasing coverage.

ResNet-34's 79.98 run is favorable, but the measurements do not make it a
one-off miracle: its final geometry and recall are reproducibly strong. Its
specific advantage is that L2-to-L3 recovers a larger fraction of missing lane
hypotheses, whereas DLA front-loads easy-lane discovery and then saturates.

## 8. Consequence for the next train

Do not spend the only full run on a raw-C2 sampler or a larger generic FPN.
The current radius-15/deep-supervision run remains a valid low-risk test because
per-layer matching directly strengthens the weak stage transition. However,
deep supervision alone may still teach the same duplicated hypotheses.

The architecture most directly implied by the evidence is a lane-discovery
auxiliary that preserves LaneRowNet's row decoder:

1. retain P2 and the structured row-token geometry path;
2. predict a lightweight coarse lane-center/occupancy map from P2;
3. give instance queries spatially distinct coarse centers (or assign them to
   distinct center peaks) before row decoding;
4. supervise this proposal signal densely at intermediate layers;
5. keep final row distributions for precise geometry and use one inference
   group, without lane NMS.

This is closer to the useful part of CondLSTR's design--query-conditioned dense
prediction and supervision at every decoder layer--than copying its neck. It
targets the measured missing-lane recovery bottleneck while leaving the
already-strong row geometry intact.

## 9. Dense centerline evidence does not yet validate a pure routing fix

We subsequently asked a stricter zero-training question: when a group-0
structured candidate misses a GT lane, is that lane nevertheless visible in the
checkpoint's independently supervised dense centerline output?

For every valid GT row, the diagnostic extracts the nearest of eight
NMS-filtered centerline peaks. Selecting the nearest peak separately at each GT
row uses GT association and is therefore an oracle diagnostic, not a detector
result. On the same 64-image subset:

| Backbone / lane bucket | Peak rows within 15 px | Dense-peak oracle R@0.50 |
|---|---:|---:|
| ResNet-34, structured hit @0.50 | 87.8% | 87.3% |
| ResNet-34, structured miss @0.50 | 57.7% | 10.8% |
| DLA-34, structured hit @0.50 | 82.2% | 73.9% |
| DLA-34, structured miss @0.50 | 57.3% | 5.3% |

The centerline probability sampled at GT coordinates also falls from `0.501`
to `0.165` for ResNet-34 hits versus misses and from `0.456` to `0.166` for
DLA-34. Thus most final misses are not clean lane traces waiting in the current
dense auxiliary map for a query router to select. DLA's missed lanes are no
more recoverable from this output than ResNet-34's.

This result narrows, rather than reverses, the conclusion in Section 7:

- instance-level hard-lane recovery remains the observed decoder symptom;
- four-group duplication remains real;
- but a **pure query-assignment/router change is not yet causally supported**;
- the current task-aligned dense readout also lacks sufficient evidence on
  most missed lanes.

The next cheap falsification test should therefore train a small discovery
probe on **frozen P2 features**, using image-disjoint train/validation subsets.
If that probe recovers a large fraction of final misses, the information is in
P2 and the current centerline supervision/readout is inadequate. If it does
not, the bottleneck includes the backbone/FPN representation of hard lanes and
a proposal-only full run is a poor use of the training budget.

## 10. Frozen-P2 discovery probe rejects the endpoint-only proposal

We implemented the proposed train/validation-disjoint diagnostic. The base
DLA-34 checkpoint and its P2 tensor remain frozen. A `121,673`-parameter probe
predicts a dense bottom-endpoint heatmap, uses the resulting spatially distinct
seeds to initialize an 80-row structured decoder, and reads frozen P2 with one
row-aware decoder block. It was trained for 250 short steps on CULane train and
evaluated on the same fixed 64-image CULane validation subset.

| DLA-34 frozen-P2 diagnostic | Result |
|---|---:|
| Endpoint seed recall within 16 / 32 / 64 px | 23.1 / 39.0 / 52.3% |
| Endpoint recall within 32 px on current group-0 misses | 28.9% |
| Probe proposal recall at IoU 0.50 / 0.70 | 0.5 / 0.0% |
| Current group-0 misses recovered at IoU 0.50 | 0 / 38 |
| Union recall gain over the frozen decoder | 0.0 points |
| GT-endpoint-seeded proposal recall at IoU 0.50 / 0.70 | 0.0 / 0.0% |

The endpoint heatmap contains a weak but real localization signal. It is not
enough to initialize a valid full-lane hypothesis: even replacing predicted
endpoints with exact GT endpoints does not produce one valid proposal. The
newly initialized row decoder had only 250 optimization steps, so this result
alone cannot prove that all trainable proposal mechanisms will fail. It does,
however, reject the claim that an endpoint router is a cheap, already-supported
ceiling fix.

## 11. Oracle intervention on the mature decoder also fails

To remove the ambiguity caused by the newly initialized probe decoder, we ran
a second zero-training intervention on the checkpoint's mature four-layer
structured decoder. Final group-0 assignments identify the query associated
with each GT lane. We then constrain that query's visual cross-attention to a
16-pixel GT corridor while leaving all weights frozen.

| GT-informed intervention | R@0.50 change | Recovered / lost @0.50 | R@0.70 change | Mean best-IoU change |
|---|---:|---:|---:|---:|
| Correct bottom endpoint, first block | 0.0 points | 0 / 0 | 0.0 points | -0.0002 |
| Correct bottom endpoint, every block | 0.0 points | 0 / 0 | 0.0 points | -0.0027 |
| Correct full curve, first block | -24.6 points | 1 / 49 | -23.1 points | -0.1840 |

These are causal diagnostics, not deployable results: the corridors use GT and
a hard attention mask is not equivalent to a learned additive proposal prior.
Nevertheless, the result is decisive for the narrow hypothesis. Giving the
mature decoder the correct endpoint does not recover any missing lane, and
hard-constraining attention around the complete GT curve causes severe
distribution shift and destroys many existing hits.

## 12. Updated go/no-go decision

The evidence now gives a **red light to an endpoint-only query router** as the
next full training run. Its required precondition was that a spatially correct
seed should let either a small P2 decoder or the mature decoder acquire missing
lanes. Neither test satisfies that condition.

This does not show that lane discovery is irrelevant. The measured symptom
remains: DLA refines acquired lanes well but recovers fewer hard missing lanes.
It shows that the cause is deeper than assigning duplicate queries to different
bottom endpoints. The hard lanes often lack a sufficiently coherent
curve-level representation in the current P2/dense outputs, and the existing
decoder cannot turn a single local cue into a new lane.

Consequently:

1. do not spend the only full run on the tested endpoint heatmap/router;
2. do not interpret a larger generic FPN as a supported fix;
3. retain the bounded matcher and train-many/infer-one changes as cleanup, not
   as claimed ceiling solutions;
4. if a new discovery architecture is pursued later, require a
   **curve-aware, sequence-level proposal** with its own intermediate
   supervision and a short subset gate before a full run;
5. for the current budget, prefer the already prepared conservative
   supervision experiment over introducing an unvalidated proposal branch.

## 13. GT-curve-aligned feature probe separates representation from acquisition

The endpoint probe in Section 10 could not distinguish two explanations:

1. the frozen features do not contain usable evidence for hard lanes; or
2. a point seed is an insufficient interface for accessing curve-level
   evidence that is already present.

We therefore trained three equal-capacity (`156,226` trainable parameters)
sequence probes while keeping the complete lane detector frozen. During probe
training and held-out evaluation, GT geometry defines a controlled
curve-aligned corridor with four synthetic perturbations: two global
translations of 32 pixels and two opposite linear tilts spanning -32 to +32
pixels. Each probe sees the complete offset profile along the curve and
predicts a smooth row sequence. The three sources are raw backbone C2, the
projected fused P2 consumed by LaneRowNet, and the trained top-down P2/P3/P4
pyramid states. All probes use the same initialization, architecture,
optimization steps, examples, offsets, and parameter count.

This is deliberately an oracle-assisted diagnostic: GT provides the
approximate curve corridor, and the best-of-four figure selects the best
corrected perturbation for each lane. It measures conditional information
availability, not deployable detector recall.

| Backbone / source, current group-0 misses | Row MAE gain | Mean IoU after correction | Pattern R@0.50 | Direction accuracy | Best-of-four R@0.50 |
|---|---:|---:|---:|---:|---:|
| ResNet-34 C2 (37 lanes) | +0.98 px | 0.233 | 10.1% | 56.0% | 35.1% |
| ResNet-34 P2 (37 lanes) | +5.55 px | 0.335 | 21.6% | 72.3% | 70.3% |
| ResNet-34 P2/P3/P4 (37 lanes) | +5.54 px | 0.335 | 21.6% | 72.0% | 67.6% |
| DLA-34 C2 (38 lanes) | +1.29 px | 0.243 | 5.9% | 56.9% | 18.4% |
| DLA-34 P2 (38 lanes) | +5.84 px | 0.339 | 19.7% | 71.5% | 60.5% |
| DLA-34 P2/P3/P4 (38 lanes) | +5.84 px | 0.339 | 20.4% | 71.8% | 63.2% |

The main result is that fused P2 is substantially more informative than raw
C2 on the lanes missed by the current decoder. For DLA-34, P2 raises
best-of-four R@0.50 from `18.4%` to `60.5%`; for ResNet-34 it rises from
`35.1%` to `70.3%`. Thus the FPN is not destroying the relevant lane signal.
Conditioned on an approximate complete curve, a small frozen-feature decoder
can recover a substantial fraction of current misses.

The result also rules out a simple multiscale-neck explanation. Adding P3 and
P4 changes DLA-34 best-of-four recovery by only one of 38 missed lanes and
slightly reduces it for ResNet-34. The learned DLA-34 scale weights assign
`94.1%` to P2. A generic larger FPN is therefore unlikely to be the highest
leverage intervention.

The diagnosis is not purely architectural. Hard misses remain much less
decodable than already acquired lanes. With DLA-34 P2, best-of-four R@0.50 is
`60.5%` for misses versus `96.2%` for hits, and direction accuracy is `71.5%`
versus `86.2%`. ResNet-34 is stronger on the corresponding miss upper bound
(`70.3%`), which provides a plausible secondary explanation for why the
current ResNet-34 run scales better than DLA-34.

The combined evidence supports the following conclusion:

- the primary bottleneck is the **curve acquisition/interface and its
  supervision**: the current lane queries do not reliably reach coherent P2
  evidence that becomes useful once the full curve neighborhood is supplied;
- hard-lane representation quality is a secondary bottleneck, particularly for
  DLA-34;
- raw C2 bypasses, endpoint-only routing, and a generic P3/P4 neck are not
  supported as the next full-run fix;
- a future intervention should be curve-aware and sequence-level, use P2 as
  its main evidence, and receive intermediate discovery/coverage supervision
  before final row-distribution refinement.

The exact diagnostic outputs are stored in
`outputs/diagnostics/gt_curve_feature_probe/{r34,dla34}_225k.json`.

## 14. Soft attention intervention locates the post-attention bottleneck

Section 13 showed that P2 contains useful curve evidence once an approximate
curve corridor is supplied, but it did not distinguish two mechanisms:

1. a query never attends to the missing lane; or
2. attention reaches the lane but its selected position is not converted into
   row geometry.

We therefore exposed the per-head row-local cross-attention maps of the frozen
four-layer decoder. Final Hungarian group-0 assignments preserve query
identity, while raw proposal geometry--before scores, thresholding, Top-K, or
NMS--defines hit and miss buckets. Attention is measured inside a 16-pixel GT
corridor. We then add small positive biases (`+0.5`, `+1`, `+2`, and `+4`) to
the L3 attention logits inside that corridor. Unlike the earlier hard mask,
all locations remain accessible and the intervention can be swept from weak
to strong.

Natural attention is correlated with successful acquisition:

| Backbone, L3 bucket | Corridor mass | Top-1 inside | Centroid error |
|---|---:|---:|---:|
| ResNet-34, L2 miss recovered at L3 | 15.2% | 40.4% | 83.3 px |
| ResNet-34, L2 miss unresolved at L3 | 12.6% | 27.7% | 143.0 px |
| DLA-34, L2 miss recovered at L3 | 15.9% | 42.6% | 87.0 px |
| DLA-34, L2 miss unresolved at L3 | 13.9% | 30.0% | 130.0 px |

Thus missing lanes receive weaker and less spatially accurate attention.
However, the causal intervention shows that this is not the complete
bottleneck:

| Backbone, final misses | Natural L3 top-1 inside | Top-1 inside with `+2` | Attention-peak R@0.50 with `+2` | Normal output recovered, L3 only / L3--L4 | Mean row-output move, L3 only / L3--L4 |
|---|---:|---:|---:|---:|---:|
| ResNet-34 (37 misses) | 28.3% | 88.2% | 37.8% | 0 / 0 | 1.64 / 2.08 px |
| DLA-34 (38 misses) | 30.7% | 92.6% | 34.2% | 1 / 2 | 1.02 / 1.60 px |

The `attention-peak` diagnostic directly converts the most-attended x-bin at
each row into a lane after oracle association. It is not a deployable
prediction, but it proves that under a moderate bias the attention maps
contain substantially more correct row location than the ordinary output
uses. The regular row prediction moves by only one to two pixels. For final
misses, the mean assigned-lane IoU gain is only `+0.005` for ResNet-34 and
`+0.011` for DLA-34 with an L3-only `+2` bias.

Increasing the bias does not solve the interface. At `+4`, L3-only bias
recovers two IoU-0.50 misses but also loses two existing hits for each
backbone. Applying `+4` at both L3 and L4 loses four ResNet-34 hits and six
DLA-34 hits while recovering only one miss. The failure is therefore not a
matter of choosing a stronger attention gate.

### 14.1 The code path explains the measured insensitivity

The current row-feature construction is:

```python
feat_value = feat
feat_key = feat_value + x_pos
```

Cross-attention then returns a weighted sum of `feat_value`, and only that sum
is added to the row token. The x-position embedding affects **where** attention
looks through the key, but neither the attention centroid nor x-position is
included in the value passed to the row state. The final absolute coordinate
distribution is predicted later from that state:

```python
q = q + cross_attn(q, feat_key, feat_value)
row_x_logits = row_x(q)
```

Consequently, moving attention from one visually similar lane marking to
another can change the attention weights without explicitly telling the row
state where the selected evidence was located. Instance-token spatial priors
can still make the trained model work, but unused or duplicate queries cannot
be reliably redirected toward a missing curve. This also explains why a
stronger backbone improves early/easy-lane evidence without consistently
raising final coverage.

### 14.2 Updated diagnosis and required architectural property

The evidence now locates the primary ceiling **after visual selection and
before row-coordinate prediction**. Natural acquisition is weaker on hard
lanes, but forcing attention to the correct corridor does not make the current
state/output path follow it. A loss-only change may improve existing query
priors, but it cannot add the missing coordinate-carrying interface.

A supported next design should therefore:

1. keep fused P2 as the primary visual source;
2. maintain an explicit per-lane, per-row reference x-coordinate;
3. feed the attended x-coordinate or a positional encoding of it back into the
   row state, rather than returning appearance values alone;
4. update the reference coordinate across decoder layers and supervise those
   intermediate coordinates/coverage;
5. preserve a bounded residual update so precise existing lanes are not
   destroyed.

This is materially different from adding P3/P4, increasing attention bias, or
adding only another final loss. The complete intervention outputs are in
`outputs/diagnostics/attention_acquisition/`, and the coordinate-response
audit is in `outputs/diagnostics/attention_acquisition_response/`.

## 15. Frozen coordinate-adapter probe does not support a post-hoc fix

Section 14 identified a real coordinate-response problem under a causal
attention intervention, but that result used a GT-defined soft corridor. We
therefore tested whether the **natural** L3/L4 attention coordinates in the
frozen ResNet-34 model contain deployable correction information.

The detector was frozen at 225k. Four identically initialized, equal-capacity
(`77,185` parameters) bounded residual adapters were fitted for 300 steps on
the CULane training split:

1. anchor x and row index only;
2. anchor plus final row state;
3. anchor plus per-head L3/L4 attention expected-x, peak-x, entropy, and peak
   probability;
4. anchor plus both row state and attention coordinates.

All adapters used the same sequence mixer, loss, optimizer, and training
examples. Evaluation used the same held-out 64-image/195-lane validation
subset as the previous diagnosis and raw group-0 geometry before scoring,
Top-K, or NMS.

| Frozen R34 probe | Raw R@0.50 | Recovered / lost @0.50 | Raw R@0.70 | Assigned-row MAE change |
|---|---:|---:|---:|---:|
| Baseline | 81.03 | -- | 65.64 | -- |
| Anchor only | 81.03 | 1 / 1 | 63.08 | **-0.22 px** |
| State only | 80.51 | 1 / 2 | 63.59 | **-0.28 px** |
| Attention only | 81.03 | 1 / 1 | 63.08 | **-0.21 px** |
| State + attention | 81.03 | 1 / 1 | 62.56 | **-0.23 px** |

Here a negative MAE change means that the correction made row error worse.
All four variants recovered only one of the 37 baseline misses. The combined
adapter improved miss recall by exactly `0.0` points over the equal-capacity
state-only control. It therefore fails the predefined positive gate.

This falsifies the **simple post-hoc version** of the hypothesis: natural
attention centroids from the already-trained decoder cannot merely be exposed
to a small frozen-model adapter and expected to unlock the ceiling. It also
shows that adding another residual head is not automatically a no-harm
operation.

The result does not erase Section 14's causal finding. The old attention maps
were themselves learned in a system whose values carry appearance but not
position; they were never optimized to serve as coordinate measurements.
Consequently, this probe cannot determine whether an explicit reference-x
path learned jointly from initialization would make attention and coordinate
updates co-adapt. It does, however, remove the justification for treating a
plug-in coordinate adapter as a likely improvement. Any remaining test of the
architectural hypothesis must be end-to-end and should first use a short
mechanistic gate before committing to a full run.

The complete output is
`outputs/diagnostics/attention_coordinate_adapter/r34_225k.json`.

## 16. Reference-guided local P2 probe rejects curve-aligned refinement as the ceiling fix

The remaining coordinate-path hypothesis was tested more directly before
committing to a full detector run. The ResNet-34 225k detector was frozen and
four identically initialized, equal-capacity (`199,233` parameters) probes
were fitted for 300 steps:

1. explicit anchor x and row index only;
2. anchor plus the intermediate row state;
3. anchor plus P2 profiles sampled at nine offsets from -64 to +64 pixels;
4. anchor plus both row state and local P2 profiles.

All variants shared exactly the same architecture, optimizer, training
assignments, coordinate inputs, and losses. Inputs disabled by an ablation
were replaced with zeros. The probes predicted bounded row-wise residuals and
were evaluated on the same held-out 64-image/195-lane subset. Evaluation
applied each probe to all deployment-group candidates without GT association
and compared them with the mature L4 raw proposals.

### 16.1 L2 reference

| Frozen R34 L2 probe | Raw R@0.50 | Raw R@0.70 | Recovered / lost @0.50 | Assigned-row MAE |
|---|---:|---:|---:|---:|
| Mature L4 output | **81.03** | **66.15** | -- | **10.10 px** |
| Anchor only | 32.31 | 0.00 | 2 / 97 | 20.41 px |
| State only | 35.38 | 0.51 | 2 / 91 | 19.62 px |
| Local P2 | 70.77 | 34.87 | 1 / 21 | 15.52 px |
| State + local P2 | 71.79 | 39.49 | 1 / 19 | 15.02 px |

Local P2 is genuinely informative at L2: relative to the state-only control,
it reduces assigned-row MAE by `4.60` pixels. It nevertheless remains `4.92`
pixels worse than the mature L4 output, loses 19 existing R@0.50 hits, and
recovers only one of 37 mature-output misses. Thus the detector does not
appear ceiling-limited by an inability to make an early curve-aligned local
correction.

### 16.2 L3 reference

| Frozen R34 L3 probe | Raw R@0.50 | Raw R@0.70 | Recovered / lost @0.50 | Assigned-row MAE |
|---|---:|---:|---:|---:|
| Mature L4 output | **81.03** | **66.15** | -- | **10.10 px** |
| Anchor only | 80.51 | 59.49 | 3 / 4 | 12.05 px |
| State only | 80.51 | 59.49 | 3 / 4 | 12.03 px |
| Local P2 | 80.00 | 62.05 | 1 / 3 | 11.92 px |
| State + local P2 | 79.49 | 62.56 | 1 / 4 | 11.84 px |

At L3 the explicit local visual path adds only `0.19` pixels of MAE
improvement over the state-only control. More importantly, it recovers fewer
mature-output misses (`1/37`) than state alone (`3/37`) and lowers total
R@0.50. The predefined positive gate therefore fails at both viable update
locations.

### 16.3 Consequence

These results reject **curve-aligned local P2 residual refinement as the
supported one-shot ceiling intervention**. They do not say that local P2 is
useless: it substantially improves an immature L2 curve. They show that the
existing L3/L4 decoder already extracts the useful part of that signal and
that the remaining mature-output misses are not recovered by searching a
local corridor around the current curve.

Accordingly, a CLRNet-style ROI gather, deformable sampling layer, or
post-hoc dynamic reference updater should not be selected as the next full
training run on the present evidence. The remaining ceiling diagnosis should
focus on proposal formation, assignment, query diversity, and supervision
rather than another local geometry-refinement branch.

The reproducible diagnostic entry point is
`scripts/probe_culane_r34_reference_guided_p2_update.sh`.

## 17. Uniform P2 interventions reject decoder image blindness

The first distribution-spanning audit asks whether the row decoder actually
uses P2, rather than reproducing learned query priors. Thirty-two validation
images were sampled uniformly across the full ordered validation list. P2 was
then zeroed, replaced by its horizontal mean, exchanged between images, or
shifted horizontally by 64 input pixels while keeping decoder weights frozen.

Zeroing P2 or removing its horizontal structure reduces raw proposal
`R@0.50` to zero for both backbones. Exchanging P2 between images makes the
decoder outputs follow the donor image, and shifting P2 produces a same-direction
coordinate response at every layer. At L4, the mean response to a 64-pixel P2
shift is approximately `98%` of the injected displacement for ResNet-34 and
`98%` for DLA-34. DLA is already strongly image-responsive at L2
(`93%`), as is ResNet-34 (`87%`).

This causally rejects a broad but tempting explanation: the structured decoder
is not globally blind to its image input, and the backbone-to-P2-to-decoder
wiring is not silently bypassed. The remaining failure must be selective--which
lane evidence is bound to which query--rather than absence of visual grounding
in general.

The exact outputs are:

- `/tmp/decoder_image_grounding_uniform32/r34_225k.json`;
- `/tmp/decoder_image_grounding_uniform32/dla34_225k.json`.

## 18. R34 and DLA miss overwhelmingly the same lanes

A second audit sampled 64 images uniformly over the full validation split,
containing 221 valid GT lanes. It compared raw group-0 proposals before scores,
Top-K, or NMS:

| Uniform-64 diagnostic | ResNet-34 | DLA-34 | Backbone oracle union |
|---|---:|---:|---:|
| R@0.50 | 63.80 | 61.09 | 68.33 |
| R@0.70 | 45.70 | 43.44 | 50.23 |

The per-lane best-IoU correlation between backbones is `0.876`. At IoU 0.50,
125 lanes are hit by both, 16 only by ResNet-34, 10 only by DLA-34, and 70 are
missed by both. Thus `87.5%` of ResNet-34 misses and `81.4%` of DLA-34 misses
are shared. Evaluating all 32 queries does not improve the cross-backbone union
over the eight-query deployment group.

The large shared-error fraction is strong evidence against backbone capacity or
a DLA-specific FPN bug as the primary ceiling. Two materially different
backbones reach nearly the same hard-instance boundary after passing through
the same query acquisition and supervision interface.

The exact output is
`/tmp/cross_backbone_error_overlap_uniform64_r34_dla34_225k.json`.

## 19. Each eight-query group collapses to four ordinal lane roles

The same uniform-64 audit traces every group-0 query. In ResNet-34, queries
`0, 2, 4, 7` are never assigned and never achieve IoU 0.30 with any GT lane;
queries `1, 3, 5, 6` specialize mostly to the four left-to-right lane ranks.
For DLA-34, the dead queries are `0, 3, 6, 7`, while `1, 2, 4, 5` occupy the
same four ordinal roles. No useful unassigned query was observed at IoU 0.30
or 0.50.

This has two distinct interpretations:

1. four active candidates are sufficient for the CULane annotation limit of
   four lanes, so “four dead queries” alone is not proof of missing capacity;
2. the model has no reserve candidate that discovers a lane when its
   corresponding ordinal specialist fails. The other four queries remain
   background rather than becoming alternate hypotheses.

The replicated four-group training contract magnifies this behavior: it trains
four copies of the same ordinal solution, explaining the severe no-NMS ranking
collapse without explaining away the common misses.

The exact output is
`/tmp/cross_backbone_query_specialization_uniform64_r34_dla34_225k.json`.

## 20. Radius-15 deep supervision improves early DLA geometry but not coverage

The original DLA-34 run and the radius-15, deeply supervised DLA-34 run were
compared at 225k on the same uniform 64 images:

| DLA-34 checkpoint | L1 R@0.50 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| Original | 0.00 | 47.06 | 59.73 | 61.09 |
| Radius-15 + deep supervision | 59.28 | 64.25 | 64.25 | 63.35 |

Deep supervision plainly works as supervision: valid lane geometry appears at
L1 instead of only after multiple blocks, and final R@0.50 rises by 2.26
points on this diagnostic. It does not solve the ceiling:

- the exact same four queries remain dead;
- final L4 loses two IoU-0.50 lanes that were present at L3;
- 73 of 221 lanes remain common misses between the two DLA variants;
- their per-lane best-IoU correlation is `0.902`;
- the completed official test run peaks around the existing 80 band rather
  than establishing a new backbone-scaling regime.

The result rules out “the final layer simply lacked any direct loss” as a
complete explanation. Deep supervision strengthens the already chosen ordinal
hypotheses but does not teach unused hypotheses to bind to missing full curves.

The exact output is
`/tmp/dla_old_vs_r15_deepsup_query_uniform64_225k.json`.

## 21. Expected-x decoding is a symptom, not the ceiling

Frozen final row distributions were decoded with argmax, soft expectations at
temperatures `0.25--1.5`, and local-mode expectations over radii
`2--16` bins. ResNet-34's normal temperature-1 expectation gives
`63.80 / 45.70` raw recall at IoU `0.50 / 0.70`. The best alternative,
temperature `0.5`, changes this to `64.25 / 46.15`: one lane at each
threshold, while assigned-row MAE becomes `0.26` pixels worse. Argmax lowers
IoU-0.70 recall and worsens MAE by `0.78` pixels. Neither original nor
deeply-supervised DLA receives a meaningful gain from an alternative decoder.

Missed lanes do have much more diffuse distributions. ResNet-34 normalized
entropy rises from `0.357` on IoU-0.50 hits to `0.514` on misses, and
expected-to-mode distance rises from `1.11` to `6.07` pixels. But switching
to the mode does not recover them. Diffuseness is therefore evidence of
uncertainty or failed association, not proof that soft expectation is averaging
away otherwise correct modes.

The exact outputs are:

- `/tmp/row_decode_r34_uniform64.json`;
- `/tmp/row_decode_dla_old_uniform64.json`;
- `/tmp/row_decode_dla_r15_deepsup_uniform64.json`.

## 22. Geometry losses do not fight, but short lanes are underweighted

For frozen final logits, each weighted geometry loss was differentiated with
respect to the row logits. On uniformly sampled images, ResNet-34 has mean
gradient norms of:

| Weighted component | Gradient norm |
|---|---:|
| Point | 0.000091 |
| LineIoU | 0.00527 |
| DFL | 0.00893 |
| Smoothness | 0.0000036 |

The LineIoU--DFL cosine is `+0.172`, not negative. Point--LineIoU is
`+0.371`. DLA gives the same qualitative result. Hence there is no evidence
for a destructive DFL-versus-LineIoU “loss war”; DFL is simply the dominant
row-logit objective, while point and smoothness are negligible at maturity.

There is a verified reduction defect. Point and DFL currently sum all valid
rows globally before division, whereas LineIoU normalizes each lane. Lanes
shorter than 80 rows therefore receive only about half the per-lane weight they
would receive under lane-balanced reduction. Replacing point/DFL reductions
with per-lane means increases rank-3 lane gradient by `33%` for ResNet-34 and
`40%` for DLA-34. However, the complete old and lane-balanced gradients retain
cosine `0.976` and differ in norm by only about `3.3%`.

Lane-balanced reduction is a justified correction, especially for short outer
lanes. Its measured magnitude is not sufficient to claim that it alone caused
the global 80-F1 ceiling.

The exact outputs are:

- `/tmp/geometry_loss_gradients_r34_uniform8.json`;
- `/tmp/geometry_loss_gradients_dla_old_uniform8.json`.

## 23. Current root-cause boundary and next falsification gate

The distribution-spanning evidence now classifies the system as follows:

| Component | Current status | Evidence |
|---|---|---|
| Backbone / P2 wiring | Functioning | Shift, zero, and image-swap interventions |
| Fused P2 task signal | Present but weaker on hard lanes | Curve-conditioned feature probes |
| Decoder image grounding | Strong | Near-proportional output response to shifted P2 |
| Final expected-x decoder | Not the ceiling | Temperature, argmax, and local-mode sweep |
| DFL versus LineIoU | No destructive conflict | Positive gradient cosine |
| Short-lane reduction | Real secondary defect | 1.6--2.0x analytical underweighting |
| Deep supervision | Improves early geometry only | Same dead queries and common misses |
| Query-to-full-curve association | Primary remaining suspect | Shared misses and rigid ordinal roles |

The highest-value remaining experiment is therefore not another neck, local
sampler, or decoding sweep. It is a frozen-base query-conditioned dense curve
probe. For every frozen lane-row state, the probe generates a dynamic vector
and scores every horizontal P2 position on the corresponding row. Only this
small head is trained, with per-lane-balanced curve supervision under the
existing group-0 assignment.

The probe includes two anti-shortcut controls:

- P2 is exchanged between validation images while the query states stay fixed;
- query states are exchanged while P2 stays fixed.

A positive gate requires both:

1. at least `+5.0` raw union-recall points over the frozen decoder at IoU 0.50;
2. correct-image dense recall at least `3.0` points above the wrong-image
   control.

Passing that gate would support an end-to-end training-time
query-conditioned dense association auxiliary, while keeping the final
LaneRowNet row-distribution head. Failing it would reject the strongest
remaining decoder-interface hypothesis and imply that the common misses are
limited more fundamentally by representation/data ambiguity.

The implemented entry point is:

```bash
bash scripts/probe_culane_query_conditioned_dense_curve_short.sh
```

## 24. Final-state dense association is image-specific but recovers no misses

The DLA-34 225k checkpoint was frozen and the query-conditioned dense curve
probe was trained for 1000 steps. Evaluation used the same uniformly spaced
64-image/221-lane validation subset as Sections 18--21.

| Probe input / output | Raw R@0.50 | Raw R@0.70 | Paired mean IoU |
|---|---:|---:|---:|
| Frozen base decoder | 61.09 | 43.44 | 0.550 |
| Dense head, correct P2 and final row state | 21.72 | 2.71 | 0.308 |
| Dense head, wrong-image P2 | 1.36 | 0.00 | 0.097 |
| Dense head, wrong-image row state | 9.05 | 0.45 | 0.189 |
| Dense head, zero P2 | 0.00 | 0.00 | 0.006 |
| 50/50 base--dense coordinate blend | 33.94 | 6.33 | 0.373 |

The controls establish that the probe learned a real image- and state-dependent
signal: correct-image recall is 20.36 points above the wrong-image control and
zeroing P2 eliminates all valid proposals. The negative result is nevertheless
decisive for the proposed use. Of the 86 base misses at IoU 0.50 and 125 at
IoU 0.70, the dense head recovers **zero**. Every dense-head hit is already a
base hit. On paired assignments it improves 29 lanes by more than 0.02 IoU but
worsens 180; blending does not provide a no-harm correction.

The predefined positive gate therefore fails by its primary criterion:
union-recall gain is `0.0`, not the required `+5.0` points. A dense output head
attached to the mature final row state should not be promoted to a full
training run on this evidence.

The result localizes the remaining ambiguity one step earlier. A missed lane's
final row state may already be committed to the wrong curve, in which case a
correctly functioning dense readout cannot rediscover it. The next and final
cheap decomposition reuses the same dense head but replaces the final row
state with the frozen image-blind `instance_token + row_token` state before
the first decoder block. A positive result would implicate decoder state
formation; another zero-recovery result would show that neither mature states
nor static ordinal query identities can extract missing full curves from the
frozen P2 representation with this supervision.

The exact final-state output is
`/tmp/dla34_query_conditioned_dense_probe.json`. The state-source control is
selected with:

```bash
STATE_SOURCE=initial \
OUTPUT_JSON=/tmp/dla34_initial_state_dense_probe.json \
SAVE_PROBE=/tmp/dla34_initial_state_dense_probe.pt \
bash scripts/probe_culane_query_conditioned_dense_curve_short.sh
```

## 25. Initial-state dense association also fails the coverage gate

The same probe was retrained from scratch with the frozen image-blind
`instance_token + row_token` states before the first decoder block. This removes
the possibility that only an already-misdirected final state prevents the
dense head from finding a missed lane.

| Probe input / output | Raw R@0.50 | Raw R@0.70 | Paired mean IoU |
|---|---:|---:|---:|
| Frozen base decoder | 61.09 | 43.44 | 0.550 |
| Dense head, correct P2 and initial state | 15.38 | 1.36 | 0.243 |
| Dense head, wrong-image P2 | 0.90 | 0.00 | 0.091 |
| Dense head, zero P2 | 0.00 | 0.00 | 0.003 |
| 50/50 base--dense coordinate blend | 24.89 | 6.33 | 0.317 |

The wrong-state result is intentionally identical to the correct-state result:
initial token states are image-independent and therefore unchanged by a batch
roll. Correct-versus-wrong P2 still shows a real image-specific signal, but the
probe again recovers `0/86` base misses at IoU 0.50 and `0/125` at IoU 0.70.
It improves 20 paired lanes by more than 0.02 IoU and worsens 186.

Together, Sections 24--25 reject both forms of the frozen dense-readout
hypothesis:

- a mature row state cannot use an added dense P2 readout to recover a missing
  lane;
- a static ordinal query identity cannot use that readout to form a new missing
  full-curve hypothesis either.

This is not evidence that P2 contains no lane information--both probes collapse
under wrong or zero P2, and the GT-corridor probe in Section 13 established a
conditional signal. It is evidence that the missing conditional variable is a
coherent **curve hypothesis**, not another pointwise output head.

Matcher instability is also too small to explain the result. In the uniform
DLA audit, the dominant query--rank assignments are approximately
`q2 -> rank0` in 53/55 cases, `q4 -> rank1` in 53/60, `q5 -> rank2` in
51/60, and `q1 -> rank3` in 38/46. ResNet-34 is more stable still. Assignment
identity is not randomly permuted from image to image.

The exact output is `/tmp/dla34_initial_state_dense_probe.json`.

## 26. Last supported loss-level test: lane-balanced short fine-tune

The only verified supervision defect not yet causally trained is the global-row
reduction in point and DFL losses (Section 22). A diagnostic option now averages
rows within each matched lane before averaging lanes, leaving the architecture,
matcher, heads, nominal loss weights, optimizer state, and checkpoint unchanged.
Historical behavior remains the default.

This test is deliberately a 5000-iteration continuation from the exact DLA
225k checkpoint, with checkpoints at 227.5k and 230k. It is not intended as a
new benchmark run. Its question is narrower: does restoring equal lane weight
begin to recover short/rank-3 lanes under a paired checkpoint comparison?

```bash
bash scripts/finetune_culane_dla34_lane_balanced_225k_short.sh
```

If neither 2.5k nor 5k continuation improves uniform raw miss recovery, loss
reduction is only a cleanup and the remaining ceiling is the absence of a
sequence-level curve proposal/initialization mechanism. If it selectively
improves rank-3 and short-lane recovery without damaging central lanes, it
becomes the lowest-risk supervision change for a later full run.

## 27. Position is essential, but missing position alone is not the ceiling

The remaining position-path ambiguity was decomposed with three frozen DLA-34
interventions. These use the same uniformly spaced validation protocol as the
previous diagnostics and must not be reported as benchmark results.

### 27.1 The learned x embedding is functional, not ignored

The 400 learned horizontal embeddings are applied to cross-attention keys but
not values. On 32 uniformly sampled images (98 GT lanes), the baseline group-0
raw recall was `62.24%` at IoU 0.50. Perturbing the positional correspondence
caused:

| Key-position intervention | Raw R@0.50 | Mean query-row movement |
|---|---:|---:|
| Baseline | 62.24 | -- |
| Zero x embedding | 2.04 | 74.52 px |
| Reverse x order | 2.04 | 133.38 px |
| Shuffle x order | 3.06 | 68.71 px |
| Roll x order by -/+ 50 P2 columns | 0.00 / 0.00 | 145.15 / 151.32 px |

The positional component RMS is `0.187`, versus `0.250` for projected P2
features. Adjacent learned embeddings have mean cosine `0.938`, while
half-width-separated embeddings have mean cosine `-0.193`. Thus the decoder
learned an ordered, spatially meaningful key-position system and depends on
it strongly. The hypothesis that the ceiling is caused by absent or ignored
horizontal position is rejected.

### 27.2 Explicit coordinate-state feedback has only a small strict-IoU effect

A second oracle intervention converted each attention distribution into a
classifier-recognizable x code and added that code to the row state. Aligned
and horizontally reversed codes used identical norms and the same GT-corridor
attention bias.

On the 32-image subset, aligned feedback at scales 1 and 2 retained
`43.88%` R@0.70, versus `41.84%` for the reversed control. Neither aligned
setting recovered a single IoU-0.50 miss. The aligned effect is directionally
consistent but small: it changes one strict-IoU lane and does not unlock
coverage. Natural-attention feedback produces no recall gain.

### 27.3 Direct attention-to-coordinate fusion fails the no-harm test

The final test bypassed row-state encoding completely. It interpolated the
attention-x distribution to the 800 output bins and mixed it directly with the
row-coordinate probability distribution. The attention was first moved toward
the assigned GT corridor with the same moderate `+2` oracle logit bias used in
Section 14. A fine sweep was evaluated on 64 uniformly sampled images and 221
lanes:

| Attention mixture | R@0.50 before -> after | Recovered / lost @0.50 | R@0.70 before -> after | Miss assigned-IoU change | Hit assigned-IoU change |
|---:|---:|---:|---:|---:|---:|
| 0 (bias only) | 61.09 -> 61.09 | 3 / 3 | 43.44 -> 42.53 | +0.011 | -0.003 |
| 0.01 | 61.09 -> 61.09 | 3 / 3 | 43.44 -> 41.18 | +0.009 | -0.016 |
| 0.025 | 61.09 -> 60.63 | 2 / 3 | 43.44 -> 41.18 | +0.008 | -0.018 |
| 0.05 | 61.09 -> 60.63 | 2 / 3 | 43.44 -> 41.18 | +0.006 | -0.024 |
| 0.10 | 61.09 -> 58.82 | 3 / 8 | 43.44 -> 38.46 | +0.003 | -0.049 |

Correctly directed x evidence can improve a few baseline misses, so the
position-transport concern is not fictitious. It is not a safe global
correction: even a one-percent mixture loses as many IoU-0.50 hits as it
recovers and damages strict localization. Larger mixtures increasingly damage
already-correct lanes. Natural, non-oracle attention fusion is worse.

The supported boundary is therefore:

- **solid:** horizontal position is represented, ordered, and heavily used;
- **secondary weakness:** hard-lane attention/position can sometimes improve a
  miss when externally corrected;
- **not supported as the primary ceiling:** merely putting position in values,
  exposing an attention centroid, or globally mixing attention coordinates;
- **primary remaining issue:** producing a reliable missing-lane curve
  hypothesis and deciding when a candidate should be redirected without
  damaging established lanes.

The next experiment should target query acquisition and refinement arbitration,
not another unconditional positional encoding. A jointly trained explicit
reference-x decoder remains a possible architecture, but it must include a
learned no-harm/uncertainty decision and must pass a short paired gate before a
full run.

Exact outputs:

- `outputs/diagnostics/position_transport/dla34_225k_uniform32_full.json`;
- `outputs/diagnostics/position_transport/dla34_225k_uniform64_fine_fusion.json`.

## 28. Lane-balanced geometry reduction is a cleanup, not the ceiling

The final verified loss-level defect from Section 26 was trained causally. Two
otherwise matched DLA-34 continuations started from the same 225k checkpoint:

- the historical global-row reduction;
- per-lane row normalization for point and DFL losses.

Both were continued for 5000 iterations and compared on the same uniformly
spaced 64-image/221-lane validation subset.

| Reduction at 230k | Raw R@0.50 | Raw R@0.70 |
|---|---:|---:|
| Historical global rows | 61.09 | 43.44 |
| Lane-balanced rows | 61.54 | 42.53 |

Lane balancing recovers one additional lane at IoU 0.50 but loses two at IoU
0.70. Per-lane best IoU is almost unchanged
(`Pearson r = 0.9964`), and 84 IoU-0.50 misses remain common to both models.
This rejects global-row reduction as the primary capacity ceiling. Per-lane
normalization remains the cleaner objective for future training, but its
observed effect is too small and does not explain why stronger backbones fail
to create new lane hypotheses.

Exact output:

`outputs/diagnostics/dla34_global_vs_lane_balanced_230k_uniform64.json`.

## 29. Error arbitration is learnable, but the available correction is weak

The next diagnostic separated two questions that were previously conflated:

1. can the model recognize a bad lane hypothesis?
2. does its current attention provide a safe replacement for that hypothesis?

The answer to the first question is yes. On the 64-image subset, low row-peak
confidence predicts an IoU-0.50 miss with AUC `0.882` and an IoU-0.70 miss
with AUC `0.860`. Low quality and row entropy each reach approximately
`0.85--0.88` AUC. The model therefore contains a useful error/uncertainty
signal.

The second answer is no. Those uncertainty signals do not predict whether an
attention-coordinate correction will help; benefit AUC is generally near
chance. The best deployable one-lane gate changes no IoU-0.50 decisions and
recovers only one lane (`+0.45` point) at IoU 0.70, while slightly reducing
mean IoU. Even a GT-defined miss gate with the natural correction recovers
only one IoU-0.50 lane.

This moves quality/gating out of the primary-suspect position. The detector can
often identify a poor hypothesis, but it does not possess a sufficiently
reliable alternative curve to substitute for it.

Exact output:

`outputs/diagnostics/refinement_arbitration/dla34_225k_uniform64.json`.

## 30. Alternative proposals expose a small natural signal and a false oracle

Attention-derived curves were next added as extra proposals instead of
overwriting the baseline lane. This removes the no-harm requirement from the
correction itself and tests whether the four dead group-0 slots could carry
useful alternatives.

The strongest natural fixed-eight bank combines the top four baseline lanes
with four weakly fused attention alternatives. It raises raw R@0.50 by
`1.81` points (four recovered, zero lost). A smaller `0.05` fusion recovers
six lanes at IoU 0.70 (`+2.71` points) but only one at IoU 0.50. Pure natural
attention-peak and attention-expectation curves recover no missing lane.
Even a 56-candidate bank assembled from every natural alternative gains only
`2.71` points at each threshold. Thus there is a small refinement signal, not
a strong missing-lane proposal source.

A GT-corridor-biased attention peak appeared to provide a much larger
`+24.89` R@0.50 gain. A counterfactual audit invalidates that interpretation:
the same gain remains when the P2 values are replaced by zeros. Wrong-image
and horizontal-mean controls also retain large gains. The oracle is copying
the injected GT coordinate through the attention distribution; it is not
recovering visual evidence. Its image-derived increment over the strongest
image-removed control is zero.

This oracle result must not be used to motivate or quantify a final
architecture. The supported natural opportunity is at most the small
fusion-bank gain above.

Exact outputs:

- `outputs/diagnostics/attention_alternative_proposals/dla34_225k_uniform64.json`;
- `outputs/diagnostics/oracle_attention_counterfactual/dla34_225k_uniform64.json`.

## 31. CondLSTR narrows the next causal question to assignment and acquisition

A source-level audit of the local CondLSTR implementation identifies four
material differences from the historical LaneRowNet run:

1. CondLSTR performs one global Hungarian assignment over 20 queries; every GT
   has one positive query and all unmatched queries receive the no-object
   target. LaneRowNet repeats each GT once in each of four isolated groups.
2. CondLSTR uses the bounded DETR matcher term `-p`, rather than `-log(p)`.
3. Every query generates dynamic parameters that correlate against the full
   spatial encoder map and produce a query-specific dense HxW localization
   field before row decoding.
4. Location is normalized within each lane and every decoder layer is
   supervised.

Items 2 and 4 have already been isolated in LaneRowNet. Bounded matching is a
valid stability correction, while lane-balanced normalization and deep
supervision did not break the coverage ceiling. A frozen post-hoc
query-conditioned dense readout also recovered zero misses, so copying a
CondLSTR-style output head onto a mature checkpoint is not supported.

The remaining clean assignment test is therefore:

- global one-to-one matching while retaining the grouped decoder;
- global one-to-one matching plus global inter-query interaction.

If the first improves unique all-query recall, duplicate positive supervision
is causal. If only the second improves, group isolation also prevents query
diversity. If neither improves after a short paired continuation, NMS/grouping
is a cleanup issue rather than the ceiling, and the remaining architectural
difference is jointly learned query-to-image curve acquisition.

The position conclusion remains deliberately narrower than “position cannot
be involved.” Static absolute x position is present, ordered, and essential.
What is still unverified is a **lane-specific, image-derived reference curve**
or equivalent dense query-image correlation before row refinement.

## 32. Independent C2/C3/P2 acquisition does not expose hidden raw-backbone coverage

The previous C2/P2/pyramid probes were conditional refinement tests: every
probe received a curve already placed near the GT. They established that P2
preserves useful local lane evidence, but they could not distinguish a
decoder-acquisition failure from an FPN representation failure.

The missing control was therefore run with an independent ordered-curve head.
It consumes only one frozen visual source plus absolute x/y coordinate
channels and explicitly excludes LaneRowNet's lane queries, row states,
decoder outputs, matcher assignments, and predicted curves. C2 and C3 are
losslessly zero-padded to 256 channels, so all three sources use the same
`552,712`-parameter head, initialization, 1500-step training schedule, and
uniform 64-image/221-lane validation subset.

| Frozen source | Independent R@0.50 (expected / argmax) | Paired mean IoU | Correct minus wrong-image paired IoU | Base misses recovered @0.50 |
|---|---:|---:|---:|---:|
| Raw C2 | 0.00 / 2.71 | 0.114 | +0.037 | 0 / 86 |
| Raw C3 | 0.90 / 6.33 | 0.137 | +0.062 | 0 / 86 |
| Fused P2 | 20.81 / 36.65 | 0.292 | +0.195 | 1 / 86 |

P2 is substantially more usable than raw C2 or C3 for independent curve
acquisition. Thus the simple FPN is not merely destroying a rich,
independently decodable lane signal that already exists in C2/C3. Extending
the P2 probe to 5000 total steps raises its standalone R@0.50 to `32.58`, but
it still recovers only two base misses; one of those is also recovered by the
wrong-image control. The correct-image-specific incremental coverage is
therefore approximately one lane (`+0.45` point).

This result does **not** prove that the decoder is innocent. It narrows the
failure to image-to-lane-reference acquisition: GT-near P2 sampling can refine
many misses, while neither the production decoder nor an independent
fixed-order P2 head discovers those lanes without a useful reference. The
evidence rules out raw C2/C3 replacement and another unconditional positional
encoding as likely fixes. A jointly learned, image-grounded reference
acquisition mechanism remains the unresolved causal intervention.

Exact outputs:

- `outputs/diagnostics/independent_feature_curve_proposals/dla34_225k_c2_uniform64_1500steps.json`;
- `outputs/diagnostics/independent_feature_curve_proposals/dla34_225k_c3_uniform64_1500steps.json`;
- `outputs/diagnostics/independent_p2_curve_proposals/dla34_225k_uniform64.json`;
- `outputs/diagnostics/independent_p2_curve_proposals/dla34_225k_uniform64_5000steps.json`.

## 33. A frozen-P2 dense separability test rejects a decoder-only diagnosis

The independent curve head still mixed two questions: detecting lane evidence
and associating row peaks into a fixed ordered curve. A smaller bypass was
therefore run directly on frozen P2. Two identically initialized
`295,297`-parameter dense heads were trained for 1000 steps in one run:

- a production-objective control using the checkpoint's Gaussian centerline
  target and BCE with `pos_weight=2`;
- a discovery control that additionally applies a direct row-wise
  log-softmax loss at every visible GT lane center.

Neither head consumes lane queries, row states, predicted curves, references,
matcher assignments, or decoder outputs. Evaluation uses the same uniform
64-image/221-lane subset and partitions lanes by whether the production
group-0 decoder already reaches IoU 0.50.

| Dense P2 reader | 15px row-peak recall on 86 base misses | Dense-peak oracle R@0.50 on base misses | Mean dense-peak oracle IoU |
|---|---:|---:|---:|
| Checkpoint head, correct image | 47.44 | 3 / 86 (3.49) | 0.279 |
| Checkpoint head, wrong image | 27.37 | 3 / 86 (3.49) | 0.157 |
| Fresh production-BCE, correct image | 46.55 | 1 / 86 (1.16) | 0.273 |
| Fresh production-BCE, wrong image | 26.31 | 3 / 86 (3.49) | 0.151 |
| Fresh discovery loss, correct image | 46.05 | 3 / 86 (3.49) | 0.272 |
| Fresh discovery loss, wrong image | 27.11 | 3 / 86 (3.49) | 0.156 |
| Fresh discovery, horizontal mean | 1.02 | 0 / 86 (0.00) | 0.005 |

The probe itself is not generally broken: over all 221 lanes, the fresh
discovery head reaches `49.32%` dense-peak oracle recall at IoU 0.50 versus
`4.98%` with wrong-image P2, and on the 135 production hits it reaches
`78.52%`. The failure is concentrated on the exact lanes already missed by
the decoder.

Correct-image P2 clearly contains **local** image-grounded evidence on the
misses: it improves 15px row-peak recall by approximately 19 points over the
wrong-image control. However, those peaks do not form a coherent lane with
IoU 0.50. A stronger, directly supervised discovery loss does not improve
miss recovery over the long-trained checkpoint head. All three predeclared
gates therefore fail:

- frozen P2 is not globally readable for the missed lanes by this modest
  dense reader;
- the production centerline objective is not isolated as the primary cause;
- joint optimization of the checkpoint centerline head is not isolated as
  the primary cause.

This rejects the strong claim that the ceiling is located **only** in the
structured decoder, matching, NMS, or positional transport. It also does not
prove that every decoder choice is optimal. The narrower causal conclusion
is:

> hard-lane failures are already shared by the frozen visual representation
> before lane-query acquisition; the decoder cannot recover coherent curves
> from evidence that remains local and fragmented.

Combined with Section 32, raw C2/C3 substitution is not a supported fix
because those sources are even less independently decodable than fused P2.
The remaining intervention should therefore alter how lane-coherent visual
features are learned or acquired **jointly**, rather than adding another
post-hoc decoder, positional embedding, quality sweep, or centerline-head
loss to the frozen checkpoint.

Exact output:

- `outputs/diagnostics/frozen_p2_centerline_separability/dla34_225k_uniform64_1000steps.json`.
