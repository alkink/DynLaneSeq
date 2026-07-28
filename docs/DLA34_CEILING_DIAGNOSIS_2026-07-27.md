# DLA-34 Ceiling Diagnosis (2026-07-27)

This note separates verified observations from architectural hypotheses. The
decoder-layer measurements below use a short, fixed subset of 64 CULane
validation images (195 valid GT lanes) and a row-wise line-IoU proxy. They are
diagnostics, not official benchmark results.

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
