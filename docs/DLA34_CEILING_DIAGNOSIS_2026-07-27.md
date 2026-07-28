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
