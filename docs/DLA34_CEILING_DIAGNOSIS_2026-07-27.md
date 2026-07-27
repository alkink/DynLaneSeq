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

The DLA backbone is already much stronger at L2, but this advantage is not
preserved by the final decoder output. For ResNet-34, L4 adds no proposal recall
at IoU 0.50 over L3 and mainly sharpens existing lanes at IoU 0.70. For DLA-34,
L4 adds only 2.6 recall points at IoU 0.50 and 2.0 points at IoU 0.70.

The normalized DFL entropy of DLA-34 falls to `0.647` at L3 but rises to `0.695`
at L4. The final block therefore makes the distribution less sharp on average
despite receiving a stronger backbone. This supports a decoder/supervision
bottleneck, not a lack of raw encoder capacity.

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

## 4. DLA detail is partly smoothed by the generic FPN

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
detail. A small frozen probe can still read some direction information from DLA
P2 (`47.8%`, versus `46.3%` for ResNet-34), so P2 is not empty. However, the raw
DLA C2 distinction is not being preserved proportionally.

ResNet-101 provides the complementary result: its P2 direction accuracy is
`59.6%`, substantially above ResNet-34, while its deployed lane result does not
scale accordingly. Therefore FPN smoothing is a DLA-specific contributor, but
the shared decoder/training path remains the broader ceiling.

## 5. Current conclusion

The evidence rejects three simple explanations:

1. **More training groups will find more lanes.** They are replicas.
2. **Changing only `-log(p)` to `-p` will break the ceiling.** Mature
   assignments are unchanged.
3. **A larger backbone merely lacks useful local evidence.** R101 and raw DLA
   features contain measurable extra evidence that the final decoder does not
   convert into better lanes.

The most defensible current diagnosis is a two-part interface bottleneck:

- the generic FPN can suppress backbone-specific high-frequency detail,
  especially for DLA-34;
- every decoder block rescans the same complete row without conditioning its
  next visual read on the geometry predicted by the previous block.

## 6. Next architecture to test after the cleanup run

Do not globally replace P2 or force all geometry through a narrow local sampler.
A lower-risk task-aligned design is:

1. L1 performs global row-local discovery from semantic P2.
2. L1 predicts a soft horizontal distribution for every lane-row state.
3. L2--L4 retain the global P2 residual path, but also read a broad-to-narrow
   differentiable band centered on the preceding distribution.
4. The local band samples a separately projected C2 detail map; for DLA this
   bypasses the smoothing measured in fused P2.
5. Each decoder layer remains supervised. The local path is mandatory rather
   than hidden behind a zero-initialized scalar gate, avoiding the previously
   observed path-of-least-resistance collapse.

This is not yet implemented in the train-many/infer-one branch. It is the next
ceiling hypothesis, and it should first be falsified with a short 25k gate
before spending a complete training budget.
