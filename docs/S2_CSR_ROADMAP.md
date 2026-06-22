# S2 CSR Roadmap

Date: 2026-06-14

This note freezes the current diagnosis and the next S2 direction so we do not
lose the thread during later experiments.

## Current Diagnosis

The structured-query frontend changed the role of S2.

Observed full-test chain:

| Stage | F1 | Interpretation |
| --- | ---: | --- |
| Structured S0 | 75.25 | Stronger recall-oriented frontend, but high FP. |
| Structured S1 | 76.02 | Meaningful topology/global correction. |
| Structured S2 | 76.23 | Positive, but small gain over S1. |

Local 2K experiments show the same pattern:

| Variant | Best 2K F1 | Notes |
| --- | ---: | --- |
| Structured 64 S0 continue | 0.5992 | Strong S0 baseline. |
| Structured 64 S1 | 0.6364 | Large jump. |
| Structured 64 S2 | 0.6506 | Useful, but smaller than S1. |
| Structured 96 S2 | 0.6484 | 96 is not proven better than 64. |
| Structured 96 Decision S2 + quality sweep | 0.6606 | Small best-case gain, not enough to justify as a main direction yet. |

The direct conclusion is not that S2 is useless. The correct conclusion is:

**Current S2 has lost its original purpose.** S0 and S1 now already perform much
of the evidence gathering and global correction that S2 was originally expected
to add. A second Transformer-like correction stage is too similar to S1 and does
not provide a sufficiently different inductive bias.

## What S2 Should Become

S2 should not compete with S1 or S3.

Target roles:

- S1: global topology and cross-lane reasoning.
- S2: physical 1D row-sequence refinement.
- S3: active corridor refinement and final quality/ranking calibration.

Therefore the next S2 should be a **Continuous Sequence Refiner (CSR)**:

- It receives S1 lanes as 72-row sequences.
- It samples local evidence around each row.
- It preserves lateral evidence instead of mean-pooling it away.
- It applies a row-wise 1D sequence model to smooth and locally correct the
  lane.
- It outputs micro `delta_x` and an IoU-style `quality_logits`.
- It should not own final existence classification.

## S2-CSR v1 Design

Keep the old S2 available behind config. Add a new config-gated backend:

```yaml
model:
  s2_refiner_type: csr_conv1d
```

### Inputs

- S1/coarse `x_rows`: `[B, N, R]`
- FPN/local feature map
- Lateral offsets, for example `[-8, -4, 0, 4, 8]`

### Evidence Sampling

Do not collapse offsets with a blind mean.

Bad:

```python
evidence = sampled.mean(dim=offset_dim)
```

Preferred:

```python
evidence = sampled.reshape(B, N, R, C * num_offsets)
evidence = offset_fuser(evidence)
```

Reason: the left/center/right profile tells the model whether the lane marking is
to the left or right of the current estimate. Mean pooling destroys this signal.

### Refiner

Use a lightweight 1D row model first:

- depthwise separable Conv1d
- residual blocks
- optional dilation
- pre-norm where useful

The first implementation should be boring and testable. Do not start with Mamba.

### Outputs

- `delta_x`: `[B, N, R]`
- `quality_logits`: `[B, N]`
- optionally `range_delta`, only if range degradation appears in analysis

Existence should initially pass through from S1:

```python
final_exist_logits = coarse_exist_logits
```

Reason: S3 is already the main quality/ranking stage. S2 should not become a
second weak classifier unless there is clear evidence it helps.

### Losses

Use the existing geometry losses where possible:

- point/x loss
- line IoU loss
- smoothness loss
- quality loss with IoU-style targets

The primary success metric is not just F1@0.5. CSR should improve:

- `R@0.7`
- mean IoU
- median IoU
- curve category
- night/crowd stability
- final S3 performance from the same S1 initialization

## Mamba / SSM Assessment

Mamba is not automatically nonsense here. It is reasonable to consider because
S2's input is now a genuine 1D sequence:

```text
[lane candidate, 72 ordered rows, feature channels]
```

That is exactly the kind of structure where sequence models can be useful.

However, Mamba should not be the first implementation.

Reasons:

1. We do not yet know whether the missing S2 gain comes from the lack of a
   row-wise inductive bias or from ranking/post-processing.
2. A Conv1d/TCN CSR baseline is simpler and will tell us whether the direction
   has signal.
3. Mamba adds dependency, implementation, and reproducibility risk.
4. "We used Mamba" is not a contribution by itself. The contribution would be:
   **row-wise physical sequence refinement between global topology S1 and
   active-corridor S3.**

Recommended order:

1. `csr_conv1d`: prove the S2 role with a simple row-sequence refiner.
2. `csr_tcn`: add dilation/longer row context if v1 works.
3. `csr_ssm` or Mamba: only after v1/v2 show a measurable bottleneck in long
   occlusions or curve continuity.

## What Not To Do Next

- Do not make 96 queries the default based only on current results.
- Do not add CondLSTR-style dynamic kernels as the main S2 idea.
- Do not add another existence classifier in S2 unless a controlled experiment
  shows it improves final S3.
- Do not judge S2 only by F1@0.5. If S2 is a local refiner, its first proof
  should be better geometry quality, not necessarily immediate F1.

## Minimum Ablation Plan

For each S2 candidate:

1. Train from the same S1 checkpoint.
2. Evaluate S2-only with score threshold sweep.
3. Run proposal recall with:
   - `TOP_K=0 RANK_BY=none`
   - `TOP_K=4 RANK_BY=score`
   - `TOP_K=4 RANK_BY=score_quality`
4. Compare:
   - F1@0.5
   - R@0.5
   - R@0.7
   - meanIoU
   - curve/night/crowd categories
5. Only then train S3 from that S2 checkpoint.

Decision rule:

- If CSR improves R@0.7/meanIoU but not S2 F1, still test S3.
- If CSR does not improve geometry metrics, stop before full S3.
- If CSR only improves F1 by quality pruning but hurts proposal recall, do not
  promote it.

