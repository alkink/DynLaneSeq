# V16.1 proposal-independent signed-registration replay

Date: 2026-08-13
Status: immutable before either held-out endpoint is evaluated

## Question

V16 put a near-global-best whole proposal inside its GT-free candidate group in
approximately 99% of the audited cases, but its learned candidate-local scorer
did not beat the raw V7 proposal.  V14 independently learned a supervised
full-width row distribution whose correct-image DFL was substantially better
than cross-clip-wrong and position-only controls, although V14's learned
proposal association did not generalize.

This audit asks one narrower question without an optimizer step:

```text
Can the fixed V14 row distribution rank the fixed V16 whole-proposal group
by direct curve-to-posterior registration?
```

This is V16.1, not V17.  It cannot authorize training, full validation, test
evaluation, threshold search, NMS search, or checkpoint selection.  It ends
after held-out-256 and validation-256 are reported.

## Frozen inputs

- V14 Stage-A fixed endpoint: iteration 227000.
- Exact V7 proposal population, route anchors, refined geometry, activity and
  writer-valid mask embedded in that checkpoint.
- Exact V16 `adaptive_voronoi_060` group builder.
- The fixed V14 held-out-256 and validation-256 lists and their deterministic
  zero-same-image/zero-same-clip negative maps.
- FP32 evaluation, threshold 0, Top-K 4, NMS 0, quality power 0.
- Test remains closed.

No model parameter is changed.  GT/targets do not enter the selection forward.

## Primary registration score

For each V7 slot `s`, row `r`, x-bin `x`, V14 produces logits `L[s,r,x]`.
The normalized row distribution is:

```text
log_p = log_softmax(L, x)
```

For every complete proposal curve `n` in the exact V16 group, bilinearly sample
`log_p` at that proposal's x coordinate.  Subtract the row maximum so that row
entropy cannot reward a candidate merely for choosing an uncertain row:

```text
q[s,n,r] = log_p[s,r, Xp[n,r]] - max_x log_p[s,r,x]
```

Rows must be visible in both the V7 slot range and the proposal range.  Row
weights are fixed and perspective-aware:

```text
w(r) = 0.10 + 0.90 * y(r)^3
```

The single predeclared primary score is:

```text
score = 0.5 * weighted_mean(q)
      + 0.5 * weighted_quantile_10pct(q)
```

The lower 10% term prevents a proposal from hiding a severe lower-road or local
misregistration behind a few easy rows.  There are no learned row weights and
no tunable lambda, uncertainty floor, score threshold, or post-hoc policy
selection.  The candidate with maximum score is gathered as one intact
proposal, including its own range.  Coordinates are never averaged or fitted.

The posterior expected x, signed candidate-minus-posterior displacement, and
absolute p90 displacement are diagnostics only; they do not alter the score.

## Causal policies

The same fixed score and groups are evaluated with:

1. correct-image V14 posterior (primary treatment),
2. deterministic cross-clip-wrong P2 posterior,
3. zero-content P2 posterior with V14 position/prior paths retained,
4. position-only P2 posterior.

Baselines are the raw V7 routed proposal and deployed V7 refined geometry.
Activity, writer-valid masks, prediction count and proposal groups are exact
V7/V16 values for every policy.

## Predeclared gate

Held-out-256 and validation-256 must each independently satisfy all of:

```text
correct registration minus raw V7 anchor:
    TP@.50 >= +3
    TP@.75 >= +6

correct registration minus cross-clip-wrong registration:
    TP@.50 >= +3
    TP@.75 >= +3

correct registration minus zero-content registration:
    TP@.50 >= +3
    TP@.75 >= +3

prediction count exact V7
selected duplicate count = 0
every selected ID belongs to its V16 group
every emitted curve/range is an exact proposal gather
optimizer steps = 0
test used = false
```

Position-only is a reported shortcut control but is not added post hoc to the
primary gate.

## Interpretation

- PASS: explicit proposal-independent curve registration is justified as the
  next architecture-level direction.  Stop and review before V17.
- FAIL: this exact V14 posterior plus fixed robust registration does not expose
  the V7 proposal oracle headroom.  It does not prove that every possible
  trainable registration model is impossible, and it must not be presented as
  such.  Stop and review before any new version.

