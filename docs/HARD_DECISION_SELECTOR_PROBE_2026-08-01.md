# Frozen Hard-Decision Selector Probe

## Motivation

The hierarchical quality probe learned broad positive/background calibration
but did not improve Top-4 decisions, even on its frozen train cache. High
global target correlation therefore did not imply correct ordering among
hard, geometrically similar candidates.

## Decision-aligned supervision

This probe removes easy per-item BCE supervision.

1. Cluster selection is autoregressive and without replacement. At every
   step, its target is the marginal one-to-one GT coverage contributed by each
   remaining NMS cluster. Selecting a duplicate of an already covered lane has
   zero marginal target.
2. Representative selection uses listwise cross-entropy only inside positive
   multi-candidate clusters with a measurable quality spread. Background
   candidates outside the cluster do not contribute to this loss.

The detector, candidate curves, descriptors, and the 20-pixel NMS partition
remain frozen at 10k. Both train and validation results are reported using the
validation-selected probe states.

## Falsification rule

- validation passes both independent arms and the combined arm: hard-decision
  supervision is supported and can justify a joint model experiment;
- train passes but validation fails: the descriptors permit memorization but
  do not generalize the hard decision;
- train and validation fail: changing only the decision loss cannot unlock
  the frozen descriptors, motivating row-preserving local-neighborhood or
  semantic evidence rather than another scalar-head patch.

## Command

```bash
bash scripts/probe_culane_dla34_hard_decision_selector_10k.sh
```

Outputs:

```text
outputs/diagnostics/dla34_hard_decision_selector_10k/
  hard_decision_selector.json
  hard_decision_selector.pt
```
