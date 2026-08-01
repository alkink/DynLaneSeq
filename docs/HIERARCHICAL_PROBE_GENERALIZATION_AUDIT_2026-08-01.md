# Hierarchical Probe Generalization Audit

## Purpose

The saved 10k hierarchical probe improved frozen validation recall by less
than one point and peaked after only 250 optimization steps. This audit
distinguishes a train-cache fit from a representation/target failure without
performing any additional optimization.

## Contract

- load the exact validation-selected cluster and representative heads;
- replay the recorded 256-image validation result exactly or stop;
- evaluate the same heads on the disjoint 2048-image train cache;
- report source-relative gains and recovered oracle headroom on both splits;
- report continuous target correlation and positive/negative score margins.

The saved checkpoint contains the useful 250-step states, not the discarded
3000-step terminal states. Therefore this audit tests the generalization of
the saved useful probe; it cannot determine how strongly a discarded later
state memorized the train cache.

## Interpretation

- train passes and validation fails: the contract is learnable in-sample but
  does not generalize;
- train and validation fail: the saved useful state does not fit the proposed
  contract even in-sample, favoring a descriptor/target redesign;
- both pass: the hierarchy is supported and can justify a joint model test.

## Command

```bash
bash scripts/audit_culane_dla34_hierarchical_cluster_selector_generalization_10k.sh
```

Output:

```text
outputs/diagnostics/hierarchical_cluster_generalization_audit.json
```
