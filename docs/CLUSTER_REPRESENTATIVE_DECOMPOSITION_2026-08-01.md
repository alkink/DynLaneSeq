# Cluster/Representative Selection Decomposition

## Scope

This is a zero-training diagnostic on the frozen 10k unified-selector cache.
It localizes the remaining NMS-to-oracle gap at that checkpoint. Because every
arm sees the same curves and scores, its within-checkpoint comparison is valid;
it does not claim that the same proportions hold at 50k or 75k.

Before reporting anything, the tool requires its recomputed raw Top-4 and NMS
recall to match the preceding four-slot report exactly. A cache/checkpoint
mismatch therefore fails loudly instead of producing a misleading diagnosis.

## Exact NMS clusters

Candidates are partitioned using the actual greedy 20-pixel lane NMS trace.
Each retained candidate is a cluster representative, and every suppressed
candidate belongs to the representative that suppressed it.

For both IoU 0.50 and 0.70, the diagnostic reports:

1. `actual_nms`: deployed score ordering, selected clusters, and source
   representatives;
2. `selected_cluster_oracle_representative`: keep the same selected clusters,
   but let GT choose the best candidate inside each cluster;
3. `oracle_cluster_source_representative`: keep source representatives, but let
   GT choose the best four clusters;
4. `oracle_cluster_and_representative`: choose both clusters and one candidate
   per cluster with GT;
5. `candidate_oracle`: choose any four candidates without the NMS partition.

The counterfactuals in items 2 and 3 are parallel interventions and must not be
added together.

## Interpretation

- Large `within_selected_cluster_representative` headroom means the model finds
  the correct lane neighborhood but scores the wrong geometric copy.
- Large `cluster_ranking_with_source_representatives` headroom means the model
  ranks distinct lane hypotheses incorrectly.
- Large `joint_interaction_beyond_best_single` means neither change alone is
  sufficient: both the selected clusters and their representatives are wrong
  on many of the same images.
- Large `nms_partition_loss` means the 20-pixel NMS relation itself merges
  candidates that must remain independently selectable.

## Command

```bash
bash scripts/decompose_culane_dla34_cluster_representative_10k.sh
```

Output:

```text
outputs/diagnostics/dla34_cluster_representative_decomposition_10k/
  cluster_representative_decomposition.json
```
