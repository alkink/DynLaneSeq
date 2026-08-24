# V34 Temporal Candidate Observability Contract

Date: 2026-08-24

## Question

Can adjacent frames distinguish an oracle-good V7 proposal from the proposal
currently selected by V7 when the current frame alone makes the wrong choice?

This is a training-free diagnostic gate. It does not claim deployable F1 gain.
Its purpose is to decide whether a learned temporal selector is justified.

## Frozen source

- Detector: mature V7 at iteration 225,000.
- Proposal geometry, route, activity, refinement, and adjacent-frame outputs are
  frozen.
- Split: CULane validation only. Test is closed.
- Frames are grouped by video clip. The 54 validation clips are deterministically
  divided into two disjoint folds.
- Up to 512 target frames are sampled independently from each fold. A target must
  have both the previous and following listed 30-frame neighbor.

## Pair population

GT is used only to identify a diagnostic pair:

1. The active V7 output slot is matched to a GT lane.
2. `wrong` is that slot's currently routed raw proposal.
3. `good` is the valid raw proposal with maximum official raster IoU to the same GT.
4. A pair enters the threshold-specific population only when
   `wrong_iou <= threshold < good_iou` and the proposal IDs differ.

GT, official IoU, and the threshold label never enter the temporal score.

## Fixed temporal score

For each target proposal:

1. Crop and resize the target and neighbor images using the exact V7 evaluation
   geometry.
2. Estimate target-to-neighbor and neighbor-to-target OpenCV DIS optical flow at
   half resolution.
3. Warp all valid proposal row points into the neighbor frame.
4. Reject row points with forward/backward error above 3 flow pixels.
5. Compute a width-30 row-wise soft line IoU against every active, refined V7 lane
   in the neighbor frame.
6. Keep the maximum support. Average previous and following support.

The primary score is therefore fixed before the full audit:

`bidirectional_selected_support`.

No score weight, distance threshold, checkpoint, or temporal direction is selected
from the result.

## Controls

- Previous-only and following-only results are reported separately.
- `all-32 bank support` tests whether temporal support exists even when the
  adjacent deployed route misses it.
- `identity/no-flow` reports whether explicit motion compensation matters.
- `wrong-clip context` replaces the real neighbor with a deterministic image from
  another clip in the same fold. It tests whether the score is using actual
  temporal correspondence rather than generic lane geometry.

## Predeclared PASS gate

All four fold/threshold cells (`A/B x .50/.75`) must independently satisfy:

- at least 30 valid diagnostic pairs;
- primary ROC AUC at least 0.70;
- primary good-over-wrong pair accuracy at least 0.65;
- primary AUC at least 0.10 above the wrong-clip AUC.

If every condition passes, a learned temporal selector is justified. If any
condition fails, this fixed temporal signal is not sufficient evidence for a new
temporal architecture; the failure must be reported rather than repaired with a
post-hoc score sweep.

