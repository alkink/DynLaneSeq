# Frozen Multi-Scale Evidence Sufficiency Probe

## Question

The earlier diagnostics established two facts that appear contradictory:

1. the 32 frozen candidates have a high cardinality oracle, so useful curves
   frequently exist; and
2. scalar rescoring, hierarchical selection, and hard-decision losses do not
   recover that oracle gap reliably on both train and validation samples.

This probe asks where the information contract breaks. It does **not** assume
that a small loss change is the answer. It independently tests:

- the ordered final decoder row-state sequence;
- exact-center P2 evidence;
- a local P2 neighborhood around every frozen curve;
- broad P3/P4 context from the trained top-down FPN path; and
- joint or decoupled use of those sources.

The detector, backbone, FPN, decoder, candidate curves, source scores, and NMS
partition remain frozen. Only diagnostic readouts are trained.

## Why this differs from the earlier P2 curve verifier

The 75k curve-verification probe added one P2 visual residual to a quality
score. The present probe instead trains the exact two hard decisions left by
NMS:

1. marginal-coverage cluster selection without replacement; and
2. listwise representative choice inside each duplicate cluster.

It also retains the row axis before sequence mixing, compares P2 with trained
P3/P4 top-down tensors, and includes wrong-image, zero-image, wrong-state, and
zero-state controls.

## Fairness contract

All five arms have identical architecture and parameter count. They differ
only by a fixed `[P2,P3,P4] x offsets` visibility mask:

- `state`: no visual evidence;
- `p2_center`: the center P2 sample only;
- `p2_strip`: all local P2 offsets;
- `p34_context`: broad P3/P4 offsets only;
- `joint`: all sources.

P3/P4 are reconstructed from the trained top-down tensors that feed P2. The
unused `pyramid_outputs` convolutions are intentionally excluded because the
source P2-only checkpoint never trained them.

The semantically decoupled decision uses P3/P4 for cluster selection and P2
for representative choice. The inverse routing is reported as a structural
control.

## Interpreting the controls

- Correct visual evidence must beat both wrong-image and zero-image evidence.
  Otherwise an apparent gain is not image-grounded.
- Correct row states must beat wrong-state and zero-state controls before the
  ordered decoder sequence can be credited.
- A useful result must be present on a fixed train subset and a disjoint fixed
  validation subset. Validation is not used to select a checkpoint.
- The raw source Top-4, source NMS, and candidate oracle are always reported.

## Remote command

```bash
cd /workspace/DynLaneSeq
conda activate clrernet

DATA_ROOT=/workspace/CULane \
TRAIN_STEPS=1000 \
BATCH_SIZE=2 \
EVAL_BATCH_SIZE=2 \
NUM_WORKERS=8 \
AMP_DTYPE=bfloat16 \
bash scripts/probe_culane_dla34_multiscale_evidence_sufficiency_10k.sh
```

The default cache inputs are produced by the earlier frozen-selector audit:

```text
outputs/diagnostics/dla34_unified_selector_frozen_10k/train_features_2048.pt
outputs/diagnostics/dla34_unified_selector_frozen_10k/val_features_256.pt
```

Outputs:

```text
outputs/diagnostics/dla34_multiscale_evidence_sufficiency_10k/
  multiscale_evidence_sufficiency.json
  multiscale_evidence_sufficiency.pt
```

## Scope limit

A negative result rejects the evidence available through this frozen 10k
checkpoint contract. It does not prove that every later checkpoint or a
jointly trained redesigned decoder must fail. This limitation is written into
the JSON verdict so the diagnostic cannot be overclaimed later.
