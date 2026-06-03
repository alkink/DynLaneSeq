# DynLaneSeq-EG

Evidence-Grounded Dynamic Lane Sequence Model for 2D lane detection.

This repository follows `docs/prd.md` as the canonical implementation
specification. Development is gated in the PRD order:

```text
Target builder -> S0 -> S1 -> S2 -> S3-B1 -> S4 optional
```

The default CULane dataset link is:

```text
dataset -> /home/alki/projects/CULane
```

First required checks:

```bash
python -m dynlaneseq_eg.tools.visualize_targets --config dynlaneseq_eg/configs/culane_s0_res34.yaml --num 50
python -m dynlaneseq_eg.tools.debug_one_batch --config dynlaneseq_eg/configs/debug/culane_s0_10img_overfit.yaml
python -m dynlaneseq_eg.tools.debug_overfit --config dynlaneseq_eg/configs/debug/culane_s0_10img_overfit.yaml
```

