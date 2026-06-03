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

## Full CULane Residual Chain

Configs do not encode checkpoint initialization by themselves. Use `--init-from`
or the helper scripts below to continue from the intended previous stage.

```bash
# S1 residual fine-tune from the strong S0 75k checkpoint.
bash scripts/run_culane_s1_strong_from_s0_75k.sh

# S2 residual fine-tune from the S1 residual 75k checkpoint.
bash scripts/run_culane_s2_strong_from_s1_residual.sh
```

Both scripts accept overrides:

```bash
DEVICE=cuda INIT_FROM=path/to/checkpoint.pt bash scripts/run_culane_s1_strong_from_s0_75k.sh
```

## Checkpoint Artifacts

Model checkpoints are not tracked in regular Git because GitHub blocks regular
repository files larger than 100 MiB. Keep checkpoints under `outputs/` locally
and distribute the required files as GitHub Release assets, Git LFS objects, or
external storage.

Required S1 initialization artifact:

```text
outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

Suggested GitHub Release upload command:

```bash
gh release create s0-strong-b16-75k \
  outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt \
  --title "S0 Strong B16 75k Checkpoint" \
  --notes "Strong S0 checkpoint used to initialize full CULane S1 residual training."
```

Suggested download command on another machine:

```bash
mkdir -p outputs/culane_s0_res34_strong_b16_giou
gh release download s0-strong-b16-75k \
  --pattern iter_0075000.pt \
  --dir outputs/culane_s0_res34_strong_b16_giou
```
