# AAAI-27 E1: Structured L4 Without Intra-Lane Attention

## Scientific change

This branch is based on frozen SOTA commit `05f43fe`.  It disables only the
vertical self-attention among row tokens belonging to the same lane instance.
Row-local visual cross-attention, grouped inter-instance self-attention, FFN,
heads, losses, matcher, augmentations, optimizer, and LR schedule are retained.

## Reproducibility

- Explicit fresh-run seed: `3407`
- Main/evidence/backbone LR: `1e-4 / 2e-4 / 1e-5`
- Scheduler: original 278k cosine horizon, 1k warmup, 0.01 minimum ratio
- Fixed comparison checkpoint: 225k
- Periodic checkpoint interval: 12.5k iterations
- Batch/accumulation: `8 x 2`
- AMP: original ResNet-34 FP16/GradScaler path; no ResNet-101 BF16 setting

The historical 79.98 run and original ResNet-101 run did not record a seed.
Therefore, if an extra GPU is available, run the included full L4 seed-3407
control in parallel for the cleanest component comparison.

## Commands

Smoke test in an isolated output directory:

```bash
MAX_ITERS=200 \
OUT_DIR=outputs/smoke_aaai27_no_intra_seed3407 \
bash scripts/run_culane_s0_structured_res34_l4_no_intra_seed3407_225k.sh
```

Full no-intra run:

```bash
bash scripts/run_culane_s0_structured_res34_l4_no_intra_seed3407_225k.sh
```

Optional same-seed full control on a second GPU/server:

```bash
bash scripts/run_culane_s0_structured_res34_l4_full_seed3407_225k.sh
```

Resume the same-seed full control:

```bash
bash scripts/resume_culane_s0_structured_res34_l4_full_seed3407_to225k.sh
```

Resume automatically from the largest zero-padded `iter_*.pt` checkpoint:

```bash
bash scripts/resume_culane_s0_structured_res34_l4_no_intra_seed3407_to225k.sh
```

Fixed-protocol validation and test:

```bash
bash scripts/eval_culane_s0_structured_res34_l4_no_intra_seed3407_225k_val.sh
bash scripts/eval_culane_s0_structured_res34_l4_no_intra_seed3407_225k_test.sh
```

Do not use intermediate test results to select a checkpoint or threshold.
