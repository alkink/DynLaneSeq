# AAAI-27 No-Inter Component Ablation

## Purpose

This branch isolates the contribution of inter-instance self-attention.  It is
paired with the seed-3407 full control and no-intra runs.

The only intended model change is:

```text
Full:     row-local cross-attention -> inter-instance -> intra-lane -> FFN
No-inter: row-local cross-attention ->                  intra-lane -> FFN
```

All backbone, input, FPN, structured-query, matcher, loss, optimizer,
scheduler, batch, augmentation, seed, and training-budget settings remain
identical to the full seed-3407 control.

## Frozen settings

```text
seed                  = 3407
backbone              = ResNet-34 (ImageNet pretrained)
input                 = 1600x640
structured layers     = 4
use_intra_attention   = true
use_inter_attention   = false
batch / accumulation  = 8 / 2
scheduler horizon     = 278000
checkpoint interval   = 12500
```

Disabled attention modules are omitted from the parameter set, but their
initialization RNG is consumed before they are discarded.  Therefore all
shared layer parameters remain seed-aligned with the full control.

## Train

```bash
bash scripts/run_culane_s0_structured_res34_l4_no_inter_seed3407_278k.sh
```

## Resume from the latest periodic checkpoint

```bash
bash scripts/resume_culane_s0_structured_res34_l4_no_inter_seed3407_to278k.sh
```

The resume script checks the embedded iteration, seed, intra flag, and inter
flag before restoring model, optimizer, scaler, and scheduler state.

## Validation selection

After training finishes, run the predeclared checkpoint and calibration grid:

```bash
VARIANT=no_inter \
CHECKPOINT_ITERS="175000 200000 225000 250000 278000" \
bash scripts/sweep_aaai_seed3407_val_checkpoints.sh
```

The selection rule is maximum validation F1@0.50, with mF1 and F1@0.70 as
tie-breakers.  Test results must not be used to select the checkpoint, score
threshold, or quality power.

## Final test

After validation has selected the checkpoint and calibration:

```bash
CKPT=outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_no_inter_seed3407_278k/iter_XXXXXXX.pt \
SCORE_THRESH=X.XX \
QUALITY_POWER=X.XX \
bash scripts/eval_culane_s0_structured_res34_l4_no_inter_seed3407_278k_test.sh
```

Report F1@0.50, precision, recall, F1@0.70, mF1, category scores, and crossroad
false positives.  Compare only against the independently trained seed-3407
full control in the component-ablation table.
