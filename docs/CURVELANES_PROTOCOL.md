# CurveLanes protocol

This branch reads the original CurveLanes JSON annotations directly. It does
not convert them to CULane text files, so the raw data remains untouched.

## Dataset layout

The configured root must contain this original structure:

```text
Curvelanes/
  train/{train.txt,images/,labels/}
  valid/{valid.txt,images/,labels/}
  test/{test.txt,images/}
```

The local default is `/mnt/d/Datasets/CurveLanes/Curvelanes`. On a remote
machine, point `DATA_ROOT` to the extracted root, for example
`/workspace/CurveLanes/Curvelanes`.

## Preprocessing and evaluation

CurveLanes has three native image geometries. Before all augmentation and
resizing, the dataset uses the same road crop policy as the public CondLaneNet
implementation:

| Native image size | Crop from top | Remaining road crop |
| --- | ---: | --- |
| 2560 x 1440 | 640 px | 2560 x 800 |
| 1570 x 660 | 180 px | 1570 x 480 |
| 1280 x 720 | 368 px | 1280 x 352 |

The road crop is resized to 1600 x 640. This is both the exact 2x counterpart
of the common 800 x 320 CurveLanes input setting and the resolution used by
CondLSTR's public CurveLanes transform; it preserves the same crop policy and
aspect transformation while retaining the project’s standard 800 x-bin head.

The reported public score is measured on `valid/valid.txt` (20,000 labelled
images), not `test/test.txt`, because the distributed test split has no labels.
The evaluator follows the public CurveLanes/CondLSTR protocol: native
polylines are scaled to 224 x 224, rasterized with width 5, Hungarian matched,
and counted as a hit only at IoU > 0.50.

## Main training runs

There are 100,000 training images. With batch size 8 and gradient accumulation
2, the effective batch is 16 and 312,500 optimizer steps equal exactly 50
epochs. The default seed is 3407. Checkpoints and visualizations are emitted
every 25,000 steps.

Structured ResNet-34:

```bash
DATA_ROOT=/mnt/d/Datasets/CurveLanes/Curvelanes \
  bash scripts/run_curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
```

Matched unstructured ResNet-34 ablation:

```bash
DATA_ROOT=/mnt/d/Datasets/CurveLanes/Curvelanes \
  bash scripts/run_curvelanes_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh
```

To resume without resetting the optimizer, scheduler, scaler, RNG state, or
global iteration, set `RESUME` to a checkpoint and use the corresponding
resume script. The script derives the remaining iterations from the checkpoint
and the 312,500-step target.

```bash
RESUME=outputs/curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0100000.pt \
  DATA_ROOT=/workspace/CurveLanes/Curvelanes \
  bash scripts/resume_curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_to50ep.sh
```

## Public validation command

```bash
CKPT=outputs/curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0250000.pt \
  SCORE_THRESH=0.30 QUALITY_POWER=0.50 EVAL_BATCH_SIZE=8 \
  DATA_ROOT=/mnt/d/Datasets/CurveLanes/Curvelanes \
  bash scripts/eval_curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_val.sh
```

The evaluator writes one prediction JSON under the result directory for every
entry in `valid/valid.txt`; it refuses to score an incomplete prediction set.
That prevents a partial write from silently inflating F1.

## Post-process selection

`top_k: 5` is a CULane-oriented default and should not be assumed optimal for
CurveLanes. The following cached validation sweep changes only the global lane
cap while keeping the initial score threshold (`0.30`), quality power (`0.50`),
and NMS policy fixed. Model inference runs once; all three post-process cases
reuse the saved raw outputs.

```bash
CKPT=outputs/curvelanes_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0250000.pt \
  DATA_ROOT=/mnt/d/Datasets/CurveLanes/Curvelanes \
  TOP_KS="5 8 0" \
  bash scripts/sweep_curvelanes_postprocess.sh
```

Here `top_k=0` means no global output cap. The script writes ranked F1 results
to `sweep.csv` and `sweep.json` beside the checkpoint. If this isolated test
shows that the cap matters, expand `SCORE_THRESHOLDS` and `QUALITY_POWERS` in a
second validation-only sweep; do not choose those settings on the unlabeled
`test/` split.
