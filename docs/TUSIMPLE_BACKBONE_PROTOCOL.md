# TuSimple Backbone Scaling Protocol

## Purpose

This experiment evaluates the frozen structured DynLaneSeq-S0 architecture with
ResNet-18, ResNet-34, ResNet-101, and DLA-34. ResNet depth is the only intended
difference among the three ResNet runs. DLA-34 is a separate backbone-family
generalization experiment; the FPN, structured decoder, losses, optimizer,
schedule, seed, and post-processing remain identical to the ResNet-34 control.

## Dataset

- Root: `/home/alki/projects/TuSimple` locally; override with `DATA_ROOT` elsewhere.
- Training: all 3,626 official training annotations from
  `label_data_0313.json`, `label_data_0601.json`, and `label_data_0531.json`.
- Test: all 2,782 entries in `test_label.json`.
- Input: crop the top 160 pixels from the 1280x720 image, then resize to 1600x640.
- Segmentation auxiliary targets are rasterized deterministically from the
  official lane JSON points. Mirror-provided segmentation labels are not used.

The three training files are intentionally used together to match the standard
TuSimple trainval protocol used by public CLRNet and CondLSTR implementations.
Consequently, `label_data_0531.json` is not used for model/checkpoint or
threshold selection in these final backbone runs.

## Controlled Variables

- Structured decoder: 4 layers
- Instances / slots: 32
- Rows: 160
- X bins: 800
- FPN / embedding width: 256
- Batch size: 8
- Gradient accumulation: 2
- Effective batch size: 16
- Seed: 3407
- Optimizer and learning rates: identical across backbones
- Schedule: cosine, 70 epochs = 15,890 optimizer iterations
- Checkpoints: every 2,270 iterations (10 epochs)
- Final checkpoint: iteration 15,890
- Postprocess: score threshold 0.30, quality power 0.50, top-k 5, lane NMS 20 input pixels

No checkpoint or threshold may be selected using TuSimple test results. The
final 70-epoch checkpoint and the fixed postprocessing configuration are used
for all four backbones.

The DLA-34 integration exposes canonical DLA levels 2--5 as C2--C5 with
strides 4/8/16/32 and channels 64/128/256/512. This is exactly the existing
ResNet-34-to-FPN interface. DLAUp, IDAUp, deformable convolution, and any
lane-specific neck changes are intentionally excluded so the experiment tests
the backbone rather than a different head/neck system.

ImageNet weights are required for the final experiment. They are downloaded as
`dla34-ba72cf86.pth`; if the legacy host is unavailable, a hash-identical mirror
is attempted. A local file can be forced with
`DYNLANESEQ_DLA34_WEIGHTS=/path/to/dla34-ba72cf86.pth`.

## Official Metrics

Report TuSimple Accuracy, FP, and FN as primary metrics. F1 is emitted only as
the derived value used by common public evaluator copies. The prediction writer
maps model-space points back through the inverse resize/crop transform and
samples each lane at the exact `h_samples` values from the annotation.

The evaluator writes `run_time=0.0`, matching the accuracy-only protocol used by
common public implementations. Runtime must be benchmarked and reported
separately with the dedicated latency tool rather than mixed into accuracy.

## Commands

Run from an activated project environment:

```bash
conda activate clrernet
```

Fresh training:

```bash
bash scripts/run_tusimple_s0_structured_query_res18_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.sh
bash scripts/run_tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.sh
bash scripts/run_tusimple_s0_structured_query_res101_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.sh
bash scripts/run_tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.sh
```

On a server with a different dataset location, prefix a command with, for
example, `DATA_ROOT=/workspace/datasets/TuSimple`.

Resume to the exact total of 15,890 optimizer iterations:

```bash
RESUME=outputs/tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0006810.pt bash scripts/resume_tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_to70ep.sh
```

Final test evaluation:

```bash
CKPT=outputs/tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt bash scripts/eval_tusimple_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep_test.sh
```

DLA-34 uses the same wrappers:

```bash
RESUME=outputs/tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0006810.pt bash scripts/resume_tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_to70ep.sh
CKPT=outputs/tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt bash scripts/eval_tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep_test.sh
```
