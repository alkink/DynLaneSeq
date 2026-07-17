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
- Shared reported postprocess: score threshold 0.20, quality power 0.25,
  top-k 5, lane NMS 20 input pixels

The completed backbone comparison uses the final 70-epoch checkpoint and one
shared post-processing configuration for every backbone. That configuration
(`score_thresh=0.20`, `quality_power=0.25`) was selected by the exploratory
ResNet-34 test-set sweep described below. It is therefore a test-selected
protocol, not an unbiased validation-selected estimate. The setting must not be
retuned separately for ResNet-18, ResNet-101, DLA-34, or the holistic baseline.

For a future unbiased protocol, train a separate ResNet-34 selection run using
only `label_data_0313.json` and `label_data_0601.json`, reserve the 358 examples
from `label_data_0531.json` for checkpoint and post-processing selection, then
freeze the selected values for all final trainval runs. A validation sweep must
not be run directly on a final trainval checkpoint because that model has
already seen the 0531 annotations.

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

Held-out ResNet-34 selection run and cached 30--70 epoch sweep:

```bash
bash scripts/run_tusimple_s0_structured_query_res34_valselect_70ep.sh
bash scripts/sweep_tusimple_s0_structured_query_res34_valselect_30to70ep.sh
```

Resume the held-out selection run if interrupted:

```bash
RESUME=outputs/tusimple_s0_structured_query_res34_valselect_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0008180.pt bash scripts/resume_tusimple_s0_structured_query_res34_valselect_to70ep.sh
```

The default grid is score threshold 0.20--0.60 in steps of 0.05 and quality
power `{0.25, 0.50, 0.75, 1.00}`. Model inference is performed once per
checkpoint and cached; the post-processing grid does not repeat GPU inference.
Official TuSimple Accuracy is the selection metric, followed by deterministic
F1/FP/FN tie breakers.

For exploratory analysis only, the full-trainval ResNet-34 checkpoints can be
swept directly on the test set:

```bash
bash scripts/sweep_tusimple_s0_structured_query_res34_test_selected_30to70ep.sh
```

This path deliberately labels its output `test_selected`. Its best value is not
an unbiased official benchmark estimate because checkpoint and post-processing
parameters are selected using test annotations. Any use in a report must state
that selection protocol explicitly.

DLA-34 uses the same wrappers:

```bash
RESUME=outputs/tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0006810.pt bash scripts/resume_tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_to70ep.sh
CKPT=outputs/tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt bash scripts/eval_tusimple_s0_structured_query_dla34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep_test.sh
```

## Matched Structured-vs-Holistic Experiment

The ResNet-34 holistic/unstructured run changes only the query representation:

- structured control: four structured instance-to-row decoder layers;
- holistic baseline: four standard lane-query decoder layers.

Dataset, input resolution, backbone, FPN, DFL, auxiliary heads, matcher, losses,
optimizer, seed, effective batch size, 70-epoch schedule, and checkpoint cadence
are inherited unchanged from the structured ResNet-34 configuration. The
holistic run must not receive a separate test-set threshold sweep. It uses the
same frozen `score_thresh=0.20` and `quality_power=0.25` setting applied to the
completed structured backbone runs.

Fresh training:

```bash
DATA_ROOT=/workspace/TUSimple BATCH_SIZE=8 GRAD_ACCUM=2 bash scripts/run_tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep.sh
```

Resume:

```bash
DATA_ROOT=/workspace/TUSimple RESUME=outputs/tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/last.pt bash scripts/resume_tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_to70ep.sh
```

Final test with the shared fixed protocol:

```bash
DATA_ROOT=/workspace/TUSimple CKPT=outputs/tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep/iter_0015890.pt bash scripts/eval_tusimple_s0_unstructured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_70ep_test.sh
```

The existing structured test-selected sweep remains a methodological
limitation and must be labelled as such in any report. Sharing its frozen
setting with the holistic baseline prevents additional baseline-specific test
tuning; it does not turn the original test-selected protocol into an unbiased
validation-selected benchmark.
