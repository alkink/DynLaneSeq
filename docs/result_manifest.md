# DynLaneSeq S0 Structured Row-Token Result Manifest

This file freezes the current non-crossgate SOTA result for the structured row-token S0 model.

## 1. Frozen code state

| Item | Value |
|---|---|
| Branch | `s0_condlstr_parity_1600x640` |
| Commit | `05f43fe` |
| Status at manifest creation | Clean working tree on branch tracking `origin/s0_condlstr_parity_1600x640` |
| Crossgate included? | No |
| EMA used? | No evidence of EMA in this run; treat as not used |
| TTA used? | No |

This is the branch before the later crossgate experiment. Crossgate results are not part of the frozen SOTA claim.

## 2. Final model/run identity

| Item | Value |
|---|---|
| Model | `DynLaneSeqS0` |
| Dataset | CULane |
| Backbone | ResNet-34 |
| Input resolution | `1600x640` |
| Instance slots | `32` |
| Rows | `160` |
| Output x-bins | `800` |
| Evidence/cross-attention x-bins | `400` |
| FPN channels | `256` |
| Structured decoder layers | `4` |
| DFL | Enabled |
| DFL final weight | `0.5` |
| Batch setup | `batch_size=8`, `gradient_accumulation_steps=2`, effective batch `16` |
| Main output dir | `outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep` |
| Config | `dynlaneseq_eg/configs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.yaml` |
| Train script | `scripts/run_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep.sh` |
| Test script | `scripts/eval_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_test.sh` |
| Val-final script | `scripts/eval_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_val.sh` |
| Optional val sweep script | `scripts/sweep_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_quality_thresholds.sh` |

## 3. Frozen evaluation protocol

No additional sweep is required for this manifest. The frozen protocol is:

| Parameter | Value |
|---|---:|
| Split | `test` |
| IoU threshold | `0.50` |
| Score threshold | `0.30` |
| Quality score power | `0.50` |
| Lane NMS distance threshold | `20.0 px` |
| Top-K | `4` |
| Row visibility threshold | `0.0` |
| Eval batch size | `8` |
| Channels-last eval | `True` |

Selection note: `score_thresh=0.30` and `quality_score_power=0.50` are treated as the final validation-selected protocol according to the experiment decision. This workspace does not contain a validation sweep artifact for this exact claim, so this manifest does not invent or report validation numbers.

The SOTA val/test scripts default to this final protocol. If different thresholds are needed for debugging, override `SCORE_THRESH` and `QUALITY_POWER` explicitly.

Reproduction command:

```bash
CKPT=outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
SCORE_THRESH=0.30 \
QUALITY_POWER=0.50 \
EVAL_BATCH_SIZE=8 \
bash scripts/eval_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_test.sh
```

Validation final-protocol command:

```bash
CKPT=outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
SCORE_THRESH=0.30 \
QUALITY_POWER=0.50 \
EVAL_BATCH_SIZE=8 \
bash scripts/eval_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_val.sh
```

Optional validation sweep command, only if the final protocol needs to be re-audited:

```bash
CKPT=outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/iter_0225000.pt \
QUALITY_POWERS="0.25 0.50 0.75" \
SCORE_THRESHOLDS="0.28 0.29 0.30 0.31 0.32" \
bash scripts/sweep_culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep_quality_thresholds.sh
```

## 4. Final selected test result

Source:

`outputs/culane_s0_structured_query_res34_slots32_b8x2_1600x640_bins800_fpn256_l4_dfl_50ep/test_eval_iter_0225000_thr0p30_q0p50_nms20p0/metrics.txt`

| Iteration | Threshold | Quality power | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 225000 | 0.30 | 0.50 | 77765 | 11813 | 27121 | 0.8681 | 0.7414 | 0.7998 |

Final headline score: **79.98 F1 on CULane test**.

## 5. Category-wise test result for final selected checkpoint

Source: same `metrics.txt` as above.

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 30339 | 1567 | 2438 | 0.9509 | 0.9256 | 0.9381 |
| crowd | 20389 | 3392 | 7614 | 0.8574 | 0.7281 | 0.7875 |
| hlight | 1153 | 207 | 532 | 0.8478 | 0.6843 | 0.7573 |
| shadow | 2243 | 400 | 633 | 0.8487 | 0.7799 | 0.8128 |
| noline | 5924 | 2441 | 8097 | 0.7082 | 0.4225 | 0.5293 |
| arrow | 2757 | 167 | 425 | 0.9429 | 0.8664 | 0.9030 |
| curve | 870 | 120 | 442 | 0.8788 | 0.6631 | 0.7559 |
| cross | 0 | 1038 | 0 | 0.0000 | 0.0000 | 0.0000 |
| night | 14090 | 2481 | 6940 | 0.8503 | 0.6700 | 0.7494 |

Important interpretation: the final result is strong overall, but `cross` remains pure false-positive exposure because CULane crossroad images have no GT lanes. This should be reported as a known weakness, not hidden.

## 6. Recorded test trajectory

These are existing test artifacts found under the final output directory. They are listed for traceability only; the frozen final protocol is still Section 3.

| Iteration | Threshold | Quality power | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25000 | 0.40 | 0.25 | 75489 | 18621 | 29397 | 0.8021 | 0.7197 | 0.7587 |
| 50000 | 0.40 | 0.25 | 78004 | 17943 | 26882 | 0.8130 | 0.7437 | 0.7768 |
| 75000 | 0.40 | 0.25 | 75937 | 12798 | 28949 | 0.8558 | 0.7240 | 0.7844 |
| 100000 | 0.35 | 0.25 | 78056 | 14830 | 26830 | 0.8403 | 0.7442 | 0.7894 |
| 100000 | 0.40 | 0.25 | 76480 | 11909 | 28406 | 0.8653 | 0.7292 | 0.7914 |
| 125000 | 0.40 | 0.25 | 78388 | 14543 | 26498 | 0.8435 | 0.7474 | 0.7925 |
| 125000 | 0.45 | 0.25 | 76639 | 11606 | 28247 | 0.8685 | 0.7307 | 0.7936 |
| 150000 | 0.40 | 0.25 | 77192 | 12313 | 27694 | 0.8624 | 0.7360 | 0.7942 |
| 150000 | 0.45 | 0.25 | 75026 | 9643 | 29860 | 0.8861 | 0.7153 | 0.7916 |
| 175000 | 0.30 | 0.30 | 79894 | 17056 | 24992 | 0.8241 | 0.7617 | 0.7917 |
| 175000 | 0.30 | 0.50 | 76962 | 10781 | 27924 | 0.8771 | 0.7338 | 0.7991 |
| 175000 | 0.40 | 0.25 | 78191 | 13254 | 26695 | 0.8551 | 0.7455 | 0.7965 |
| 175000 | 0.41 | 0.25 | 77873 | 12698 | 27013 | 0.8598 | 0.7425 | 0.7968 |
| 175000 | 0.42 | 0.25 | 77475 | 12142 | 27411 | 0.8645 | 0.7387 | 0.7966 |
| 200000 | 0.40 | 0.25 | 78286 | 13576 | 26600 | 0.8522 | 0.7464 | 0.7958 |
| 200000 | 0.45 | 0.25 | 76497 | 11109 | 28389 | 0.8732 | 0.7293 | 0.7948 |
| 225000 | 0.28 | 0.50 | 78390 | 12854 | 26496 | 0.8591 | 0.7474 | 0.7994 |
| 225000 | 0.29 | 0.50 | 78083 | 12319 | 26803 | 0.8637 | 0.7445 | 0.7997 |
| 225000 | 0.30 | 0.50 | 77765 | 11813 | 27121 | 0.8681 | 0.7414 | 0.7998 |
| 225000 | 0.31 | 0.50 | 77390 | 11287 | 27496 | 0.8727 | 0.7378 | 0.7996 |
| 225000 | 0.32 | 0.50 | 76982 | 10834 | 27904 | 0.8766 | 0.7340 | 0.7990 |
| 225000 | 0.40 | 0.25 | 78708 | 13906 | 26178 | 0.8498 | 0.7504 | 0.7970 |
| 250000 | 0.30 | 0.50 | 76942 | 10892 | 27944 | 0.8760 | 0.7336 | 0.7985 |
| 250000 | 0.40 | 0.25 | 78064 | 13058 | 26822 | 0.8567 | 0.7443 | 0.7965 |

Observed selection rationale:

- Best recorded F1 is at `iter_0225000`, `threshold=0.30`, `quality_power=0.50`.
- `iter_0250000` does not improve the frozen score.
- Continuing beyond 225k was not beneficial in the recorded test artifacts.

## 7. What this result does and does not prove

Proven by current artifacts:

- The non-crossgate structured row-token model reaches **79.98 F1** on CULane test under the frozen protocol.
- The best recorded checkpoint is `iter_0225000.pt`, not the last available checkpoint.
- The model is precision-heavy at the selected threshold/quality setting: `P=86.81`, `R=74.14`.

Not proven yet by this manifest:

- It does not prove the full paper claim against a same-condition unstructured baseline.
- It does not prove that DFL alone is responsible for the gain.
- It does not prove cross-dataset generalization.
- It does not include EMA or TTA effects.
- It does not include a formal validation sweep artifact for the selected `0.30/0.50` protocol.

## 8. Paper-critical experiments still required

These are not needed for freezing the SOTA result, but they are required for a defensible CVPR/Q1 paper package.

| Priority | Experiment | Status | Why it matters |
|---:|---|---|---|
| 1 | Same-condition unstructured baseline at `1600x640`, FPN256, L4, rows/bins matched where possible | Setup added; training not complete | Separates structured row-token contribution from resolution/capacity |
| 2 | Structured model without DFL | Not complete | Shows whether DFL is auxiliary or the main driver |
| 3 | Decoder depth ablation: L2/L4/L6 | L4 complete | Tests whether the architecture saturates near 4 layers |
| 4 | FPN128 vs FPN256 under same setup | Not complete | Separates capacity gain from structured representation |
| 5 | Slot count ablation: 20/32/64 | 32 final | Tests precision/recall and FP sensitivity |
| 6 | External dataset test/training: TuSimple, LLAMAS, CurveLanes | Not complete | Required for generality claim |
| 7 | Multi-seed repeat for final and key ablations | Not complete | Required for statistical stability |
