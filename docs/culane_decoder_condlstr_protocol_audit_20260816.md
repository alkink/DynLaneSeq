# Official CULane decoder and CondLSTR protocol audit — 2026-08-16

## Population contract

- Training is the untouched official `list/train.txt`: 88,880 images.
- Validation is the untouched official `list/val.txt`: 9,675 images.
- No image exclusion, deduplication, checkpoint selection, threshold sweep, or test-set selection is used.
- Train/validation overlap: 0; duplicate list entries: 0.
- Empty annotation files are retained: 10,459 train and 739 validation images.
- `train.txt` SHA-256: `8a8b0d7cbe85f3c1e571015597ab55211ca423888c1bed8f715be08b415d3ac0`.
- `val.txt` SHA-256: `01f04df210879e3c52bc3994b834eba71135250c2a06c56bc8fef0450dd7610d`.

## DynLaneSeq full-validation decoder parity

Fixed checkpoint/config, fixed score threshold, and the same 9,675 cached validation records were used for every arm. Official CULane C evaluation used `-p 1` so every image was processed.

| Decoder treatment | F1@.50 | Delta | F1@.75 | Delta |
|---|---:|---:|---:|---:|
| Current DynLaneSeq | 82.2498 | — | 62.1345 | — |
| CLRNet spline writer only | 82.1303 | -0.1195 | 60.6977 | -1.4368 |
| CLRNet-style bottom range extension only | 73.8123 | -8.4375 | 44.8659 | -17.2686 |
| CLRNet 50-pixel-at-800 NMS distance only | 82.2511 | +0.0013 | 62.1355 | +0.0010 |
| All three CLR-style treatments | 75.4835 | -6.7663 | 45.5878 | -16.5467 |

Paired image results show the same pattern. At IoU .50, the spline writer improved 38 images and worsened 58; the range treatment improved 14 and worsened 2,620; changing only the NMS distance changed no image-level TP count.

The current writer's disk serialization accounts for a small but real `-10/+10/+10` TP/FP/FN difference relative to the cache's in-memory float geometry at IoU .50. The official C evaluator and the Python raster evaluator agree after written predictions are reloaded.

## Evaluator coverage bug

The local compiled CULane evaluator fork computes `batch_size = NUM / NUM_PROCESS` and launches exactly `NUM_PROCESS` equal floor-sized chunks. With 9,675 validation images and `-p 20`, it evaluates only 9,660 images.

| Evaluator mode | Effective images | TP/FP/FN @.50 | F1@.50 |
|---|---:|---:|---:|
| `-p 1` | 9,675 | 25,474 / 3,787 / 7,208 | 82.2498 |
| `-p 20` | 9,660 | 25,421 / 3,781 / 7,201 | 82.2367 |

The dropped 15-image remainder contains 60 GT lanes and 59 predictions. All official comparisons in this audit therefore force `-p 1`.

## CondLSTR protocol corrections

The upstream checkout cannot be used as a fair official CULane control unchanged:

1. Its preprocessing merges `train_gt.txt` and `val.txt` for training and maps `test.txt` to validation.
2. It does not preserve the official empty-label population safely: imgaug can segfault on empty line strings and mmcv can segfault while padding an `H×W×0` mask.
3. Its default mask collation assumes a fixed number of lane instances, while CULane lane counts vary per image.
4. Its training-time validation metric is Chamfer F1, not the official CULane strip-IoU evaluator.

The committed protocol patch changes the split to official `train.txt → val.txt`, retains every empty image, keeps variable instance masks as per-image tensors, adds lane attributes required by the model, and runs official C evaluation at IoU .50/.75 with `-p 1`. A mixed batch containing one empty-label and one three-lane official frame passed train/validation transforms, collation, the full model forward/backward step, and the validation evaluator. The two-image toy run then exited only at upstream's final `model_best` lookup because a zero toy metric never satisfies its strict `>` save rule; a real positive endpoint creates that file.

Full CondLSTR training is intentionally endpoint-only for model selection: validation is opened at the final epoch, followed by conversion of all 9,675 outputs and official C evaluation. No test result is used for selection.

## Reproducibility

- Branch: `codex/culane-official-parity-20260816`
- Commits: `67053ae`, `56f1b71`
- Full local JSON: `outputs/diagnostics/culane_decoder_parity_full_val_20260816/decoder_parity_audit.json`
- Full local Markdown: `outputs/diagnostics/culane_decoder_parity_full_val_20260816/decoder_parity_audit.md`
- CondLSTR population manifest: `outputs/diagnostics/condlstr_culane_official_protocol_audit.json`

The remote CondLSTR launch remains blocked until the rented server authorizes the supplied SSH public key; the server currently rejects it with `Permission denied (publickey)`.
