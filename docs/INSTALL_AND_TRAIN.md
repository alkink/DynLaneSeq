# DynLaneSeq-EG Install And Train Guide

This guide assumes Ubuntu/WSL and a CULane copy available on the same machine.
Run the commands in a WSL/Linux terminal, not in Windows PowerShell.

## 1. Clone

```bash
cd ~/projects
git clone https://github.com/alkink/DynLaneSeq.git
cd DynLaneSeq
```

## 2. Create Conda Environment

```bash
conda create -n clrernet python=3.10 -y
conda activate clrernet
python -m pip install --upgrade pip
```

Install PyTorch. Pick the CUDA wheel/channel that matches your machine. This is
the common CUDA 12.1 conda install:

```bash
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the project dependencies and editable package:

```bash
pip install -r requirements.txt
pip install -e .
```

Quick environment check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 3. Link CULane

The configs expect `dataset` at the repository root.

```bash
cd ~/projects/DynLaneSeq
ln -sfn /home/alki/projects/CULane dataset

test -f dataset/list/train_gt.txt
test -f dataset/list/val.txt
test -f dataset/list/test.txt
```

For the 2k debug configs, these files should also exist:

```bash
test -f dataset/list/train_2k.txt
test -f dataset/list/test_2k.txt
```

## 4. Smoke Tests

```bash
conda activate clrernet
cd ~/projects/DynLaneSeq

python -m pytest -q dynlaneseq_eg/tests

python -m dynlaneseq_eg.tools.visualize_targets \
  --config dynlaneseq_eg/configs/debug/culane_s0_10img_overfit.yaml \
  --num 10
```

## 5. Download The S0 Checkpoint

Checkpoints are not stored in regular Git. The S1 full run expects this local
file:

```text
outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

If the checkpoint was uploaded as a GitHub Release asset, install/login to
GitHub CLI first:

```bash
sudo apt update
sudo apt install -y gh
gh auth login
```

Download the checkpoint:

```bash
cd ~/projects/DynLaneSeq
mkdir -p outputs/culane_s0_res34_strong_b16_giou

gh release download s0-strong-b16-75k \
  --pattern iter_0075000.pt \
  --dir outputs/culane_s0_res34_strong_b16_giou
```

If the release asset is not available, copy the checkpoint manually to:

```bash
outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

Verify:

```bash
ls -lh outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

## 6. Full Training Chain

Always use the scripts or pass `--init-from` explicitly. Running only
`python -m dynlaneseq_eg.tools.train --config ...` starts from scratch.

### S1 residual from strong S0

```bash
conda activate clrernet
cd ~/projects/DynLaneSeq

bash scripts/run_culane_s1_strong_from_s0_75k.sh
```

This script initializes from:

```text
outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

Manual equivalent:

```bash
python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s1_res34_strong_b16_from_s0_75k.yaml \
  --device cuda \
  --init-from outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt
```

### S2 residual from S1

Run this after S1 produces `iter_0075000.pt`:

```bash
bash scripts/run_culane_s2_strong_from_s1_residual.sh
```

Manual equivalent:

```bash
python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s2_res34_strong_b16_from_s1_residual.yaml \
  --device cuda \
  --init-from outputs/culane_s1_res34_strong_b16_from_s0_75k/iter_0075000.pt
```

## 7. Evaluation

S1 test evaluation:

```bash
python -m dynlaneseq_eg.tools.evaluate_culane \
  --config dynlaneseq_eg/configs/culane_s1_res34_strong_b16_from_s0_75k.yaml \
  --checkpoint outputs/culane_s1_res34_strong_b16_from_s0_75k/iter_0075000.pt \
  --split test \
  --device cuda \
  --score-thresh 0.5 \
  --pred-dir outputs/culane_s1_res34_strong_b16_from_s0_75k/culane_pred_test_75k_thr0.5 \
  --categories
```

S2 test evaluation:

```bash
python -m dynlaneseq_eg.tools.evaluate_culane \
  --config dynlaneseq_eg/configs/culane_s2_res34_strong_b16_from_s1_residual.yaml \
  --checkpoint outputs/culane_s2_res34_strong_b16_from_s1_residual/iter_0075000.pt \
  --split test \
  --device cuda \
  --score-thresh 0.5 \
  --pred-dir outputs/culane_s2_res34_strong_b16_from_s1_residual/culane_pred_test_75k_thr0.5 \
  --categories
```

## 8. Optional: Train S0 From Scratch

This is not needed if you use the downloaded S0 checkpoint.

```bash
python -m dynlaneseq_eg.tools.train \
  --config dynlaneseq_eg/configs/culane_s0_res34_strong_b16.yaml \
  --device cuda
```

Evaluate S0:

```bash
python -m dynlaneseq_eg.tools.evaluate_culane \
  --config dynlaneseq_eg/configs/culane_s0_res34_strong_b16.yaml \
  --checkpoint outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt \
  --split test \
  --device cuda \
  --score-thresh 0.5 \
  --pred-dir outputs/culane_s0_res34_strong_b16_giou/culane_pred_test_75k_thr0.5 \
  --categories
```

## 9. Upload The S0 Checkpoint As A Release Asset

Run after pushing the repository and logging in with `gh auth login`:

```bash
gh release create s0-strong-b16-75k \
  outputs/culane_s0_res34_strong_b16_giou/iter_0075000.pt \
  --title "S0 Strong B16 75k Checkpoint" \
  --notes "Strong S0 checkpoint used to initialize full CULane S1 residual training."
```
