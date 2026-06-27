#!/bin/bash
set -e

# Kullanım: bash scripts/eval_culane_igsr_joint_controlled_noharm.sh [CHECKPOINT_YOLU]
# Eğer argüman verilmezse otomatik olarak iter_0025000.pt'ye bakar.

export PYTHONPATH="$(pwd):$PYTHONPATH"

CONFIG="dynlaneseq_eg/configs/culane_igsr_joint_controlled_noharm_25k.yaml"
CHECKPOINT="${1:-outputs/culane_igsr_joint_controlled_noharm_25k/iter_0025000.pt}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Hata: Checkpoint bulunamadı -> $CHECKPOINT"
    exit 1
fi

ITER_NAME=$(basename "$CHECKPOINT" .pt)
PRED_DIR="outputs/culane_igsr_joint_controlled_noharm_25k/culane_pred_test_${ITER_NAME}_thr0p50_q0p0"

echo "Değerlendiriliyor: $CHECKPOINT"
echo "Çıktı klasörü: $PRED_DIR"

python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --device cuda \
    --score-thresh 0.50 \
    --quality-score-power 0.0 \
    --output-stage final \
    --pred-dir "$PRED_DIR" \
    --categories
