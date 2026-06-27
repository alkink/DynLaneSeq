#!/usr/bin/env bash
# =============================================================================
# Faz 1: Representation Kapasitesi Testi
#
# SORU: Mevcut S0 row_token representation'ı, doğru quality target verilince
#       TP/FP ayrımını öğrenebilir mi?
#
# YÖNTEMİ:
#   1. S0 ağırlıkları yüklenir (--init-from)
#   2. structured_query_head.quality hariç HER ŞEY dondurulur
#   3. w_allprop_quality + w_ranking loss ile quality head eğitilir
#
# BEKLENEN ÇIKTI:
#   - 10k iterasyon sonrası: loss_allprop_quality < 0.35 ise öğreniyor
#   - Val quality AUC / separation metrikleri raporlanır
#   - Son checkpoint evaluate_culane ile değerlendirilir (quality_score_power=1.0)
# =============================================================================

set -e

S0_CHECKPOINT="outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt"
CONFIG="dynlaneseq_eg/configs/culane_faz1_soft_quality.yaml"
OUTPUT_DIR="outputs/culane_faz1_soft_quality"

if [ ! -f "$S0_CHECKPOINT" ]; then
    echo "ERROR: S0 checkpoint bulunamadı: $S0_CHECKPOINT"
    exit 1
fi

echo "============================================================"
echo "Faz 1: Soft Quality + Ranking Loss"
echo "S0 checkpoint: $S0_CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# S0'ı yükle ve sadece quality head'i eğit
python -c "
import torch
from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_model
from dynlaneseq_eg.engine.checkpoint import load_compatible_model_weights

cfg = load_config('$CONFIG')
model = build_model(cfg)
stats = load_compatible_model_weights('$S0_CHECKPOINT', model)
print(f'Yükleme istatistikleri: {stats}')

# Tüm parametreleri dondur
total_frozen = 0
for name, param in model.named_parameters():
    param.requires_grad_(False)
    total_frozen += param.numel()

# SADECE quality head'i aç
# structured_query_head.quality = nn.Sequential(Linear, GELU, Linear)
quality_params = 0
for name, param in model.named_parameters():
    if 'structured_query_head.quality' in name:
        param.requires_grad_(True)
        quality_params += param.numel()
        print(f'  Eğitilecek: {name} ({param.numel():,} parametre)')

print(f'Toplam frozen: {total_frozen:,}')
print(f'Toplam eğitilecek (quality head): {quality_params:,}')
print('Freeze testi başarılı.')
" 2>&1

echo ""
echo "Quality head freeze testi geçti. Eğitim başlıyor..."
echo ""

# Gerçek eğitim: freeze işlemi train.py'ye entegre değil,
# bu yüzden özel bir train scripti kullanıyoruz
python -m dynlaneseq_eg.tools.train_faz1 \
    --config "$CONFIG" \
    --init-from "$S0_CHECKPOINT" \
    --device cuda

echo ""
echo "============================================================"
echo "Eğitim tamamlandı. Değerlendirme yapılıyor..."
echo "============================================================"

# İlk değerlendirme: quality_score_power=0 (baseline, quality kapalı)
python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --split val \
    --device cuda \
    --score-thresh 0.40 \
    --quality-score-power 0.0 \
    --output-stage main \
    --pred-dir "$OUTPUT_DIR/val_pred_q0p0" \
    --categories 2>&1 | tee "$OUTPUT_DIR/eval_q0p0.log"

echo ""
# İkinci değerlendirme: quality_score_power=1.0 (quality head aktif)
python -m dynlaneseq_eg.tools.evaluate_culane \
    --config "$CONFIG" \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --split val \
    --device cuda \
    --score-thresh 0.10 \
    --quality-score-power 1.0 \
    --output-stage main \
    --pred-dir "$OUTPUT_DIR/val_pred_q1p0" \
    --categories 2>&1 | tee "$OUTPUT_DIR/eval_q1p0.log"

echo ""
echo "============================================================"
echo "SONUÇ KARŞILAŞTIRMASI:"
echo "  quality_score_power=0.0 (baseline, quality kapalı):"
grep -E "^(F1|Precision|Recall|All)" "$OUTPUT_DIR/eval_q0p0.log" 2>/dev/null || echo "  Log dosyasına bakın: $OUTPUT_DIR/eval_q0p0.log"
echo ""
echo "  quality_score_power=1.0 (quality head aktif):"
grep -E "^(F1|Precision|Recall|All)" "$OUTPUT_DIR/eval_q1p0.log" 2>/dev/null || echo "  Log dosyasına bakın: $OUTPUT_DIR/eval_q1p0.log"
echo "============================================================"
