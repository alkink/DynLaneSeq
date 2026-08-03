# Unified Lane-Set V4.1: Frozen-Geometry Score Gate

Tarih: 3 Ağustos 2026

## Amaç

V4 50k checkpoint'i geometri açısından sağlıklıdır, fakat deploy edilen skor
32 aday içinden doğru ve benzersiz dört şeridi seçememektedir. Bu deney V4
geometrisini değiştirmeden iki soruyu ayrı ayrı ölçer:

1. Adayları birlikte gören bir set scorer gerekli mi?
2. Matcher kimliğini kopyalamak yerine score-independent unique-IoU hedefi
   gerekli mi?

Bu bir final eğitim değildir. Sağlıklı 50k geometri üzerinde nedensellik
deneyidir.

## Gradyan sınırı

```text
V4 final row states ---------------- detach --+
V4 final x/range/confidence -------- detach --+
curve-aligned P2 evidence ---------- detach --+
trained P4/P5 decision descriptor -- frozen --+
                                             |
                                             v
                              V4.1 score descriptor
                                             |
                           +-----------------+-----------------+
                           |                                   |
                           v                                   v
                 independent candidate MLP          candidate-set Transformer
                           |                                   |
                           +-----------------+-----------------+
                                             |
                                             v
                                  one deployment score
```

Yalnız `structured_query_head.set_selection_head.*` trainable'dır. Backbone,
FPN, lane-state, reference, bounded-delta head, range head ve eski existence
head dondurulur. Dondurulan detector `eval()` modunda tutulduğu için BatchNorm
running stat'ları ve dropout da geometriyi değiştiremez.

`detach_geometry_features=true` artık yalnız x koordinatını değil aşağıdaki
bütün selector girdilerini kapsar:

- final row tokens;
- final lane query;
- range ve row logits;
- reference curve;
- sampled P2 curve evidence.

Bu nedenle selection loss geometriye dolaylı bir yol üzerinden de geri dönmez.

## 2×2 deney

| Kol | Candidate interaction | Target/loss |
|---|---|---|
| A | Independent MLP | Final geometry matcher assignment, quality floor 0.5 |
| B | Independent MLP | Score-independent unique IoU assignment, negative weight 2, ranking |
| C | Set Transformer | Final geometry matcher assignment, quality floor 0.5 |
| D | Set Transformer | Score-independent unique IoU assignment, negative weight 2, ranking |

Karşılaştırmalar:

```text
B - A = target/loss etkisi, bağımsız scorer
D - C = target/loss etkisi, set scorer
C - A = set interaction etkisi, shared target
D - B = set interaction etkisi, unique target
```

Independent ve Transformer kolları aynı descriptor'ı görür. Aralarındaki tek
mimari fark candidate ekseninde iletişim olup olmamasıdır. Hiçbir kol legacy
existence veya quality skoruyla çarpım yapmaz.

## Çalıştırma

Sunucuda V4 50k checkpoint mevcutken:

```bash
cd /workspace/DynLaneSeq
conda activate clrernet

DATA_ROOT=/workspace/CULane \
SOURCE_CHECKPOINT=outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt \
TRAIN_STEPS=3000 \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
MAX_BATCHES=64 \
bash scripts/run_culane_dla34_unified_lane_set_v4_1_score_gate_50k.sh
```

Script:

- checkpoint içindeki iteration'ın tam 50,000 olduğunu doğrular;
- checkpoint SHA256 değerini loglar;
- dört config sözleşmesini fail-fast denetler;
- bütün kolları aynı seed ve veri protokolüyle 50k→53k eğitir;
- her 500 adımda checkpoint alır;
- aynı uniform-256 validation örneğinde bütün kolları değerlendirir;
- `reports/summary.json` içine faktör etkilerini ve geçiş kararını yazar.

Yarım kalan bir kol otomatik resume edilmez. Checkpoint DataLoader ve
augmentation cursor taşımadığı için resume, paired-data sözleşmesini sessizce
bozardı. Script yeniden çalıştırılırsa tamamlanmamış kol aynı 50k kaynaktan ve
aynı seed'den başlar; tamamlanmış kollar atlanır.

## Çıktılar

```text
outputs/diagnostics/unified_lane_set_v4_1_score_gate_50k/
  a_mlp_shared/
  b_mlp_unique/
  c_set_shared/
  d_set_unique/
  reports/
    source_v4_uniform256.json
    a_mlp_shared_uniform256.json
    b_mlp_unique_uniform256.json
    c_set_shared_uniform256.json
    d_set_unique_uniform256.json
    summary.json
```

Her rapor şunları içerir:

- Direct score Top-4 precision/recall/F1 @0.50 ve @0.75;
- All-32 Oracle Top-4 kapasitesi;
- unique-candidate AP;
- foreground probability mass;
- duplicate/near-miss/background FP ayrımı.

## Gate sınırları

D kolunun ilk hızlı geçiş koşulları:

```text
All-32 geometry               kaynak V4 ile aynı
Direct Top-4 recall @0.50     >= 0.70
Direct Top-4 recall @0.75     >= 0.60
diagnostic F1 @0.50           >= 0.65
unique candidate AP @0.50     >= 0.50
foreground probability mass  3.0–4.5
```

Bu eşikler resmi test sonucu değildir. D kolu geçerse sıradaki işlem full
9,675-image validation'dır. Full validation da pozitif olursa score contract
dondurulur ve final model sıfırdan, 278k scheduler ile eğitilir.

## Dürüst sınır

V4'ün yüksek Oracle kapasitesi, iyi bir scorer'ın 80 F1'ı garanti ettiği
anlamına gelmez. Daha önce frozen selector deneyleri başarısız olmuştur. Bu
gate'in amacı yeni fikri varsaymak değil, şu üç sonucu birbirinden ayırmaktır:

```text
loss-only kazanır  -> ana sorun target/negative calibration
set-only kazanır   -> ana sorun candidate-set karşılaştırması
combined kazanır   -> ikisi birlikte gerekli
hiçbiri kazanmaz   -> frozen descriptor benzersiz seçim için yetersiz
```

Son durumda geometry'ye score gradyanı açılmaz; descriptor ve ownership
bilgisinin nerede kaybolduğu yeniden incelenir.
