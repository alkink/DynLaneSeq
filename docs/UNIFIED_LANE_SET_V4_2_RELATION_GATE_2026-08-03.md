# Unified Lane-Set V4.2: Relation-Aware Selection Gate

Tarih: 3 Ağustos 2026

## Neden bu müdahaleyi yapıyoruz?

V4, V3'teki uzun-eğitim geometri çöküşünü durdurdu. V4-50k üzerinde 32 adayın
oracle kapasitesi güçlü kaldı; fakat scalar Top-4 aynı gerçek şeridin birkaç
kopyasını seçti. En güçlü karşı-olgusal sonuç şudur:

```text
aynı V4 geometrisi + aynı D skoru

scalar Top-4                  F1@.50 = 50.26
soft MMR (sigma20, p0.5)     F1@.50 = 75.61

duplicate FP                 367 -> 8
```

Değişen detector veya candidate quality değildir. Değişen tek şey, seçim
anında aday-aday eğri yakınlığının kullanılmasıdır. Bu yüzden V4.2 geometriyi
yeniden tasarlamaz. MMR'nin eksik bıraktığını gösterdiği açık ilişkiyi, frozen
geometri sınırının arkasındaki score-only modele verir.

## Yeni tensor akışı

```text
V4 final curves/ranges/row confidence/P2/P4/P5
                       |
                       v
                    DETACH
                       |
          +------------+-------------+
          |                          |
          v                          v
 unary candidate descriptor   pairwise curve relation
      [B, 32, 1074]             [B, 32, 32, 6]
          |                          |
          +-------------+------------+
                        v
       relation-biased score-only Transformer
                        |
                        v
             one scalar score / candidate
```

Relation kanalları:

```text
1. common-range mean |x_i - x_k|
2. common-range top-weighted distance
3. common-range bottom-weighted distance
4. visible-row overlap
5. range IoU
6. exp(-|x_i-x_k| / 20px) soft strip similarity
```

Her attention head'i kendi `phi_h(R_ik)` bias'ını öğrenir:

```text
A_ik^h = Q_i^h K_k^h / sqrt(d_h) + phi_h(R_ik)
```

Bu hard NMS değildir. İki gerçek yakın lane'i sabit 20 piksel kuralıyla silmez;
range, top/bottom ayrımı ve unary localization quality ile yumuşak karar verir.

## Yeni selection loss

Candidate-to-GT range-aware kalite matrisi `Q[N,M]`, Hungarian sonrasında hemen
tek boyutlu target'a atılmaz. Dört yardımcı terim için korunur:

```text
quality:    candidate'ın localization kalitesi
coverage:   her GT en az bir yüksek olasılıklı candidate ile kapsansın
duplicate:  aynı GT'yi açıklayan iki candidate birlikte yükselmesin
winner:     aynı cluster içinde strict-IoU'su en iyi temsilci kazansın
count:      empty scene/değişken cardinality için küçük global yardımcı sinyal
```

Başlangıç ağırlıkları:

```text
quality                       1.00
existing scalar ranking       0.25
coverage                      0.50
duplicate                     1.00
winner                        0.25
count                         0.10
```

Count bilinçli olarak küçüktür. Geometry tamamen detached olsa da tek başına
count, bütün slotlara benzer probability dağıtan eski kestirmeyi yeniden
üretebilir.

## Düzeltilen config uyuşmazlığı

V4.1 kodu final P4/P5 semantic score adaptörlerinin güvenli biçimde
eğitilmesine izin veriyordu; girdiler geometry/FPN'den detach edilmişti.
Ancak gate config'i yalnız `set_selection_head` parametrelerini açtığı için bu
adaptörler fiilen frozen kaldı.

V4.2 R1-R3'te yalnız final katmanın şu score-only parçaları açılır:

```text
lane_state_layers.3.semantic_attention
lane_state_layers.3.semantic_router
lane_state_layers.3.semantic_scale_embedding
lane_state_layers.3.norm_semantic_query / norm_semantic_ffn
lane_state_layers.3.semantic_ffn
decision_norm
set_selection_head
```

Lane set geometry, row delta heads, reference decoder, FPN ve backbone frozen
kalır.

## R0-R3 nedensellik matrisi

```text
R0  generic Transformer + unique scalar loss + semantic frozen
    -> yalnız 3k yerine 10k eğitmenin etkisi

R1  R0 + final semantic score adapters trainable
    -> V4.1 config uyuşmazlığının etkisi

R2  R1 + explicit [B,N,N,6] relation attention bias
    -> eksik curve-relation tensorının etkisi

R3  R2 + coverage/duplicate/winner/count losses
    -> set-level supervision'ın ek etkisi
```

Bütün kollar aynı V4-50k checkpoint'inden, aynı veri/seed ile başlar. Her 500
adımda compact delta checkpoint ve optimizer state kaydedilir. Source geometry
yeniden yazılmaz; böylece disk dolduğunda yarım 500-MB checkpoint bırakma riski
azalır ve atomic checkpoint writer korunur.

## Çalıştırma

```bash
cd /workspace/DynLaneSeq
conda activate clrernet

DATA_ROOT=/workspace/CULane \
SOURCE_CHECKPOINT=outputs/culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k/iter_0050000.pt \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
AMP_DTYPE=bfloat16 \
bash scripts/run_culane_dla34_unified_lane_set_v4_2_relation_gate_50k.sh
```

Varsayılan deney tek seed (`3407`) ve 10k adımdır. Nedensel kazanan belirlendikten
sonra üç seed doğrulaması şu şekilde yapılabilir:

```bash
SEEDS="3407 4703 7304" ... \
bash scripts/run_culane_dla34_unified_lane_set_v4_2_relation_gate_50k.sh
```

İlk aşamada dört kolu üç seed ile körlemesine koşturmak yerine tek seed causal
gate kullanılır; yalnız kazanan yapı çoklu seed'e taşınır.

Ana çıktılar:

```text
outputs/diagnostics/unified_lane_set_v4_2_relation_gate_50k/
  gradient_contract/*.json
  seed_3407/reports/summary.json
  seed_3407/reports/trajectory_summary.json
```

## Fail-fast sözleşmesi

Run başlamadan gerçek batch gradient audit'i şunları zorunlu kılar:

```text
selector gradient                         > 0
semantic score adapter gradient (R1-R3)  > 0
geometry/FPN/backbone gradient            = 0
```

R3 geçiş kriterleri:

```text
Direct Top-4 R@.50             >= 70%
Direct Top-4 R@.75             >= 60%
duplicate FP fraction          < 25%
Unique AP@.50                  >= 0.50
foreground mass                3.0 - 4.5
MMR'nin direct score kazancı   < 5 F1
```

Son kriter en önemlisidir: relation-aware scorer gerçekten MMR davranışını
içselleştirdiyse, inference'ta MMR eklemek artık büyük kazanç sağlamamalıdır.

## Ne yapmıyoruz?

- Score gradientini geometry'ye geri açmıyoruz.
- Hard 20px NMS'i mimariye gömmüyoruz.
- V4-50k geometry checkpoint'ini atmıyoruz.
- R3 başarılı olmadan 278k ana koşu başlatmıyoruz.
- Diagnostic subset sonucunu resmî test sonucu saymıyoruz.

R2/R3 10k sonunda hâlâ MMR'den 10-20 F1 gerideyse scalar scoring sözleşmesi
yetersiz kabul edilecek. O durumda sonraki adım relation özelliklerini tekrar
yamamak değil; `NO-LANE/STOP` içeren dört adımlı differentiable MMR veya
pointer-style subset selector olacaktır.

