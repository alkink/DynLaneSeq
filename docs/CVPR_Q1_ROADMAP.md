# Structured Row-Token Lane Decoder: CVPR / Q1 Roadmap

Son güncelleme: 2026-07-03

Bu doküman, mevcut structured S0 modelini CVPR veya Q1 journal seviyesinde savunulabilir bir makaleye çevirmek için yapılması gereken işleri sıralı ve operasyonel şekilde tanımlar.

Ana hedef: modeli yeni modüllerle şişirmek değil; mevcut fikrin gerçekten çalıştığını temiz, adil ve tekrar edilebilir deneylerle kanıtlamak.

## 0. Mevcut durum

Şu anki en güçlü model ailesi:

```text
S0 structured row-token decoder
ResNet34
input 1600x640
slots32
num_rows 160
x_bins 800
FPN 256
structured decoder layers 4
DFL enabled
no EMA
no TTA
```

En iyi test sonucu şu an yaklaşık:

```text
CULane test
iter 175k
SCORE_THRESH ~= 0.30
QUALITY_POWER ~= 0.50
F1 ~= 79.9x
P ~= 87.7
R ~= 73.4
```

Önemli uyarı: Bu eşik ve quality power seçimi test set üzerinde keşfedildiyse final paper protokolü olarak kullanılamaz. Final protokolde threshold / quality power / checkpoint seçimi validation set üzerinde yapılmalı, test set sadece son raporlama için kullanılmalıdır.

## 1. Makalenin ana iddiası

Makale şu iddiayı savunmalı:

```text
Whole-lane queries are structurally under-specified for 2D lane geometry.
We instead decompose each lane into a global instance token and ordered row-level geometry tokens.
This factorization gives a stronger inductive bias for row-wise lane localization without handcrafted line anchors.
```

Türkçe karşılığı:

```text
Bir şeridin tüm geometrisini tek bir global query vektörüne sıkıştırmak zayıf bir temsil.
Biz lane kimliğini instance token ile, lane geometrisini ise sıralı row token dizisiyle ayırıyoruz.
Bu yapı, el yapımı line anchor kullanmadan row-wise geometriyi daha iyi öğreniyor.
```

Makale “DFL ekledik, skor arttı” diye yazılmamalı. DFL yardımcı lokalizasyon supervision’ı olarak konumlandırılmalı. Ana katkı structured instance-to-row decoder olmalı.

Önerilen isimler:

```text
Structured Row-Token Decoding for 2D Lane Detection
Instance-to-Row Token Decoding for Line-Anchor-Free Lane Detection
Structured Instance-to-Row Decoder for Lane Geometry Modeling
```

## 2. Yapılmayacaklar

Bu aşamada aşağıdaki yönlere girilmemeli:

- Yeni orthogonal evidence / verifier modülü eklemek.
- Geometry refiner eklemek.
- Büyük bir cascade mimarisi kurmak.
- Test set üzerinde sürekli threshold seçip final skor aramak.
- Her başarısızlığı “scoring problemi” diye yamamak.
- Modeli açıklaması zor hale getirecek yeni başlıklar eklemek.

Bundan sonraki işin ana odağı:

```text
kanıt, ablation, protokol, genelleme, yazım
```

olmalı.

## 3. Faz 1 — Kod ve sonuç dondurma

Amaç: current best modelin gerçekten tekrar üretilebilir olduğundan emin olmak.

### 3.1 Branch ve commit dondurma

Yapılacak:

```bash
git status
git diff
```

Sonra mevcut çalışır hali ayrı branch veya tag ile dondur:

```bash
git checkout -b paper_structured_row_token_freeze
git add -A
git commit -m "Freeze structured row-token CULane experiments"
```

Eğer branch zaten varsa:

```bash
git add -A
git commit -m "Update structured row-token paper roadmap and scripts"
```

Kabul kriteri:

- Mevcut config, train script, eval script, checkpoint pathleri kaybolmayacak.
- En iyi sonucu veren config ve script net bulunabilir olacak.

### 3.2 Mevcut checkpoint/result manifest oluşturma

Bir manifest dosyası oluştur:

```text
docs/RESULT_MANIFEST.md
```

İçerik:

- checkpoint path
- config path
- eval command
- score threshold
- quality power
- F1 / P / R
- category metrics
- EMA var mı
- TTA var mı
- input resolution
- notes

Örnek satır:

```text
CULane test | ResNet34 | 1600x640 | slots32 | L4 | iter_0175000 | thr=0.30 | q=0.50 | F1=79.9x | no EMA | no TTA
```

Kabul kriteri:

- Bir ay sonra aynı sonucu üretmek için hangi komut çalışacak belli olmalı.

## 4. Faz 2 — Validation protokolünü temizleme

Amaç: final skorun test-set tuning gibi görünmesini engellemek.

### 4.1 Val sweep scriptleri

Şu kombinasyonlar validation set üzerinde çalıştırılmalı:

```text
checkpoint:
100k, 125k, 150k, 175k, 200k, 225k

quality_power:
0.25, 0.40, 0.50, 0.60, 0.75

score_thresh:
0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45

nms_distance:
20, 25, 30
```

Başta tüm kombinasyonları koşmak pahalıysa iki aşama yapılabilir:

1. coarse sweep:

```text
q = 0.25, 0.50, 0.75
thr = 0.30, 0.40, 0.45
nms = 20
```

2. fine sweep:

```text
en iyi checkpoint çevresinde
q = 0.40, 0.50, 0.60
thr = 0.25, 0.28, 0.30, 0.32, 0.35
nms = 20, 25, 30
```

Kabul kriteri:

- Final checkpoint, threshold, quality power ve NMS validation set üzerinden seçilmiş olmalı.
- Test set sadece final seçilen ayar ile raporlanmalı.

### 4.2 Test final run

Validation ile seçilen tek ayar test set üzerinde çalıştırılır.

Örnek:

```bash
CKPT=outputs/.../iter_0175000.pt \
SCORE_THRESH=<val_selected_thr> \
QUALITY_POWER=<val_selected_q> \
NMS_DISTANCE_THRESH_PX=<val_selected_nms> \
CATEGORIES=--categories \
bash scripts/eval_..._test.sh
```

Kabul kriteri:

- Paper’daki ana test skoru bu tek final protokolden gelmeli.
- Test üzerinde sonradan “daha iyi q/thr” aranmayacak.

## 5. Faz 3 — Ana internal ablation: structured vs unstructured

Bu faz makalenin omurgasıdır.

Amaç: 75.25 → 79.9x farkının gerçekten structured decoder fikrinden geldiğini göstermek.

### 5.1 Aynı koşulda unstructured baseline

Şu model eğitilmeli:

```text
ResNet34
input 1600x640
FPN256
same optimizer
same schedule
same augmentations
same row/bin setting mümkünse
no structured row-token decoder
unstructured holistic query / MLP head
```

Beklenen paper sorusu:

```text
Structured decoder gerçekten faydalı mı, yoksa büyük input/FPN/DFL mı faydalı?
```

Bu deney bu soruyu cevaplar.

Kabul kriteri:

- Unstructured aynı güçlü koşulda structured modelden açıkça düşük kalmalı.
- Eğer fark çok azalırsa makale iddiası yeniden yazılmalı.

### 5.2 Eski baseline manifest

Eski baseline:

```text
S0 unstructured
50 epoch
F1 ~= 75.25
```

Bu sonuç paper’da kullanılabilir ama tek başına yeterli değil. “Historical internal baseline” olarak geçebilir. Ana kıyas için aynı koşulda yeni unstructured run gerekir.

## 6. Faz 4 — Contribution isolation ablation

Amaç: reviewer’ın “bu sadece resolution/DFL/model size” eleştirisini kapatmak.

Önerilen ablation tablosu:

| Deney | Değişiklik | Amaç |
|---|---|---|
| A | Unstructured holistic query | ana baseline |
| B | Instance token only | global lane identity ayrımı işe yarıyor mu |
| C | Instance + ordered row tokens, no DFL | structured decoder etkisi |
| D | Instance + ordered row tokens + DFL | DFL katkısı |
| E | D + FPN128 | channel etkisi |
| F | D + FPN256 | final channel |
| G | D + 2 decoder layer | depth etkisi |
| H | D + 4 decoder layer | final depth |
| I | D + 6 decoder layer | depth plato kontrolü |
| J | slots20 | CondLSTR query sayısına yakın |
| K | slots32 | final |
| L | slots64 | fazla slot recall/FP etkisi |

Kabul kriteri:

- Ordered row tokens çıkarıldığında skor anlamlı düşmeli.
- DFL çıkarıldığında özellikle localization / tight IoU metrikleri düşmeli.
- FPN/input büyümesi tek başına tüm kazancı açıklamamalı.
- 4 layer güçlü olmalı; 6 layer sadece marjinal iyileşme verirse makale daha temiz olur.

## 7. Faz 5 — Resolution ve capacity fairness

Amaç: “1600x640 olduğu için kazandı” eleştirisini yönetmek.

### 7.1 Resolution ablation

Koşul:

```text
ResNet34
structured decoder
same training
```

Denenecek:

```text
800x288 / 800x320
1024x384
1600x640
```

Her resolution’da:

- x_bins / rows oranı net yazılmalı.
- pixel-per-bin hesaplanmalı.
- FLOPs / FPS raporlanmalı.

Kabul kriteri:

- Structured model düşük resolution’da da unstructured baseline’dan iyi olmalı.
- 1600x640 sadece final scaling olarak sunulmalı.

### 7.2 x_bins / rows ablation

Denenecek:

```text
rows: 96, 128, 160
x_bins: 512, 640, 800
```

Kabul kriteri:

- Artış sadece daha fazla bin sayısından gelmemeli.
- Daha fazla binin özellikle localization/tight IoU katkısı varsa bu açık yazılmalı.

## 8. Faz 6 — Diğer datasetler

Amaç: yöntemin CULane’e özel bir tuning olmadığını göstermek.

Önerilen sıra:

1. TuSimple
2. LLAMAS
3. CurveLanes

### 8.1 TuSimple

Amaç:

- Temiz lane datasetinde modelin stabil çalıştığını göstermek.
- Genelleme tablosu için düşük maliyetli ilk dış dataset.

Yapılacak:

- Dataset loader kontrolü.
- Evaluation formatı kurulacak.
- ResNet34 structured ve unstructured aynı koşulda eğitilecek.

Kabul kriteri:

- Structured > unstructured olmalı.
- Public baselines ile rekabetçi olmalı.

### 8.2 LLAMAS

Amaç:

- Large-scale lane dataset üzerinde robustness göstermek.

Yapılacak:

- Loader / annotation conversion.
- Official metric script doğrulama.
- ResNet34 structured final ayar.

Kabul kriteri:

- CULane dışındaki büyük dataset üzerinde fikir çalışmalı.

### 8.3 CurveLanes

Amaç:

- Ordered row-token fikrinin curve geometry üzerinde özellikle anlamlı olduğunu göstermek.

Yapılacak:

- Curve-heavy category analysis.
- 2/4/6 decoder layer ablation burada özellikle önemli.

Kabul kriteri:

- CurveLanes veya curve-heavy split üzerinde structured decoder belirgin avantaj göstermeli.

## 9. Faz 7 — Backbone scaling

Amaç: katkının sadece ResNet34’e özel olmadığını göstermek.

Minimum:

```text
ResNet18
ResNet34
ResNet50
```

Opsiyonel:

```text
DLA34
ResNet101
```

Paper’da ana tablo:

| Backbone | Unstructured | Structured | Gain |
|---|---:|---:|---:|
| ResNet18 | ... | ... | ... |
| ResNet34 | ... | ... | ... |
| ResNet50 | ... | ... | ... |

Kabul kriteri:

- Gain sadece ResNet34’te değil, en az iki backbone’da tekrar etmeli.
- Eğer büyük backbone’da gain azalırsa bu dürüstçe “structured prior is most beneficial under moderate backbone capacity” diye yazılabilir.

## 10. Faz 8 — Public baseline karşılaştırması

Amaç: makaleyi literatür içinde doğru konumlandırmak.

Karşılaştırılacak yöntem aileleri:

- UFLD / UFLDv2
- LaneATT
- CondLaneNet
- CLRNet
- CondLSTR
- LaneFormer
- BézierLaneNet / parametric curve methods
- Lane2Seq
- newer sparse/query lane methods

Tabloda mutlaka yazılacak:

```text
Backbone
Input resolution
EMA/TTA var mı
F1
Precision
Recall
FPS
Params
FLOPs
```

Kabul kriteri:

- Bizim model “SOTA’yı ezdi” diye yazılmamalı.
- Doğru iddia: “competitive ResNet34 result with a new structured decoder.”
- Eğer EMA/TTA ile 80+ gelirse ayrı satırda verilmeli.

## 11. Faz 9 — EMA ve TTA

Amaç: final skorun üst sınırını görmek, ama ana katkıyı test trick’e bağlamamak.

### 11.1 EMA

Mevcut checkpointlerde EMA yok. EMA için yeni eğitim veya erken checkpointten itibaren EMA tutulması gerekir.

Yapılacak:

- Training loop’a EMA state ekle.
- Checkpoint içine `ema_model` veya `model_ema` kaydet.
- Eval script’e `--use-ema` ekle.

Kabul kriteri:

- EMA sonucu ayrı satırda raporlanmalı.
- Ana ablation no-EMA kalmalı.

### 11.2 Horizontal flip TTA

Yapılacak:

- Eval sırasında image normal + horizontal flipped çalıştırılır.
- Flipped prediction x koordinatları geri çevrilir.
- İki prediction seti merge edilir.
- NMS/top-k uygulanır.

Risk:

- Lane order / left-right semantic yoksa teknik olarak yönetilebilir.
- Ancak merging iyi yapılmazsa FP artabilir.

Kabul kriteri:

- TTA sonucu ayrı satır.
- No-TTA ana skor korunur.

## 12. Faz 10 — Tight localization metrics

Amaç: DFL ve row-token decoder’ın sadece F1@0.5 değil, lokalizasyon kalitesi sağladığını göstermek.

Raporlanacak:

```text
F1@0.5
F1@0.7 veya F1@0.75
mean IoU / mF1
category-wise CULane
```

Kabul kriteri:

- DFL ve high-bin settings tight metriclerde daha anlamlı iyileşme göstermeli.
- Eğer göstermiyorsa DFL iddiası zayıflatılmalı.

## 13. Faz 11 — Failure analysis

Amaç: modelin güçlü ve zayıf yönlerini dürüst göstermek.

Analiz edilecek kategoriler:

- normal
- crowd
- highlight
- shadow
- no-line
- arrow
- curve
- cross
- night

Özellikle:

```text
cross FP
no-line FP/FN
night recall
curve localization
```

Görsel figürler:

1. structured modelin unstructured baseline’a göre lane tamamladığı örnekler.
2. cross/no-line hallucination failure örnekleri.
3. row-token x-distribution heatmap.
4. row-token attention / evidence visualization.
5. threshold-quality trade-off curve.

Kabul kriteri:

- Failure analysis modelin zayıflığını saklamamalı.
- Reviewer’a “authors understand their model” mesajı vermeli.

## 14. Faz 12 — Paper writing plan

### 14.1 Introduction

Ana problem:

```text
Holistic lane queries compress complex lane geometry into a single vector.
This is weak for long, curved, partially occluded lane markings.
```

Çözüm:

```text
Factorize lane representation into instance identity and ordered row-wise geometry.
```

### 14.2 Related work

Bölümler:

- segmentation-based lane detection
- anchor-based lane detection
- row-wise classification methods
- transformer/query-based lane detection
- vectorized map / hierarchical query analogies

Kritik ayrımlar:

- CondLaneNet: row-wise output var, ama decoder state olarak ordered row token yok.
- CondLSTR: lane query dynamic kernel üretiyor, ama lane geometry explicit row-token sequence değil.
- LaneFormer: row/column attention encoder tarafında, lane-specific row decoder değil.
- UFLD: row-wise classification var, instance-conditioned structured decoder yok.

### 14.3 Method

Bölümler:

1. overview
2. instance token
3. ordered row tokens
4. row-local cross attention
5. inter-instance/group attention
6. intra-lane row attention
7. row-wise x distribution and DFL
8. existence/range/quality heads
9. training objective
10. inference and postprocess

### 14.4 Experiments

Sıra:

1. datasets and metrics
2. implementation details
3. main comparison
4. ablation
5. cross-dataset generalization
6. speed/complexity
7. diagnostics/failure analysis

### 14.5 Limitations

Dürüst yazılacak:

- lane-absent scenes can still produce hallucinated lane hypotheses
- high resolution setting increases compute
- model relies on row-wise camera-view assumption
- final performance may require careful validation-based calibration

## 15. Tahmini zaman planı

Bu zaman planı “tek kişi, hazır altyapı, güçlü GPU erişimi var” varsayımıyla yazılmıştır.

### Hafta 1 — Freeze + val protocol

- result manifest
- best current config freeze
- validation sweep
- final test protocol

Çıktı:

```text
VAL_SELECTED_FINAL.md
```

### Hafta 2 — Core ablation

- unstructured same-condition baseline
- no DFL
- no ordered row token
- 2/4/6 layer
- slots20/32/64

Çıktı:

```text
ABLATION_CORE.md
```

### Hafta 3 — Resolution/capacity ablation

- 1024x384 vs 1600x640
- FPN128 vs FPN256
- bins/rows table

Çıktı:

```text
ABLATION_CAPACITY.md
```

### Hafta 4 — Other datasets

- TuSimple
- LLAMAS
- CurveLanes başlangıç

Çıktı:

```text
CROSS_DATASET_RESULTS.md
```

### Hafta 5 — Backbone scaling + EMA/TTA

- ResNet18/34/50
- EMA experiment
- TTA experiment

Çıktı:

```text
SCALING_AND_TRICKS.md
```

### Hafta 6 — Diagnostics + paper draft

- failure figures
- row distribution figures
- category plots
- first full paper draft

Çıktı:

```text
paper/draft_v1.tex
```

## 16. Go / no-go karar noktaları

### Go for CVPR submission

Aşağıdakiler sağlanırsa CVPR submission mantıklı:

- CULane final no-EMA/no-TTA >= ~79.9
- EMA/TTA veya final tricks ile 80+ mümkünse iyi
- structured vs unstructured aynı koşulda açık fark gösteriyor
- en az iki dış dataset üzerinde structured gain var
- ablationlar kazancın structured decoder’dan geldiğini gösteriyor
- failure analysis dürüst ve ikna edici

### Q1 journal için yeterli paket

Aşağıdakiler sağlanırsa Q1 journal gerçekçi:

- CULane güçlü ve temiz
- TuSimple/LLAMAS/CurveLanes sonuçları var
- backbone scaling var
- ablation detaylı
- speed/params/FLOPs raporlu
- writing net

### No-go / yeniden düşün

Aşağıdaki durumlarda paper iddiası zayıflar:

- same-condition unstructured baseline structured’a çok yaklaşırsa
- external datasetlerde gain kaybolursa
- ablationlar kazancı resolution/FPN/DFL’a bağlarsa
- cross/no-line FP aşırı kötüleşip açıklanamazsa

## 17. En yakın yapılacaklar listesi

Sıradaki somut işler:

1. `RESULT_MANIFEST.md` oluştur.
2. 175k/200k için validation q/thr/NMS sweep çalıştır.
3. Val seçimiyle tek final test sonucu üret.
4. Same-condition unstructured baseline config hazırla.
5. No-DFL structured config hazırla.
6. 2/4/6 layer configs hazırla.
7. slots20/32/64 configs hazırla.
8. CULane category-wise delta plot scripti yaz.
9. Row distribution visualization scripti yaz.
10. Paper outline başlat.

Bu sıranın dışına çıkılmamalı. Yeni mimari fikri gelirse önce bu liste bitmeli.
