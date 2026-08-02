# Unified Lane-Set V3 Collapse-Onset Forensics

**Tarih:** 2 Ağustos 2026  
**Durum:** 25k–50k zaman çizelgesi ve 30k→35k üç kollu nedensellik kapısı tamamlandı; scale-factor ayrıştırma deneyi sırada
**İlgili branch:** `diagnostic_v3_collapse_localization`  
**İlgili commit:** `a03594f`  
**Önceki teknik kayıt:** `docs/UNIFIED_V3_25K_TO75K_COLLAPSE_RESCUE_2026-08-02.md`

Bu rapor, Unified Lane-Set V3 modelinin neden 25k'da kullanılabilir lane
geometrileri üretirken 50k'da sıfır recall'a düştüğünü ara checkpointlerle
lokalize eder. Ölçülen sonuçları, kuvvetli çıkarımları ve henüz test edilmemiş
hipotezleri özellikle birbirinden ayırır.

---

## 1. Yönetici özeti

Yeni ara-checkpoint deneyi çöküşü kesin olarak şu pencereye sıkıştırdı:

```text
25k: sağlıklı
30k: sağlıklı ve deploy F1 açısından 25k'dan biraz daha iyi
35k: ağır geometri çöküşü
40k: bütün candidate geometrisi ölü
45k–50k: ölü çözüm devam ediyor
```

En önemli yeni sonuç şudur:

> Model önce skorlarını eşitleyip sonra geometriyi kaybetmiyor. Geometri
> 30k–35k arasında çökerken model kötü geometrilere hâlâ çok yüksek ve ayrışmış
> foreground skorları veriyor. Uniform-score/cardinality dengesi ancak 40k
> civarında ortaya çıkıyor.

Parametre zaman çizelgesi, performansın kırıldığı aynı 30k–35k aralığında
paylaşılan `row_norm -> row_x` hattının anormal hızla büyüdüğünü gösteriyor:

```text
row_readout L2 normu: 80.80 -> 160.54
row_norm.bias normu:   5.06 -> 14.90
row_x.weight normu:   74.20 -> 153.25
```

Aynı sırada:

- learned instance-token effective rank'i çökmüyor;
- learned reference-anchor çeşitliliği çökmüyor;
- `reference_logit_scale` sabit kalıyor;
- checkpoint/config/optimizer/scheduler zinciri tutarlı;
- bütün parametreler sonlu, NaN/Inf yok.

Bu nedenle şu anki en güçlü yakın neden:

> **Lane-state ve paylaşılan row readout'un birlikte kararsız ölçek/yön
> geliştirmesi; bunun yanlış ve aşırı güvenli row dağılımları üretip detached
> local-reference döngüsünü yanlış koridora kilitlemesi.**

Fakat parametre büyümesinin ilk tetikleyicisinin yüksek decoder LR'si mi,
score/matcher/deep-supervision geri beslemesi mi, yoksa ikisinin birleşimi mi
olduğu henüz kanıtlanmış değildir. Hazırlanan 30k→35k üç kollu deney tam olarak
bu ayrımı yapacaktır.

---

## 2. Kullanılan artefact'lar ve sabit protokol

Ana girdiler:

```text
outputs/diagnostics/trajectory_summary.json
outputs/diagnostics/parameter_trajectory.json
```

Checkpointler:

```text
iter_0025000.pt
iter_0030000.pt
iter_0035000.pt
iter_0040000.pt
iter_0045000.pt
iter_0050000.pt
```

Prediction trajectory protokolü:

```text
split                 = validation
örnekleme             = aynı uniform 64 görüntü
score thresholds      = 0.20 ve tarihsel 0.30
quality power         = 0
Top-K                 = 4
NMS                   = kapalı
checkpoint başına seçim = yok
gradient images       = 0
```

Dolayısıyla aşağıdaki F1 değerleri resmî full-test sonucu değildir. Bunlar aynı
örnekler üzerinde checkpoint dinamiğini karşılaştıran frozen teşhis
ölçümleridir. Threshold her checkpoint için yeniden seçilmemiştir.

---

## 3. Prediction-space zaman çizelgesi

### 3.1 Ana tablo

| Iter | F1@0.50, score=.20 | P@0.50 | R@0.50 | All-32 R@.50 | Direct-4 R@.50 | All-32 R@.75 | Matched mean IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25k | 70.12 | 77.17 | 64.25 | 94.12 | 70.59 | 76.92 | 0.773 |
| 30k | **72.82** | **81.11** | **66.06** | 94.12 | **72.40** | 72.85 | 0.763 |
| 35k | 31.88 | 30.80 | 33.03 | 38.91 | 33.03 | 12.67 | 0.384 |
| 40k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.134 |
| 45k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.107 |
| 50k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.114 |

### 3.2 Bu tablonun kanıtladığı şeyler

1. **25k tavan değildir.** Model 30k'ya kadar deploy F1, precision, recall ve
   Direct Top-4 recall açısından hâlâ iyileşiyor.
2. **Çöküş yavaş bir overfit değildir.** All-32 R@.50 yalnız 5k adımda
   `%94.12 -> %38.91` düşüyor ve sonraki 5k'da sıfıra iniyor.
3. **35k'da sorun artık ranking değildir.** All-32 oracle yalnız `%38.91`;
   doğru lane'lerin çoğu 32 adayın içinde bile bulunmuyor.
4. **40k'da selection ile kurtarılabilecek geometri kalmıyor.** Direct-4,
   Oracle-4 ve All-32 aynı anda sıfır.
5. **75k'ya devam etmek iyileşme sağlamaz.** Önceki 75k audit'i de geometri
   kapasitesinin geri dönmediğini göstermişti.

### 3.3 Çöküşün iki fazı

#### Faz A — 30k→35k: yanlış geometriye yüksek güven

| Ölçüm | 30k | 35k |
|---|---:|---:|
| Matched mean score | 0.410 | **0.646** |
| Unmatched mean score | 0.080 | **0.050** |
| Matched mean IoU | 0.763 | **0.384** |
| Foreground probability mass | 3.706 | 3.648 |
| Mean GT lane count | 3.453 | 3.453 |

35k'da score head matched ve unmatched query'leri güçlü biçimde ayırıyor.
Fakat matched query'nin geometrisi artık kötüdür. Hungarian her GT için mevcut
adayların içinden yine bir “en az kötü” query seçtiği için, binary existence
loss bu kötü geometriyi pozitif olarak güçlendirebilir.

Bu gözlem aşağıdaki iddiayı reddeder:

```text
"Önce bütün skorlar M/32 civarında eşitlendi, geometri de bu yüzden çöktü."
```

35k'da skorlar eşit değildir. İlk gözlenen kırılma geometri tarafındadır.

#### Faz B — 40k ve sonrası: ölü count-calibrated denge

40k'da:

```text
matched score   = 0.131
unmatched score = 0.111
foreground mass = 3.629
GT lane count   = 3.453
All-32 recall   = 0
```

Bu aşamada model hangi query'nin lane olduğunu ayıramazken toplam foreground
kütlesini yaklaşık doğru tutmaktadır. Differentiable cardinality loss bu ölü
çözümü başlatmış olmak zorunda değildir; fakat identity/geometri sinyali
bozulduktan sonra onu stabilize edebilecek tam matematiksel yapıya sahiptir.

---

## 4. Checkpoint bütünlük denetimi

Bütün 25k–50k checkpointlerinde:

- model tensor sayısı `664`;
- model eleman sayısı `69,079,693`;
- config-contract SHA256 aynı:
  `8e52fd3e4e5415d49bd277a43834e926b86fd9a722cb0400b7214e7b1f7a33f5`;
- optimizer state entry sayısı `500`;
- optimizer step'i checkpoint iteration ile birebir aynı;
- scheduler `last_epoch` ve `_step_count` doğru;
- bütün parametreler sonlu.

Evidence LR kesintisiz ve yavaş biçimde azalıyor:

| Iter | Evidence LR |
|---:|---:|
| 25k | 1.9636e-4 |
| 30k | 1.9469e-4 |
| 35k | 1.9273e-4 |
| 40k | 1.9047e-4 |
| 45k | 1.8793e-4 |
| 50k | 1.8510e-4 |

Dolayısıyla aşağıdaki basit altyapı açıklamaları desteklenmiyor:

- yanlış iteration yazılmış checkpoint;
- optimizer restart;
- scheduler/warmup restart;
- config değişimi;
- NaN/Inf parametre bozulması;
- 25k'dan sonra LR'nin aniden yükselmesi.

Bu sonuç, kullanılan kodun veya objective'in doğru olduğu anlamına gelmez;
yalnızca checkpoint zincirinin kendi içinde süreklilik taşıdığını gösterir.

---

## 5. Parametre-space zaman çizelgesi

### 5.1 Modül grupları

`relative delta`, ilgili grubun 25k parametre normuna göre değişim normudur;
grubun mevcut L2 norm oranı değildir.

| Iter | Row-readout norm | Row-readout relative delta | Lane-state core relative delta | Row-reference relative delta | FPN running-buffer relative delta |
|---:|---:|---:|---:|---:|---:|
| 25k | 52.98 | 0.000 | 0.000 | 0.000 | 0.000 |
| 30k | 80.80 | 0.675 | 0.186 | 0.083 | 0.450 |
| 35k | **160.54** | **2.408** | 0.297 | 0.168 | 1.391 |
| 40k | 187.32 | 2.979 | 0.357 | 0.213 | 3.496 |
| 45k | 196.18 | 3.162 | 0.411 | 0.248 | 4.535 |
| 50k | 221.61 | 3.698 | 0.458 | 0.269 | 4.486 |

En anormal değişim row readout'tadır. Lane-state core ve row-reference yolu da
değişmektedir; fakat onların toplam normlarında benzer bir patlama yoktur.

FPN BatchNorm running buffer'ları da güçlü biçimde sürüklenmektedir. Physical
batch `4` olduğu için bu ayrıca incelenmesi gereken gerçek bir risk olsa da
önceki module-swap deneyi failed decoder'ın sağlıklı encoder ile bile recall'ı
öldürmeye yeterli olduğunu göstermiştir. Bu nedenle FPN/BN şu aşamada ana
neden değil, muhtemel katkı olarak tutulmalıdır.

### 5.2 Paylaşılan row readout içindeki tensorlar

| Iter | `row_norm.weight` | `row_norm.bias` | `row_x.weight` | `row_x.bias` |
|---:|---:|---:|---:|---:|
| 25k | 25.07 | 1.33 | 46.62 | 1.61 |
| 30k | 31.38 | 5.06 | 74.20 | 3.40 |
| 35k | **44.64** | **14.90** | **153.25** | **8.49** |
| 40k | 55.67 | 24.84 | 176.84 | 9.89 |
| 45k | 60.17 | 29.05 | 184.16 | 10.34 |
| 50k | 63.30 | 32.02 | 209.64 | 11.35 |

Yalnız 30k→35k arasında:

```text
row_readout normu  yaklaşık +%99
row_norm.bias      yaklaşık +%195
row_x.weight       yaklaşık +%107
```

Bu, performans kırılmasıyla aynı zaman penceresidir.

---

## 6. Tensor düzeyinde neden bu hat kritik olabilir?

Kodda row logitleri şu biçimde üretiliyor:

```text
z      = normalize(row_state)
h      = gamma * z + beta              # affine LayerNorm: row_norm
logits = W_x h + b_x + reference_prior # shared row_x
```

Yani yaklaşık olarak:

\[
\text{logits}
=
W_x(\gamma\odot z + \beta)+b_x+\text{prior}.
\]

Hem `gamma/beta` hem `W_x/b_x` birlikte büyüdüğünde iki sorun oluşabilir:

1. Görüntüye bağlı normalize edilmiş içerik `z`, büyüyen affine ve sabit bias
   bileşenlerinin yanında etkisini kaybedebilir.
2. Çok keskin fakat yanlış x-bin dağılımları oluşabilir.

Basit bir teşhis göstergesi olarak
`mean(abs(gamma)) * std(W_x)` kullanıldığında:

| Iter | Yaklaşık ortak gain göstergesi | 25k'ya oran |
|---:|---:|---:|
| 25k | 0.159 | 1.00x |
| 30k | 0.315 | 1.98x |
| 35k | 0.929 | **5.85x** |
| 40k | 1.347 | 8.49x |
| 45k | 1.519 | 9.57x |
| 50k | 1.820 | 11.47x |

Bu gerçek operator normu veya tek başına nedensellik kanıtı değildir. Ancak
readout'un effective logit gain'inin performansın kırıldığı sırada birkaç kat
arttığını gösteren yararlı bir statik göstergedir.

V3'te bu head dört decoder katmanı tarafından paylaşılır ve final ile üç ara
katmanın geometry loss'larını alır. Aynı structured head parametrelerinin
tamamı `evidence_lr ~= 2e-4` grubundadır. Üstelik her katman bağımsız Hungarian
assignment kullanır. Bu kombinasyon aşağıdaki kararsız döngüyü mümkün kılar:

```text
shared row head'in ölçeği/yönü sürüklenir
        |
        v
bir katmanda yanlış fakat keskin x dağılımı oluşur
        |
        v
soft expected-x yanlış konuma gider
        |
        v
prediction detach edilip sonraki layer'ın reference'ı olur
        |
        v
sonraki local sampler gerçek lane yerine yanlış P2 koridoruna bakar
        |
        v
görüntü kanıtı zayıflar, assignment ve state daha da sürüklenir
```

Önceki temperature deneyi 50k geometrisini kurtaramamıştı. Bu çelişki değildir:
sorun yalnız ortak bir sıcaklık ise temperature yeterli olurdu. Burada weight
yönleri, affine bias ve sonraki local-reference koordinatları da değişmiştir.

---

## 7. Hangi hipotezler zayıfladı?

### 7.1 Learned instance token'lar birbirinin aynısı olmadı

| Iter | Instance-token effective rank | Off-diagonal cosine | Mean pairwise L2 |
|---:|---:|---:|---:|
| 25k | 25.65 | 0.024 | 0.825 |
| 35k | 26.02 | 0.034 | 0.977 |
| 50k | 25.66 | 0.086 | 1.124 |

Effective rank yaklaşık sabit ve pairwise uzaklık azalmıyor. Bu nedenle
“learned query embedding'leri tek vektöre çöktü” açıklaması desteklenmiyor.

Bu ölçüm runtime lane-state activation'larının benzeşmediğini kanıtlamaz.
Sadece statik query parametrelerinin çökmediğini kanıtlar.

### 7.2 Learned reference anchor'lar çökmüyor

Reference-anchor candidate standard deviation yaklaşık `274 px`, pairwise L2
yaklaşık `4,288–4,300` ve zaman boyunca sabittir. Dolayısıyla bütün anchor
template'lerin aynı eğriye çökmesi gözlenmiyor.

### 7.3 Reference prior scale patlamıyor

`reference_logit_scale`:

```text
25k: 1.857
30k: 1.841
35k: 1.842
50k: 1.844
```

Bu scalar kök neden değildir.

### 7.4 Backbone parametre normu patlamıyor

Backbone toplam normu `99.48 -> 99.67` değişmektedir. FPN running statistic
drift'i ayrı bir risk olsa da model ağırlıklarının tamamında genel bir norm
patlaması yoktur.

---

## 8. Şu anda ne kesin, ne kuvvetli, ne yalnız hipotez?

| Seviye | İfade |
|---|---|
| **Doğrudan kanıt** | Geometri 30k–35k arasında ağır biçimde, 40k'ya kadar tamamen çöküyor. |
| **Doğrudan kanıt** | Threshold, Top-K, NMS ve Oracle selection 40k+ modeli kurtaramaz. |
| **Doğrudan kanıt** | Row readout parametreleri aynı 30k–35k penceresinde anormal hızla büyüyor. |
| **Doğrudan kanıt** | Query-token ve anchor parametre çeşitliliği korunuyor. |
| **Doğrudan kanıt** | Uniform foreground score, geometri kırıldıktan sonra ortaya çıkıyor. |
| **Kuvvetli çıkarım** | Proximal failure decoder row-readout/lane-state uyumsuzluğudur. |
| **Kuvvetli çıkarım** | Detached local-reference döngüsü ilk hatayı geri besliyor olabilir. |
| **Açık hipotez** | İlk tetikleyici objective/matcher/intermediate-score geri beslemesidir. |
| **Açık hipotez** | İlk tetikleyici lane-state/readout için fazla büyük LR'dir. |
| **Açık hipotez** | Runtime lane-state'ler attention içinde homojenleşmektedir. |
| **Açık hipotez** | FPN BatchNorm drift'i çöküşü hızlandırmaktadır. |

Dolayısıyla henüz “kesin çözüm persistent lane-ID eklemektir” veya “kesin çözüm
yalnız LR düşürmektir” denemez.

---

## 9. Önceki dış analizin güncellenmiş değerlendirmesi

Önceki uzun analizde doğru olan kısımlar:

- 50k problemi score threshold değil, candidate geometry çöküşüdür.
- Cardinality ölü uniform-score çözümünü stabilize edebilir.
- 50k veya 75k'dan devam edilmemelidir.
- Local-only reference hattı yanlış koridorda self-reinforcing olabilir.
- Checkpoint/resume bütünlüğü ve ara-checkpoint zaman çizelgesi önce
  denetlenmelidir.

Yeni kanıtla zayıflayan veya fazla kesin söylenen kısımlar:

- Learned query-ID parametrelerinin çöktüğü iddiası desteklenmiyor.
- Learned reference anchor'ların tek curve'e çöktüğü iddiası desteklenmiyor.
- Cardinality'nin ilk tetikleyici olduğu iddiası 35k skorlarıyla uyuşmuyor.
- Persistent identity/global acquisition içeren V4'e doğrudan geçmek için
  henüz nedensellik kanıtı yok.

Yeni kanıtın eklediği en önemli parça, **row readout runaway'in çöküş anıyla
doğrudan zaman eşleşmesidir.**

---

## 10. Hazırlanan 30k→35k nedensellik deneyi

Eski 25k→30k gate artık yeterli değildir; çünkü exact tarihsel model 30k'da
hâlâ sağlıklıdır. Yeni script üç kolu doğrudan bilinen kırılma penceresinde
çalıştırır.

### 10.1 Source

```text
iter_0030000.pt
```

Bu checkpoint:

- All-32 R@.50 `%94.12`;
- matched mean IoU `0.763`;
- frozen F1@.50 `72.82`;
- tam optimizer ve scheduler state'i bulunan son sağlıklı noktadır.

### 10.2 Arm A — Control

```text
objective aynı
optimizer aynı
scheduler aynı
mimari aynı
30k -> 35k
```

Amaç, yeni process/data-order altında da çöküşün tekrar oluşup oluşmadığını
görmektir.

### 10.3 Arm B — Contract rescue

```yaml
matcher.lambda_obj: 0.0
loss.w_intermediate_exist: 0.0
loss.w_cardinality: 0.0
loss.w_score_margin: 0.0
```

Korunanlar:

- final foreground focal supervision;
- intermediate point, DFL, LineIoU ve range supervision;
- bağımsız layer-local geometry Hungarian assignment;
- mimari ve optimizer LR'leri.

Bu kol score/matcher/intermediate-classification geri beslemesinin geometri
çöküşünü başlatıp başlatmadığını test eder.

### 10.4 Arm C — Scale rescue

Objective tamamen aynı kalır. Yalnız:

```text
structured_query_head.lane_state_layers.*
structured_query_head.row_norm.*
structured_query_head.row_x.*
```

için base LR `2e-4 -> 5e-5` olur. AdamW moments parametre bazında korunur ve
scheduler 30k fazına hizalanır.

Bu kol parametre runaway'in objective değişmeden önlenip önlenemediğini test
eder.

### 10.5 Karar matrisi

| 35k sonucu | Yorum |
|---|---|
| Control çöker, contract korunur, scale çöker | İlk tetikleyici training contract / matching geri beslemesi |
| Control çöker, scale korunur, contract çöker | İlk tetikleyici decoder update ölçeği |
| Control çöker, ikisi de korunur | İki mekanizma ayrı ayrı yeterli koruma sağlıyor; 40k'ya ayrı uzatılmalı |
| Üçü de çöker | Combined gate, sonra structural V4 acquisition/identity deneyi |
| Control çökmez | Yeni replay tarihsel yolu yeniden üretmedi; kazanan seçilmez, bütün kollar 40k'ya uzatılır veya farklı seed ile tekrarlanır |

Başarı yalnız F1 ile değil aşağıdakilerle değerlendirilecektir:

```text
All-32 recall @0.50 / @0.75
matched mean official IoU
Direct Top-4 recall
row_norm / row_x norm eğimi
lane-state relative parameter delta
foreground score ayrımı
```

---

## 11. Reproducibility komutu

Branch ve commit:

```text
branch: diagnostic_v3_collapse_localization
commit: a03594f
```

Uzak sunucu komutu:

```bash
cd /workspace/DynLaneSeq

git fetch origin refs/heads/diagnostic_v3_collapse_localization
git switch -C diagnostic_v3_collapse_localization FETCH_HEAD

DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
AUDIT_MAX_BATCHES=16 \
AMP_DTYPE=bfloat16 \
bash scripts/run_culane_dla34_unified_lane_set_v3_rescue_30k_to35k.sh
```

Beklenen sonuç:

```text
outputs/diagnostics/unified_lane_set_v3_rescue_30k_to35k/summary.json
```

Script üç adet 5k kolu ardışık çalıştırır; toplam maliyet 15k optimizer
adımıdır.

### Resume karşılaştırma uyarısı

Checkpoint model, optimizer ve scheduler state'ini taşır; fakat DataLoader/RNG
cursor'unu taşımamaktadır. Bu nedenle replay control tarihsel kesintisiz
30k→35k yolunun bit düzeyinde aynısı olmayabilir. Üç yeni kol aynı source ve
aynı yeni veri sırasını kullandığı için kendi aralarında eşleştirilmiştir.
Control 35k'da sağlıklı kalırsa bu, müdahalenin başarısı değildir; kollar
bilinen çöküş görünene kadar birlikte uzatılmalıdır.

---

## 12. Şu anda yapılmaması gerekenler

- 50k veya 75k checkpoint'ten fine-tune ile kurtarma beklemek.
- Test threshold'u değiştirerek geometri çöküşünü gizlemeye çalışmak.
- Parametre normu büyük diye doğrudan weight clipping ekleyip sonucu kök neden
  kanıtı gibi sunmak.
- Contract ve LR müdahalelerini ilk koşuda tek kolda birleştirmek.
- Minimal nedensellik gate sonucu gelmeden büyük V4 mimarisini from scratch
  başlatmak.
- 64-image teşhis F1'ını resmî CULane sonucu olarak raporlamak.

---

## 13. Mevcut dürüst hüküm

Unified V3'ün 25k'da iyi, 30k'da daha iyi, 35k'da ağır biçimde bozuk olması
modelin temsil kapasitesinin doğal tavanına ulaşması değildir. Bu davranış
**eğitim dinamiği kararsızlığıdır.**

Şu an görünen en somut sorun, paylaşılan row readout'un lane-state dağılımıyla
birlikte kontrolden çıkmasıdır. Bu readout kötüleştikten sonra detached local
reference zinciri hatayı sonraki decoder layer'larına taşıyıp büyütmektedir.
Ancak readout patlamasının nedeni henüz yalnız korelasyonla bilinmektedir.

Bu yüzden sıradaki deney “daha fazla eğitim” veya “hemen baştan yeni mimari”
değil; aynı 30k state'inden başlayan contract-vs-scale nedensellik kapısıdır.
Bu gate, bir sonraki büyük mimari kararın tahmine değil doğrudan kanıta
dayanmasını sağlayacaktır.

---

## 14. 30k→35k üç kollu nedensellik kapısının gerçek sonucu

Bu bölüm 3 Ağustos 2026'da, aşağıdaki yeni artefact'lar geldikten sonra
eklenmiştir:

```text
outputs/diagnostics/summary.json
outputs/diagnostics/control_uniform64.json
outputs/diagnostics/contract_uniform64.json
outputs/diagnostics/scale_uniform64.json
```

Üç kol da:

- aynı `iter_0030000.pt` source state'inden başlamıştır;
- 35k iteration'da ölçülmüştür;
- aynı 64 validation görüntüsünü ve aynı dataset indekslerini kullanmıştır;
- aynı frozen evaluation sözleşmesiyle değerlendirilmiştir.

Bu nedenle kolların kendi aralarındaki fark anlamlıdır. Bununla birlikte
64-image sonuçları resmî CULane validation/test sonucu değildir ve bu koşuda
`gradient_images=0` olduğundan layerwise gradient çatışması henüz ölçülmüş
değildir.

### 14.1 Ana sonuç tablosu

| Ölçüm | Source 30k | Control 35k | Contract 35k | Scale 35k |
|---|---:|---:|---:|---:|
| All-32 R@.50 | **94.12** | **0.00** | 41.63 | **72.40** |
| All-32 R@.75 | **72.85** | **0.00** | **0.00** | **33.03** |
| Direct Top-4 R@.50 | **72.40** | **0.00** | 29.41 | **50.23** |
| Direct Top-4 R@.75 | **51.13** | **0.00** | **0.00** | **23.08** |
| Matched mean IoU | **0.763** | 0.110 | 0.430 | **0.568** |
| Matched score | 0.410 | 0.500 | 0.437 | 0.428 |
| Unmatched score | 0.080 | 0.063 | 0.157 | **0.067** |
| Score–IoU Pearson | 0.435 | 0.317 | **0.582** | 0.434 |
| Foreground probability mass | 3.706 | 3.521 | **5.978** | **3.389** |
| Frozen F1@.50, score 0.20 | **72.82** | **0.00** | 27.37 | **49.47** |

En önemli üç gözlem:

1. **Control çöküşü yeniden üretti.** Bu kez 35k'da All-32 recall tamamen
   sıfır oldu. Dolayısıyla tarihsel çöküş tek bir eski DataLoader sırasına özgü
   tesadüf değildir; sistem aynı sağlıklı 30k state'inden yeniden kararsız
   çekim noktasına girebilmektedir.
2. **Contract müdahalesi etkisiz değildir ama yetersizdir.** All-32 R@.50'yi
   `%0` yerine `%41.63` seviyesinde tutmuş ve score–IoU korelasyonunu
   iyileştirmiştir. Fakat R@.75 tamamen ölmüş, matched IoU `0.430`'a düşmüş ve
   readout norm büyümesi neredeyse control kadar sürmüştür.
3. **Scale müdahalesi açık ara en güçlü nedensel sinyaldir.** Objective
   değiştirilmeden yalnız lane-state ve row-readout LR'sinin dörtte bire
   indirilmesi All-32 R@.50'nin `%72.40`, R@.75'in `%33.03` olarak kalmasını
   sağlamıştır. Bu, güncelleme ölçeğinin çöküşte yalnız korelasyon değil gerçek
   bir nedensel faktör olduğunu gösterir.

### 14.2 Parametre büyümesiyle geometry kaybının müdahale cevabı

| Parametre | Source 30k | Control oranı | Contract oranı | Scale oranı |
|---|---:|---:|---:|---:|
| `row_norm.weight` | 31.38 | 1.421x | 1.393x | **1.089x** |
| `row_norm.bias` | 5.06 | 2.915x | 2.666x | **1.368x** |
| `row_x.weight` | 74.20 | 2.079x | 2.092x | **1.272x** |
| `row_x.bias` | 3.40 | 2.510x | 2.515x | **1.386x** |
| Readout relative delta | — | 1.265 | 1.257 | **0.332** |
| Lane-state relative delta | — | 0.209 | 0.192 | **0.078** |

Contract kolunda objective değişmesine rağmen `row_x.weight` yine yaklaşık
iki katına çıkmıştır. Buna karşılık scale kolunda readout relative delta
control'ün yaklaşık dörtte birine düşmüş ve kullanılabilir geometry'nin büyük
bölümü korunmuştur. Müdahale ile hem parametre runaway'in hem geometry
çöküşünün birlikte azalması, update scale hipotezini doğrudan güçlendirir.

Ancak scale kolu bir başarı koşusu değildir:

```text
30k -> scale 35k
All-32 R@.50: 94.12 -> 72.40
All-32 R@.75: 72.85 -> 33.03
matched IoU:  0.763 -> 0.568
```

Yani dört kat düşük LR yalnızca felaketi yavaşlatmış veya kısmen engellemiş;
sağlıklı 30k geometrisini korumamıştır. Otomatik özetin
`neither_minimal_intervention_is_sufficient` etiketi bu katı başarı eşiği
açısından doğrudur. Fakat bu etiket, scale kolunun diğer koldan çok daha güçlü
nedensel bilgi taşıdığını gizlememelidir.

### 14.3 Güncellenmiş nedensellik hükmü

Artık şu iddialar savunulabilir:

| Güven | Sonuç |
|---|---|
| **Kesin** | Control aynı 30k source'tan yeniden çökmüştür; kararsızlık tekrarlanabilirdir. |
| **Kesin** | Score/matcher/intermediate-score contract'ı değiştirmek tek başına geometry'yi korumaz. |
| **Kesin** | Lane-state/readout güncelleme ölçeğini azaltmak çöküşü büyük ölçüde bastırır. |
| **Kuvvetli** | Yüksek effective update, row-readout runaway ve geometry kaybı için gerekli veya çok güçlü bir büyütücüdür. |
| **Kuvvetli** | Contract geri beslemesi ikincil katkı yapmaktadır; scale ile birleştiğinde ek kazanç verebilir. |
| **Açık** | Asıl hassas grup lane-state mi, `row_norm/row_x` readout mu, yoksa karşılıklı co-adaptation mı? |
| **Açık** | Düşük LR yapısal gradient çatışmasını çözüyor mu, yoksa yalnızca daha yavaş mı biriktiriyor? |
| **Açık** | Detached local-reference zinciri scale kolundaki kalan strict-IoU kaybının ne kadarını büyütüyor? |

Basit mekanik anlatımı:

```text
30k'da iyi lane state + iyi row readout
                 |
       yüksek güncelleme ölçeği
                 v
lane state ile readout birlikte hızla yer değiştiriyor
                 |
                 v
yanlış fakat keskin x koordinatları
                 |
                 v
sonraki layer yanlış yerel koridoru örnekliyor
                 |
                 v
geometry tamamen çöküyor
```

Contract değişikliği bu zincirin assignment/score tarafından gelen kısmını
hafifletiyor; fakat parametre runaway'i durdurmuyor. Scale değişikliği zincirin
enerjisini doğrudan azaltıyor; bu yüzden daha fazla geometry koruyor.

### 14.4 Bir sonraki en yüksek bilgi değerli deney

Amaç yalnız hızlı bir kurtarma checkpoint'i üretmek olsaydı sıradaki kol
`contract + scale` birleşimi olurdu. Amaç kök nedeni bulup sağlam bir nihai
model kurmak olduğu için önce mevcut scale müdahalesi ikiye ayrılmalıdır.

> **3 Ağustos strateji notu:** Bu ayrıştırma, yalnız mevcut V3'ün adli kök
> nedenini tamamlamak için en yüksek bilgi değerli kısa deneydir. Projenin
> kabul ölçütü 278k boyunca kararlı çalışan nihai bir model olarak
> sabitlendiğinde artık ana geliştirme yolu değildir. Bölüm 15'teki yapısal V4
> tasarımı bu planın önüne geçmiştir.

Aynı 30k source ve aynı replay manifest ile iki yeni kol:

| Kol | Lane-state LR | `row_norm/row_x` LR | Sorduğu soru |
|---|---:|---:|---|
| Readout-low only | `2e-4` | `5e-5` | Runaway'i doğrudan readout update'i mi başlatıyor? |
| Lane-state-low only | `5e-5` | `2e-4` | Readout, değişen lane-state dağılımını kovalamaya mı çalışıyor? |

Mevcut sonuçlar 2×2 matrisin diğer iki köşesini zaten vermektedir:

```text
lane high + readout high = control, tam çöküş
lane low  + readout low  = scale, kısmi koruma
```

Karar:

- Readout-low only, both-low kadar iyiyse ana hassas aktör readout update'idir.
- Lane-state-low only, both-low kadar iyiyse ana tetikleyici lane-state
  dağılım driftidir.
- İki tekil kol da çöker, yalnız both-low korunursa problem karşılıklı
  co-adaptation'dır.

Bu iki kolun ardından:

1. Kazanan scale sözleşmesi `contract + scale` ile birleştirilir.
2. 35k'da korunursa 40k ve 50k'ya uzatılır.
3. Aynı anda layerwise geometry-gradient cosine, logit
   `content/bias/prior` ayrıştırması ve reference `±96 px` coverage ölçülür.
4. Düşük LR yalnızca çöküşü geciktiriyorsa shared auxiliary readout,
   bounded local-delta prediction veya reference trust-region yapısal gate'i
   açılır.
5. Uzun vadeli sözleşme doğrulandıktan sonra model düşük LR gruplarıyla
   **from scratch** eğitilir; 30k rescue checkpoint nihai benchmark modeli
   olarak kullanılmaz.

### 14.5 Bu sonuç neyi henüz kanıtlamaz?

- Scale kolunun resmî full-test F1'ını kanıtlamaz.
- Modelin 80+ olacağını kanıtlamaz.
- “Sadece LR'yi düşür, problem çözüldü” sonucunu desteklemez.
- Persistent lane identity veya global acquisition'ın gereksiz olduğunu
  kanıtlamaz; yalnız bunlara geçmeden önce daha küçük ve daha doğrudan bir
  scale-factor ayrımı olduğunu gösterir.
- Shared-head gradient çatışmasını doğrulamaz; bu koşuda gradient audit
  çalıştırılmamıştır.

En dürüst son cümle:

> **İlk kez gerçek bir müdahale ile çöküşü belirgin biçimde baskıladık. Ana
> yön artık yalnız bir şüphe değil: lane-state/readout update ölçeği kritik.
> Fakat scale kolunun strict geometry kaybı hâlâ büyük; dolayısıyla sağlam
> çözümün hangi alt grubu stabilize etmesi ve hangi yapısal geri beslemeyi
> sınırlaması gerektiğini iki kısa ayrıştırma koluyla belirlemeliyiz.**

---

## 15. Stratejik düzeltme: hedef 5k kurtarma değil, 278k kararlılığı

Mevcut V3 için dürüst cevap şudur:

> **Hayır. 30k–35k arasında katastrofik biçimde çöken mevcut hesaplama
> grafiğinin, yalnız LR ayarıyla 278k boyunca güvenilir çalışacağı
> varsayılamaz.**

Scale kolu arızanın bir parçasını kanıtlamıştır; nihai çözümü kanıtlamamıştır.
Bu nedenle “readout-low mu, lane-state-low mu?” ayrımı artık yalnız açıklayıcı
bir yan deneydir. Ana yol, hatanın oluşmasını yapısal olarak imkânsızlaştıran
bir decoder sözleşmesidir.

### 15.1 Mevcut V3'te 278k açısından kabul edilemez döngü

Koddan doğrulanan akış:

```text
full-row image-conditioned initial reference
                    |
                    v
lane-set state -> row state injection
                    |
                    v
yalnız reference ±96 px içindeki 7 P2 noktası okunuyor
                    |
                    v
ortak row_norm -> ortak serbest Linear(256, 800)
                    |
                    v
görüntünün herhangi bir yerinde mutlak x tahmini
                    |
                    v
detach -> sonraki katmanın sampling reference'ı
```

Buradaki yapısal uyumsuzluk:

- gözlem alanı yereldir;
- çıktı alanı bütün görüntüdür;
- çıktı kafası dört farklı decoder state dağılımı tarafından paylaşılır;
- her ara katman bağımsız Hungarian kimliğiyle aynı kafaya gradient verir;
- yanlış mutlak çıktı bir sonraki katmanın ne göreceğini belirler;
- yanlış sıçramayı sınırlayan hard trust region yoktur.

Bu sistem küçük LR ile daha yavaş bozulabilir, fakat 278k kararlılığı için
matematiksel bir güvence içermez.

### 15.2 MapTR ve CondLSTR karşılaştırmasının öğrettiği şey

Yerel kaynak kod karşılaştırması:

```text
/home/alki/projects/lanemodels/MapTR/
/home/alki/projects/lanemodels/CondLSTR/
```

MapTR iterative refinement sırasında:

- reference'a residual ekler;
- yeni reference'ı detach eder;
- `with_box_refine=true` olduğunda her decoder katmanına ayrı regression
  branch kopyalar;
- farklı layer state dağılımlarını tek bir ortak coordinate head'e zorlamaz.

CondLSTR ise lane koordinatını query tarafından üretilen yoğun mask/regression
alanından çıkarır. Geometri doğrudan image feature üzerinde tanımlıdır; serbest
bir global coordinate classifier'ın kendi çıktısını sonraki local görüş alanı
olarak tekrar beslediği bir döngü yoktur.

Dolayısıyla problem “Transformer decoder prensipte çalışmıyor” değildir.
Problem V3'ün şu özgül birleşimidir:

```text
local observation
+ shared absolute coordinate head
+ independent layer matching
+ detached recursive reference
+ high common LR
```

### 15.3 Unified Lane-Set V4: 278k-dayanıklı geometri sözleşmesi

V4 tek bir tutarlı tasarım olarak aşağıdaki invariants'ları sağlamalıdır.

#### A. Birinci katman gerçek global acquisition

İlk geometry katmanı her row için bütün P2 yatay dizisini görür. İlk koordinat
görüntüye bağlı dense similarity dağılımından çıkar:

```text
row query ---- normalized dot product ---- P2 row keys over full width
                                  |
                                  v
                         absolute x distribution
```

Burada serbest `Linear(256, 800)` yerine, ayrı coordinate query/key
projection'larıyla görüntüye bağlı logit üretilmesi tercih edilir. Böylece
coordinate logiti gerçek bir uzamsal P2 konumuna bağlıdır.

#### B. Sonraki katmanlar yalnız bounded local delta üretir

Refinement katmanı reference çevresinde yoğun bir profil okur; örneğin
`[-96,+96]` içinde 25–33 offset. Çıktı global 800-bin x değildir:

```text
delta_logits = similarity(row_query, sampled_P2[offset])
delta_x      = expectation(delta_logits, offsets)
x_next       = clamp(x_reference + delta_x, 0, W-1)
```

Temel invariant:

```text
|x_next - x_reference| <= refinement_radius
```

Model görmediği bir bölgeye tek adımda sıçrayamaz. Local observation ile local
action aynı fiziksel koordinat sisteminde olur.

#### C. Katman başına ayrı coordinate readout

```text
Layer 0 -> global coordinate query/key head
Layer 1 -> local delta head 1
Layer 2 -> local delta head 2
Layer 3 -> local delta head 3
```

Ara katmanlar aynı `row_norm/row_x` affine sistemini paylaşmaz. Bu, layer-state
dağılımları ve layerwise geometry gradientleri arasındaki çatışmayı tek bir
head içinde biriktirmeyi önler.

#### D. Kalıcı kimlik ile değişen content ayrılır

Her lane için iki ayrı büyüklük korunur:

```text
e_i   = değişmeyen learned lane identity
q_i^l = görüntüyle değişen lane content
```

Set attention Q/K girdisi:

```text
LN(q_i^l) + e_i + curve_position_code(x_i^l)
```

Value yalnız content taşır. Böylece attention content'i güncellese bile lane
kimliği ve mevcut curve konumu kaybolmaz.

#### E. Lane-to-row etkisi sınırlı residual olur

Lane state row geometry'yi kontrolsüz tam-genlikli affine ile sürüklemez.
Projection normalize edilir ve sabit/bounded bir residual ölçeği kullanılır:

```text
row <- row + alpha * normalized(lane_to_row(lane_state))
0 < alpha <= güvenli sabit
```

Bu yol açıktır fakat row feature dağılımını 30k–35k'daki gibi hızla başka bir
koordinat sistemine taşıyamaz.

#### F. Geometry matching ve score geri beslemesi ayrılır

```text
Hungarian assignment: point + range + LineIoU; score cost = 0
Intermediate layers: geometry-only loss
Final layer: geometry + tek foreground/quality score
Intermediate assignment: final query identity yeniden kullanılır
```

Final score, detached final geometry IoU'suna bağlı continuous target öğrenir.
Score head'in girdisi geometry core'dan detach edilir; score loss geometry
state'ini değiştiremez. Cardinality ve binary margin başlangıçta kapalıdır.

Sonuç olarak:

```text
geometry -> score'a bilgi verir
score -X-> matcher/geometry'ye geri kumanda vermez
```

#### G. Optimizer, görev sınırlarını yansıtır

Başlangıç sözleşmesi:

| Grup | Base LR |
|---|---:|
| DLA backbone | `1e-5` |
| FPN/P2 projection | `1e-4` |
| Row visual decoder | `1e-4` |
| Lane-set state / collect / inject | `5e-5` |
| Global/local coordinate heads | `5e-5` |
| Instance/row identity embeddings | `5e-5` |
| Final semantic score branch | `1e-4` |

Norm ve bias parametreleri decay almaz; global gradient clipping korunur.
Kritik fark bütün `structured_query_head.*` parametrelerinin tek `2e-4`
evidence grubuna atılmamasıdır.

### 15.4 Yeni mimari ağacı

```text
DLA-34 + SimpleFPN
        |
        +---------------- P2: yüksek çözünürlüklü geometri
        |                         |
        |                         v
        |              Global row acquisition (Layer 0)
        |                         |
        |                         v
        |                absolute image-grounded curve
        |                         |
        |          +--------------+--------------+
        |          |              |              |
        |          v              v              v
        |      Local Δ L1     Local Δ L2     Local Δ L3
        |      bounded        bounded        bounded
        |      own head       own head       own head
        |          |              |              |
        |          +--------------+--------------+
        |                         |
        |                         v
        |                 final ordered rows
        |                         |
        |                         +------> range
        |                         |
        +---- P4/P5 semantic ----+------> detached final score
                                          |
                                          v
                                  Top-K / optional NMS
```

### 15.5 Tek ana eğitim: 0→278k, ilk 50k yapısal kabul kapısı

Yeni tasarım bir 5k rescue olarak denenmemelidir. Sıfırdan, scheduler horizon
başından itibaren `278000` olarak eğitilir. İlk 50k bu aynı koşunun erken kabul
kapısıdır; başarılı olursa optimizer sıfırlanmadan aynı koşu 278k'ya devam
eder.

Checkpoint/audit sıklığı:

```text
0–50k: her 5k
50–150k: her 10k veya 25k
150–278k: her 25k
```

İlk 50k geçiş şartları yalnız F1 değildir:

| Sözleşme | Fail-fast koşulu |
|---|---|
| All-32 R@.50 | 30k sonrası katastrofik düşüş yok |
| All-32 R@.75 | iki ardışık checkpointte sert düşüş yok |
| Local trust region | `|x_next-x_ref|` hiçbir örnekte radius'u aşmaz |
| Reference coverage | layer boyunca monoton çöküş yok |
| Head normları | 5k pencerede 2x runaway yok |
| Update/weight | kritik gruplarda sürekli büyüyen trend yok |
| Layer identity | final assignment reuse sözleşmesi ihlal edilmiyor |
| Score feedback | score gradientinin geometry core'a normu tam sıfır |

Bu koşu 50k'ya kadar sağlıklı kalmazsa 278k'ya devam edilmez. Sağlıklı
kalırsa bu yalnız “kısa test başarılı” demek değildir; aynı optimizer state ve
aynı 278k schedule ile gerçek uzun koşunun ilk bölümünün geçtiği anlamına
gelir.

### 15.6 Nihai karar

Mevcut V3'ü LR yamalarıyla 278k'ya taşımak ana plan değildir. V3 adli
deneyleri, yeni tasarımın hangi invariants'lara sahip olması gerektiğini
öğretmiştir. Bundan sonraki ana mühendislik işi:

```text
shared absolute readout'u kaldır
local observation'ı bounded local action ile eşleştir
katman başına ayrı coordinate head kullan
score'u geometry geri beslemesinden ayır
kritik decoder gruplarını düşük ve ayrı LR ile eğit
sıfırdan başlayan tek 278k koşusunu 50k fail-fast kapısıyla izle
```

Bu, küçük bir hiperparametre yaması değil; V3'ün kararsız geri besleme
döngüsünü ortadan kaldıran genel mimari çözümdür.
