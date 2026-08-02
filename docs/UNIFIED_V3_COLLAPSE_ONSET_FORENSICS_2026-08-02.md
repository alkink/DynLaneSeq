# Unified Lane-Set V3 Collapse-Onset Forensics

**Tarih:** 2 Ağustos 2026  
**Durum:** 25k–50k checkpoint zaman çizelgesi tamamlandı; 30k→35k nedensellik kapısının sonucu bekleniyor  
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
