# V35 Ham-RGB Temporal Observability Gate — Nihai Rapor

**Tarih:** 24 Ağustos 2026  
**Durum:** İki yön, üç kol ve üç seed tamamlandı; gate FAIL  
**Test split:** Kullanılmadı

## Kısa cevap

V35'in sonucu nettir:

> Candidate çevresindeki ham hedef görüntü, oracle-good proposal ile V7'nin
> current-wrong proposal'ını yeni clip'lerde güvenilir biçimde ayıramadı.
> Önceki ve sonraki ham kareyi eklemek de sonucu iyileştirmedi.

Bu yalnız küçük bir dalgalanma değildir. Dört bağımsız hücrenin tamamında
önceden kilitlenen `AUC >= 0.70` ve `pair accuracy >= 0.65` sınırları geniş
farkla kaçtı. En iyi AUC yalnız `0.5711` oldu. Üç-kare kolu, tek-kare kolunu
dört hücrenin üçünde daha da kötüleştirdi.

Net karar:

```text
Yeni frozen selector / reranker:                 KAPALI
Candidate ribbon / pair-corridor:                KAPALI
AGF-V7 detached score FPN:                       KAPALI
Frozen V7 üstünde temporal voting/selector:      KAPALI
V7 bankasını seçerek kurtarma ana araştırma yolu: KAPALI
```

Bu sonuç yeni bir end-to-end detector'ın veya gerçek video modelinin prensipte
imkânsız olduğunu kanıtlamaz. Fakat mevcut 32'lik bankadaki lucky-tail adayı
sonradan seçme ailesine daha fazla mimari eklemek artık savunulabilir ana yön
değildir.

## 1. Ne yaptık?

V34'te iki önemli bulgu vardı:

1. Oracle-good proposal komşu karelerin GT lane'iyle fiziksel olarak tutarlıydı.
2. Frozen V7 tahminleri komşu karelerde de current-wrong fiziksel yapıya daha
   fazla inanıyordu.

V35 bu ikisinin arasında kalan son soruyu sordu:

> V7 feature ve skorlarını tamamen çıkarırsak, ham RGB görüntü doğru proposal'ı
> gösterebilir mi? Üç kare tek kareden daha fazla bilgi taşır mı?

### 1.1 Clip-disjoint cross-fit

V34'ün mature V7 tarafından hiç görülmemiş 54 validation clip'i iki sabit gruba
ayrıldı:

```text
A→B: 27 clip'te train, diğer 27 clip'te evaluation
B→A: yön ters çevrildi
```

Scorer hiçbir evaluation clip'ini eğitimde görmedi. Evaluation sonucuyla
checkpoint, step, seed, loss veya threshold seçilmedi.

Kullanılan veri:

```text
Training pair:              1.650
Exact V34 evaluation pair:    980
Benzersiz candidate:         3.302
Seed:                   3 matched seed
Sabit endpoint:         2.000 step
```

Validation GT, training fold'da candidate kalite label'ı oluşturdu. Bu nedenle
deney bir cross-fit diagnostic'tir; official validation F1 veya deployment
sonucu değildir.

### 1.2 Candidate ribbon

Her candidate'ın 64 row'u boyunca iki ham RGB koridoru örneklendi:

```text
fine ribbon:   ±32 input pixel
coarse ribbon: ±160 input pixel
```

Önceki ve sonraki karede candidate curve, V34 ile aynı DIS optical-flow
kontratıyla taşındı. Scorer şu bilgileri görmedi:

- proposal ID,
- V7 route score,
- V7 existence/quality score,
- V7 image feature,
- komşu V7 prediction'ları.

### 1.3 Üç matched kol

Üç kol aynı `521.345` parametreli scorer'ı, optimizer'ı, seed'leri ve step
sayısını kullandı:

```text
G — Geometry-only:
    RGB sıfır; yalnız candidate x/eğim/eğrilik/maskesi.

S — Single-frame:
    Yalnız hedef ham RGB ribbon'u.

T — Temporal:
    Flow-aligned previous + target + following ham RGB ribbon'ları.
```

Wrong-image kontrolünde candidate geometrisi sabit tutulup ham görüntü başka
clip ile değiştirildi.

## 2. Ne oldu?

### 2.1 Ana sonuç

| Yön | IoU | Pair | G AUC | S acc | S AUC | T acc | T AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B | .50 | 195 | 0.5376 | 0.5487 | 0.5487 | 0.5282 | 0.5556 |
| A→B | .75 | 366 | 0.5664 | 0.5792 | **0.5711** | 0.5410 | 0.5498 |
| B→A | .50 | 229 | 0.5349 | 0.5721 | 0.5514 | 0.4934 | 0.5291 |
| B→A | .75 | 423 | 0.5561 | 0.5414 | 0.5225 | 0.5177 | 0.5200 |

Önceden kilitli minimumlar:

```text
Pair accuracy >= 0.65
AUC           >= 0.70
```

Hiçbir hücre bu iki sınırdan birini bile geçemedi.

### 2.2 Ham görüntü geometri kontrolünü geçemedi

Tek-kare AUC'sinin geometry-only AUC'ye farkı:

```text
A→B .50: +0.0112
A→B .75: +0.0047
B→A .50: +0.0165
B→A .75: -0.0336
```

Gate en az `+0.05` istemişti. Dört farkın bootstrap aralığı da sıfırı içeriyor.

Yani scorer yeni clip'lerde esas olarak ham lane görünümünden değil, zayıf
candidate-geometri kısa yollarından faydalandı.

### 2.3 Wrong-image kontrolü

Doğru görüntü AUC'si ile başka clip görüntüsü AUC'si arasındaki fark:

| Yön | IoU | S doğru−yanlış | T doğru−yanlış |
| --- | ---: | ---: | ---: |
| A→B | .50 | +0.0204 | +0.0026 |
| A→B | .75 | +0.0471 | +0.0427 |
| B→A | .50 | +0.0129 | **-0.0158** |
| B→A | .75 | +0.0165 | +0.0228 |

Gate `>= +0.10` istemişti. Hiçbir hücre yaklaşamadı.

Bu kontrolün sade anlamı:

> Modelin doğru fotoğrafı görmesi, başka bir clip'in fotoğrafını görmesine göre
> doğru proposal sıralamasını anlamlı ve tutarlı biçimde iyileştirmedi.

### 2.4 Üç kare yeni bilgi katmadı

Temporal AUC eksi single-frame AUC:

```text
A→B .50: +0.0069
A→B .75: -0.0213
B→A .50: -0.0223
B→A .75: -0.0025
```

Temporal pair-accuracy farkı dört hücrenin tamamında negatiftir:

```text
-0.0205, -0.0383, -0.0786, -0.0236
```

Dolayısıyla V34'te görülen zamansal kararlılık, correct belief sinyaline
dönüşmedi. Ham komşu kareler bu candidate-centric modelde yalnız daha fazla
değişkenlik ekledi.

### 2.5 Model eğitimi öğrenmedi mi?

Hayır. Training problemi öğrenildi, fakat yeni clip'e taşınmadı.

Sabit endpoint'te bütün kolların train good-minus-wrong marjı yaklaşık:

```text
3.9 – 4.5 logit
```

seviyesine çıktı. Train loss yaklaşık `0.16–0.20` aralığına indi. Buna karşılık
held-out AUC `0.52–0.57` düzeyinde kaldı.

Bu desenin adı basitçe ezberlemedir:

> Ağ training clip'lerindeki candidate çiftlerini çok güçlü ayırdı, fakat
> öğrendiği kural başka clip'lerde geçerli değildi.

Seed sonucu da bunu destekliyor. Özellikle B→A single-frame AUC'si üç seed'de
`0.553`, `0.503`, `0.517`; temporal AUC `0.551`, `0.501`, `0.528` kaldı.

## 3. Bu sonuç ne anlama geliyor?

### Kanıtlanan

1. Candidate-local ham RGB bilgisi, current-wrong ile oracle-good farkını
   clip-disjoint biçimde güvenli seçime çevirmiyor.
2. Geniş ribbon eklemek, P1/P2 corridor başarısızlığını çözmüyor.
3. Önceki ve sonraki ham kare bu ayrımı güçlendirmiyor.
4. Model kapasitesiz veya eğitimsiz kalmadı; training çiftlerini güçlü biçimde
   fit etti ama unseen clip'lere genelleyemedi.
5. Doğru proposal'ın adjacent GT ile uyumu, onun deployed V7 veya küçük bir
   raw-image hakem tarafından gözlemlenebilir yüksek-inanç modu olduğu anlamına
   gelmiyor.

### Kanıtlanmayan

1. Hiçbir end-to-end video detector çalışamaz.
2. Daha güçlü bir backbone hiçbir zaman fayda veremez.
3. CULane GT tamamen görüntüden gözlemlenemez.
4. V7 dışı yeni bir detector 81+ yapamaz.

V35 belirli ve önemli bir aileyi sınadı: sabit V7 proposal bankasını sonradan
candidate-centric bir hakemle düzeltmek.

## 4. Sorun nerede?

### En olası neden — Genellenebilir candidate-local karar kuralı yok

V7'nin wrong proposal'ı çoğu kez rastgele gürültü değil. Görüntüde gerçek,
keskin ve zaman içinde devam eden bir fiziksel yapıyı izliyor. Oracle-good
proposal ise annotation'a daha yakın olmasına rağmen V7 posteriorunda düşük
kütleli bir tail örneği.

Bu nedenle küçük hakem training clip'lerinde şu ayrımı ezberleyebiliyor:

```text
Bu görüntüde P10 doğru, P12 yanlış.
```

Ama yeni clip'te kullanabileceği ortak kuralı bulamıyor.

### İkinci olası neden — Candidate hakemliği yanlış problem formu

Doğru lane'i seçmek, hazır iki curve'e bakıp kalite puanı vermekten daha temel
bir görüntü-anlama problemi olabilir. Lane'in topolojik kimliği, yolun tamamı,
annotation devamlılığı ve diğer lane'lerle ilişkisi üretim sırasında birlikte
kurulmalıdır.

Yalnız ribbon bağlamı eksik olabilir. Fakat V28/V29'da tam görüntüyü okuyan ayrı
DLA/FPN belief tower da OOF'ta başarısız oldu. Bu nedenle “biraz daha geniş
context ekleyelim” açıklaması artık zayıftır.

### Üçüncü olası neden — Veri miktarı

Her yönde yalnız 27 clip ile scorer eğitildi. Daha fazla clip fayda verebilir.
Bu gerçek bir sınırlamadır.

Fakat bunu ana açıklama saymıyoruz; çünkü:

- V28 full-train belief tower source V7'yi geçmedi,
- V29 OOF belief tower source TP'leri fazla kaybetti,
- V19 quality prediction yüksek korelasyona rağmen F1 getirmedi,
- frozen corridor ve selector ailesi farklı veri hacimlerinde aynı güvenlik
  sorununu tekrarladı.

## 5. Gradyan teorisi hakkında güncel karar

V31 şu gerçeği göstermişti:

```text
Selection gradienti tamamen kapalıysa bilgi representation'a ulaşmıyor.
Selection gradienti aynı çizim sistemine sürekli giderse uzun vadede sistemi bozuyor.
```

Bu hâlâ doğrudur.

Ama V28/V29 ve şimdi V35 şunu ekledi:

> Gradyanı ayrı bir seçim beynine vermek, tek başına doğru winner bilgisini
> genellenebilir hâle getirmiyor.

Dolayısıyla:

```text
“Gradient önemlidir.”                 EVET
“Ana sorun yalnız gradient yoludur.”  HAYIR
“SPLIT/AGF bunu kesin çözer.”          HAYIR
```

V31'in 35K'daki geçici kazancı artık en iyi şu şekilde okunur:

> Kısa süreli objective değişimi optimizasyon yolunu faydalı bir noktaya itti;
> fakat stabil ve genellenebilir yeni bir selection bilgisi üretmedi.

## 6. Neyi kapatıyoruz?

Yeni isim veya küçük ayrıntı farkıyla şu deneyler yeniden açılmamalı:

- frozen V7 üstünde MLP/Transformer reranker,
- candidate P1/P2/P4/P5 corridor,
- ribbon genişliği veya offset sweep'i,
- mode/member clustering hakemi,
- AGF-V7 detached score FPN,
- SPLIT-Lane frozen support belief tower,
- frozen V7 tahminleri üzerinde temporal vote/consistency,
- QFL/ordinal quality head ile aynı bankayı yeniden puanlama.

V35 tek başına değil, V16/V19/V21/V26/V27/V28/V29/V34 ile birlikte bu kararı
destekliyor.

## 7. Sonraki mantıklı deney

Bir sonraki adım yeni model yazmak değil. Önce V7'nin **training contract'ının
yanlış ana moda neden kütle verdiğini** ölçmeliyiz.

# V36 — Assignment–Posterior Contract Audit

V34/V35'teki aynı good/current-wrong çiftleri için training-time matcher ve loss
yeniden oynatılacak. Şunlar ölçülecek:

1. Good ve wrong candidate training'de pozitif owner alıyor mu?
2. İkisi birden pozitifse loss onları eşdeğer mi görüyor?
3. DFL/point/LineIoU gradyanı hangisini GT'ye daha güçlü taşıyor?
4. Wrong candidate'ın yüksek olasılık modu training objective tarafından mı
   korunuyor, yoksa objective'e rağmen mi oluşuyor?
5. One-to-many assignment doğru tail candidate'a ne kadar probability mass
   veriyor?

Bu audit üç olası yolu ayıracak:

```text
Wrong pozitif, good zayıf/negatif:
  matcher/assignment kontratı düzeltilmeli.

Good daha güçlü doğru supervision alıyor ama yine kaybediyor:
  problem loss isminden çok representation/optimization kapasitesidir;
  yeni detector/backbone gerekir.

Good ve wrong aynı hedef sınıfında görülüyor:
  proposal sayısını yeniden seçmek yerine üretim posteriorunu official kalite
  tier'larına göre şekillendiren yeni bir detector objective'i gerekir.
```

V36 training-free ve ucuzdur. Sonucu görmeden başka selector veya büyük training
run'ı açılmamalıdır.

## 8. Çalışma zamanı ve provenance

```text
Ribbon cache:       63.43 saniye
Toplam ana koşu:   222.90 saniye
GPU:               RTX 5090
Test split:        kullanılmadı
Checkpoint seçimi: yapılmadı
Threshold seçimi:  yapılmadı
```

Kod branch'i:

```text
codex/v34-temporal-observability-20260824
```

Predeclared contract:

```text
docs/V35_RAW_RGB_TEMPORAL_OBSERVABILITY_CONTRACT_20260824.md
```

Ham çıktılar:

```text
outputs/diagnostics/v35_raw_rgb_temporal_observability/
  v35_raw_rgb_temporal_observability.json
  v35_raw_rgb_temporal_observability.md
  ribbon_cache/
  training/
```
