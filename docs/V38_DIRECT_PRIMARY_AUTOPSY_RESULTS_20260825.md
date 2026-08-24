# V38 Direct-Primary 50K Otopsi Sonucu

## Kısa karar

V38'in ana sorunu tek başına confidence değildir. Özellikle strict `.75`
ölçümünde sorun açık biçimde **geometri kapasitesidir**: dört direct query'nin
ürettiği eğriler yeterince hassas değildir.

- `.50` kaybı karışıktır: yaklaşık yarısı geometri, yarısı confidence/emission.
- `.75` kaybının `%94.6` kadarı, skorları tamamen yok saysak bile kalan
  geometri eksikliğidir.
- Range başlangıç/bitiş hatası ikincildir. Asıl hata lane'in `x` konumunda ve
  özellikle görüntünün alt yarısında büyümektedir.
- Deployed duplicate ve crossing oranları yaklaşık `%3.5–3.7` olduğu için ana
  darboğaz değildir.

Bu sonuç, **QFL veya başka bir score loss'un tek başına V38'i kurtaramayacağını**
gösterir. Bir sonraki minimal deney confidence değil, dört query'nin başlangıç
eğrisini görüntüye göre üretmeyi hedeflemelidir.

## 1. Ne yaptık?

Aynı V38 `iter_0050000.pt` checkpoint'ini hiç eğitmeden, official CULane
validation'ın bütün `9.675` görüntüsünde üç politika ile değerlendirdik:

1. `deployed`: Mevcut V38; existence skoru `0.50` üzerindeki lane'leri yazar.
2. `forced_all_valid`: Dört lane'in tamamını skoruna bakmadan yazar.
3. `cardinality_oracle`: GT'yi bilen, deploy edilemez bir oracle; aynı dört
   eğri içinden official IoU eşiğini geçen en iyi eşleşmeleri seçer.

Oracle'ın amacı final model skoru üretmek değildir. Yalnız şu soruyu yanıtlar:

> Skor tamamen kusursuz olsaydı, mevcut dört eğri içinde kaç doğru lane vardı?

Güvenlik kontrolleri:

- Test split kullanılmadı.
- Eğitim yapılmadı.
- Threshold veya checkpoint seçimi yapılmadı.
- FP32 ve official V38 evaluator ile aynı `channels_last` yürütme kullanıldı.
- Diagnostic rasterizer official CULane'ın `int32` truncation davranışıyla
  eşleştirildi.
- Cache'den yeniden hesaplanan deployed TP/FP/FN, mevcut official V38 raporuyla
  iki eşikte de birebir eşleşti.

## 2. Ne oldu?

### Official set-level sonuç

| Politika | TP@.50 | FP@.50 | FN@.50 | F1@.50 | TP@.75 | FP@.75 | FN@.75 | F1@.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V38 deployed | 25.045 | 6.674 | 7.637 | 77.778 | 16.107 | 15.612 | 16.575 | 50.021 |
| Dört lane'i zorla yaz | 25.542 | 13.158 | 7.140 | 71.564 | 16.249 | 22.451 | 16.433 | 45.527 |
| Best-of-4 oracle | 25.542 | 0 | 7.140 | 87.737* | 16.249 | 0 | 16.433 | 66.416* |
| V7 50K reference | 26.000 | 6.263 | 6.682 | 80.068 | 18.759 | 13.504 | 13.923 | 57.769 |

`*` Oracle F1 deploy edilebilir bir performans değildir; oracle GT'yi bildiği
için yalnız doğru eşleşmeleri yazar ve FP'si yapay olarak sıfırdır. Bilimsel
olarak önemli olan oracle'ın TP/support sayısıdır.

### V7 TP açığının ayrıştırılması

| Eşik | V7 − deployed TP | V7 − best-of-4 TP | Best-of-4 − deployed TP | Geometri payı |
| --- | ---: | ---: | ---: | ---: |
| `.50` | 955 | 458 | 497 | `%47.96` |
| `.75` | 2.652 | 2.510 | 142 | `%94.65` |

Basit okuma:

- `.50`: V38'in V7'ye göre kaybettiği 955 TP'nin 497'si mevcut ama confidence
  yüzünden yazılmıyor; 458'i dört eğrinin hiçbirinde yok.
- `.75`: Kaybedilen 2.652 TP'nin yalnız 142'si confidence yüzünden kaçıyor;
  2.510'u mevcut dört eğrinin hiçbirinde strict kalitede yok.

Support'un deployed çıktıya dönüşümü zaten yüksektir:

```text
@.50: %98.05
@.75: %99.13
```

Bu yüzden `.75` için router/confidence ana darboğaz değildir.

## 3. Existence skoru gerçek kaliteyi biliyor mu?

Kısmen, fakat yeterince temiz bilmiyor:

```text
Existence score ↔ official IoU Spearman: 0.515
AUC, IoU > .50:                         0.790
AUC, IoU > .75:                         0.733
```

Skorlar ayrıca aşırı doygun:

| Candidate kalitesi | Existence medianı |
| --- | ---: |
| IoU ≤ `.50` | 0.824 |
| `.50` < IoU < `.75` | 0.9995 |
| IoU > `.75` | 0.9998 |

Yani confidence kalibrasyonu gerçekten kötüdür. Background/near-miss birçok
aday çok yüksek skor almaktadır. Fakat bir image'daki adaylar one-to-one
eşleştiğinde, kusursuz confidence'ın kurtarabileceği **benzersiz** strict TP
yalnız 142'dir. Candidate-level yanlış confidence sayısını doğrudan kazanılacak
TP sanmamak gerekir.

## 4. Sorun geometride tam olarak nerede?

### Range mi, x konumu mu?

Best-of-4 eğrilerde range sınırı hatası:

```text
Başlangıç hatası median: 3 row
Bitiş hatası median:     1 row
```

Strict `.75` desteği olmayan GT'lerde bile median range hatası yaklaşık
`4 row / 2 row` seviyesindedir. Range kusursuz değildir ama ana kayıp değildir.

### X hatası görüntünün altına doğru büyüyor

Her GT için dört eğrinin en iyisinin median mutlak x hatası:

| Row bandı | Median x hatası |
| --- | ---: |
| üst `0–39` | 3.50 px |
| üst-orta `40–79` | 4.12 px |
| alt-orta `80–119` | 6.43 px |
| alt `120–159` | 10.46 px |

`.75` desteği hiç bulunmayan GT'lerde tablo çok daha serttir:

| Row bandı | Median x hatası |
| --- | ---: |
| üst `0–39` | 6.86 px |
| üst-orta `40–79` | 12.33 px |
| alt-orta `80–119` | 21.01 px |
| alt `120–159` | 29.30 px |

Karşılaştırma için deployed `.75` TP'lerde alt bandın median hatası yalnız
`4.73 px`tir.

Bu sonuç şunu söylüyor:

> V38'in strict başarısızlığı lane'in nerede başlayıp bittiğini bilmemesinden
> çok, eğriyi özellikle alt ve alt-orta satırlarda yanlış yatay konuma
> yerleştirmesidir.

## 5. Duplicate ve crossing ana sorun mu?

Hayır.

```text
Deployed duplicate-image oranı: %3.535
Deployed crossing-image oranı:  %3.700
```

Dördüncü lane zorla açıldığında ikisi de yaklaşık `%26.6`ya çıkıyor. Bu,
mevcut existence kararının birçok bozuk dördüncü eğriyi faydalı biçimde
bastırdığını gösterir. `forced-active`, NMS veya MMR ana çözüm değildir.

## 6. Kanıt ve hipotezi ayıralım

### Kanıtlanan

1. V38 direct-primary 50K, V7 50K'nın gerisindedir.
2. `.75` TP açığının `%94.6`sı best-of-4 oracle altında bile kalır.
3. Strict kayıplarda yatay hata alt satırlara doğru ciddi biçimde büyür.
4. Existence skoru kötü kalibredir ama strict TP açığının ana nedeni değildir.
5. Dört lane'i zorla yazmak recall'ı az artırıp FP'yi çok büyütür.
6. Deployed duplicate/crossing ana public darboğaz değildir.

### Henüz yalnız hipotez

1. Sabit query/anchor başlangıcının lane'in global eğrisini yeterince iyi
   kuramadığı.
2. Görüntüye bağlı tam-eğri başlangıcının alt satır x hatasını azaltabileceği.
3. Global yol bağlamı + candidate çevresi evidence'ının daha sonra ek yarar
   sağlayabileceği.

Bu otopsi ikinci ve üçüncü maddeleri henüz kanıtlamaz; yalnız bir sonraki
deneyde hangi değişkenin hedeflenmesi gerektiğini gösterir.

## 7. Sonraki mantıklı deney

Tam MIND-Lane paketini tek seferde kurmak yanlış olur. İlk causal gate yalnız
şu parçayı test etmelidir:

### V39-PQI — Image-Conditioned Full-Curve Query Initialization

Bu parça ne yapacak?

> Dört lane query'si sabit bir başlangıçtan çıkmak yerine, görüntünün global
> yol bilgisinden üretilen tam bir 160-row başlangıç eğrisiyle decoder'a
> girecek.

Modele neden lazım?

> Otopsi, strict başarısızlığın esas olarak alt/alt-orta row'larda yanlış x
> konumundan geldiğini gösterdi. Bu modül doğrudan o geometriyi hedefler.

Nereye bağlanacak?

```text
Global image feature
    ↓
Train-only 16 curve prototype için ağırlıklar
    ↓
4 × 160 başlangıç eğrisi
    ↓
Mevcut V38 decoder
```

İlk gate'te özellikle olmayacaklar:

```text
QFL yok
Dense field yok
Denoising yok
Multi-inquiry yok
Attention-flow loss yok
Yeni selector/router yok
MMR/NMS yok
```

Adil deney:

- ortak V38 50K checkpoint'inden iki exact-paired continuation;
- control: normal V38;
- treatment: yalnız zero-init pattern-query residual;
- aynı manifest, augmentation, optimizer, LR, seed ve 65K endpoint;
- ana karar yalnız predeclared 65K endpoint'inde.

Önerilen PASS:

```text
All-4 support TP @.75: control'den en az +300
Best-of-4 median IoU:  control'den en az +0.01
Unsupported-.75 alt-band median x hatası: en az %15 azalma
Official F1@.75:      en az +0.50 puan
Official F1@.50:      non-regression
Duplicate/crossing:   belirgin artmamalı
```

Bu gate geçerse global–center–periphery multi-inquiry denenebilir. Geçmezse
pattern initialization kapatılır; full MIND-Lane'e kör biçimde yatırım
yapılmaz.

## Nihai hüküm

V38 bize şunu öğretti:

```text
Modelin dört lane'e verdiği güven kusurlu.
Ama strict kaybın asıl nedeni bu değil.

Asıl neden:
Dört lane'in kendisi yeterince doğru yerde çizilmiyor.
Özellikle görüntünün alt yarısında.
```

Bu yüzden sıradaki yatırım score loss'una değil, **görüntüye bağlı tam-eğri
query başlangıcına** gitmelidir. QFL daha sonra, geometri desteği gerçekten
arttıktan sonra confidence/FP temizliği için yeniden değerlendirilebilir.

