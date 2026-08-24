# V37 Deep-Supervision Trajectory Audit — Sonuç Raporu

Tarih: 2026-08-24  
Karar: `DEEP_SUPERVISION_NOT_PRIMARY`  
Test split: kullanılmadı

## Kısa cevap

V37, V7'nin ara decoder katmanlarındaki bağımsız eşleştirmelerin finalde doğru proposal olacak query'yi yanlış yönde eğitip eğitmediğini ölçtü.

Sonuç:

> Ara katmanlarda query kimliği bazen değişiyor; fakat ara ve final existence loss'larının toplamı doğru proposal'ın logitini yükseltiyor, yanlış proposal'ın logitini düşürüyor.

Bu nedenle V7'nin `wrong-but-confident` davranışının ana nedeni deep supervision çatışması değildir. Matcher, route target ve deep-supervision sözleşmesini yeniden düzenleyen deney ailesi burada kapatılmalıdır.

## 1. Ne yaptık?

V36 şu güçlü sonucu vermişti:

- Route target doğru proposal'ı biliyor.
- Final matcher doğru proposal'ı intended-positive olarak işaretliyor.
- Final output seviyesindeki existence gradienti doğru proposal'ı yukarı, yanlış proposal'ı aşağı itiyor.
- Buna rağmen eğitilmiş model yanlış proposal'a çok daha yüksek existence veriyor.

V37 kalan son sözleşme açığını test etti:

> Final katman doğru query'yi pozitif yapsa bile, ara decoder katmanları aynı query'yi negatif yapıp toplam eğitim sinyalini bozuyor olabilir mi?

Bunun için mature V7 `iter_0225000.pt` checkpoint'i üzerinde:

1. Dört decoder katmanının matcher assignment'ları ayrı ayrı tekrarlandı.
2. V36'daki 980 `good-versus-wrong` proposal çifti izlendi.
3. Her katmanda good ve wrong query'nin existence ve quality değerleri kaydedildi.
4. Gerçek training ağırlıklarıyla ara ve final existence loss'larının output-logit update yönleri hesaplandı.
5. 128 dengeli görüntüde, paylaşılan `head.exist` parametreleri için ara-katman ve final-katman gradientlerinin cosine benzerliği ölçüldü.
6. Sonuçlar iki clip-disjoint fold ve `.50/.75` alt gruplarında ayrı tekrarlandı.

## 2. Replay güvenilirliği

İlk replay yalnız 625 hedef görüntüyü yeniden forward etmişti. V34 cache'i farklı GPU/batch kompozisyonuyla BF16 üretilmiş olduğu için nadir autoregressive row outlier'ı oluştu ve hard parity gate geçmedi. Bu ilk run bilimsel sonuç üretmek için kullanılmadı.

İkinci run, V34'ün exact `temporal_union_val.txt` listesini ve batch sırasını yeniden kullandı: 3.072 görüntü forward edildi, 625 hedef görüntü ve 980 çift seçildi.

Stabilite sonuçları:

| Ölçüm | Sonuç |
| --- | ---: |
| V36 pair assignment-state agreement | `%99.90` |
| Tüm proposal'larda ortalama `pred_x` farkı | `0.156 px` |
| Good/wrong çiftlerinde ortalama curve farkı | `0.113 px` |
| Good quality ortalama mutlak farkı | `0.00144` |
| Wrong quality ortalama mutlak farkı | `0.00043` |
| Good existence ortalama mutlak farkı | `0.00092` |
| Wrong existence ortalama mutlak farkı | `0.00043` |
| Range maksimum mutlak farkı | `0.0273` |

Tüm bankada tek bir nadir row outlier'ı nedeniyle maksimum `pred_x` farkı `100.845 px` kaldı. Ancak karar verilen 980 pair'in assignment state'i `%99.90` aynı, pair curve farkı `0.113 px` ve score farkları çok küçüktür. Bu nedenle sonuçlar V36 pair population'ı için güvenilir kabul edildi; bit-exact replay iddiası yapılmadı.

## 3. Ne oldu?

### 3.1 Assignment katmandan katmana iyileşiyor

| Decoder katmanı | Good intended-positive | Wrong intended-positive | Good `p_exist` | Wrong `p_exist` | Good quality | Wrong quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Layer 1 | `%65.51` | `%8.37` | `0.484` | `0.805` | `0.526` | `0.257` |
| Layer 2 | `%80.31` | `%3.88` | `0.495` | `0.831` | `0.605` | `0.253` |
| Layer 3 | `%88.06` | `%2.55` | `0.503` | `0.836` | `0.636` | `0.252` |
| Final | `%91.33` | `%2.14` | `0.508` | `0.841` | `0.645` | `0.252` |

Ara katmanlarda assignment churn vardır:

- Good proposal'ın bütün katmanlarda intended-positive kalma oranı `%62.45`.
- En az bir ara katmanda negatif olma oranı `%36.53`.
- Intended query'nin bütün katmanlarda aynı kalma oranı `%68.06`.
- Ortalama query switch sayısı `0.341`.

Fakat bu churn ilerledikçe düzelmektedir. Good-positive oranı `%65.51 → %91.33`, wrong-positive oranı `%8.37 → %2.14` olur.

### 3.2 Kritik final-positive cohort

En temiz nedensel alt grup, final katmanda good proposal'ın gerçekten positive olduğu 895 çifttir.

Bu grupta:

| Ölçüm | Sonuç |
| --- | ---: |
| Good bütün katmanlarda intended-positive | `%68.38` |
| Good en az bir ara katmanda negatif | `%31.62` |
| Good toplam update yukarı | **`%100.00`** |
| Good toplam update aşağı | **`%0.00`** |
| Wrong toplam update aşağı | **`%99.89`** |
| Wrong toplam update yukarı | `%0.11` |

Bu tablo V37'nin en önemli sonucudur:

> Ara katmanlardan biri good query'yi geçici olarak negatif yapsa bile, gerçek loss ağırlıklarıyla ara+final toplam existence update'i final-good query'yi tek bir vakada bile aşağı itmedi.

### 3.3 Bütün 980 pair

Finalde good positive olmayan vakalar da dahil edildiğinde:

| Ölçüm | Sonuç | Clip bootstrap `%95` |
| --- | ---: | ---: |
| Good toplam update yukarı | `%93.57` | `%91.83–95.09` |
| Good toplam update aşağı | `%6.43` | `%4.91–8.17` |
| Wrong toplam update aşağı | `%98.16` | `%97.23–99.03` |
| Wrong toplam update yukarı | `%1.84` | `%0.97–2.77` |

Yani genel population'da da eğitim yönü ezici biçimde doğrudur.

### 3.4 Gerçek parameter-gradient ölçümü

Output-logit işaretleri tek başına yeterli olmayabileceği için aynı audit, paylaşılan `head.exist` parametrelerinde gerçek autograd gradientlerini de ölçtü.

128 dengeli görüntüde ara-katman toplam gradienti ile final-katman gradienti:

| Ölçüm | Sonuç |
| --- | ---: |
| Cosine mean | `0.756` |
| Cosine median | `0.836` |
| Cosine q10 / q90 | `0.472 / 0.961` |
| Negatif cosine oranı | **`%1.56`** (`2/128`) |
| Ara/final norm oranı median | `0.504` |

Katman bazında erken decoder daha gürültülüdür:

| Ara katman | Cosine median | Negatif oran | Ara/final norm median |
| --- | ---: | ---: | ---: |
| Layer 1 | `0.454` | `%10.94` | `0.107` |
| Layer 2 | `0.667` | `%8.59` | `0.222` |
| Layer 3 | `0.909` | `%2.34` | `0.278` |

Layer 1'de yerel çatışma vardır; fakat gradient normu final gradientinin yalnız yaklaşık `%10.7`si düzeyindedir. Üç ara katman birlikte ele alındığında toplam gradient final gradientiyle güçlü biçimde aynı yöndedir.

### 3.5 Fold ve threshold tekrarı

Sonuç tek bir clip grubuna veya IoU eşiğine bağlı değildir:

| Alt grup | Good toplam update yukarı | Wrong toplam update aşağı |
| --- | ---: | ---: |
| Fold A | `%92.66` | `%97.55` |
| Fold B | `%94.65` | `%98.89` |
| `.50` çiftleri | `%96.46` | `%99.53` |
| `.75` çiftleri | `%93.03` | `%97.97` |

## 4. Bu sonuç ne anlama geliyor?

Kanıtlananlar:

1. V7'nin matcher'ı katmanlar arasında tamamen stabil değildir.
2. Erken decoder katmanlarında az miktarda local gradient çatışması vardır.
3. Fakat toplam deep-supervision objective doğru proposal'ın existence logitini yükseltir.
4. Aynı objective yanlış proposal'ın existence logitini düşürür.
5. Ara ve final existence parameter gradientleri neredeyse her batch'te aynı yöndedir.
6. Buna rağmen mature model wrong proposal'a çok daha yüksek existence verir: final katmanda ortalama `0.841` versus good `0.508`.

Dolayısıyla:

> Model yanlış proposal'a, eğitim kontratı onu yanlış yönde ittiği için inanmıyor. Eğitim sinyali doğru yönde olmasına rağmen öğrendiği representation/logit düzeni yanlış proposal'ı tercih ediyor.

Bu V36'nın `POST_ASSIGNMENT_BELIEF_FAILURE` kararını güçlendirir.

## 5. Sorun nerede değil?

V36 ve V37 birlikte aşağıdaki rescue fikirlerini kapatır:

```text
Matcher cost'u tekrar ayarlamak
Final matcher'ı başka türlü kurmak
Ara decoder matching'i kaldırmak
Deep supervision ağırlığını azaltmak
No-object ağırlığını tek başına değiştirmek
Point/LineIoU target'ını yeniden tanımlamak
Selection ve proposal loss'larının işaretini düzeltmek
```

Bu parçaların kusursuz olduğu söylenmiyor. Söylenen şey daha dar ve güçlüdür:

> Ölçülen wrong-belief hatasını açıklayacak büyüklükte ters bir assignment veya deep-supervision gradienti bulunmadı.

## 6. Sorun muhtemelen nerede?

Kanıt sırasına göre:

### En olası açıklama: representation/optimization başarısızlığı

Modelin candidate-local feature'ları yanlış proposal üzerindeki güçlü görüntü işaretlerini kolayca kullanıyor. Doğru proposal için gelen supervision doğru olsa da, bu feature düzeni unseen görüntülerde doğru ayrımı genelleştiremiyor.

Bu bir hipotezdir; V36/V37 bunun tam sinirsel nedenini tek başına kanıtlamaz. Ancak yanlış target ve ters gradient açıklamalarını büyük ölçüde elediği için en güçlü kalan açıklamadır.

### İkinci açıklama: tek görüntüde gözlemlenebilirlik sınırı

Bankadaki oracle-best proposal bazen gerçek bir görüntü modu değil, düşük olasılıklı lucky-tail örneği olabilir. Böyle bir durumda doğru candidate'ın doğru olduğunu tek kareden genellenebilir biçimde anlamak mümkün olmayabilir.

Bu ihtimal V28/V29 ayrı trainable belief tower, V34 temporal-frozen observability ve V35 raw-RGB crossfit başarısızlıklarıyla güçlenmiştir.

## 7. Sınırlamalar

V37'nin iddia etmediği şeyler:

1. Bütün backbone/FPN parametrelerinin 225K adımlık training boyunca hiç gradient çatışması yaşamadığını kanıtlamaz.
2. Yalnız mature checkpoint snapshot'ında exact existence sözleşmesini ve paylaşılan `head.exist` parametrelerini denetler.
3. BF16 cross-device replay bit-exact değildir; nadir bir tüm-bank outlier'ı vardır.
4. Assignment trajectory GT kullanan diagnostic'tir; deploy metriği değildir.

Buna rağmen V37'nin test ettiği spesifik hipotez için sonuç yeterince nettir:

> Intermediate existence supervision final-good query'yi sistematik biçimde aşağı itmiyor.

## 8. Sonraki mantıklı deney

Proposal-selection kurtarma ailesini daha büyük bir head, SPLIT/AGF tower, corridor veya yeni matcher ile sürdürmek artık düşük değerdedir. Özellikle V28/V29 zaten frozen support yanında route target'la baştan sona eğitilen ayrı bir DLA/FPN belief tower'ı test etmiş ve OOF'ta başarısız olmuştur.

Bir sonraki rasyonel soru şudur:

> V7'nin 32 proposal bankasını seçmeye çalışmak yerine, az sayıda one-to-one primary lane'i doğrudan üreten model yeterince uzun eğitildiğinde mature V7'ye yaklaşabiliyor mu?

V33 direct-primary kontrolü yalnız yaklaşık `13.9K` adımda ve kısa component gate amacıyla ölçüldü. Bu nedenle önce V25/V33 checkpoint ve optimizer sözleşmesi denetlenmeli. Eğer direct-primary model için gerçek matched mature horizon hiç çalıştırılmadıysa, V38 tek önceden kilitli endpoint ile bu eksikliği test etmelidir.

PASS için yalnız öğrenme eğrisi değil, full validation official F1 ve proposal/primary support birlikte kullanılmalıdır. Direct-primary mature eğitimde de belirgin biçimde V7 altında kalırsa, yeni detector yönü daha temel biçimde yeniden tasarlanmalıdır.

## Nihai karar

```text
Deep-supervision conflict ana neden      → REDDEDİLDİ
Matcher/assignment rescue               → KAPAT
V7 wrong-belief gerçektir               → DOĞRULANDI
Target ve lokal gradient yönü doğrudur  → DOĞRULANDI
Başka SPLIT/AGF selector yaz             → HAYIR
Mature direct-primary eksikliği denetle → EVET
```

