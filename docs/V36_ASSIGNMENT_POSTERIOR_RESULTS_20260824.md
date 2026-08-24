# V36 Assignment–Posterior Contract Audit — Sonuç Raporu

## Kısa karar

V36'nın sonucu nettir:

> **V7'nin proposal matcher'ı ve four-slot route target'ı, deploy sırasında
> kaybedilen oracle-good proposal'ı zaten doğru biçimde biliyor.**

980 exact route-recoverable good/current-wrong çiftinde:

| Ölçüm | Sonuç | Clip-bootstrap %95 |
| --- | ---: | ---: |
| Route target good'u wrong'dan üstün tutuyor | **%99.90** | `%99.66–100.00` |
| Route target top-1 = good | **%97.76** | `%96.75–98.63` |
| Configured matcher cost good'u tercih ediyor | **%97.35** | `%96.22–98.39` |
| Good proposal intended GT'ye atanıyor | **%91.43** | `%89.50–93.13` |
| Wrong proposal intended GT'ye atanıyor | **%2.14** | `%1.20–3.24` |
| Current slot refiner intended GT'ye loss alıyor | **%98.67** | `%97.51–99.66` |

Buna rağmen learned route posterior good'u wrong'dan yalnız tie-aware `%4.74`
oranında üstün tutuyor. Strict `good > wrong` sayısı **sıfırdır**. Proposal
existence olasılığında da good, wrong'u yalnız `%4.69` çiftte geçmektedir.

Önceden kilitlenen karar ağacının sonucu:

```text
POST_ASSIGNMENT_BELIEF_FAILURE
```

Bu nedenle:

```text
yeni matcher                          → ana yön değil
daha fazla positive auxiliary query  → ana yön değil
one-to-many'i kör biçimde büyütmek    → ana yön değil
route target'ı yeniden icat etmek     → ana yön değil
```

## 1. Ne yaptık?

Yeni model eğitmedik ve prediction üretmedik. Mature V7 `225K` checkpoint'inin
V34 sırasında cache'lenmiş exact çıktılarını kullandık.

Veri:

- 54 validation clip;
- 625 görüntü;
- 980 unique oracle-good/current-wrong çift;
- `.50` recoverable: 424 çift;
- `.75` recoverable: 789 çift;
- bazı çiftler iki threshold kohortunda birden bulunur;
- test split kullanılmadı.

Her çift için V7'nin üç training sözleşmesini yeniden oynattık:

1. 32-proposal Hungarian matcher;
2. four-slot soft route target;
3. routed-reference → GT final geometry Hungarian.

Target fixed-row lane'leri official annotation lane'lerine raster Hungarian ile
eşlendi. 2,320 lane eşleşmesinin ortalama raster IoU'su `0.9721` oldu; 980
çiftin hiçbiri mapping nedeniyle düşmedi.

## 2. Route target doğru adayı biliyor mu?

Evet, neredeyse kusursuz biçimde.

V7 route target kontratı:

```text
all-GT target
near-best cluster delta = 0.10
temperature = 0.03
```

Good ve wrong candidate arasındaki ortalama target probability margin:

```text
target_mass(good) - target_mass(wrong) = +0.9537
median                                  = +1.0000
```

Kırılımlar:

| Kohort | Target good preference | Target top-1 good |
| --- | ---: | ---: |
| Fold A | `%99.81` | `%97.55` |
| Fold B | `%100.00` | `%98.00` |
| IoU `.50` | `%99.76` | `%97.88` |
| IoU `.75` | `%100.00` | `%98.23` |

Bu sonuç route target semantiğini ana şüpheli olmaktan çıkarır.

## 3. Proposal matcher doğru adayı pozitif yapıyor mu?

Büyük çoğunlukta evet.

Configured V7 matcher:

```text
0.25 * object
+ 5.0 * point
+ 1.0 * range
+ 2.0 * LineIoU
```

Sonuç:

```text
896 / 980:
good intended GT'ye positive,
wrong negative

63 / 980:
good ve wrong ikisi de intended GT positive değil

21 / 980:
wrong intended GT'ye positive,
good negative
```

Başka bir deyişle:

```text
good positive rate  = %91.43
wrong positive rate =  %2.14
```

`.50` recoverable çiftlerde good positive oranı `%95.05`; `.75`te `%90.75`.

Dolayısıyla “oracle-good tail matcher tarafından çoğunlukla background
sayılıyor” hipotezi yanlışlandı.

## 4. Matcher bileşenleri ne söylüyor?

Good'u tercih etme oranları:

| Maliyet bileşeni | Good preference |
| --- | ---: |
| Object/existence | **%4.69** |
| Point | **%100.00** |
| Range | `%60.51` (+ `%23.16` tie) |
| LineIoU | **%99.90** |
| Weighted configured total | **%97.35** |

Bu tablo önemli bir ayrım gösterir:

```text
Geometri maliyetleri:
good proposal doğru diyor.

Existence confidence:
wrong proposal doğru diyor.
```

Object term yanlış yönde olsa da point ve LineIoU sinyali daha güçlü olduğu
için toplam Hungarian çoğunlukla good'u atıyor.

## 5. En sarsıcı sonuç: doğru label öğrenilmiş skora dönüşmüyor

Good proposal matcher positive olduğu hâlde current route hâlâ wrong'u seçiyor.

980 çiftte:

- `804` çiftte good proposal intended positive;
- buna rağmen learned route posterior doğrudan wrong'u good'dan yüksek tutuyor;
- current wrong proposal negative olmasına rağmen existence olasılığı good'dan
  yüksek olan çift oranı **%93.16**.

Ortalama confidence marginleri:

```text
p_exist(good) - p_exist(wrong) = -0.3328
p_route(good) - p_route(wrong) = -0.2297
```

Yani model yalnız route head'de değil, proposal ownership confidence'ında da
doğru supervision'ın tersine inanıyor.

## 6. Loss'un mevcut checkpointte istediği yön ne?

Output-logit seviyesinde descent yönünü tekrar hesapladık.

Good candidate için:

```text
route target good'u yükseltmek istiyor
proposal positive target good'u yükseltmek istiyor
```

bu iki yön çiftlerin `%91.43`ünde aynı anda geçerli.

Wrong candidate için iki loss'un da skoru düşürmek istediği çift oranı
`%97.86`.

Yalnız:

```text
route good'u yükseltirken proposal target good'u düşürüyor: %8.47
route wrong'u düşürürken proposal target wrong'u yükseltiyor: %2.04
```

Dolayısıyla ana problem iki target'ın çoğu örnekte birbirine ters olması
değildir. Çoğu çiftte ikisi de doğru yönü istemektedir.

## 7. Refiner yanlış GT'ye mi eğitiliyor?

Hayır.

Current slot'un routed raw reference'ı üzerinden tekrar oynatılan exact
range-aware Hungarian, slotu intended GT'ye `%98.67` oranında atadı.

Kırılımlar:

```text
.50: %96.93
.75: %99.37
```

Bu nedenle failure'ın ana açıklaması:

```text
router yanlış proposal seçiyor,
refiner da onu başka GT'ye çekiyor
```

değildir. Refiner çoğunlukla doğru lane target'ını alıyor; fakat ±24 px bounded
correction yanlış başlangıç geometrisini her zaman kurtaramıyor.

## 8. Counterfactual matcher sonuçları

| Assignment | Good intended positive | Wrong intended positive |
| --- | ---: | ---: |
| Configured | `%91.43` | `%2.14` |
| Object term kapalı | `%97.86` | `%0.10` |
| Point-only | `%98.27` | `%0.00` |
| Target row-IoU | `%97.76` | `%0.10` |
| Official raster oracle | `%100.00` | `%0.00` |

Point-only configured matcher'a göre:

```text
71 failure çiftini good-positive yapıyor
4 mevcut good-positive çifti kaybediyor
net +67 / 980 = +6.84 puan
```

Bu iyileştirme gerçektir fakat önceden kilitlenen `+10 puan` starvation
kapısını geçmez. Daha önemlisi, configured matcher'ın zaten good'u positive
yaptığı `896` çiftin çoğunda deployed belief yine yanlıştır. Matcher'ı kusursuz
yapsak bile ana failure kitlesi kalır.

## 9. Bu sonuç ne anlama geliyor?

V36 üç açıklamayı kapatıyor:

```text
“Training target yanlış candidate'ı istiyor.”
→ Hayır.

“Proposal Hungarian good tail'i çoğunlukla background yapıyor.”
→ Hayır.

“Final refiner çoğunlukla yanlış GT'ye loss alıyor.”
→ Hayır.
```

Daha doğru teşhis:

> **V7'nin objective'i doğru candidate'ı büyük ölçüde tanıyor, fakat bu doğru
> label candidate confidence ve final route posterioruna öğrenilmiş bir inanç
> olarak yerleşmiyor.**

Bu, “gradient yolu önemlidir” görüşünü destekler; fakat “daha fazla gradient
vermek çözer” sonucunu desteklemez. V31 continuous bridge, V32 pulse ve V28/V29
ayrı trainable belief tower zaten bu basit rescue biçimlerini yanlışladı.

## 10. En önemli sınır

V36 bir **mature snapshot audit**'idir.

Şunu biliyoruz:

```text
225K final decoder output'unda good query positive.
```

Fakat henüz şunu bilmiyoruz:

```text
Aynı query decoder'ın önceki üç katmanında da positive miydi?
```

V7 her decoder layer için yeniden Hungarian assignment yapıyor. Exist head
katmanlar arasında paylaşılmışsa şu durum mümkün:

```text
Layer 1: final-good query negative
Layer 2: final-good query negative
Layer 3: final-good query başka GT
Layer 4: final-good query intended positive
```

Bu durumda final snapshot target doğru görünürken deep supervision aynı query
state zincirine çelişkili label geçmişi vermiş olabilir.

Bu şu an yalnız hipotezdir; V36 bunu ölçmedi.

## 11. Sonraki mantıklı deney

# V37 Deep-Supervision Assignment Trajectory Audit

Mature V7 aynı 625 görüntüde bir kez yeniden forward edilmelidir. Dört decoder
layer'ın her biri için:

- final-good query intended/any assignment;
- final-wrong query assignment;
- intended GT'nin query kimliği;
- good/wrong geometry quality trajectory;
- intermediate existence target işareti;
- configured intermediate layer weightleriyle net positive/negative label
  baskısı;
- assignment switch count

ölçülmelidir.

Karar:

```text
Good final query erken layerlarda sıkça negative ise:
→ deep-supervision identity/assignment contract doğrudan hedeflenir.

Good query bütün layerlarda tutarlı positive ise:
→ assignment ailesi tamamen kapanır;
  sorun label erişimi değil, representation/optimization sınırıdır.
```

V37 training-free ve kısa bir audittir. Sonucu görülmeden yeni architecture veya
uzun training açılmamalıdır.

## Son hüküm

V36'nın en sade özeti:

```text
Matcher doğruyu biliyor.
Route target doğruyu biliyor.
Refiner hangi GT'yi düzeltmesi gerektiğini biliyor.

Ama modelin iki confidence sistemi de
yanlış proposal'a daha çok inanıyor.
```

Bu nedenle yeni matcher ya da daha fazla proposal supervision ana çözüm değildir.
Şimdi ölçülmesi gereken son açık training-contract ihtimali, layerlar arasındaki
assignment/label tutarlılığıdır.
