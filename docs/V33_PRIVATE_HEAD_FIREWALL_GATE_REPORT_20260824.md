# V33-PH Private-Head Cold-Start Firewall Gate — Nihai Rapor

Tarih: 24 Ağustos 2026

## 1. Kısa cevap

V33-PH deneyinin sonucu iki parçalıdır:

1. **Rastgele auxiliary head'in başlangıç problemi gerçekten düzeltildi.** Yalnız
   private proposal head eğitildiğinde, ortak ağa gidecek primary ve auxiliary
   gradyanlar belirgin biçimde hizalandı.
2. **Bu düzeltme official validation F1 kazancına dönüşmedi.** Pretrained-head
   kolu `.50`de yalnız küçük ve belirsiz bir artış üretti; `.75`te ise geriledi.

En sade hüküm:

> Cold-start çatışması gerçekti, fakat V33-B'nin başarısızlığının ana ve yeterli
> açıklaması değildi. Auxiliary proposal objective hizalandıktan sonra primary
> modele yararlı yeni bilgi katmıyor; geniş toleransta küçük bir düzenleme
> yaparken hassas geometriyi bozuyor.

Bu nedenle:

```text
V33-PH uzun eğitime açılmayacak.
Auxiliary loss/pretraining süresi sweep edilmeyecek.
Direct-primary + shared auxiliary geometry ailesi kapatılacak.
Test split'i açılmayacak.
```

## 2. Neyi test ettik?

Önceki V33-GI audit'inde şu problem ölçülmüştü:

```text
Random auxiliary head
        ↓
ilk auxiliary gradyanlar
        ↓
özellikle FPN/P2/fine-resolution shared feature'ları
primary'nin istemediği yöne itiyor
```

Fakat auxiliary head öğrenildikten sonra gradyanlar hizalanıyordu. V33-PH şu
tek soruyu test etti:

> Auxiliary head'i önce tek başına eğitip cold-start zararını kaldırırsak,
> training-only auxiliary supervision primary final lane'leri gerçekten
> iyileştirir mi?

### Karşılaştırılan kollar

| Kol | Başlangıç | Ana 2,778 adım |
| --- | --- | --- |
| A | Exact G0 parent | Primary-only objective |
| B | Exact G0 parent + random private head | Primary + auxiliary objective |
| D | Exact G0 parent + privately pretrained head | Primary + auxiliary objective |

A ve B daha önceki exact-paired V33 gate'inden gelir. D bu deneyde üretildi.

## 3. Private head'i nasıl hazırladık?

Faz 0'da yalnız şu parametre grubu trainable bırakıldı:

```text
detector.proposal_memory
```

Kontrat:

```text
Private parametre:          143,296
Private parametre tensörü:  26
Private pretraining:        2,778 step
Loss:                       yalnız proposal coverage
Shared update:              0
Backbone/FPN/primary:       frozen
```

Doğrulamalar:

```text
Frozen parent state bit-exact:       true
Frozen tensörde gradient:            0 / 291
Nonzero private gradient tensörü:    26 / 26
Checkpoint selection:                yok
Threshold selection:                 yok
Test split:                          kullanılmadı
```

Private auxiliary loss:

```text
İlk adım:                  6.591
İlk 5 kayıt ortalaması:   5.644
Son 20 kayıt ortalaması:  3.310
Son adım:                 3.528
```

Bu faz yaklaşık `33.8 img/s` hızla çalıştı.

## 4. Cold-start gerçekten kalktı mı?

Evet, önceden kilitlenen mekanizma gate'i geçti.

32 exact train/probe batch çifti üzerinde, aynı shared parametreler için
primary ve auxiliary gradyanlar ayrı ayrı ölçüldü.

| Ölçüm | Parent + random head | Parent + pretrained private head |
| --- | ---: | ---: |
| Shared gradient cosine, medyan | `+0.028` | **`+0.635`** |
| Negatif gradyan çifti | `%37.5` | **`%3.1`** |
| Auxiliary virtual-step primary harm | `%68.75` | **`%59.38`** |

Cosine'ın basit anlamı:

```text
0'a yakın:
primary ve auxiliary farklı yönlere çekiyor.

+0.635:
iki görev büyük ölçüde benzer feature değişimini istiyor.
```

Burada önemli bir nüans var. Cosine iyileşmesi güçlüdür; fakat virtual-harm
oranı `%59.38` ile predeclared `%60` sınırını yalnız kıl payı geçmiştir.
Dolayısıyla “auxiliary artık kesin yararlıdır” diyemeyiz. Doğru ifade:

> Ölçülen random-head cold-start sorunu kaldırıldı; fakat auxiliary yönün probe
> batch'lerde güvenilir biçimde yararlı olduğu kanıtlanmadı.

### Ana eğitimin başlangıcında beklenen etki görüldü

Ana continuation'ın ilk beş kayıt ortalaması:

| Kol | Proposal coverage loss |
| --- | ---: |
| B — random head | `5.600` |
| D — pretrained head | **`3.365`** |

Ana eğitimin sonunda iki kol birbirine yaklaştı:

```text
B son 20 ortalama: 3.344
D son 20 ortalama: 3.261
```

Yani private pretraining tam hedeflenen işi yaptı: B'nin ilk adımlardaki
cold-start yükünü ortadan kaldırdı. Deney başarısızsa bunun nedeni artık
“head hiç öğrenemedi” değildir.

## 5. Ana training kontratı adil miydi?

A, B ve D:

```text
Exact aynı parent SHA256:
5d327f73c7920c3f3d5d32bea86850a33bda96938c2864fea3fd9d2c834d8935

Başlangıç iterasyonu: 11,110
Bitiş iterasyonu:     13,888
Shared update:        2,778
Official epoch:       0.250045
Runtime seed:         3407021331
```

B ve D aynı config'i ve aynı primary+auxiliary ana objective'i kullandı. D'deki
tek nedensel fark, ana stream başlamadan önce `proposal_memory` tensörlerinin
private-pretrained olmasıdır. D transplant kontrolü parent/shared tensörlerin
değişmediğini doğruladı.

D ana eğitimi yaklaşık `20.8 img/s` hızla tamamlandı.

## 6. Full official validation sonucu

Population:

```text
9,675 görüntü
54 video clip
filtreleme yok
deduplication yok
test split kapalı
```

| Model | F1@.50 | F1@.75 |
| --- | ---: | ---: |
| A — Primary-only | **72.290** | **37.249** |
| B — Random-head joint | 72.106 | 37.309 |
| D — Pretrained-head joint | **72.436** | 36.782 |

D farkları:

| Karşılaştırma | Δ F1@.50 | Δ F1@.75 |
| --- | ---: | ---: |
| D − A | `+0.146` | **`−0.468`** |
| D − B | **`+0.330`** | **`−0.527`** |

Predeclared PASS:

```text
D − A @.50 >= +0.30       FAIL
D − A @.75 >=  0.00       FAIL
D − B @.50 >= +0.20       PASS
D − B @.75 >=  0.00       FAIL
D TP@.50 >= A             PASS
D TP@.75 >= A             FAIL
```

Sonuç:

```text
Mechanism gate: PASS
Official F1 gate: FAIL
```

## 7. TP/FP/FN bize ne söylüyor?

### D − A, IoU `.50`

```text
TP: +23
FP: −91
FN: −23
Prediction count: −68
```

Bu küçük `.50` artışı yalnız daha fazla lane yazmaktan gelmiyor. D biraz daha az
lane yazıp FP sayısını azaltmış. Fakat net TP kazancı yalnız `23`; bu, 9,675
görüntülük population için küçük bir farktır.

GT transition audit:

```text
Kurtarılan source FN: 662
Kaybedilen source TP: 639
Kurtarılan/kaybedilen: 1.036
Source TP retention: %97.29
```

Yani çok sayıda lane yer değiştirmiş, fakat kazanılan ve kaybedilen neredeyse
birbirini götürmüştür.

### D − A, IoU `.75`

```text
TP: −165
FP: +97
FN: +165
Prediction count: −68
```

GT transition audit:

```text
Kurtarılan source FN:  862
Kaybedilen source TP: 1,027
Kurtarılan/kaybedilen: 0.839
Source TP retention:  %91.54
```

Basit anlamı:

> Auxiliary objective bazı yeni strict lane'ler buluyor, fakat daha fazla mevcut
> strict doğru lane'i kaybediyor. Modelin hassas geometrisi stabil değil.

Geometri güvenlik ölçümleri de aynı yönü gösterdi:

```text
Crossing image:
A 635 → D 763

Duplicate-geometry image:
A 439 → D 559
```

Bu sayılar tek başına official hatanın nedeni değildir; fakat D'nin primary
geometri koordinasyonunu iyileştirmediğini destekler.

## 8. İstatistiksel belirsizlik

10,000 tekrarlı paired bootstrap çalıştırıldı. CULane frameleri clip içinde
korelasyonlu olduğu için ana yorum clip-level bootstrap'tır.

### F1@.50, D − A

```text
Observed:              +0.146 puan
Image-level %95 CI:    −0.071 .. +0.358 puan
Clip-level %95 CI:     −0.294 .. +0.644 puan
Clip pozitif olasılık: %71.72
```

`.50` yönü pozitif görünse de güvenilir bir kazanç değildir.

### F1@.75, D − A

```text
Observed:              −0.468 puan
Image-level %95 CI:    −0.727 .. −0.202 puan
Clip-level %95 CI:     −1.093 .. +0.101 puan
Clip pozitif olasılık: %5.65
```

Görüntü seviyesinde kayıp tamamen negatiftir. Clip korelasyonu aralığı
genişletir; yine de pozitif yön olasılığı yalnız `%5.65`tir.

Uniform256 ayrıca uyarıcı bir örnektir:

```text
Uniform256 @.50: −0.400 puan
Uniform256 @.75: +0.487 puan

Kalan 9,419 @.50: +0.161 puan
Kalan 9,419 @.75: −0.494 puan
```

Yani küçük subset'e bakılsaydı strict sonucun yönü yanlış okunabilirdi. Full
9,675 validation bu karar için zorunluydu.

## 9. Bu deney neyi kanıtladı?

### Kanıtlanan

1. Random auxiliary head'in cold-start gradyan problemi gerçektir.
2. Yalnız private head'i eğitmek shared ağı değiştirmeden gradyanları hizalar.
3. Private pretraining B'nin ilk auxiliary loss yükünü kaldırır.
4. Buna rağmen auxiliary supervision primary-only A'yı geçecek yararlı F1
   bilgisi üretmez.
5. Strict `.75` geometri açısından sonuç nettir: yardımcı objective zarar verir.

### Kanıtlanmayan

Bu deney şunları test etmedi:

```text
V7'nin 32 proposal bankasından winner seçme problemi
SPLIT-Lane / ayrı trainable Belief Tower
OOF proposal selection'ın full image encoder ile öğrenilebilirliği
Temporal/video evidence
```

V33-PH bir selection deneyi değildir. Direct-primary modelin shared encoder'ına
training-only auxiliary geometry vermeyi test eder.

Bu nedenle sonuç:

```text
“Bütün gradient-level fikirler yanlış.”
```

demek değildir.

Doğru sonuç:

```text
“Direct-primary modele aynı shared representation üzerinden auxiliary geometry
vermek, cold-start düzeltilse bile yararlı değildir.”
```

## 10. Başarısızlığın en olası nedenleri

### 1. En olası — Auxiliary objective primary için redundant

**Kanıt:** Gradyan cosine `+0.635`e çıktı ve ilk loss sorunu kalktı; buna rağmen
D, A'yı geçmedi.

Basit anlamı:

> Auxiliary kol artık yanlış talimat vermiyor olabilir, ama primary'nin kendi
> geometri loss'unda olmayan yeni bir bilgi de öğretmiyor.

### 2. İkinci olası — Coverage objective coarse geometriyi destekliyor

**Kanıt:** `.50` küçük pozitifken `.75` belirgin negatif; strict recovered/lost
oranı `0.839`.

Bu açıklama bir çıkarımdır. Auxiliary coverage loss'u “GT çevresinde bir aday
bulunsun” davranışını teşvik ederken primary dört lane'in milimetrik merkezini
korumaya yardım etmiyor olabilir.

### 3. Daha düşük olasılık — Endpoint çok kısa

Bu ihtimal teorik olarak vardır. Ancak daha uzun koşu açmak için kısa causal
gate'in önce pozitif olması gerekiyordu. D `.75`te hem A hem B'den geride ve
clip-level pozitif olasılık yalnız `%5.65`. Bu yüzden “belki uzun eğitimde
düzelir” gerekçesi maliyetli bir long run için yeterli değildir.

## 11. Ne yapmamalıyız?

```text
Private pretraining süresini sweep etme
Auxiliary loss ağırlığını sweep etme
V33-D'yi 50K/100K'ya uzatma
PCGrad ekleme
Auxiliary memory fusion'ı büyütme
Direct-primary query sayısını kör biçimde artırma
Test split'ini açma
```

Neden? Ölçülen cold-start problemi zaten giderildi. Kalan sonuç yeni bir
optimizasyon ayarı değil, objective'in primary için yetersiz/redundant olduğunu
gösteriyor.

## 12. Sonraki mantıklı deney

V33-PH başarısızlığı sonrası direct-primary auxiliary ailesi kapanmalıdır.

Hâlâ test edilmemiş olan tek gradient-ownership hipotezi şudur:

> V7 proposal bankasını bit-exact koruyup, proposal seçimi için kendi trainable
> görüntü encoder'ına sahip ayrı bir Belief Tower kurmak.

İlk deney dev SPLIT/TRIDENT modeli olmamalıdır. Minimum causal gate:

```text
Support/V7:
  tamamen frozen
  aynı 32 proposal
  aynı activity/count
  aynı refiner/writer

Belief:
  ayrı, trainable image encoder
  yalnız mevcut route target loss'u
  candidate geometry Belief'e detached girer

Kapalı parçalar:
  field loss
  support unfreeze
  precision tower
  discover queries
  new activity head
  QFL/ranking ek loss'ları
```

Bu deney eski OOF selector'ın tekrarı değildir. Eski deney yalnız frozen
representation üstüne yeni head koymuştu. Yeni gate'te selection loss ham
görüntüden başlayan ayrı encoder'ın tamamını eğitebilir; selection gradyanı V7
geometrisine geçemez.

Bu gate de iki clip-disjoint OOF yönde selection ve official F1 kazancı
üretemezse:

> Gradient-graph ile proposal-selection kurtarma ailesi kapatılmalı; doğru tail
> proposal'ın tek görüntüden genellenebilir biçimde ayırt edilemediği kabul
> edilmelidir.

## 13. Nihai karar

```text
Random-head cold-start teorisi:        DOĞRULANDI
Private-head firewall mekanizması:     ÇALIŞTI
Official F1 faydası:                   FAIL
Direct-primary auxiliary ailesi:       KAPALI
Uzun V33-PH koşusu:                    AÇILMAYACAK
Test split:                            KAPALI
Sıradaki tek araştırma gate'i:         Frozen-support, trainable-Belief S0
```

En kısa cümle:

> Head'i hazırlayarak yanlış başlangıç gradyanını düzelttik; fakat model daha iyi
> olmadı. Demek ki sorun yalnız cold-start değildi. Auxiliary geometri görevi
> primary'ye gereken yeni bilgiyi taşımıyor.

## 14. Kod, checkpoint ve ham çıktılar

Branch:

```text
codex/v33-gradient-interaction-20260824
```

Uygulama commit'i:

```text
17c5c31  Add V33 private auxiliary cold-start gate
```

Ana checkpoint SHA256:

```text
Private-head endpoint:
074b6249da6f72301bf8a40a399ab336208cacae0cc77fadf8e64f250628dc3e

D joint endpoint:
7d7ee7bfa8f5e944327c23cf3240e6ef7df429b28206b961fbae1d85b4a4d333
```

Ham çıktılar:

```text
outputs/diagnostics/v33_private_head_firewall_gate/
  v33_private_head_summary.json
  pretrain_gradient_audit_32pairs.json
  a_vs_d_paired_clip_bootstrap.json
  a_vs_d_gt_transitions.json
  private_proposal_memory_pretrain/
    private_head_endpoint.pt
    private_training_report.json
  arm_d_pretrained_head_joint/
    component_endpoint.pt
    training_report.json
  eval_a_vs_d/
    a_vs_d_official_val.json
  eval_b_vs_d/
    b_vs_d_official_val.json
```

Local checkpoint kopyalarının SHA256 değerleri remote kaynakla exact eşleşmiştir.

