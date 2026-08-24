# V33-GI Shared Gradient Interaction Audit — Nihai Rapor

Tarih: 24 Ağustos 2026

## 1. Kısa cevap

V33-B'nin başarısızlığını açıklayan gerçek bir eğitim problemi bulduk:

> Rastgele başlatılan auxiliary proposal head, eğitimin başında ortak görüntü
> ağını primary görevin yararına olmayan bir yöne itiyor.

Bu sonuç yalnız loss değerlerine bakılarak çıkarılmadı. Aynı batch üzerinde
primary ve auxiliary loss'ların ortak parametrelere gönderdiği gradyanlar ayrı
ayrı ölçüldü. Sonra yalnız bu gradyan yönlerinde küçük sanal adımlar atılıp bir
sonraki, görülmemiş batch'in primary loss'u tekrar hesaplandı.

Ancak ikinci sonuç da en az bunun kadar önemli:

> Auxiliary head öğrenildikten sonra bu çatışma kayboluyor. Gradyanlar güçlü
> biçimde aynı yöne bakıyor; auxiliary gradyan primary gradyanı büyüklük olarak
> da domine etmiyor.

Bu nedenle güncel hüküm şudur:

```text
Kalıcı primary–auxiliary gradient çatışması var.       HAYIR
Auxiliary loss primary gradyanı büyüklükle eziyor.     HAYIR
Rastgele auxiliary cold-start başlangıçta zararlı.     EVET
Yalnız auxiliary head'i eğitmek çatışmayı kaldırıyor.  EVET
Bu tek başına F1 kazancını kanıtlıyor.                  HAYIR
```

Mevcut V33-B hâlâ FAIL'dir. Audit, V33-B'nin F1 sonucunu değiştirmez. Yalnızca
başarısızlığın eğitim yolunda nerede başladığını çok daha net gösterir.

## 2. V33 neden bu audite ihtiyaç duydu?

V33 direct-primary deneyinde üç kol vardı:

| Kol | İçerik | F1@.50 | F1@.75 |
| --- | --- | ---: | ---: |
| A | Primary-only | 72.290 | 37.249 |
| B | Primary + training-only auxiliary proposal loss | 72.106 | 37.309 |
| C | Primary + auxiliary spatial memory | 71.870 | 37.453 |

B'nin A'ya farkı:

```text
F1@.50: −0.184
F1@.75: +0.059
```

Yani 32 auxiliary proposal'ın geometri loss'u ortak encoder'a ulaştı, fakat
primary final lane'leri anlamlı biçimde iyileşmedi.

Bu sonuç iki farklı açıklamayı ayırmıyordu:

1. Auxiliary gradyan primary gradyanla çatışıyor olabilir.
2. Auxiliary gradyan uyumlu olabilir ama primary'ye yeni bilgi katmıyor olabilir.

V33-GI bu iki açıklamayı doğrudan ayırmak için yapıldı.

## 3. Ne yaptık?

### 3.1 Üç durum karşılaştırıldı

#### Durum 1 — Parent + rastgele auxiliary head

```text
Primary/shared ağırlıklar:
V33 G0 parent, iter 11,110

Auxiliary proposal_memory:
yeni ve rastgele başlatılmış
```

Bu durum V33-B'nin cold-start anını temsil eder.

#### Durum 2 — Hibrit causal counterfactual

```text
Primary/shared ağırlıklar:
exact aynı G0 parent, iter 11,110

Auxiliary proposal_memory:
eğitilmiş B endpoint'inden alınan 27 tensor, iter 13,888
```

Burada yalnız private auxiliary head değiştirildi. Backbone, FPN, primary
decoder ve primary head parent ile aynı kaldı.

Kontroller:

```text
Transplant sonrası parent/shared tensor parity: true
Parent–hibrit primary loss farkı, 32/32 çiftte: 0.0
Train/probe batch metadata eşleşmesi: true
```

Bu nedenle parent ile hibrit arasındaki gradyan farkını ortak backbone'un daha
önce eğitilmiş olmasıyla açıklayamayız. Değişen tek esas parça private auxiliary
head'dir.

#### Durum 3 — Tam eğitilmiş V33-B endpoint

```text
Primary/shared ağırlıklar:
B endpoint, iter 13,888

Auxiliary proposal_memory:
B endpoint, iter 13,888
```

Bu durum ortak sistem kısa auxiliary eğitiminden sonra nereye geldi sorusunu
ölçer.

### 3.2 Aynı veri kullanıldı

Her checkpoint durumu için:

```text
32 exact train/probe batch çifti
256 toplam görüntü
aynı batch manifest SHA256:
1b9e08c18069ae495f2037455e3089164097b2d9972815af6c4ddc0e543adc34
```

Resmî CULane population kontratı korunmuştur:

```text
Train: 88,880 satır
Val:    9,675 satır
Filtreleme/deduplication: yok
Test split: kullanılmadı
```

### 3.3 Hangi gradyanlar ölçüldü?

Aynı train batch üzerinde:

```text
g_primary = gradient(L_primary, shared parameters)
g_aux     = gradient(L_auxiliary_proposal_coverage, shared parameters)
```

Ölçülen ana değerler:

- Gradient cosine: İki gradyan aynı yöne mi, ters yöne mi bakıyor?
- Auxiliary/primary norm oranı: Auxiliary gradyan büyüklük olarak baskın mı?
- Negatif batch oranı: Kaç batch'te iki gradyanın yönü ters?
- Layer grupları: Sorun backbone, FPN veya yüksek çözünürlük yolunun neresinde?
- Cross-batch virtual step: Yalnız bir gradyan yönünde küçük adım atınca sonraki
  batch'in primary loss'u iyileşiyor mu, kötüleşiyor mu?

Cosine'ın basit anlamı:

```text
+1'e yakın: iki görev aynı parametre değişimini istiyor.
 0'a yakın: görevler büyük ölçüde farklı yönlere bakıyor.
−1'e yakın: görevler birbirine ters talimat veriyor.
```

### 3.4 Virtual-step testi

Train batch'te bulunan gradyan yönü, parametre normuna göre normalize edilerek
üç küçük relative scale ile uygulandı:

```text
1e-5, 3e-5, 1e-4
```

Sonra aynı train batch değil, bir sonraki probe batch'in primary loss'u ölçüldü.
Ana rapor değeri en büyük fakat hâlâ küçük olan `1e-4` ölçümüdür.

Bu test gerçek Adam optimizer adımı değildir. Yerel, normalize edilmiş bir SGD
yön testidir. Bu yüzden mekanizma kanıtı olarak değerlidir, fakat doğrudan F1
sonucu değildir.

## 4. Ana sonuçlar

### 4.1 Birleşik tablo

| Durum | Shared cosine | Negatif batch | Aux/primary norm | Aux yönü primary'yi kötüleştirdi | Virtual primary-loss medyan değişimi |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent + random aux | `+0.027` | `%37.5` | `0.083×` | **`%71.9`** | **`+0.000878`** |
| Parent + trained private aux | **`+0.656`** | **`%3.1`** | `0.165×` | `%56.3` | `+0.000232` |
| Tam eğitilmiş B | **`+0.742`** | **`%0.0`** | `0.174×` | `%43.8` | `−0.000673` |

Pozitif virtual-loss değişimi zarardır; negatif değişim iyileşmedir.

### 4.2 Paired-bootstrap belirsizliği

32 batch çifti üzerinde 20,000 tekrar örneklemeli paired bootstrap kullanıldı.
Audit scriptinin kullandığı lower-median kontratı korunmuştur.

#### Shared gradient cosine

```text
Parent:
  merkez  +0.027
  %95 CI  −0.019 .. +0.053

Hibrit:
  merkez  +0.656
  %95 CI  +0.607 .. +0.734

Trained:
  merkez  +0.742
  %95 CI  +0.562 .. +0.780
```

Hibrit ve trained durumlarda güven aralığı tamamen pozitiftir.

#### Auxiliary virtual step'in primary loss'a etkisi, scale `1e-4`

```text
Parent:
  merkez  +0.000878
  %95 CI  +0.000131 .. +0.001313
  → güvenilir biçimde zararlı

Hibrit:
  merkez  +0.000232
  %95 CI  −0.002107 .. +0.002164
  → yön belirsiz; güvenilir zarar yok

Trained:
  merkez  −0.000673
  %95 CI  −0.002293 .. +0.001500
  → yön belirsiz; güvenilir zarar yok
```

En temiz değişim şudur:

> Rastgele auxiliary head'te sonraki batch'e taşınan güvenilir zarar vardır.
> Yalnız private auxiliary head eğitildiğinde bu güvenilir zarar kaybolur.

## 5. Çatışma hangi katmanlardaydı?

### 5.1 Rastgele auxiliary başlangıcı

| Parametre grubu | Medyan cosine | Negatif batch | Aux/primary norm |
| --- | ---: | ---: | ---: |
| Bütün shared | `+0.027` | `%37.5` | `0.083×` |
| Backbone | `+0.063` | `%37.5` | `0.044×` |
| FPN | `−0.013` | `%53.1` | `0.124×` |
| Fine stem | `−0.078` | `%56.3` | `0.237×` |
| P2 projection | `−0.020` | `%59.4` | `0.192×` |
| Image fusion | `−0.006` | `%53.1` | `0.569×` |

Global cosine çok negatif değildir. Sorun daha yereldir:

> Rastgele auxiliary head özellikle FPN, fine-resolution stem, P2 projection ve
> image-fusion yolunda primary'nin istediği feature değişimiyle ters düşüyor.

Bu, “auxiliary gradient her yerde devasa ve primary'yi eziyor” hikâyesini
desteklemez. Norm oranı bütün shared parametrelerde yalnız `0.083×` idi.

### 5.2 Yalnız private auxiliary head eğitilince

| Parametre grubu | Medyan cosine | Negatif batch | Aux/primary norm |
| --- | ---: | ---: | ---: |
| Bütün shared | `+0.656` | `%3.1` | `0.165×` |
| Backbone | `+0.676` | `%3.1` | `0.160×` |
| FPN | `+0.457` | `%3.1` | `0.193×` |
| Fine stem | `+0.458` | `%6.3` | `0.196×` |
| P2 projection | `+0.375` | `%3.1` | `0.205×` |
| Image fusion | `+0.241` | `%6.3` | `0.257×` |

Ortak backbone hiç eğitilmediği hâlde bütün kritik gruplar pozitif yöne geçti.

Bu sonuç çok önemlidir:

> Alignment için ortak representation'ın önce auxiliary loss'la değişmesi şart
> değildir. Auxiliary head'in doğru geometriyi okumayı öğrenmesi tek başına
> gradyan yönünü büyük ölçüde düzeltmeye yetmiştir.

### 5.3 Tam eğitilmiş B endpoint

Tam eğitilmiş sistemde:

```text
Shared cosine:               +0.742
Negatif shared batch oranı:  %0
Aux/primary norm oranı:      0.174×
```

Yani V33-B endpoint'inde persistent gradient conflict yoktur.

## 6. Bu sonuç ne anlama geliyor?

### Kanıtlanan bölüm

1. V33-B'nin ilk auxiliary adımları primary için zararlı bir cold-start sinyali
   üretiyor.
2. Bu zarar auxiliary gradyanın çok büyük olmasından gelmiyor.
3. Yalnız private proposal-memory head'ini eğitmek, ortak ağırlıkları değiştirmeden
   gradyanları güçlü biçimde hizalamaya yetiyor.
4. Tam eğitilmiş endpoint'te primary ve auxiliary gradyanlar artık çatışmıyor.

### Henüz kanıtlanmayan bölüm

Şu zincir hâlâ bir hipotezdir:

```text
Rastgele auxiliary cold-start
→ shared representation erken zarar görüyor
→ model daha sonra hizalansa bile bu hasarı telafi edemiyor
→ V33-B F1 kazanmıyor
```

Audit bu açıklamayı güçlü biçimde mümkün kıldı, fakat henüz F1 ile doğrulamadı.

Ayrıca trained endpoint'te gradyanların hizalı olması, auxiliary loss'un yararlı
yeni bilgi kattığını göstermez. Aynı geometri hedefini primary loss'la tekrar
ediyor ve artık büyük ölçüde redundant kalıyor olabilir.

## 7. En olası başarısızlık nedenleri

### 1. En olası — Auxiliary cold-start trajectory'yi bozuyor

**Kanıt:** Parent'te auxiliary virtual step batch'lerin `%71.9`unda primary
loss'u artırıyor ve bootstrap aralığı tamamen pozitif. Eğitilmiş private head
transplant edilince bu güvenilir zarar kayboluyor.

### 2. İkinci olası — Alignment sonrası auxiliary objective redundant

**Kanıt:** Tam eğitilmiş B'de cosine güçlü pozitif olmasına rağmen B, A'yı F1'de
geçmedi.

Basit anlamı:

> Auxiliary artık yanlış yöne itmiyor olabilir; fakat primary'nin bilmediği yeni
> bir şey de öğretmiyor olabilir.

### 3. Daha az olası — Kısa endpoint yararı göstermedi

V33 yalnız kısa bir sufficiency gate idi. Fakat kısa endpoint gerekçesiyle kör
biçimde uzun training açmak doğru değildir. Önce cold-start zararını kaldıran
exact causal gate pozitif F1 üretmelidir.

## 8. Neyi yapmamalıyız?

Bu audit aşağıdakileri desteklemiyor:

```text
PCGrad eklemek
auxiliary loss ağırlığı sweep etmek
kalıcı layer detach sınırı seçmek
mevcut V33-B'yi uzun eğitime sürmek
mevcut C random spatial fusion'u büyütmek
test split'ini açmak
```

Neden?

- Sorun persistent bir gradyan çatışması değil.
- Sorun auxiliary norm dominance değil.
- Eğitilmiş head zaten gradyanı hizalıyor.
- Dolayısıyla genel gradient surgery gerçek ölçülen probleme fazla geniş saldırır.

## 9. Sonraki tek mantıklı deney

# V33-PH — Private-Head Cold-Start Firewall Gate

### Bu parça ne yapacak?

Auxiliary proposal head önce ortak görüntü ağına dokunmadan geometri okumayı
öğrenecek. Head yeterince eğitildikten sonra auxiliary gradyanın shared encoder'a
ulaşmasına izin verilecek.

Basit şema:

```text
Faz 0 — Private head pretraining

shared backbone/FPN: frozen
primary model:       frozen
proposal_memory:     trainable
loss:                yalnız auxiliary coverage

                  ↓

Faz 1 — Exact paired joint continuation

shared backbone/FPN: trainable
primary model:       trainable
proposal_memory:     trainable
loss:                primary + auxiliary
```

### Modele neden lazım?

Rastgele head'in ortak encoder'a zararlı gradyan göndermesini engeller. Auxiliary
head ancak measured-aligned hâle geldikten sonra shared representation'ı
etkileyebilir.

### Nasıl adil test edilir?

Aynı G0 parent ve aynı main-training manifestiyle:

| Kol | Başlangıç | Main continuation |
| --- | --- | --- |
| A | Exact parent | Primary-only |
| B | Exact parent + random aux | Primary + auxiliary |
| D | Exact parent + privately pretrained aux | Primary + auxiliary |

Private pretraining için fixed `2,778` step kullanılmalıdır. Bu sayı, audit'te
alignment sağlayan donor head'in gördüğü training uzunluğuyla eşleşir. Bu bir
loss-weight veya checkpoint sweep'i değildir.

Main continuation üç kolda da:

```text
aynı görüntü sırası
aynı augmentation
aynı shared-update sayısı
aynı LR/optimizer schedule
aynı tek endpoint
```

olmalıdır.

### Önceden kilitli PASS koşulu

D kolu için:

```text
D − A F1@.50 >= +0.30
D − A F1@.75 >=  0.00

D − B F1@.50 >= +0.20
D − B F1@.75 >=  0.00

proposal support non-regression
primary TP retention >= %99
test split kapalı
```

Ek mekanizma şartı:

```text
Main joint training'in ilk %10'unda
shared cosine > +0.10
ve auxiliary virtual-harm fraction < %60
```

### STOP koşulu

D yalnız B'nin cold-start zararını silip A'yı geçemezse:

> Auxiliary objective zararlı değil ama redundant kabul edilir ve direct-primary
> auxiliary ailesi kapatılır.

D, A'yı geçerse ancak o zaman daha uzun matched endpoint düşünülür.

## 10. SPLIT/Belief yönü hakkında etkisi

Bu audit SPLIT-Lane'i doğrulamaz veya reddetmez. V33'ün sorusu selection değil,
direct-primary geometry için training-only auxiliary supervision idi.

Ancak araştırma sırası açısından şu değişti:

```text
Önce: V33 başarısız → auxiliary muhtemelen redundant/çatışmalı.

Şimdi: Rastgele private head'in ölçülen cold-start zararı var;
       bunu kaldıran tek ucuz causal gate henüz yapılmadı.
```

Bu nedenle büyük iki-tower SPLIT sistemi veya temporal model açmadan önce yalnız
V33-PH gate'i haklıdır. V33-PH de A'yı geçmezse auxiliary-gradient yolu kapanır;
sonra yeni bilgi kaynağı veya ayrı belief representation tartışılır.

## 11. Nihai karar

V33-GI'nin en dürüst sonucu:

> Auxiliary geometri görevi primary ile doğası gereği sürekli çatışmıyor.
> Problem, rastgele auxiliary okuyucunun eğitimin başında ortak görüntü ağına
> yanlış talimat vermesi. Auxiliary okuyucu öğrendiğinde gradyanlar hizalanıyor;
> fakat hizalanmış olmak tek başına F1 faydası demek değil.

Bu yüzden:

```text
Mevcut V33-B:                     FAIL ve kapalı
Persistent-conflict teorisi:      reddedildi
Norm-dominance teorisi:           reddedildi
Private-head cold-start teorisi:  güçlü biçimde desteklendi
F1 çözümü:                        henüz kanıtlanmadı
Sıradaki deney:                   yalnız V33-PH causal gate
```

## 12. Kod ve provenance

Kod branch'i:

```text
codex/v33-gradient-interaction-20260824
```

Commit'ler:

```text
3a4bbd9  Add V33 primary auxiliary gradient interaction audit
c1d38cb  Add reproducible V33 gradient audit runner
131e0ba  Add V33 auxiliary cold-start counterfactual
```

Test:

```text
dynlaneseq_eg/tests/test_v33_gradient_interaction.py
3 passed
```

Ana ham çıktı:

```text
outputs/diagnostics/v33_primary_aux_sufficiency_gate/
  gradient_interaction/
    v33_gradient_interaction_32pairs_3way.json
```

Ham çıktı SHA256:

```text
f5800da48fa06c0d90b478ee85a1ea7417b77ad58db7002b20bdcbf3cf35c660
```

Checkpoint SHA256:

```text
G0 parent:
5d327f73c7920c3f3d5d32bea86850a33bda96938c2864fea3fd9d2c834d8935

V33-B endpoint / private-head donor:
30b67e9a067260c262b2d0d3c09dd6ddacd8af0535bfad90c56cac44a83fa36c
```

Audit'in otomatik birleşik kararı:

```text
private_auxiliary_cold_start_conflict_supported
```

