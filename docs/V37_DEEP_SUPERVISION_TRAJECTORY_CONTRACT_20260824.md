# V37 Deep-Supervision Assignment Trajectory Audit

## Amaç

V36 mature final output'ta şunu gösterdi:

```text
route target good'u tercih ediyor:       %99.90
proposal matcher good'u tercih ediyor:   %97.35
good final proposal positive:             %91.43
```

Buna rağmen proposal existence ve final route confidence wrong proposal'ı
tercih ediyor.

V37 kalan tek assignment-contract ihtimalini test eder:

> Final-good query son decoder layerında positive olsa bile, aynı query ilk üç
> decoder layerında negative veya başka-GT target alıyor mu?

## Sabit protokol

- mature V7 `iter_0225000.pt`;
- V36'daki exact 625 görüntü / 980 pair;
- V34 cache'in üretildiği exact `temporal_union_val.txt` sırası ve batch
  kompozisyonu;
- dört decoder output: üç intermediate + final;
- her layer kendi configured V7 Hungarian assignment'ını kullanır;
- AMP: BF16, model eval, augmentation yok;
- final proposal output V34 cache ile cross-device BF16 replay stability
  kontrolünden geçer;
- test split kapalı;
- training veya checkpoint seçimi yok.

## Ölçümler

Her final-good/current-wrong çiftinde:

- good/wrong query'nin layer başına intended/any-positive durumu;
- intended GT'ye layer başına atanan query ID;
- final-good query'nin ilk kez hangi layerda positive olduğu;
- dört layer boyunca assignment identity switch sayısı;
- good/wrong target-quality ve existence probability trajectory'si;
- exact intermediate loss katsayılarıyla candidate-level existence logit update;
- final ve aggregate intermediate existence loss'unun shared `exist` head
  parametrelerindeki gradient cosine ve norm oranı.

Exact layer katsayıları:

```text
L1 = 0.5 * 1/6 = 0.08333
L2 = 0.5 * 2/6 = 0.16667
L3 = 0.5 * 3/6 = 0.25000
L4 = 1.00000
```

No-lane CE class weight `0.10` aynen korunur.

Gradient ölçümü iki folddan deterministik toplam 128 görüntüde yapılır. Bu
ölçüm yalnız shared proposal existence head içindir; bütün backbone gradienti
olduğu şeklinde yorumlanmaz.

V34 cache önceki GPU üzerinde üretilmiş olabileceği için tek bir autoregressive
row'un maksimum piksel farkını bit-exact gate olarak kullanmıyoruz. Replay şu
üç koşulu birlikte sağlamalıdır:

```text
all-proposal mean absolute coordinate difference <= 1.0 px
V36 pair assignment-state agreement              >= %95
good/wrong target-quality mean absolute error     <= 0.03
```

İlk denemede farklı batch kompozisyonuyla `max abs = 100.845 px` görüldüğü için
koşu sonuç üretmeden durduruldu. Yukarıdaki cross-device sözleşme ve exact V34
batch sırası bu başarısız replay sonrasında, bilimsel metriğe bakılmadan önce
kilitlendi.

## Önceden kilitli karar ağacı

### `DEEP_SUPERVISION_SCORE_CONFLICT`

Şunlardan biri güçlü biçimde gerçekleşirse:

```text
final-good-positive çiftlerin >= %25'inde
exact weighted total existence update good'u aşağı itiyor
```

veya:

```text
final vs intermediate exist-head gradient:
negative fraction >= %50
ve median intermediate/final norm ratio >= 0.25
```

Deep supervision scoring contract ana adaydır. Sonraki causal run, geometry
deep supervision'ı koruyup intermediate existence loss'unu kapatır veya final
assignment identity'sini reuse eder.

### `ASSIGNMENT_CHURN_WITH_ALIGNED_SCORE_GRADIENT`

Şunların ikisi birlikte gerçekleşirse:

```text
final-good query'nin all-layer intended-positive oranı < %65
```

ama:

```text
weighted total update good'u aşağı itme oranı < %10
ve gradient negative fraction < %25
```

Assignment kimliği değişmektedir fakat shared score head'i net olarak yanlış
yöne itmiyordur. Churn gözlemi tek başına yeni run açmak için yeterli değildir.

### `DEEP_SUPERVISION_NOT_PRIMARY`

Şunların tamamı gerçekleşirse:

```text
weighted total update good'u yukarı itiyor >= %90
weighted total update wrong'u aşağı itiyor >= %90
gradient negative fraction < %25
```

Deep supervision ana açıklama değildir. Matcher/assignment rescue ailesi
kapatılır.

Diğer sonuçlar `MIXED_DEEP_SUPERVISION_EFFECT` olarak raporlanır.

## Sonuç sınırı

Bu audit shared `exist` head ve assignment trajectory'sini ölçer. Four-slot
route representation'ın tek başına neden generalize olmadığını doğrudan çözmez.
Fakat proposal score'un doğru positive label'a rağmen ters kalmasının son açık
assignment-temelli açıklamasını test eder.
