# V38 Direct-Primary Maturity Gate — Önceden Kilitli Kontrat

Tarih: 2026-08-24
Test split: kapalı

## Kısa cevap

V36 ve V37, V7'nin wrong-belief hatasını matcher, target veya deep-supervision işaret hatasıyla açıklayamadı. V28/V29 ise frozen V7 support bankası yanında tamamen ayrı ve trainable bir belief image tower'ın OOF'ta doğru proposal seçimini genelleştiremediğini gösterdi.

Bu nedenle proposal-selection rescue ailesi tekrar açılmayacaktır.

V38 başka bir soru sorar:

> Dört lane'i 32 proposal bankasından seçmeden doğrudan üreten one-to-one primary detector, V7 ile aynı 50K optimizer adımı ve aynı effective batch altında yeterli kapasiteye ulaşabiliyor mu?

V33'te bu direct-primary çekirdek yalnız 13,888 adıma, yani yaklaşık 1.25 train-list epoch'a kadar ölçüldü. V33'ün bilimsel sorusu primary modele training-only auxiliary proposal supervision eklemenin faydasıydı. Auxiliary kol başarısız oldu; direct-primary çekirdeğin olgun eğitim kapasitesi test edilmedi.

## 1. Ne değişiyor?

V38 modeli:

```text
Görüntü
  ↓
DLA-34 + yüksek çözünürlüklü FPN
  ↓
4 persistent lane object
  ↓
Her lane için 160 row dağılımı, existence ve range
  ↓
Row expectation
  ↓
En fazla 4 final lane
```

Modelde şunlar yoktur:

```text
32 proposal bankası
proposal selector/reranker
field loss
selection bridge
auxiliary proposal branch
proposal memory
MMR/NMS
V7 writer veya V7 geometry bağımlılığı
```

Bu nedenle V38, eski proposal'ları daha iyi seçmeye çalışan başka bir head değildir. Yeni detector/generator yönünün en sade testidir.

## 2. Neden bu deney gerekli?

Bilinen sonuç:

```text
V33 direct-primary, 13.9K:
72.290 F1@.50
37.249 F1@.75
```

Bu değer V7'nin altındadır. Ancak 13.9K endpoint yalnız kısa component gate'tir. V25'in orijinal planı da bir epoch sonucunun final kapasiteyi kanıtlamayacağını açıkça belirtmiştir.

V38'in amacı “belki daha uzun eğitirsek düzelir” şeklinde sınırsız umut üretmek değildir. Tek bir eşit-horizon gate ile bu açıklığı kapatmaktır.

## 3. Eğitim kontratı

| Özellik | V38 |
| --- | ---: |
| Başlangıç | ImageNet DLA-34, iterasyon 0 |
| Seed | `3407` |
| Official train list | `88,880` satır, filtre yok |
| Endpoint | `50,000` optimizer step |
| Effective batch | `16` |
| Toplam görüntü maruziyeti | `800,000` |
| Yaklaşık train-list epoch | `9.00` |
| LR schedule horizon | `278,000` step |
| Warmup | `1,000` step |
| Checkpoint seçimi | yok |
| Threshold seçimi | yok |
| NMS | kapalı |
| Test | kapalı |

V7 exact-50K referansı da effective batch `16`, global step `50,000` ve `278,000` step cosine schedule kullanır. Mimariler farklı olduğu için augmentation akışı bit-exact paired değildir; V38 “exact paired training” iddiası taşımaz. Adım, veri maruziyeti, seed ve schedule horizon eşleştirilmiştir.

V38 primary objective, V33 Arm-A ile aynıdır:

```text
existence
row distribution
point geometry
strip IoU
range
smoothness/order/duplicate safety
```

Quality50/75, competition, slot interaction ve auxiliary proposal loss kapalıdır.

## 4. Değerlendirme kontratı

Population:

```text
Official validation: 9,675 görüntü
Subset yok
Deduplication yok
Test yok
```

Inference:

```text
row expectation
existence threshold = 0.5
top_k = 4
NMS = kapalı
float32 inference
```

Decode politikası V38 başlamadan önce kilitlendi. Exact G0 1-epoch checkpoint'i
üzerindeki full-validation replay'de aynı logits için:

```text
Expectation: 72.831 @.50 / 40.506 @.75
Hard path:   72.611 @.50 / 37.687 @.75
```

Hard path, continuity ölçümlerini iyileştirse de önceden tanımlı G1 F1 gate'ini
geçmedi. Bu nedenle V38 endpoint'inde decode seçimi yapılmayacak; expectation
baştan sabit deploy politikasıdır.

Reference:

```text
Exact V7 50K:
F1@.50 = 80.06775
F1@.75 = 57.76888
```

## 5. Önceden kilitli karar

### Strong win

```text
V38 − V7 @.50 >= +0.30 puan
V38 − V7 @.75 >=  0.00 puan
```

Karar: ikinci seed ve mature endpoint hak eder.

### Maturity pass

```text
V38 − V7 @.50 >= −0.50 puan
V38 − V7 @.75 >= −0.50 puan
```

Karar: direct-primary 50K'da V7'ye yeterince yaklaşmıştır; yalnız bir adet önceden kilitli uzun endpoint hak eder.

### Fail

İki threshold'dan herhangi biri V7'nin `0.50` puandan fazla gerisindeyse:

```text
DIRECT_PRIMARY_50K_FAIL
```

Karar:

> Mevcut dört-object direct-primary detector kapatılır. Query sayısı, loss ağırlığı veya endpoint sweep'i yapılmaz.

## 6. Bu deney neyi kanıtlamaz?

- 50K'da V7'ye yaklaşmak eventual 81+ garantisi değildir.
- V38'in fail olması bütün direct detector tasarımlarını reddetmez; yalnız mevcut V25/V33 dört-object formunu reddeder.
- V38 proposal lucky-tail hipotezini doğrudan ölçmez; proposal-selection paradigmasının dışına çıkar.
- V38 test seti hakkında sonuç üretmez.

## 7. Sonraki karar ağacı

```text
Strong win
  → ikinci seed
  → matched mature endpoint

Maturity pass
  → tek uzun endpoint
  → gain yoksa kapat

Fail
  → V25/V33 direct-primary formunu kapat
  → yeni detector için daha temel temsil/decoder tasarımı
```
