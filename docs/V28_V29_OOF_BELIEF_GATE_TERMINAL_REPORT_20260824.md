# V28/V29 Frozen-Support Belief Gate Terminal Report

**Tarih:** 24 Ağustos 2026  
**Durum:** OOF A→B gate tamamlandı ve FAIL; ters yön önceden tanımlanan stop kuralı nedeniyle çalıştırılmadı  
**Test split:** Kullanılmadı

## 1. Kısa cevap

V28/V29, daha sonra SPLIT-Lane adıyla tekrar önerilen temel mekanizmayı büyük
ölçüde zaten sınadı:

```text
Frozen V7 support beyni
        ↓
bit-exact 4×32 final-ready candidate bankası
        ↓
ayrı ve tamamen trainable DLA-34 + FPN seçim beyni
        ↓
row-ordered image reasoning
        ↓
hard unique proposal seçimi
```

Bu deney yalnız küçük bir MLP/head deneyi değildi. Selection loss, ayrı seçim
ağının DLA-34 backbone'una kadar ulaştı. Buna rağmen OOF değerlendirmede yeni
seçim sistemi frozen kaynak V7'yi geçemedi ve mevcut doğru V7 lane'lerini fazla
sık bozdu.

Net karar:

> Aynı frozen-support + ayrı trainable belief-tower çekirdeğini yeni bir adla
> tekrar çalıştırmamalıyız.

## 2. Ne yaptık?

### 2.1 Frozen support tarafı

V7 şu parçaların tek sahibi olarak kaldı:

- 32 proposal üretimi,
- dört final slot ve activity/count,
- slot başına counterfactual refinement,
- visible range,
- writer geometrisi.

Support modelinin bütün parametreleri frozen kaldı. Candidate koordinatları,
range ve validity tensorları seçim ağına detached olarak verildi. Bu nedenle
belief loss proposal geometrisini değiştiremedi.

### 2.2 Trainable belief tarafı

Yeni seçim ağı şunları içeriyordu:

- V7 ağırlıklarıyla warm-start edilen ayrı `DLA34Backbone`,
- ayrı `SimpleFPN`,
- slot ve row embedding'leri,
- tam görüntü üzerinde row-wise spatial belief field,
- iki katmanlı vertical Transformer,
- slotlar arası attention,
- her final-ready candidate'ın 160 row boyunca field üzerinde skorlanması,
- exact injective dört-slot route decode.

Optimizer, belief backbone'u ve geri kalan router parametrelerini ayrı öğrenme
oranlarıyla birlikte eğitti. Gate-zero kontrolü selection gradientinin belief
backbone, FPN, field key ve vertical encoder'a ulaştığını; frozen V7 tarafında
ise hiçbir gradient veya state değişimi olmadığını doğruladı.

Bu nedenle deneyin sorduğu gerçek soru şuydu:

> Proposal bankası hiç değişmeden kalırken, selection target tarafından baştan
> sona eğitilen ayrı bir görüntü temsili doğru proposal'ı yeni görüntülerde
> seçebilir mi?

### 2.3 Arm B ve Arm C

```text
Arm B:
ayrı trainable image-belief tower + candidate-level route loss

Arm C:
Arm B ile aynı graph + her slot/row için dense spatial field loss
```

Arm C'nin ek loss'u foreground/background segmentasyonu değildi. Her lane'in
her row'da hangi yatay konumda bulunması gerektiğini öğreten DFL-benzeri row
position supervision'ıydı.

## 3. V28 full-train sonucu

Her iki belief kolu 6.000 adım, yani 48.000 görüntü gördü. Frozen V7 proposal
bankası, count ve writer iki kolda aynıydı.

| Model | F1@.50 | F1@.75 |
| --- | ---: | ---: |
| Frozen V7 | **81.385** | **60.462** |
| Arm B | 79.685 | 58.932 |
| Arm C | 80.439 | 59.689 |
| Arm C, wrong image | 54.945 | 21.013 |

Arm C, Arm B'yi geçti:

```text
+0.754 F1@.50
+0.757 F1@.75
```

Fakat Arm C hâlâ V7'den gerideydi:

```text
-0.946 F1@.50
-0.773 F1@.75
```

Arm C yeni TP'ler buldu ama daha fazla mevcut V7 TP'sini kaybetti:

```text
@.50: 403 yeni TP, 708 source TP kaybı
@.75: 1.072 yeni TP, 1.321 source TP kaybı
```

Source TP kayıp oranları:

```text
@.50: %2.70
@.75: %6.78
```

Önceden belirlenen güvenlik sınırı `%1` idi.

Candidate-score marjının faydalı ve zararlı switch'i ayırma AUC'si de rastgele
seviyedeydi:

```text
genel kalite farkı: 0.496
@.50 geçişi:       0.529
@.75 geçişi:       0.477
```

Wrong-image sonucu, ağın görüntüyü gerçekten kullandığını kanıtladı. Ancak
görüntüyü kullanmak, güvenli winner discrimination için yeterli olmadı.

## 4. V29 OOF deneyi

### 4.1 OOF neden gerekliydi?

V28'de full V7 ile belief ağı aynı official train popülasyonuyla ilişkiliydi.
Belief ağı, V7'nin training görüntülerine özgü proposal hata desenlerini
ezberlemiş olabilirdi.

Bu yüzden train clip'leri ikiye ayrıldı:

```text
Support-A yalnız Fold-A üzerinde eğitildi.
Support-A, hiç görmediği Fold-B üzerinde proposal üretti.
Belief B/C, bu gerçek unseen support dağılımında eğitildi ve değerlendirildi.
```

İlk A→B yönü başarısız olursa ikinci support modelini eğitmemek önceden
tanımlanan stop kuralıydı.

### 4.2 OOF support yeterli miydi?

1.024 unseen Fold-B görüntüsünde Support-A 30K:

```text
All-32 oracle recall @.50: %94.85
All-32 oracle recall @.75: %77.27
Deployed top-4 recall @.50: %73.26
Kurtarılabilir @.50 boşluğu: %21.59
```

Dolayısıyla `.50` sonucu başarısız olursa “bankada doğru proposal yoktu”
açıklaması güçlü değildi. Banka mekanizma testi için yeterliydi.

### 4.3 OOF ana sonuç

| Model | F1@.50 | F1@.75 |
| --- | ---: | ---: |
| Arm B | 75.467 | 52.474 |
| Arm C | 77.117 | 53.734 |
| Frozen source V7-30K | **77.816** | **55.031** |

Field supervision OOF koşulunda da Arm B'yi iyileştirdi:

```text
Arm C - Arm B:
+1.651 F1@.50
+1.260 F1@.75
```

Ancak Arm C kaynak V7'yi geçemedi:

```text
Arm C - source V7:
-0.699 F1@.50
-1.297 F1@.75
```

Arm C lane kararlarının yaklaşık `%58`ini değiştirdi. Bunun sonucu source TP
kayıpları çok yüksekti:

```text
@.50 source TP loss: %4.97
@.75 source TP loss: %11.50
```

Target-best proposal top-1 oranı da yükselmedi:

```text
Arm B: %28.09
Arm C: %27.74
```

Switch-confidence AUC:

```text
@.50: 0.513
@.75: 0.505
```

Zararlı switch oranını post-hoc olarak `%1` altında tuttuğumuz noktada net
kazanım pratik olarak sıfırdı:

```text
@.50: 173 kazanım, 170 kayıp, net +3
@.75: 20 kazanım, 14 kayıp, net +6
```

Bu eşikler validation üzerinde sonradan bulunduğu için deploy sonucu olarak da
kullanılamaz.

## 5. Bu sonuç ne anlama geliyor?

### Kanıtlanan

1. Dense row-position supervision, route-only ayrı seçim ağına gerçek katkı
   veriyor. Etki OOF ortamında da tekrarlandı.
2. Ayrı seçim ağı görüntüyü gerçekten kullanıyor; wrong-image kontrolü bunu
   açık biçimde gösteriyor.
3. Selection loss, yalnız son head'i değil ayrı DLA/FPN görüntü temsilini
   eğitti.
4. Buna rağmen sistem doğru candidate değişikliklerini zararlı değişikliklerden
   ayıramadı ve frozen V7'yi geçemedi.

### Kanıtlanmayan

1. “Bütün sorun gradient kopukluğudur.”
2. “İki ayrı tower kurmak tek başına sorunu çözer.”
3. “Daha uzun aynı belief eğitimi V7'yi geçer.”
4. “Joint/alternating iki canlı tower kesin başarısız olur.”

Son madde özel olarak test edilmedi. Ancak frozen support bankası zaten çok
yüksek oracle kapasitesine sahipken seçim ağı güvenli winner bulamadı. Support'u
aynı anda yeniden eğitmek bu bilgi eksikliğini otomatik çözmez; ayrıca yeni
candidate-distribution drift'i ve support kaybı riski ekler.

## 6. Daha sonraki SPLIT önerisiyle karşılaştırma

| SPLIT bileşeni | V28/V29'da durum |
| --- | --- |
| Frozen V7 support bankası | Test edildi |
| 4×32 final-ready refined candidate | Test edildi |
| Ayrı trainable DLA-34/FPN belief tower | Test edildi |
| Selection gradientinin belief backbone'a ulaşması | Test edildi |
| Selection gradientinin support geometrisine ulaşmaması | Test edildi |
| Row sırasını koruyan sequence encoder | Test edildi |
| Structured exact unique dört-slot seçim | Test edildi |
| Correct-image / wrong-image nedensellik kontrolü | Test edildi |
| Clip-disjoint OOF support | Test edildi, A→B yönünde FAIL |
| Geniş candidate ribbon sampling | Birebir test edilmedi |
| PROTECT/CORRECT/DISCOVER triage | Test edilmedi |
| Row-reversal ve ordinal quality loss | Test edilmedi |
| İki live tower'ın alternating joint training'i | Test edilmedi |

Dolayısıyla tam SPLIT metninde yeni ayrıntılar vardır. Fakat temel bilimsel
hipotezi — ayrı ve trainable belief görüntü temsilinin frozen güçlü bankada
winner discrimination yapabilmesi — V29 zaten sınamış ve geçememiştir.

Ribbon, triage veya farklı ranking loss'u eklemek yeni bir deney olur; fakat
bu artık “ilk kez gradient ownership test ediyoruz” diye sunulamaz. Geçmiş
pair-corridor ve quality-ranking başarısızlıkları nedeniyle bu eklerin başarı
olasılığı da düşük kabul edilmelidir.

## 7. Terminal karar

```text
V28/V29 frozen-support belief router       KAPALI
Aynı SPLIT sufficiency gate'ini tekrar et  HAYIR
İkinci OOF support yönünü bu model için aç  HAYIR
Field'in route-only'ye katkısı              GERÇEK AMA YETERSİZ
Gradient problemi                           KISMİ; TEK KÖK NEDEN DEĞİL
```

Bir sonraki pahalı deney, yalnız aşağıdakilerden birini sağlamalıdır:

1. Candidate seçimine gerçekten yeni gözlemlenebilir bilgi eklemek; veya
2. 32'lik bankadan winner seçme probleminden çıkıp yeni bir detector ailesini
   test etmek.

Sadece daha büyük belief tower, başka bir sequence head veya yeni isimli aynı
frozen-bank router bilimsel olarak tekrar olacaktır.

## 8. Provenance notu

V28/V29 ham çıktı dizinleri yeni RTX 5090 sunucusuna taşınmamıştı. Yukarıdaki
sayısal sonuçlar, 20–21 Ağustos 2026 tarihli timestamp'li Codex oturum
kayıtlarındaki tamamlanmış terminal raporlardan kurtarıldı. Kod ve gradient
kontratı mevcut branch'teki kaynaklardan doğrudan doğrulandı. Bu rapor yeni bir
recomputation değildir; kayıp deney özetinin canonical olarak yeniden
oluşturulmasıdır.

Ana kaynak kodlar:

- `dynlaneseq_eg/modeling/v28_refined_belief_router.py`
- `dynlaneseq_eg/modeling/dynlaneseq_v28.py`
- `dynlaneseq_eg/tools/train_v28_refined_belief_router.py`
- `dynlaneseq_eg/tools/evaluate_v28_refined_belief_gate.py`
- `dynlaneseq_eg/tools/audit_v28_switch_confidence.py`
- `dynlaneseq_eg/tools/build_v29_oof_folds.py`

