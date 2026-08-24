# V35 Raw-RGB Temporal Observability Gate — Predeclared Contract

**Tarih:** 24 Ağustos 2026  
**Durum:** Sonuçlar görülmeden önce kilitlendi  
**Test split:** Kullanılmayacak

## 1. Soru

V34 iki olguyu birlikte gösterdi:

1. Oracle-good proposal komşu karelerin GT anotasyonlarıyla fiziksel olarak
   tutarlı.
2. Frozen V7 tahminleri ise komşu karelerde de current-wrong proposal'ı tercih
   ediyor.

V35 şu kalan soruyu ölçer:

> V7 feature veya prediction'larını kullanmadan, proposal çevresindeki ham RGB
> bilgisi oracle-good ile current-wrong proposal'ı yeni clip'lerde ayırabiliyor
> mu? Komşu iki ham kare hedef kareye ek bilgi katıyor mu?

Bu bir final detector eğitimi değil, observability/falsification gate'idir.

## 2. Veri ayrımı

- V34'ün sabit 54 validation clip'i ve 1.024 hedef görüntüsü kullanılacak.
- Mature V7 bu validation görüntülerini eğitimde görmedi.
- Clip'ler V34 manifestindeki sabit A/B ayrımında kalacak: 27 + 27 clip.
- İki bağımsız yön çalışacak:

```text
A→B: scorer Fold-A clip'lerinde eğitilir, yalnız Fold-B'de ölçülür.
B→A: scorer Fold-B clip'lerinde eğitilir, yalnız Fold-A'da ölçülür.
```

- Aynı clip iki tarafta bulunamaz.
- Evaluation label'ları eğitime, checkpoint seçimine veya threshold seçimine
  giremez.
- Son checkpoint sabit step'te değerlendirilecek; early stopping yoktur.

Validation GT bu diagnostic cross-fit içinde training fold'un candidate kalite
label'larını üretir. Bu nedenle sonuç official validation F1 veya deploy sonucu
olarak sunulmayacak. Ama evaluation fold'un clip'leri scorer için tamamen
unseen kalacaktır.

## 3. Sabit candidate bankası ve label

- Candidate geometrileri mature V7-225K target-main cache'inden gelir.
- Her slot ile GT, mevcut official slot eşleştirmesiyle eşlenir.
- Train fold'unda official candidate IoU yalnız pair label'ı oluşturur.
- Evaluation, V34'te önceden tanımlanan exact oracle-good versus
  current-wrong route-recoverable çiftlerinde yapılır.
- Proposal ID embedding, source route/existence/quality score ve V7 image
  feature'ı scorer'a verilmeyecek.

## 4. Ham görüntü girdisi

Her candidate curve için 64 dikey konumda iki yatay ribbon örneklenir:

```text
fine:   candidate merkezinin ±32 input-pixel çevresi
coarse: candidate merkezinin ±160 input-pixel çevresi
```

Her ribbon 25 yatay örnek taşır. Candidate'ın x, eğim, eğrilik ve geçerli-row
maskesi ayrı geometry kanalıdır.

Komşu karelerde candidate koordinatları V34 ile aynı DIS optical-flow ve
forward/backward consistency kontratıyla taşınır. Raw RGB scorer hiçbir V7
neighbor prediction'ı görmez.

## 5. Üç matched kol

Üç kol aynı scorer mimarisi, parametre sayısı, optimizer, seed, batch ve step
sayısını kullanır. Yalnız image tensorunun içeriği değişir.

### G — Geometry-only control

```text
RGB tensoru = 0
candidate x/eğim/eğrilik/maskesi = açık
```

Proposal geometrisinden öğrenilebilen kısa yolları ölçer.

### S — Single-frame raw RGB

```text
target RGB ribbon üç temporal konuma kopyalanır
```

Tek görüntünün proposal'ları ayırmaya yeterli olup olmadığını ölçer.

### T — Three-frame raw RGB

```text
flow-aligned previous + target + following raw RGB ribbons
```

Komşu karelerin gerçekten yeni bilgi katıp katmadığını ölçer.

Yanlış-görüntü kontrolünde aynı candidate geometrisi başka bir clip'in ham
üç-kare dizisinde örneklenir. Bu kontrol yalnız evaluation içindir.

## 6. Model ve eğitim

- Candidate-shared compact 2D ribbon CNN + 1D geometry encoder.
- Aynı scorer good ve wrong candidate'a uygulanır.
- Candidate sırasına veya proposal ID'sine özel parametre yoktur.
- Ana loss paired logistic ranking loss'tur.
- Yardımcı absolute-quality loss, good ve wrong official IoU target'larına
  uygulanır.
- Sabit optimizer endpoint'i: 2.000 step.
- Batch: 64 pair.
- Seed ensemble: 3407, 5741, 9011.
- Model seçimi yapılmaz; üç sabit endpoint logitlerinin ortalaması raporlanır.

## 7. Ana metrikler

Her direction ve threshold (`.50`, `.75`) ayrı raporlanır:

- exact V34 pair count,
- good-vs-wrong pair accuracy,
- candidate-level ROC AUC,
- good-minus-wrong mean margin,
- candidate-only AUC,
- correct-image ve wrong-image AUC,
- single-frame ve temporal farkı,
- clip-paired bootstrap güven aralıkları.

## 8. Önceden kilitli karar kuralları

### Ham tek-kare evidence PASS

S kolu dört hücrenin tamamında (iki yön × iki threshold):

```text
pair count >= 150
pair accuracy >= 0.65
AUC >= 0.70
AUC(S correct) - AUC(S wrong-image) >= 0.10
AUC(S correct) - AUC(G) >= 0.05
```

### Temporal incremental PASS

T kolu önce yukarıdaki bütün raw-image koşullarını geçmeli. Buna ek olarak dört
hücrenin tamamında:

```text
AUC(T) - AUC(S) >= 0.05
pair_accuracy(T) - pair_accuracy(S) >= 0.03
```

### Karar

```text
S geçer, T ek katkı vermez:
  raw-image candidate evaluator için tek bir deployment gate hak edilir;
  temporal detector gerekçelendirilmez.

T geçer, S geçmez ve temporal incremental koşullar geçer:
  yalnız temporal raw-image belief modeli için deployment gate hak edilir.

S ve T ikisi de geçmez:
  proposal-selection rescue ailesi kapatılır. Yeni frozen head, ribbon,
  corridor, mode-member, AGF veya temporal voting deneyi açılmaz.
```

Bu gate official F1 kazancı iddia etmez. PASS olursa bile sonraki deney, aynı
bankada source-TP protection ve exact structured route replay olmak zorundadır.

## 9. Yasaklar

- Test split açılmayacak.
- Evaluation fold ile checkpoint/step/loss ağırlığı seçilmeyecek.
- Sonuç sonrası seed, fold, pair veya threshold elenmeyecek.
- Inverted score post-hoc deploy skoru olarak kullanılmayacak.
- V7 route/existence/quality score'u raw RGB scorer'a eklenmeyecek.
- PASS olmadan full detector veya yeni tower eğitimi başlatılmayacak.
