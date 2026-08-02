# Coherent Lane State'ten Unified Lane-Set Decoder V3'e

**Tarih:** 2 Ağustos 2026  
**Kapsam:** Son mimari teşhis kaydından sonra yapılan iki ana tasarım adımı  
**Korunan kontrol branch'i:** `experiment_coherent_lane_state_25k`  
**Kontrol commit'i:** `33a7dec`  
**Yeni mimari branch'i:** `experiment_unified_lane_set_decoder_v3_25k`  
**İlk V3 implementasyon commit'i:** `4844d04`

Bu belge iki ayrı çalışmayı tek bir teknik hikâye içinde kaydeder:

1. **Coherent Primary Lane-State 25k:** Geometri, query kimliği, Hungarian
   eşleşmesi ve deployment skorunu aynı aday kimliğinde birleştiren kontrollü
   deney.
2. **Unified Lane-Set Decoder V3:** Coherent deneyde açık kalan nedensellik ve
   set seçimi problemlerini gideren yeni decoder tasarımı.

Belgenin amacı yalnızca hangi dosyaların değiştiğini listelemek değildir.
Forward akışını, gradyan yollarını, assignment ve loss sözleşmesini, eski
tasarımla farkları ve 25k deneyinin neyi kanıtlayıp neyi kanıtlamayacağını
ayrıntılı biçimde tanımlar.

---

## 1. En kısa özet

İlk model ailesi iyi lane geometrileri üretebiliyordu; fakat eğitim sırasında
çoklu query gruplarıyla aynı lane'in birden fazla kopyasını üretmeye teşvik
ediliyor, deployment sırasında bu kopyalar `quality` ve NMS ile temizleniyordu.
Bu yüzden yüksek F1 alabilse de gerçek anlamda NMS'siz bir lane seti
üretmiyordu.

`Coherent Lane State` deneyi bu dağınık sözleşmeyi sadeleştirdi:

- yalnızca 32 gerçek deployment query'si;
- global bire-bir Hungarian assignment;
- her katmanda bağımsız eşleşme;
- query başına kalıcı lane state;
- doğrudan existence skoru;
- quality, ikinci selector ve NMS yok.

Fakat bu ilk düzeltmede lane state, row geometrisini yalnızca **okuyordu**.
Geometriyi üreten row state'lere geri yazamıyor ve diğer lane adaylarıyla bütün
eğri düzeyinde rekabet edemiyordu. Dolayısıyla score'u üreten state ile curve'ü
üreten state tam anlamıyla aynı nedensel nesne değildi.

`Unified Lane-Set Decoder V3` bu eksik iki kenarı kapattı:

```text
lane state --> kendi row state'leri --> P2 geometrisi
     ^                                  |
     |__________________________________|

ve

lane state <--> diğer 31 lane state
```

Böylece tek bir lane state artık aynı anda:

- diğer adaylarla set düzeyinde rekabet eder;
- kendi row geometrisini etkiler;
- kendi row'larından gelen görüntü kanıtını geri toplar;
- görünür aralığı taşır;
- tek foreground skorunu üretir;
- aynı Hungarian kimliğiyle eğitilir;
- aynı skorla eşleştirilir, sıralanır ve test edilir.

Temel problem artık “FPN daha güçlü olmalı” şeklinde ele alınmıyor. Ana hedef,
FPN'in ürettiği kullanılabilir kanıtı **benzersiz, sıralanmış ve NMS'siz bir
lane setine** çevirebilen tutarlı bir decoder kurmaktır.

---

## 2. Üç nesil mimarinin büyük resmi

### 2.1 Tarihsel grouped S0

```text
Girdi görüntüsü
  |
  v
DLA-34 + SimpleFPN
  |
  v
Yüksek çözünürlüklü P2
  |
  v
Instance token + ordered row token
  |
  +--> birden fazla assignment grubu
  |       |
  |       +--> aynı GT lane için birden fazla pozitif kopya
  |
  v
Row geometrisi + existence + quality
  |
  v
existence x quality
  |
  v
NMS ile duplicate temizleme
  |
  v
Top-K lane çıktısı
```

Bu yapı tamamen anlamsız değildi. Kendi içinde tutarlı bir one-to-many
proposal sistemi kuruyor ve duplicate temizliğini açıkça post-processing'e
bırakıyordu. Sorunu, backbone kapasitesini ölçerken decoder'ın benzersiz set
üretme kapasitesini gizlemesiydi. NMS kaldırıldığında gerçek duplicate yükü
ortaya çıkıyordu.

### 2.2 Coherent Primary Lane-State

```text
32 primary query
  |
  +--> query i + row embedding j
  |          |
  |          v
  |      row state r[i,j]
  |          |
  |          v
  |      P2 row-reference decoder --> x[i,j]
  |          |
  |          v
  +<----- row-to-lane attention
             |
             v
       persistent lane state q[i]
             |
             +--> existence
             +--> visible range

Global 1:1 Hungarian + direct existence + NMS-free Top-4
```

Bu aşamada query kimliği, assignment, score ve range tek bir primary lane
üzerinde birleştirildi. Ancak lane state'in geometriden önce row state'lere
yazdığı bir yol ve lane state'lerin birbirini karşılaştırdığı global bir yol
yoktu.

### 2.3 Unified Lane-Set Decoder V3

```text
                             P4 / P5
                               |
                          pooled semantic
                               |
                               v
q[1..32] --> global lane-set attention -----------------------+
   |                                                          |
   | lane-to-row fixed residual                               |
   v                                                          |
r[i,1..160] --> curve-aligned P2 decoding --> x[i,1..160]     |
   |                                                          |
   +--> same-row candidate interaction                        |
   +--> same-lane vertical interaction                        |
   |                                                          |
   +--> rows-to-same-lane collection --> updated q[i] --------+
                                             |
                                             +--> range
                                             |
                                  temporary semantic view
                                             |
                                             +--> one score s[i]

sigmoid(s) --> Hungarian + focal + count + ranking + threshold + Top-4
```

Buradaki önemli nokta P4/P5'in x koordinatı üretmemesidir. P2 “lane nereden
geçiyor?” sorusunu, P4/P5 ise yalnızca “bu bütün eğri gerçek ve benzersiz bir
lane mi?” sorusunu destekler.

---

## 3. Birinci adım: Coherent Primary Lane-State 25k

### 3.1 Bu deney neden yapıldı?

Önceki kısa teşhisler şu tabloyu ortaya çıkarmıştı:

- 32 adayın içinde GT'yi karşılayabilecek geometriler bulunabiliyordu;
- oracle Top-K ile deploy edilen skor Top-K arasında belirgin fark vardı;
- query ownership eğitim boyunca yeterince kararlı değildi;
- frozen scalar selector, hierarchical selector ve hard selector gibi sonradan
  eklenen okuyucular train verisine uyum sağlasa da validation'a genellenemedi;
- multi-scale frozen evidence probe'un daha uzun eğitilmesi train uyumunu
  artırırken validation sonucunu kötüleştirdi;
- CondLSTR, MapTR ve DETR ailesinde geometry, assignment ve foreground kararı
  birbirinden kopuk query kimliklerine bırakılmıyordu.

Bu nedenle ilk kontrollü soru şuydu:

> Tek bir primary query; row geometrisini, lane state'i, Hungarian hedefini,
> foreground skorunu ve visible range'i baştan sona sahiplenirse scoring ve
> ownership problemi çözülür mü?

### 3.2 Coherent sözleşmesinde değiştirilenler

- `num_instances=32`, `num_groups=1` yapıldı.
- Train-only auxiliary query grupları kaldırıldı.
- Strict global one-to-one Hungarian assignment kullanıldı.
- Matcher object cost `-log(p)` yerine bounded `-p` ailesinde tutuldu.
- `matcher.lambda_obj=0.5` kullanıldı.
- Her decoder katmanı kendi mevcut çıktısı için yeniden Hungarian match yaptı.
- Query başına `PersistentLaneStateLayer` eklendi.
- Lane state yalnızca kendi 160 row state'ini cross-attention ile topladı.
- Existence ve visible range bu kalıcı lane state'ten üretildi.
- Quality loss, ayrı set selector ve score multiplication kapatıldı.
- Deployment skoru doğrudan foreground existence oldu.
- NMS kapatıldı ve doğrudan Top-4 kullanıldı.
- Bir katmanın ürettiği x eğrisi sonraki katmanın P2 örnekleme referansı oldu;
  katmanlar arasındaki referans `detach` edildi.
- Deep supervision korundu ve ara katmanlar bağımsız eşleştirildi.

### 3.3 Coherent forward akışı

Bir query `i` için:

```text
Başlangıç:
  q_i^0 = learned instance token i
  r_i,j^0 = instance token i + ordered row token j

Her decoder katmanı l:
  1. Önceki curve referansı çevresinden P2 kanıtı örneklenir.
  2. r_i,j^l row-local ve same-lane işlemlerle güncellenir.
  3. q_i^l yalnızca kendi güncel row state'lerine attention yapar.
  4. q_i^l -> existence ve visible range.
  5. r_i,j^l -> 800-bin x dağılımı ve beklenen x koordinatı.
  6. x tahmini detach edilerek sonraki katmanın referansı yapılır.
```

### 3.4 Coherent tasarımın gerçekten düzelttiği şey

Bu deney önemli bir temizlik yaptı. Artık:

- eğitimde var olup testte kaybolan proposal kimliği yoktu;
- quality ile existence farklı candidate ordering üretmiyordu;
- final selector başka bir query temsilinden skor çıkarmıyordu;
- matched query foreground, unmatched query background olarak açık biçimde
  denetleniyordu;
- inference kodu eğitimde optimize edilen score'u doğrudan kullanıyordu.

Yani **kimlik ve supervision sözleşmesi** büyük ölçüde düzeldi.

### 3.5 Coherent tasarımda açık kalan mimari hata

`PersistentLaneStateLayer` aşağıdaki yönde çalışıyordu:

```text
row geometry --> lane state --> score
```

Ama ters yön yoktu:

```text
lane state -X-> row geometry
```

Ayrıca lane state'ler arasında bütün eğri düzeyinde rekabet yoktu:

```text
q_i -X-> q_k,  i != k
```

Bunun üç sonucu vardı:

1. Score'u üreten state, skorladığı eğrinin oluşumunu nedensel olarak kontrol
   etmiyordu.
2. Bir aday, komşu adayın aynı lane'in duplicate'i olup olmadığını tüm curve
   boyunca karşılaştıramıyordu.
3. Quality ve NMS kaldırılınca existence head'den aynı anda existence,
   localization quality, uniqueness ve görüntü başına lane sayısını tek başına
   çözmesi beklendi.

Bu, bir DETR set decoder'ı kurmadan DETR gibi NMS'siz davranmasını istemekti.

### 3.6 25k sinyalinin doğru okunması

Kullanıcının paylaştığı NMS'siz coherent 25k sonucu:

| Ölçüt | Coherent 25k, NMS=0 |
|---|---:|
| TP | 71,181 |
| FP | 39,343 |
| FN | 33,705 |
| Precision | 64.40 |
| Recall | 67.87 |
| F1@0.50 | 66.09 |
| F1@0.75 | 49.29 |

Tarihsel DLA-34 25k sonucu ise yaklaşık olarak:

| Ölçüt | Tarihsel DLA-34 25k |
|---|---:|
| TP | 72,752 |
| FP | 13,488 |
| FN | 32,134 |
| Precision | 84.36 |
| Recall | 69.36 |
| F1@0.50 | 76.13 |

Bu iki satır **birebir aynı post-processing protokolü değildir**. Tarihsel
sonuç `score=0.30`, `quality_power=0.50` ve `NMS=20` ile alınmıştı; coherent
sonuç `score=0.30`, `quality_power=0` ve `NMS=0` ile alınmıştır. Dolayısıyla
“yeni geometry doğrudan 10 F1 kötüleşti” sonucu çıkarılamaz.

Yine de hata tipinin büyüklüğü çok nettir:

- TP farkı: `-1,571`
- FN farkı: `+1,571`
- FP farkı: `+25,855`

Ana çöküş geometri kapasitesinden çok, duplicate/foreground set kararındadır.
Bu sonuç V3'ün tasarım gerekçesini oluşturdu.

---

## 4. İkinci adım: Unified Lane-Set Decoder V3

### 4.1 Tasarım hedefleri

V3 aşağıdaki beş şartı aynı anda sağlamayı hedefler:

1. Score'u üreten lane state, kendi row geometrisini de etkilemelidir.
2. Row geometrisini üreten görüntü kanıtı tekrar aynı lane state'e dönmelidir.
3. Bütün lane state'ler duplicate ve cardinality kararı için birbirini
   görebilmelidir.
4. P2 geometrik hassasiyeti korunmalı; P4/P5 yalnızca semantik karar vermelidir.
5. Matching, classification, ranking, threshold ve Top-K aynı tek skoru
   kullanmalıdır.

### 4.2 Tensörler ve sabit boyutlar

Ana deney ayarlarında:

```text
B = batch size
N = 32 lane adayı
R = 160 ordered image row
D = 256 kanal
K = 800 yatay DFL bin
L = 4 decoder katmanı
```

Temel state'ler:

```text
lane_state q       : [B, N, D]
row_state r        : [B, N, R, D]
row_x_logits       : [B, N, R, K]
foreground_logit s : [B, N]
range              : [B, N, 2]
```

1600x640 girişte yaklaşık uzaysal ölçekler:

```text
P2: 160 x 400   -> yüksek çözünürlüklü geometri
P4:  40 x 100   -> orta/coarse semantik bağlam
P5:  20 x  50   -> geniş bağlam
```

P4 ve P5 doğrudan tam çözünürlükte attention memory yapılmaz. Her biri
`10 x 25` boyutuna adaptive average pooling ile indirilir. Böylece semantik
kolun aktivasyon maliyeti sınırlanır.

### 4.3 Her decoder katmanının gerçek sırası

Her `l` katmanı şu sırayı izler:

```text
q^l
 |
 |  (1) global lane-set self-attention, N=32
 v
q_set^l
 |
 |  (2) sabit lane-to-row residual
 v
r_owned^l = r^l + W(q_set^l)
 |
 |  (3) P2 üzerinde curve-aligned row-reference decoder
 |      - önceki x referansı çevresinden lokal örnekleme
 |      - aynı satırdaki adaylar arası interaction
 |      - aynı lane'in satırları boyunca vertical interaction
 v
r_visual^l
 |
 |  (4) rows-to-same-lane cross-attention
 v
q_base^(l+1)
 |
 +------------------> range head
 |
 |  (5) pooled P4/P5 semantic attention
 v
q_decision^(l+1) ----> one-logit foreground head

r_visual^l ----------> row distribution head --> x^(l+1)
x^(l+1).detach() ----> sonraki katmanın sampling referansı
q_base^(l+1) --------> sonraki katmanın kalıcı lane state'i
```

Bu sırada iki farklı lane görünümü vardır:

- **`q_base`:** Kalıcı geometry-owned lane state. Sonraki decoder katmanına
  taşınır ve range üretir.
- **`q_decision`:** `q_base` üzerine P4/P5 bağlamı eklenmiş geçici score
  görünümü. Yalnızca foreground kararı üretir; sonraki geometry katmanına
  taşınmaz.

Bu ayrım bilinçlidir. Semantik bilgi skoru düzeltebilir ama düşük
çözünürlüklü P4/P5 bilgisinin x koordinatlarını bulanıklaştırmasına izin
verilmez.

### 4.4 Global lane-set attention

Coherent modelde interaction ağırlıklı olarak row eksenindeydi. V3'te her
katmanın başında 32 complete lane state tek bir set olarak self-attention'a
girer:

```text
q_set = q + MHA(LN(q), LN(q), LN(q))
q_set = q_set + FFN(LN(q_set))
```

Bu katman bir lane adayının diğer adaylara göre:

- aynı lane'in duplicate'i olup olmadığını;
- soldan/sağdan hangi lane sırasına oturduğunu;
- görüntüde kaç farklı lane bulunduğunu;
- hangi adayın foreground olarak kalması gerektiğini

bütün eğri temsili üzerinden değerlendirebilmesi için eklenmiştir.

### 4.5 Lane-to-row: score state geometrinin sahibi oluyor

Lane state her kendi row state'ine projekte edilerek eklenir:

```text
r_i,j <- r_i,j + W_lane_to_rows(LN(q_i))
```

Burada öğrenilebilir `gamma`, `alpha` veya residual gate yoktur. Bunun nedeni
önceki dynamic/refinement deneylerinde optimizer'ın gürültülü yeni yolu
`gamma -> 0` yaparak devre dışı bırakabildiğinin görülmesidir.

Bu sabit residual sayesinde:

- geometry loss lane state'e geri ulaşır;
- lane state artık curve'ün yalnızca gözlemcisi değildir;
- query kimliği her row koordinatının üretimine doğrudan katılır.

### 4.6 P2 row-reference geometrisi

Geometri kolu mevcut yüksek çözünürlüklü mekanizmayı korur. Önceki katmanın
eğrisi P2 üzerinde bir referans oluşturur ve her row için bu referansın
çevresindeki yedi yatay offset örneklenir:

```text
[-96, -48, -24, 0, 24, 48, 96] piksel
```

Buradaki sampling offset'leri ile `LineIoU radius` aynı parametre değildir.
Localization loss ve matcher tarafındaki `LineIoU radius=15` olarak kalır.

Bu bölümün korunma nedeni:

- P3-first deneylerinde erken geometri bulanıklaşmıştı;
- row-reference deneyleri doğru görüntüye bakıldığını kanıtlamıştı;
- ana yeni sorun aday capacity'den çok set seçimi ve score ownership idi.

V3 bu nedenle geometriyi P4/P5'e taşımaz. P2 hâlâ tek coordinate-producing
memory'dir.

### 4.7 Rows-to-lane: görüntü kanıtı score'a geri dönüyor

P2 ile güncellenen 160 row state tekrar yalnızca kendi lane state'i tarafından
toplanır:

```text
q_i <- q_i + CrossAttn(q_i, r_i,1..R, r_i,1..R)
q_i <- q_i + FFN(LN(q_i))
```

Bu yol sayesinde foreground loss yalnızca soyut learned query'yi değil,
curve'ü gerçekten üreten row evidence yolunu da denetler.

### 4.8 P4/P5 score-only semantic view

P4 ve P5 ayrı projeksiyon ve attention modüllerinden geçer. Scale router iki
ölçeği birleştirir. Router sıfır ağırlık/bias ile başlatıldığı için ilk adımda
iki ölçeğe eşit davranır.

```text
c4 = CrossAttn(q_base, pooled(P4))
c5 = CrossAttn(q_base, pooled(P5))
w  = softmax(router(q_base))
c  = w4*c4 + w5*c5
q_decision = q_base + c + FFN(...)
```

`q_decision`:

- existence/uniqueness kararı verebilir;
- cross, no-line, shadow ve geniş bağlam gerektiren sahneleri okuyabilir;
- x koordinatı üretemez;
- range üretmez;
- sonraki decoder katmanına taşınmaz.

Dolayısıyla semantik dal geometry üzerinde “uzaktan kumandalı düzeltme” yapmaz.
Sadece aynı lane kimliğinin foreground kararını zenginleştirir.

---

## 5. Gradyan akışı

### 5.1 Forward ve backward birlikte

```text
GEOMETRY FORWARD
q --> lane-set --> lane-to-row --> row decoder/P2 --> x
                         ^               |
                         |               v
                         +---- row collection ----> q_base

GEOMETRY BACKWARD
L_point/L_DFL/L_IoU
  --> x head
  --> row states
  --> P2 evidence
  --> lane-to-row
  --> lane-set state

SCORE FORWARD
P2 --> rows --> q_base --> P4/P5 decision view --> s

SCORE BACKWARD
L_exist/L_count/L_margin
  --> same score s
  --> q_decision
  --> P4/P5 semantic context
  --> q_base
  --> row collection
  --> row states and P2 evidence
```

### 5.2 Loss bazında izin verilen yollar

| Loss | Ulaşması gereken yol | Bilinçli olarak kapalı yol |
|---|---|---|
| Point / DFL / LineIoU | x head -> row -> P2 ve lane-to-row -> lane-set | P4/P5 semantic residual -> x |
| Range | range head -> `q_base` | `q_decision` -> range |
| Foreground focal | score -> `q_decision` -> `q_base` -> rows/P2 | ikinci quality/selector head |
| Foreground focal | score -> P4/P5 semantic attention | P4/P5 -> sonraki geometry block |
| Cardinality | aynı foreground olasılıklarının toplamı | ayrı count head |
| Margin | aynı foreground logitleri | ayrı ranking score'u |

### 5.3 Detached row reference neyi kesiyor?

Katman `l` tarafından üretilen `x^l`, katman `l+1` için örnekleme konumudur.
Bu koordinat katmanlar arasında detach edilir:

```text
x^l --detach--> sampler^(l+1)
```

Bu, bütün öğrenmeyi kesmez. Her katman kendi point/DFL/LineIoU loss'unu alır
ve lane-state residual yolu dört blok boyunca differentiable kalır. Detach
yalnızca tekrarlı `grid_sample` koordinatları üzerinden oluşacak kontrolsüz
yüksek dereceli gradyan yolunu engeller.

---

## 6. Assignment ve supervision sözleşmesi

### 6.1 Hungarian matching

V3 aşağıdaki sözleşmeyi coherent kontrolden korur:

- bir görüntüdeki 32 query tek global aday setidir;
- her GT lane yalnızca bir query ile eşleşir;
- bir query en fazla bir GT lane alır;
- unmatched 28 civarı query açıkça background'dur;
- object matcher cost bounded foreground probability kullanır;
- `lambda_obj=0.5` ile geometry'nin matcher'da ezilmesi önlenir;
- her decoder katmanı kendi çıktısına göre yeniden eşleşir.

Ara katmanın final katman permutation'ını zorla devralmaması önemlidir. Erken
katman hangi curve'ü gerçekten çiziyorsa supervision o katmanın mevcut
geometrisine göre verilir.

### 6.2 Tek foreground skoru

Model tek scalar logit üretir:

```text
s_i in R
p_i = sigmoid(s_i)
```

Eski kodla uyumluluk için dışarıya şu view verilir:

```text
exist_logits_i = [s_i, 0]
softmax([s_i, 0])[0] = sigmoid(s_i)
```

Dolayısıyla matcher, focal loss ve inference matematiksel olarak aynı
olasılığı görür.

### 6.3 Cardinality loss

Bağımsız focal classification her query'yi tek tek denetler; fakat görüntüde
toplam kaç lane skoru taşınması gerektiğini doğrudan söylemez. V3 bu nedenle
ayrı bir count head eklemek yerine aynı score'ların toplamını kullanır:

```text
predicted_count = sum_i sigmoid(s_i)
target_count    = number_of_GT_lanes
L_cardinality   = SmoothL1(predicted_count, target_count)
```

Ağırlık:

```text
w_cardinality = 0.10
```

Bu özellikle cross ve no-line görüntülerde gereksiz foreground kütlesini
azaltmayı hedefler.

### 6.4 Hard-negative score margin

Her matched query, en yüksek skorlu unmatched duplicate'lerden daha yukarıda
olmalıdır:

```text
L_margin = mean softplus(margin - s_positive + s_hard_negative)
```

Her görüntüde en zor sekiz unmatched query kullanılır:

```text
margin = 0.50
hard negatives = 8
w_score_margin = 0.25
```

Bu bir ikinci selector değildir. Aynı `s_i` logitini doğrudan Top-K için
eğitir.

### 6.5 Aktif ve kapalı score bileşenleri

```text
AKTİF
  w_exist        = 2.0
  w_cardinality  = 0.10
  w_score_margin = 0.25

KAPALI
  w_quality       = 0.0
  w_set_selection = 0.0
  quality_power   = 0.0
  NMS distance    = 0.0
```

Deep supervision ve mevcut localization loss'ları korunur. LineIoU radius 15
olarak kalır.

---

## 7. Training ve inference artık nasıl eşleşiyor?

```text
TRAIN
  32 query
    -> global 1:1 Hungarian
    -> matched = foreground
    -> unmatched = background
    -> aynı s ile focal/count/margin
    -> her katmanda geometry supervision

INFERENCE
  aynı 32 query
    -> p = sigmoid(s)
    -> score threshold
    -> direct Top-4
    -> quality multiplication yok
    -> ikinci selector yok
    -> NMS yok
```

Bu tasarımda eğitimde optimize edilip deployment'ta atılan ayrı bir query
grubu veya score yolu bulunmaz.

---

## 8. Karşılaştırmalı bileşen tablosu

| Özellik | Tarihsel grouped S0 | Coherent lane state | Unified V3 |
|---|---|---|---|
| Deployable aday | 32 (4x8 grup) | 32 primary | 32 primary |
| Train-only query grubu | Tarihsel baz koşuda yok | Yok | Yok |
| Assignment | Dört izole grupta ayrı eşleşme; aynı GT için kopyalar | Global 1:1 | Global 1:1 |
| Katman başına fresh match | Son sürümlerde var | Var | Var |
| Kalıcı lane state | Yok/pooled summary | Var | Var |
| Lane state diğer lane'leri görür | Sınırlı/row düzeyi | Hayır | Evet, global set attention |
| Lane state row geometrisine yazar | Hayır | Hayır | Evet, fixed residual |
| Row evidence lane state'e döner | Pooling/yan yol | Evet | Evet |
| P2 curve-aligned geometry | Son rowref sürümlerinde var | Var | Var |
| P4/P5 semantik | Geometry ile uyumsuz denemeler | Yok | Yalnız score view |
| Foreground çıkışı | 2 logit | 2 logit | Tek logit |
| Quality | Aktif | Kapalı | Kapalı |
| Ayrı selector | Bazı sürümlerde aktif | Kapalı | Kapalı |
| Count supervision | Yok | Yok | Aynı score üzerinde var |
| Hard-negative ranking | Ayrı selector deneyleri | Yok | Aynı score üzerinde var |
| NMS | Gerekli | Kapalı, fakat FP patladı | Tasarım hedefi NMS'siz |
| Deployment score | exist x quality | exist | sigmoid(single score) |

---

## 9. Kod haritası

| Dosya | Görevi |
|---|---|
| `dynlaneseq_eg/modeling/unified_lane_set.py` | Global set attention, lane-to-row, rows-to-lane ve score-only P4/P5 semantic view |
| `dynlaneseq_eg/modeling/structured_queries.py` | V3 bloğunu dört decoder katmanına bağlar; tek logit ve decision/base state ayrımını uygular |
| `dynlaneseq_eg/modeling/dynlaneseq_s0.py` | Structured head'e P2 ile birlikte seçili multi-scale feature'ları taşır |
| `dynlaneseq_eg/losses/loss_s0.py` | Cardinality ve hard-negative margin loss'larını uygular |
| `dynlaneseq_eg/factory.py` | Yeni loss config alanlarını criterion'a bağlar |
| `dynlaneseq_eg/configs/culane_s0_structured_query_dla34_unified_lane_set_v3_25k.yaml` | 25k V3 deney sözleşmesi |
| `dynlaneseq_eg/tests/test_unified_lane_set_decoder.py` | Shape, skor eşitliği ve gradyan contract testleri |
| `scripts/run_culane_dla34_unified_lane_set_v3_fromscratch_25k.sh` | Audit, fresh train ve auto-resume |
| `scripts/evaluate_culane_dla34_unified_lane_set_v3_25k_val.sh` | Tek inference ile NMS'siz threshold taraması ve cached NMS karşı-olgusu |
| `scripts/eval_culane_dla34_unified_lane_set_v3_full_test.sh` | Dondurulmuş tek ayarla resmi CULane test |

Coherent kontrolün ana dosyaları:

| Dosya | Görevi |
|---|---|
| `dynlaneseq_eg/configs/culane_s0_structured_query_dla34_coherent_lane_state_25k.yaml` | İlk coherent contract |
| `scripts/run_culane_dla34_coherent_lane_state_fromscratch_25k.sh` | Coherent eğitim/audit |
| `scripts/evaluate_culane_dla34_coherent_lane_state_25k_gate.sh` | Coherent diagnostic gate |
| `dynlaneseq_eg/tests/test_coherent_lane_state_contract.py` | İlk kimlik ve inference sözleşmesi testleri |

---

## 10. Uygulama doğrulamaları

V3 implementasyonu yalnızca config düzeyinde kontrol edilmedi.

Yapılan doğrulamalar:

- tüm `dynlaneseq_eg/tests` paketi: **340 test geçti**;
- küçük V3 head için `torch.compile` forward/backward smoke testi geçti;
- tam DLA-34 model synthetic inference çıktıları finite bulundu;
- tam model + matcher + criterion + backward testi geçti;
- geometry loss'un lane-to-row ve lane-set katmanına ulaştığı doğrulandı;
- foreground loss'un rows-to-lane, P2 ve P4/P5 yollarına ulaştığı doğrulandı;
- saf geometry loss altında P4/P5 gradient'inin `None` kaldığı doğrulandı;
- tek logit compatibility eşitliği kontrol edildi;
- shell script syntax, Python compile ve config contract audit geçti.

Örnek smoke testte sıfır olmayan gradient normları gözlendi:

```text
lane_to_rows projection : 0.1350
rows_to_lane attention  : 0.1707
lane-set attention      : 0.1428
P4 semantic path        : 0.0191
```

Bu değerler performans sonucu değildir. Yalnızca tasarlanan gradyan yollarının
gerçekte açık olduğunu kanıtlar.

---

## 11. Parametre ve maliyet farkı

```text
Coherent toplam parametre : 61.421M
Coherent structured head  :  9.585M

V3 toplam parametre       : 69.061M
V3 structured head        : 17.225M

Artış                     :  7.640M parametre
FP32 weight karşılığı     : yaklaşık 30.6 MB
```

Artışın ana kaynağı dört decoder bloğundaki:

- global lane-set attention;
- lane-to-row projection;
- rows-to-lane attention;
- P4/P5 semantic attention ve FFN katmanlarıdır.

P4/P5 memory `10x25` havuzlandığı için semantik kol full-resolution attention
kadar activation büyütmez. Bununla birlikte gerçek eğitim hızı ve peak VRAM
sunucu üzerinde ölçülmeden kesin performans tahmini yapılmamalıdır.

---

## 12. 25k deney protokolü

Bu koşu baştan eğitilir. Coherent checkpoint'ten resume edilmez; çünkü yeni
lane-to-row, set-attention ve semantic yolların baştan aynı identity contract
altında öğrenmesi gerekir.

Sabit tutulanlar:

- DLA-34 backbone;
- 1600x640 input;
- SimpleFPN ve P2 geometry;
- 32 aday;
- 160 ordered row;
- 800 DFL bin;
- 4 decoder katmanı;
- row-reference offset'leri `[-96,-48,-24,0,24,48,96]`;
- matcher/loss LineIoU radius 15;
- deep supervision;
- seed 3407;
- effective batch 16;
- 1k warmup;
- 278k cosine scheduler horizon.

İlk gate 25k'da durur; scheduler 25k'ya sıkıştırılmaz. Yani bu bir final eğitim
takvimi değil, eşlenmiş mimari sinyal testidir.

### 12.1 Validation akışı

Validation script'i sabit uniform-256 subset üzerinde modeli yalnızca bir kez
çalıştırır. Aynı cached predictions üzerinde:

- NMS'siz finite threshold grid;
- aynı threshold altında NMS=20 counterfactual

hesaplanır.

Ana çıktılar:

```text
outputs/diagnostics/unified_lane_set_v3_25k/nmsfree_uniform256.json
outputs/diagnostics/unified_lane_set_v3_25k/nms20_uniform256.json
outputs/diagnostics/unified_lane_set_v3_25k/summary.json
```

### 12.2 Başarı ölçütleri

V3 başarılı sayılmak için yalnızca threshold sweep'te iyi bir sayı bulmamalı.
Aşağıdaki yapısal sinyaller birlikte görülmelidir:

1. NMS'siz precision ve F1 coherent kontrolden belirgin yüksek olmalı.
2. Direct score Top-K ile oracle Top-K arasındaki boşluk küçülmeli.
3. Aynı threshold'a NMS eklenince gelen F1 kazancı küçülmeli.
4. F1@0.75 ve candidate geometry capacity çökmemeli.
5. Toplam foreground probability, GT lane sayısını takip etmeli.
6. Cross ve no-line örneklerde gereksiz foreground kütlesi azalmalı.

### 12.3 Sonuçlar nasıl yorumlanacak?

```text
NMS'siz precision yükselir + NMS kazancı küçülür
  -> unified set sözleşmesi doğru yönde.

Geometry iyi kalır ama NMS hâlâ büyük kazanç sağlar
  -> duplicate uniqueness hâlâ çözülmemiş.

Precision yükselir ama F1@0.75 çöker
  -> score iyileşirken geometry yolu zarar görmüş.

All-32/oracle capacity iyi, direct Top-K kötü
  -> score/assignment sorunu sürüyor.

All-32 capacity de düşer
  -> decoder artık yalnızca selection değil geometry de bozuyor.

Train loss düşer fakat validation selection kötüleşir
  -> ek score yolları genellenebilir kanıt öğrenmiyor.
```

---

## 13. Çalıştırma komutları

### 13.1 Branch'i uzak sunucuda çekme

```bash
cd /workspace/DynLaneSeq
conda activate clrernet

git fetch origin experiment_unified_lane_set_decoder_v3_25k
git switch -C experiment_unified_lane_set_decoder_v3_25k FETCH_HEAD
```

### 13.2 Yalnız contract audit

```bash
AUDIT_ONLY=1 \
bash scripts/run_culane_dla34_unified_lane_set_v3_fromscratch_25k.sh
```

### 13.3 Baştan eğitim veya otomatik resume

```bash
DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
AUTO_RESUME=1 \
bash scripts/run_culane_dla34_unified_lane_set_v3_fromscratch_25k.sh
```

### 13.4 25k validation gate

```bash
DATA_ROOT=/workspace/CULane \
EVAL_BATCH_SIZE=4 \
NUM_WORKERS=8 \
MAX_BATCHES=64 \
bash scripts/evaluate_culane_dla34_unified_lane_set_v3_25k_val.sh
```

### 13.5 Dondurulmuş threshold ile resmi test

Önce `summary.json` içindeki `best_nmsfree.score_threshold` okunur. Ardından:

```bash
DATA_ROOT=/workspace/CULane \
SCORE_THRESH=<best_nmsfree_threshold> \
EVAL_BATCH_SIZE=8 \
AMP_DTYPE=none \
bash scripts/eval_culane_dla34_unified_lane_set_v3_full_test.sh
```

---

## 14. Ne çözüldü, ne henüz kanıtlanmadı?

### Kod ve sözleşme düzeyinde çözülenler

- Score state ile geometry state arasında iki yönlü nedensel bağ kuruldu.
- Query'ler bütün-lane düzeyinde aynı set içinde rekabet etmeye başladı.
- P2 geometry ile P4/P5 semantic score görevleri ayrıştırıldı.
- İki-logit serbestliği tek foreground logitine indirildi.
- Matcher, focal, ranking ve inference aynı skora bağlandı.
- Cardinality ve hard-negative ranking ayrı score head açmadan eklendi.
- Quality, late selector, one-to-many train grupları ve NMS deployment'tan
  çıkarıldı.
- Tasarlanan gradyan yolları testlerle doğrulandı.

### Henüz kanıtlanmayanlar

- V3'ün coherent kontrolden daha yüksek validation F1 vereceği henüz deneyle
  gösterilmedi.
- NMS bağımlılığını tamamen kaldıracağı henüz gösterilmedi.
- 25k pozitif sinyalin 278k boyunca korunacağı bilinmiyor.
- 80+ CULane F1 garanti değildir.
- P4/P5 semantic view'un özellikle cross/no-line precision'ını artıracağı bir
  hipotezdir; sonucu validation gate belirleyecektir.

Bu nedenle doğru ifade şudur:

> V3, gözlenen precision çöküşüne doğrudan karşılık veren, gradyan yolları
> doğrulanmış bir mimari düzeltmedir; fakat başarı iddiası 25k paired
> validation ve ardından tek dondurulmuş full-test sonucu gelmeden yapılamaz.

---

## 15. Son mimari ağaç

```text
DynLaneSeq / Unified Lane-Set Decoder V3
|
+-- Backbone: DLA-34
|
+-- Neck: SimpleFPN
|   |
|   +-- P2 --> yalnız geometry-producing visual memory
|   +-- P4 --> pooled score semantics
|   +-- P5 --> pooled score semantics
|
+-- Token initialization
|   |
|   +-- 32 learned lane/instance state
|   +-- 160 learned ordered row token per lane
|   +-- image-conditioned initial curve reference
|
+-- Decoder x4
|   |
|   +-- global lane-set self-attention
|   +-- fixed lane-state -> own rows residual
|   +-- curve-aligned P2 sampling, 7 yatay offset
|   +-- same-row inter-candidate interaction
|   +-- same-lane vertical interaction
|   +-- own rows -> same lane-state collection
|   +-- temporary P4/P5 semantic decision view
|   +-- per-layer prediction + independent Hungarian supervision
|   `-- detached x reference to next block
|
+-- Heads
|   |
|   +-- row DFL head: D -> 800 bins per row
|   +-- range head: q_base -> start/end
|   `-- foreground head: q_decision -> one scalar
|
+-- Training
|   |
|   +-- global strict 1:1 Hungarian
|   +-- focal foreground/background
|   +-- cardinality regularization
|   +-- hard-negative score margin
|   +-- point + DFL + LineIoU + range
|   `-- deep supervision at decoder layers
|
`-- Deployment
    |
    +-- sigmoid(single foreground logit)
    +-- frozen threshold
    +-- direct Top-4
    +-- quality multiplication: off
    +-- second selector: off
    `-- lane NMS: off
```

Bu ağaç V3'ün ana ilkesini özetler: **bir lane adayı, görüntü kanıtından
geometrisine, assignment'ından skoruna kadar tek ve nedensel bir state zinciri
olmalıdır.**
