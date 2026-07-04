# Structured Row-Token Decoder ile 2D Lane Detection

Danışman hocaya teknik açıklama dokümanı  
Son güncelleme: 2026-07-03

Bu doküman, DynLaneSeq projesindeki son ana modelin mimarisini sıfırdan açıklar. Hedef okuyucu CNN, feature map, segmentation, detection gibi klasik computer vision kavramlarını bilen; fakat transformer, query, token, self-attention ve cross-attention kavramlarına aşina olmayan bir araştırmacıdır.

Dokümanın amacı şudur:

- Modelin neden tasarlandığını açıklamak.
- “Instance token”, “row token”, “ordered row tokens”, “cross-attention”, “DFL” gibi terimleri proje bağlamında netleştirmek.
- Modelin input görüntüden final lane çizgilerine nasıl gittiğini adım adım göstermek.
- Hangi loss’ların neyi öğrettiğini anlatmak.
- Mimari kararların gerekçesini açık hale getirmek.
- Hocanın sorabileceği “neden böyle yaptınız?” sorularına doğrudan cevap vermek.

Bu doküman paper metni değildir; paper’a dönüştürülebilecek teknik açıklama metnidir. İstenirse sonradan IEEE/LaTeX formatına çevrilebilir.

---

## 1. Kısa özet

Lane detection probleminde amaç, kamera görüntüsündeki şerit çizgilerini bulmaktır. CULane gibi veri setlerinde her lane genellikle farklı `y` satırlarında karşılık gelen `x` koordinatlarıyla temsil edilir.

Klasik yaklaşımlar bu problemi farklı biçimlerde çözer:

- segmentation tabanlı yöntemler her piksele lane / background etiketi verir;
- anchor tabanlı yöntemler önceden tanımlı lane şablonlarını düzeltir;
- row-wise yöntemler her yatay satır için lane’in `x` konumunu tahmin eder;
- transformer query tabanlı yöntemler ise her lane’i bir query vektörüyle temsil etmeye çalışır.

Bizim yaklaşımımızın ana fikri:

```text
Bir lane’i tek bir global vektöre sıkıştırmak yerine,
lane kimliğini instance token ile,
lane geometrisini ise sıralı row token dizisiyle temsil ediyoruz.
```

Yani bir lane şu şekilde modelleniyor:

```text
lane_i =
  instance_token_i
  +
  [row_token_1, row_token_2, ..., row_token_R]
```

Burada:

- `instance_token_i`: “bu hangi lane adayı?” bilgisini taşır.
- `row_token_r`: “bu lane’in r’inci yatay satırdaki geometrisi nedir?” bilgisini taşır.
- `R`: kullanılan satır sayısıdır. Son güçlü modelde `R = 160`.

Bu yapı sayesinde model her lane için tek seferde bütün eğriyi ezberlemek zorunda kalmaz. Lane geometrisini satır satır ama birbirinden kopuk olmayacak şekilde öğrenir.

Son güçlü konfigürasyon:

```text
Backbone: ResNet-34
Input: 1600 x 640
FPN channel: 256
Lane candidate / instance slot: 32
Row count: 160
X-bin count: 800
Structured decoder layer: 4
DFL: enabled
EMA: yok
TTA: yok
```

Bu yapı CULane test üzerinde yaklaşık `79.9 F1` bandına çıkmıştır. Bu skor tek başına makale için yeterli argüman değildir; asıl argüman, structured row-token decoder fikrinin unstructured baseline’a göre sistematik olarak daha iyi çalıştığını ablationlarla göstermek olmalıdır.

---

## 2. Problem tanımı

### 2.1 Lane detection çıktısı nasıl temsil edilir?

Bir yol görüntüsü düşünelim. Lane çizgisi düz bir kutu değildir; çoğu zaman uzun, ince, eğri ve perspektiften dolayı şekli değişen bir yapıdır.

Biz lane’i şu şekilde temsil ediyoruz:

```text
lane = [(x_1, y_1), (x_2, y_2), ..., (x_R, y_R)]
```

Burada:

- `y_r` önceden belirlenmiş yatay satırdır.
- `x_r` modelin tahmin ettiği lane konumudur.
- bazı satırlarda lane görünmüyor olabilir; bunun için valid mask kullanılır.

Son modelde:

```text
input_h = 640
num_rows = 160
```

Yani yaklaşık her `4 px` yükseklikte bir lane noktası tahmin edilir:

```text
640 / 160 = 4 px per row
```

Yatay eksende:

```text
input_w = 1600
x_bins = 800
```

Yani model her satırda lane’in `x` konumunu yaklaşık `2 px` çözünürlükle modelleyebilir:

```text
1600 / 800 = 2 px per bin
```

Bu temsil lane detection için doğal bir temsildir; çünkü lane çizgileri kamera görüntüsünde genellikle yukarıdan aşağıya devam eden eğrilerdir. Her yatay satırda lane’in x konumunu bilmek, çizgiyi yeniden oluşturmak için yeterlidir.

---

## 3. Eski / basit yaklaşım neden sınırlı?

Basit transformer tabanlı bir lane detector şöyle düşünülebilir:

```text
Her lane adayı için bir vektör üret.
Bu vektörden bütün lane geometrisini tahmin et.
```

Şema:

```text
image features
     |
 transformer decoder
     |
 lane_query_i  ----->  [x_1, x_2, ..., x_R]
```

Bu yaklaşımda `lane_query_i` tek bir vektördür. Bu vektör hem şunları tutmaya çalışır:

- lane kimliği,
- lane var mı yok mu bilgisi,
- lane’in nerede başladığı ve bittiği,
- lane’in tüm satırlardaki x koordinatları,
- eğrilik,
- visibility / occlusion bilgisi.

Bu temsil fazla sıkıştırılmıştır. Bir lane’in 160 satırlık geometrisini tek bir vektöre yüklemek zorunda kalır.

Problem şudur:

```text
Tek vektör = bütün lane eğrisini tek hafızaya sıkıştırmak.
Structured row tokens = lane eğrisini satır satır temsil etmek.
```

CNN bilen biri için benzetme:

- Tek query yaklaşımı, bütün image segmentation maskesini tek bir global feature vektöründen üretmeye benzer.
- Structured row-token yaklaşımı ise her spatial konum için ayrı ama ilişkili feature tutmaya benzer.

Bizim model bu yüzden lane’i tek query yerine yapılandırılmış token grubuyla temsil eder.

---

## 4. Transformer kavramları: sıfırdan açıklama

Bu bölüm, transformer bilmeyen okuyucu için yazılmıştır.

### 4.1 Token nedir?

Transformer’da `token`, modelin işlem yaptığı vektördür.

NLP’de token genellikle kelimedir:

```text
"araba" -> token vector
"yol"   -> token vector
```

Computer vision’da token şu anlamlara gelebilir:

- görüntünün bir patch’i,
- bir object query,
- bir lane query,
- bir row query,
- bir learnable temsil vektörü.

Bizim modelde tokenlar görüntüden doğrudan kesilmiş patch değildir. Bazıları öğrenilen vektörlerdir:

```text
instance token: öğrenilen lane adayı vektörü
row token: öğrenilen row pozisyonu vektörü
```

Bu tokenlar eğitim sırasında optimize edilir.

### 4.2 Query, key, value nedir?

Attention mekanizmasında üç kavram vardır:

```text
Query:  ne arıyorum?
Key:    nerede ne var?
Value:  bulduğum bilgiden ne alacağım?
```

Basit sezgi:

Bir öğrenci kütüphaneye gidiyor:

- Query = öğrencinin sorduğu soru.
- Key = kitapların indeks/başlık bilgisi.
- Value = kitapların içeriği.

Öğrenci sorusuna en uygun kitaplardan bilgi alır.

Computer vision’da:

- Query bir object/lane/row token olabilir.
- Key ve value genellikle CNN feature map’ten gelir.
- Attention, query’nin feature map’in hangi bölgelerine bakacağını öğrenir.

### 4.3 Self-attention nedir?

Self-attention’da tokenlar birbirleriyle konuşur.

Örnek:

```text
lane row 10 tokenı
lane row 11 tokenına bakabilir
lane row 12 tokenına bakabilir
...
```

Bu, lane çizgisinin sürekliliğini öğrenmek için önemlidir. Çünkü lane’in bir satırdaki konumu, komşu satırlardaki konumlarından bağımsız değildir.

### 4.4 Cross-attention nedir?

Cross-attention’da tokenlar dış bir kaynak feature map’e bakar.

Bizim durumda:

```text
row token = query
CNN/FPN image features = key/value
```

Yani row token görüntüye soru sorar:

```text
"Ben lane 5'in 80. satırdaki temsilcisiyim.
Bu satırda lane çizgisi yatayda nerede olabilir?"
```

Feature map cevap verir:

```text
"Bu satırdaki şu x bölgeleri lane benzeri görünüyor."
```

Cross-attention bu bilgi aktarımını öğrenir.

---

## 5. Modelin genel akışı

Son modelin ana akışı:

```text
Input image
   |
ResNet-34 backbone
   |
FPN
   |
Feature projection
   |
Structured Row-Token Decoder
   |
Per-lane outputs:
   - existence score
   - row-wise x distributions
   - predicted x rows
   - vertical valid range
   - quality score
   |
Post-processing:
   - score threshold
   - lane NMS
   - top-k selection
   |
Final lane predictions
```

Daha görsel hali:

```text
                 ┌────────────────────┐
                 │ input image         │
                 │ 1600 x 640          │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ ResNet-34 backbone  │
                 │ C2, C3, C4, C5      │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ FPN                │
                 │ p2 feature map      │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ 1x1 projection      │
                 │ 256-dim features    │
                 └─────────┬──────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │ Structured Row-Token Decoder         │
        │ instance token + ordered row tokens  │
        └─────────┬───────────────────────────┘
                  │
                  ▼
        ┌─────────────────────────────────────┐
        │ 32 lane candidates                   │
        │ each has 160 row-wise x predictions  │
        └─────────────────────────────────────┘
```

---

## 6. Backbone ve FPN

### 6.1 ResNet-34 backbone

Model ResNet-34 ile başlar. ResNet-34 görüntüyü giderek daha soyut feature map’lere çevirir:

```text
C2: daha yüksek çözünürlük, düşük seviye detay
C3: orta seviye
C4: daha semantik
C5: en düşük çözünürlük, en yüksek semantik bilgi
```

Bizim backbone dosyasında ResNet-34 şu feature seviyelerini üretir:

```text
C2: 64 channel
C3: 128 channel
C4: 256 channel
C5: 512 channel
```

### 6.2 FPN neden kullanılıyor?

FPN, farklı çözünürlükteki feature seviyelerini birleştirir. Lane detection için bu önemlidir:

- Lane çizgileri incedir; düşük seviye spatial detay gerekir.
- Zor görüntülerde semantik bağlam gerekir.
- FPN hem detay hem bağlamı birleştirmeye çalışır.

Bizim son modelde FPN çıkışı:

```text
fpn_channels = 256
```

Sonra `1x1 conv` ile model boyutuna projekte edilir:

```text
dim = 256
```

Yani structured decoder’a giren image feature boyutu:

```text
[B, 256, H_feature, W_feature]
```

Sonra bu feature, row-token decoder için yeniden örneklenir:

```text
[B, 256, 160, 400]
```

Burada:

- `160`: row sayısı.
- `400`: attention için yatay evidence bin sayısı.
- `256`: channel / embedding dim.

Not: Output x tahmini için `x_bins = 800` kullanılır; attention evidence için `evidence_x_bins = 400` kullanılır. Bunun sebebi compute maliyetini kontrol etmektir. Model görüntüden 400 yatay evidence konumuna bakar, ama final x dağılımını 800 bin üzerinden üretir.

---

## 7. Structured row-token decoder’ın ana fikri

### 7.1 Temel temsil

Son modelde `32` lane adayı vardır:

```text
num_instances = 32
```

Her lane adayı için `160` row token vardır:

```text
num_rows = 160
```

Dolayısıyla her görüntü için decoder içinde şu kadar lane-row token oluşur:

```text
32 x 160 = 5120 token
```

Her token `256` boyutlu vektördür:

```text
token_dim = 256
```

### 7.2 Instance token nedir?

`instance token`, lane adayının kimliğini temsil eder.

Örnek:

```text
instance_token_0 -> birinci lane adayı
instance_token_1 -> ikinci lane adayı
...
instance_token_31 -> otuz ikinci lane adayı
```

Bu tokenlar görüntüden gelmez; learnable parametrelerdir. Eğitim boyunca öğrenilirler. Amaçları şudur:

- farklı lane adaylarının birbirinden ayrışmasını sağlamak,
- modelin “bu slot sol lane olsun, bu slot sağ lane olsun” gibi görev paylaşımı öğrenmesine izin vermek,
- her lane adayına global bir kimlik vermek.

### 7.3 Row token nedir?

`row token`, dikey eksendeki satır pozisyonunu temsil eder.

Örnek:

```text
row_token_0   -> görüntünün üst tarafına yakın row
row_token_80  -> orta row
row_token_159 -> alt tarafa yakın row
```

Bu tokenlar da learnable parametrelerdir. Amaçları şudur:

- her row’un farklı geometrik rolü olduğunu modele bildirmek,
- alt taraftaki lane noktaları ile üst taraftaki lane noktalarının farklı perspektif davranışına sahip olmasını sağlamak,
- lane geometrisini ordered yani sıralı hale getirmek.

### 7.4 Neden instance token ile row token toplanıyor?

Kodda temel satır şudur:

```python
row_tokens = instance[:, None, :] + row[None, :, :]
```

Yani her lane-row token şu şekilde kurulur:

```text
token(i, r) = instance_token(i) + row_token(r)
```

Bu çok kritik bir tasarım kararıdır.

Bu toplamın anlamı:

```text
token(i, r) =
  "i numaralı lane adayının"
  +
  "r numaralı satırdaki temsilcisi"
```

Örneğin:

```text
token(3, 80)
```

şunu temsil eder:

```text
3 numaralı lane adayının 80. satırdaki geometri tokenı.
```

#### Neden toplama, concat değil?

İki seçenek olabilirdi:

```text
concat: [instance_token_i ; row_token_r]  -> 512 dim
sum:    instance_token_i + row_token_r    -> 256 dim
```

Toplama seçilmesinin sebepleri:

1. Transformer embeddinglerinde pozisyon bilgisini eklemek standart bir yöntemdir.
   NLP’de de kelime embedding’i ile pozisyon embedding’i genellikle toplanır.

2. Toplama, token boyutunu sabit tutar.
   `concat` yapsaydık boyut iki katına çıkardı veya ek projection gerekirdi.

3. Toplama, instance bilgisi ile row bilgisini aynı embedding uzayında birleştirir.
   Böylece attention katmanları tek bir vektör üzerinden hem “hangi lane?” hem “hangi row?” bilgisini görür.

4. Bu yapı ayrıştırılabilir ama birlikte işlenebilir bir temsil üretir.
   Aynı row token farklı instance tokenlarla birleşince farklı lane-row tokenları oluşur.

Basit benzetme:

```text
instance_token = öğrencinin kimliği
row_token      = sınavdaki soru numarası
toplam token   = bu öğrencinin bu soruya vereceği cevap için temsil
```

Öğrenci kimliği tek başına yetmez; soru numarası da tek başına yetmez. İkisi birlikte anlamlıdır.

### 7.5 Ordered row token ne demek?

Row tokenlar rastgele bir set değildir. Her biri belirli bir `y` satırına karşılık gelir:

```text
row_0, row_1, row_2, ..., row_159
```

Bu sıra geometrik olarak anlamlıdır:

- `row_150`, görüntünün altına yakındır.
- `row_20`, görüntünün üstüne yakındır.
- komşu row’lar fiziksel olarak komşu y koordinatlarına denk gelir.

Bu yüzden model “set” değil “ordered sequence” gibi davranır. Lane eğrisi boyunca süreklilik öğrenebilir.

---

## 8. Decoder katmanının iç yapısı

Her structured decoder katmanı üç ana attention işlemi yapar:

```text
1. Row-local cross-attention
2. Inter-instance attention
3. Intra-lane vertical attention
```

Son modelde bu katmandan `4` tane vardır:

```text
structured_query.num_layers = 4
```

### 8.1 Row-local cross-attention

Amaç: Her row token’ın görüntünün aynı yatay satırındaki feature’lara bakmasıdır.

Kod seviyesinde:

```python
q = row_tokens for a fixed row
key/value = image features from the same row
cross_attn(q, key, value)
```

Şekil:

```text
row r için:

lane token 0 ┐
lane token 1 ├── attention ──> image feature row r, all x positions
lane token 2 ┤
...          │
lane token N ┘
```

Yani her row için model yatay eksende arama yapar:

```text
Bu row’da lane çizgisi hangi x konumunda olabilir?
```

Bu, lane detection için güçlü bir prior’dır. Çünkü her row’da lane’in esas bilinmeyeni çoğunlukla `x` koordinatıdır.

### 8.2 Inter-instance attention

Amaç: Aynı row’daki farklı lane adaylarının birbirinden haberdar olmasıdır.

Örneğin aynı satırda:

- sol lane,
- orta lane,
- sağ lane

birbirine yakın olabilir. Modelin aynı lane’i iki kez üretmemesi gerekir.

Inter-instance attention şunu sağlar:

```text
Aynı row’daki farklı lane candidate tokenları birbirleriyle konuşur.
```

Şema:

```text
row r:

lane_0_token_r  <──> lane_1_token_r <──> lane_2_token_r <──> ...
```

Son modelde instance’lar `4` gruba ayrılmıştır:

```text
num_groups = 4
num_instances = 32
group size = 8
```

Gruplama sebebi, training sırasında grouped one-to-many assignment kullanmamızdır. Böylece her grup kendi içinde öğrenir; gruplar birbirini aşırı bastırmaz. Bu, recall ve convergence için kullanılmıştır.

### 8.3 Intra-lane vertical attention

Amaç: Aynı lane’in farklı row’ları arasında süreklilik kurmaktır.

Bir lane fiziksel olarak düzgün devam eder. Row 80’deki x konumu, row 81 ve row 82’den tamamen bağımsız değildir.

Bu nedenle model aynı lane’in row tokenlarını birbirine bağlar:

```text
lane i:

row_0_token  <──> row_1_token <──> ... <──> row_159_token
```

Bu attention sayesinde model:

- eğrilik,
- süreklilik,
- kaybolup tekrar görünen lane,
- perspektif davranışı

gibi bilgileri öğrenebilir.

### 8.4 Feed-forward network

Her attention bloğundan sonra klasik transformer FFN kullanılır:

```text
Linear -> GELU -> Dropout -> Linear
```

Bu, attention ile toplanan bilgiyi nonlineer olarak işler.

---

## 9. Decoder’ın çıktı başları

Structured decoder sonunda elimizde şu tensor vardır:

```text
row_tokens: [B, N, R, C]
```

Burada:

- `B`: batch size
- `N`: lane instance sayısı, son modelde 32
- `R`: row sayısı, son modelde 160
- `C`: channel, son modelde 256

Bu tokenlardan farklı çıktılar üretilir.

### 9.1 Row-wise x distribution

Her lane-row token için bir x dağılımı tahmin edilir:

```python
row_x_logits = Linear(row_tokens)
```

Tensor şekli:

```text
row_x_logits: [B, N, R, x_bins]
```

Son model:

```text
[B, 32, 160, 800]
```

Bu şu demektir:

```text
Her lane adayı,
her row için,
800 yatay bin arasından lane x konumunu seçer.
```

### 9.2 Expected x

Model doğrudan tek bir `x` sayısı üretmez. Önce 800 bin üzerinde dağılım üretir. Sonra bu dağılımın beklenen değeri alınır:

```python
probs = softmax(row_x_logits)
expected_bin = sum(probs * bin_index)
pred_x = expected_bin * bin_width
```

Son modelde:

```text
bin_width = input_w / x_bins = 1600 / 800 = 2 px
```

Yani beklenen bin değeri piksel koordinatına çevrilir.

### 9.3 Lane-level query

Existence, range ve quality gibi lane-level kararlar row tokenların tamamından çıkarılır.

Kodda:

```python
lane_query =
  LayerNorm(
    mean(row_tokens over rows)
    + max(row_tokens over rows)
    + instance_token
  )
```

Mantık:

- `mean`: lane’in genel ortalama bilgisini taşır.
- `max`: belirgin lokal aktivasyonları yakalar.
- `instance_token`: lane adayının kimliğini korur.

Bu `lane_query` şu çıktıları üretir:

```text
exist_logits   -> lane var mı yok mu?
range_norm     -> lane'in dikey başlangıç/bitiş aralığı
quality_logits -> tahmin kalitesi
```

### 9.4 Existence head

Existence head iki sınıflı karar verir:

```text
lane / no-lane
```

Neden gerekir?

Model 32 lane adayı üretir, ama gerçek görüntüde genellikle 2-4 lane vardır. Geri kalan adaylar bastırılmalıdır.

### 9.5 Range head

Lane her row’da görünmez. Örneğin bazı lane’ler görüntünün sadece alt yarısında vardır.

Range head şunu tahmin eder:

```text
y_start, y_end
```

Bu bilgi lane’in hangi dikey bölgede geçerli olduğunu modellemeye yarar.

### 9.6 Quality head

Quality head tahmin edilen lane’in geometrik kalitesiyle ilişkilidir. Eğitimde matched lane’ler için LineIoU benzeri hedeflerle öğrenir.

Inference sırasında skor şu mantıkla kullanılabilir:

```text
final_score = existence_score * quality_score^alpha
```

Burada `alpha`, quality’nin skora ne kadar etki edeceğini belirler. Bu değer final protokolde validation set üzerinden seçilmelidir.

---

## 10. Neden DFL kullanılıyor?

DFL = Distribution Focal Loss.

Buradaki amaç, row-wise x tahminini sadece regresyon gibi değil, dağılım öğrenme problemi gibi ele almaktır.

### 10.1 Problem: expected value tek başına yeterli değil

Model 800 bin üzerinde olasılık dağılımı üretir. Sonra expected value alınır.

Fakat expected value tek başına bazı belirsizlikleri gizleyebilir.

Örnek:

```text
GT bin = 30

Model A:
bin 30'a %100 olasılık verir.
expected = 30

Model B:
bin 20'ye %50, bin 40'a %50 verir.
expected = 30
```

İkisi de expected value açısından doğru görünür. Ama Model B aslında emin değildir ve dağılımı yanlıştır.

SmoothL1 sadece expected x’e bakarsa bu farkı yeterince cezalandırmayabilir.

### 10.2 DFL ne yapıyor?

DFL, ground truth x değerini en yakın iki bin arasında soft target olarak verir.

Örnek:

```text
GT x, bin 30.25'e denk gelsin.

target:
bin 30 -> 0.75
bin 31 -> 0.25
```

Loss:

```text
-(0.75 * log P(bin30) + 0.25 * log P(bin31))
```

Bu sayede model:

- doğru konum etrafında keskin dağılım üretmeyi öğrenir,
- belirsiz ve yayvan dağılımlardan uzaklaşır,
- x lokalizasyonunu daha precise hale getirir.

### 10.3 DFL ana katkı mı?

Hayır.

Bu proje açısından ana katkı:

```text
instance token + ordered row token structured decoder
```

DFL ise bu temsilin doğal lokalizasyon supervision’ıdır. Paper’da ana hikaye DFL üzerine kurulursa çalışma zayıflar; çünkü DFL genel object detection literatüründe bilinen bir fikirdir.

Doğru konumlandırma:

```text
Ana katkı: structured row-token decoder
Yardımcı teknik: row-wise distribution / DFL supervision
```

---

## 11. Training: model nasıl öğreniyor?

Training sırasında her görüntüde model 32 lane adayı üretir. Ground truth ise genellikle daha az sayıda lane içerir.

Bu nedenle önce hangi prediction’ın hangi GT lane ile eşleşeceğini belirlemek gerekir.

### 11.1 Matcher

Matcher’ın görevi:

```text
prediction lane candidates <-> ground truth lanes
```

eşleştirmesini yapmaktır.

Bizde matcher maliyeti şu parçalardan oluşur:

```text
cost =
  lambda_obj      * objectness cost
  + lambda_point  * row-wise x distance
  + lambda_range  * vertical range distance
  + lambda_line_iou * line IoU cost
```

Son config:

```text
lambda_obj = 2.0
lambda_point = 5.0
lambda_range = 1.0
lambda_line_iou = 1.0
```

### 11.2 Grouped one-to-many assignment

Son modelde:

```text
assignment = grouped_one_to_many
num_groups = 4
num_instances = 32
```

32 prediction slot 4 gruba ayrılır:

```text
group 1: slot 0-7
group 2: slot 8-15
group 3: slot 16-23
group 4: slot 24-31
```

Her grup kendi içinde Hungarian matching yapar.

Bunun amacı:

- training sırasında her GT lane’in birden fazla grupta pozitif örnek alabilmesi,
- convergence’ın kolaylaşması,
- recall’ın artmasıdır.

Risk:

- one-to-many assignment fazla pozitif davranışa yol açabilir,
- duplicate / false positive riski artabilir.

Bu yüzden inference sırasında threshold + NMS + top-k gerekir.

### 11.3 Existence loss

Modelin 32 adayından sadece eşleşenler lane olarak işaretlenir. Geri kalanı no-lane kabul edilir.

Son modelde focal loss kullanılır:

```text
exist_loss_type = focal
focal_alpha = 0.25
focal_gamma = 2.0
```

Focal loss zor örneklere daha çok ağırlık verir. Lane/no-lane dengesizliği için uygundur.

### 11.4 Point loss

Matched prediction ile GT lane arasında row-wise x farkı hesaplanır:

```text
SmoothL1(pred_x_rows, gt_x_rows)
```

Sadece valid row’lar hesaba katılır.

### 11.5 LineIoU loss

Lane sadece nokta nokta doğru değil, çizgi olarak da iyi olmalıdır. LineIoU loss, predicted ve GT lane çizgilerinin row-wise overlap’ini ölçer.

Bu, CULane resmi değerlendirmesine daha yakın bir sinyal verir.

### 11.6 Range loss

Predicted vertical range ile GT vertical range karşılaştırılır:

```text
SmoothL1(pred_range, gt_range)
```

### 11.7 Smoothness loss

Lane çizgileri fiziksel olarak çoğu zaman pürüzsüzdür. Smoothness loss, row’lar arasındaki aşırı zigzag davranışı azaltır.

Bu loss’un ağırlığı küçük tutulur:

```text
w_smooth = 0.05
```

Çünkü aşırı smoothness gerçek eğrileri de bastırabilir.

### 11.8 Quality loss

Quality head matched prediction’ların geometrik kalitesini öğrenir. Hedef, row-wise overlap / line quality’den türetilir.

Amaç:

- iyi lokalize edilmiş lane’lere yüksek kalite,
- kötü lane’lere düşük kalite vermek.

### 11.9 Segmentation auxiliary loss

Modelin ana çıktısı segmentation değildir. Fakat training sırasında feature extractor’ın lane bölgelerini daha iyi öğrenmesi için auxiliary segmentation head kullanılır.

Son config:

```text
seg_aux.enabled = true
w_seg = 1.0
```

Bu head inference’ın ana çıktısı değildir; training supervision sağlar.

### 11.10 Centerline auxiliary loss

Centerline auxiliary head row/x grid üzerinde lane merkez haritası üretir. Bu da FPN feature’larını lane çizgilerine duyarlı hale getirmek için kullanılır.

Son config:

```text
centerline_aux.enabled = true
w_centerline = 0.25
```

---

## 12. Inference: final lane nasıl seçiliyor?

Model inference sırasında 32 lane adayı üretir:

```text
32 candidates
each candidate has 160 row-wise x values
```

Sonra şu adımlar uygulanır:

### 12.1 Skor hesaplama

Her lane için existence score ve quality score kullanılır.

Genel mantık:

```text
score = lane_existence_score * quality_score^quality_power
```

`quality_power` validation set üzerinde seçilmelidir.

### 12.2 Score threshold

Düşük skorlu lane adayları atılır:

```text
score < threshold -> discard
```

Bu threshold final paper’da test üzerinde seçilmemelidir. Validation’da seçilip testte sabit uygulanmalıdır.

### 12.3 Lane NMS

Benzer lane adayları birbirini tekrar ediyorsa NMS ile bastırılır.

Son config:

```text
lane_nms_distance_thresh_px = 20.0
lane_nms_min_overlap_points = 5
```

### 12.4 Top-k

CULane’de genellikle en fazla 4 lane raporlanır:

```text
top_k = 4
```

Final output:

```text
image -> up to 4 lane predictions
```

---

## 13. Neden bu mimari mantıklı?

### 13.1 Lane geometrisi doğal olarak row-wise’dır

Lane detection’da her y satırında x koordinatı tahmin etmek doğal bir formülasyondur.

Structured decoder bu prior’ı doğrudan modele koyar:

```text
Her lane için her row’a ayrı temsil ver.
```

Bu, tamamen global bir lane vektöründen daha açıklanabilir ve daha yapılandırılmıştır.

### 13.2 Instance ve geometry ayrımı

Lane detection iki ayrı problemi içerir:

```text
1. Hangi lane adayları var?
2. Her lane’in satır satır geometrisi nerede?
```

Biz bunu mimaride ayırıyoruz:

```text
instance token -> lane identity / objectness / global routing
row token      -> local row geometry
```

Bu ayrım paper’ın ana mimari argümanıdır.

### 13.3 Cross-attention row-local yapılıyor

Her row token tüm görüntüye bakmak yerine kendi satırındaki horizontal evidence’a bakar.

Bu hem hesaplamayı daha kontrollü yapar hem de inductive bias verir:

```text
Bu row’da lane nerede?
```

### 13.4 Vertical attention süreklilik sağlar

Row’lar tamamen bağımsız tahmin edilseydi çizgi zigzag olurdu.

Intra-lane vertical attention row’ları birbirine bağlar:

```text
Bu lane’in üst ve alt row’ları birlikte düşünülür.
```

Bu, eğri sürekliliği için kritiktir.

### 13.5 DFL hassas lokalizasyon sağlar

Row tokenlar her row için x dağılımı ürettiğinden, DFL bu dağılımı doğru bin etrafında keskinleştirir.

Bu özellikle yüksek çözünürlüklü ayarda önemlidir:

```text
1600 px width / 800 bins = 2 px precision
```

---

## 14. Modelin eski deneylerden farkı

Projede daha önce orthogonal evidence / verifier gibi farklı fikirler denendi. Bu dokümanın ana modeli onlardan farklıdır.

### 14.1 Orthogonal evidence neydi?

Önceki denemelerde modelin draft lane tahmininin etrafından lokal görüntü evidence’ı alınarak geometriyi düzeltmeye çalıştık.

Problem:

- Eğer draft yanlış yere, örneğin bariyere oturduysa, evidence da yanlış yerden geliyordu.
- Lokal evidence bazen lane ile bariyeri ayırmakta yetersiz kalıyordu.
- Geometriye doğrudan müdahale jitter ve FP artışı yaratabiliyordu.

### 14.2 Structured row-token decoder neyi farklı yapıyor?

Yeni model geometriyi sonradan yama gibi düzeltmiyor. Temsili baştan değiştiriyor.

Eski fikir:

```text
Önce lane tahmin et, sonra evidence ile düzelt.
```

Yeni fikir:

```text
Lane temsilini baştan instance + ordered row tokens olarak kur.
Her row token doğrudan image evidence okuyarak x dağılımını üretir.
```

Bu yüzden mevcut ana katkı orthogonal evidence değil, structured representation’dır.

---

## 15. Tensor akışı: kodla uyumlu özet

Son modelde forward pass şu şekildedir.

### 15.1 Encoder

```python
feats = ResNet34Backbone(images)
fpn = SimpleFPN(feats)
features = Conv1x1(fpn)
```

Şekil:

```text
images:   [B, 3, 640, 1600]
features: [B, 256, Hf, Wf]
```

Sonra structured head içinde:

```python
features = interpolate(features, size=(num_rows, evidence_x_bins))
```

Şekil:

```text
row features: [B, 160, 400, 256]
```

### 15.2 Token initialization

```python
instance = Embedding(32, 256)
row = Embedding(160, 256)

row_tokens = instance[:, None, :] + row[None, :, :]
```

Şekil:

```text
row_tokens: [32, 160, 256]
batch sonrası:
row_tokens: [B, 32, 160, 256]
```

### 15.3 Structured decoder layers

4 kez:

```python
row_tokens = RowAwareCrossAttentionLayer(row_tokens, row_features)
```

Her layer içinde:

```text
row-local cross-attention
inter-instance attention
intra-lane vertical attention
FFN
```

### 15.4 Prediction heads

```python
row_x_logits = Linear(row_tokens)
pred_x_rows = soft_expected_x(row_x_logits)

lane_query = mean(row_tokens) + max(row_tokens) + instance_token

exist_logits = ExistHead(lane_query)
range_norm = RangeHead(lane_query)
quality_logits = QualityHead(lane_query)
```

Output:

```text
exist_logits:   [B, 32, 2]
row_x_logits:   [B, 32, 160, 800]
pred_x_rows:    [B, 32, 160]
range_norm:     [B, 32, 2]
quality_logits: [B, 32]
```

---

## 16. Basit örnek üzerinden anlatım

Bir görüntüde 3 gerçek lane olduğunu düşünelim.

Model 32 lane adayı üretir:

```text
candidate 0
candidate 1
...
candidate 31
```

Her candidate için 160 row vardır:

```text
candidate 7:
  row 0   -> x distribution
  row 1   -> x distribution
  ...
  row 159 -> x distribution
```

Eğitim sırasında matcher şunu yapar:

```text
candidate 4  -> GT lane 0
candidate 12 -> GT lane 1
candidate 21 -> GT lane 2
diğerleri    -> no-lane
```

Matched candidate’lar:

- x coordinate loss alır,
- DFL loss alır,
- range loss alır,
- LineIoU loss alır,
- existence “lane” hedefi alır.

Unmatched candidate’lar:

- existence “no-lane” hedefi alır.

Inference sırasında:

- existence/quality skoru düşük olanlar atılır,
- benzer olanlar NMS ile temizlenir,
- en iyi 4 lane raporlanır.

---

## 17. Hocanın sorabileceği kritik sorular ve cevapları

### Soru 1: “Instance token ile row token niye toplanıyor?”

Çünkü her lane-row token’ın iki bilgiye aynı anda ihtiyacı var:

```text
hangi lane adayı?
hangi y satırı?
```

Toplama şu temsili üretir:

```text
token(i, r) = lane identity i + row position r
```

Bu transformer’larda pozisyon embedding’i ekleme mantığıyla aynıdır. Concat yapsaydık boyut büyür ve ek projection gerekirdi. Toplama daha temiz, daha ucuz ve standarttır.

### Soru 2: “Row token görüntüden gelmiyorsa nasıl lane öğreniyor?”

Row token başlangıçta learnable bir sorgudur. Görüntü bilgisini cross-attention ile alır.

Yani token görüntüden başlamaz; görüntüye soru sorar:

```text
Ben 50. row’daki lane temsilcisiyim. Bu row’da lane nerede?
```

Feature map’ten aldığı bilgiyle güncellenir.

### Soru 3: “Bu segmentation değil mi?”

Hayır. Model piksel maskesi üretmiyor. Ana çıktı lane başına row-wise x koordinatlarıdır.

Segmentation sadece auxiliary training loss olarak kullanılır. Final prediction lane çizgileridir.

### Soru 4: “Bu anchor-based mi?”

Klasik line-anchor anlamında hayır.

Model önceden tanımlı lane çizgisi şablonlarını düzeltmez. Lane adayları learnable instance tokenlarla temsil edilir.

Ama row-grid ve x-bin discretization vardır. Bu yüzden paper’da “completely anchor-free” yerine daha dikkatli ifade kullanılmalıdır:

```text
line-anchor-free
structured row-wise decoder
```

### Soru 5: “Neden 32 lane adayı var, görüntüde 4 lane yok mu?”

32 aday training ve matching için kullanılır. Her görüntüde hepsi final output olmaz.

Inference sonunda:

```text
score threshold + NMS + top_k=4
```

ile genellikle en iyi 4 lane raporlanır.

### Soru 6: “DFL olmadan olmaz mı?”

Olur, ama DFL row-wise x dağılımını daha iyi kalibre eder. Expected x tek başına dağılımın yayvanlığını cezalandırmayabilir.

DFL özellikle yüksek çözünürlüklü x-bin sisteminde lokalizasyonu keskinleştirir.

### Soru 7: “Cross-attention neden tüm görüntüye değil de row-local?”

Lane detection’da her row için ana bilinmeyen yatay x konumudur. Row-local attention, probleme uygun bir prior koyar:

```text
row r'daki token, row r'daki horizontal evidence'a baksın.
```

Tüm görüntüye bakmak daha genel ama daha gürültülü ve maliyetli olabilir.

### Soru 8: “Vertical attention olmazsa ne olur?”

Row’lar bağımsızlaşır. Bu durumda lane çizgisi satırdan satıra titreyebilir veya süreklilik kaybedebilir.

Vertical attention, aynı lane’in row’ları arasında bilgi paylaşımı sağlar.

### Soru 9: “Bu model neden klasik CNN head’den farklı?”

Klasik CNN head genellikle dense map üretir veya global feature’dan koordinat çıkarır. Burada ise explicit lane-instance ve row-level token state vardır.

Model sadece output grid üretmiyor; her lane’in her row’u için decoder içinde ayrı latent temsil tutuyor.

### Soru 10: “Paper’da asıl novelty nerede?”

Asıl novelty:

```text
2D lane detection için instance token + ordered row token şeklinde yapılandırılmış decoder temsili.
```

DFL, yüksek çözünürlük, FPN256 gibi parçalar destekleyici unsurlardır. Ana hikaye bunlara indirgenmemelidir.

---

## 18. Güçlü yanlar

Modelin güçlü yanları:

1. Lane geometrisine uygun inductive bias verir.
2. Whole-lane query’ye göre daha açıklanabilir bir temsil sunar.
3. Row-wise lokalizasyonu doğal şekilde yapar.
4. Transformer cross-attention ile image feature’lardan adaptif bilgi okur.
5. DFL ile x tahminini dağılımsal ve precise hale getirir.
6. ResNet-34 gibi standart backbone ile rekabetçi skor verir.
7. Ablation için temiz parçalara ayrılabilir:
   - instance token,
   - row token,
   - DFL,
   - decoder layer sayısı,
   - row/bin çözünürlüğü,
   - FPN kapasitesi.

---

## 19. Zayıf yanlar ve açık riskler

Bu model kusursuz değildir. Paper veya tez yazımında bu noktalar bilinçli yönetilmelidir.

### 19.1 Scoring / false positive riski

Model güçlü row-wise prior nedeniyle lane olmayan yerlerde de lane benzeri yapı bulmaya çalışabilir.

Özellikle:

- crossroad,
- no-line,
- night,
- heavy shadow

gibi kategorilerde false positive riski vardır.

### 19.2 One-to-many training duplicate riski

Grouped one-to-many assignment recall ve convergence için faydalıdır; fakat duplicate prediction ve precision yönetimini zorlaştırabilir.

Bu yüzden postprocess ve threshold protokolü önemlidir.

### 19.3 Test threshold tuning riski

Test set üzerinde threshold seçilirse akademik olarak zayıf olur.

Doğru protokol:

```text
val set üzerinde checkpoint + threshold + quality_power seç
test seti sadece final raporlama için kullan
```

### 19.4 Yakın literatür riski

Bu fikir tamamen boşlukta değildir. Yakın çalışmalar:

- UFLD / UFLDv2: row-wise classification
- CondLaneNet: instance-first row-wise lane shape
- CondLSTR: transformer query + row-wise heatmap
- LaneFormer: transformer ve row/column attention
- MapTR: instance + point-level hierarchical queries

Bu yüzden contribution dikkatli yazılmalıdır:

```text
"Row-wise lane detection’ı ilk biz yaptık" denmemeli.
"Transformer query’yi ilk biz kullandık" denmemeli.
"2D lane detection için lane identity ve ordered row geometry tokenlarını decoder state olarak açıkça ayırıyoruz" denmeli.
```

---

## 20. Paper için önerilen yöntem anlatımı

Paper’da yöntem şu sırayla anlatılmalı:

### 20.1 Problem

```text
Existing holistic lane queries compress the full lane geometry into a single vector.
This is structurally weak for long, thin, row-wise lane curves.
```

### 20.2 Öneri

```text
We factorize each lane into:
1. an instance token for lane identity,
2. ordered row tokens for geometry.
```

### 20.3 Decoder

```text
Each row token performs row-local cross-attention over image features,
interacts with other lane candidates at the same row,
and exchanges vertical context with other rows of the same lane.
```

### 20.4 Output

```text
Each row token predicts a discrete x distribution.
The lane-level token aggregates row information for existence, quality and valid range.
```

### 20.5 Training

```text
Hungarian/grouped assignment aligns candidates with ground truth lanes.
Matched lanes receive point, DFL, LineIoU, range and quality losses.
Unmatched lanes receive no-lane existence supervision.
Auxiliary segmentation/centerline losses improve image features.
```

### 20.6 Claim

```text
The structured factorization gives better localization and stronger inductive bias
than an unstructured query/MLP lane decoder under the same backbone.
```

---

## 21. Önerilen şekiller

Hocaya veya paper’a koymak için aşağıdaki figürler faydalı olur.

### Figure 1: Genel mimari

```text
Image -> ResNet34 -> FPN -> Structured Row-Token Decoder -> Lane outputs
```

### Figure 2: Token factorization

```text
instance_token_i + row_token_r = lane-row token(i,r)
```

### Figure 3: Decoder layer

```text
row-local cross-attention
inter-instance attention
vertical intra-lane attention
FFN
```

### Figure 4: Row-wise x distribution

Her row için x-bin probability heatmap:

```text
row axis vs x-bin axis
```

### Figure 5: Ablation diagram

```text
unstructured query
-> +instance-row factorization
-> +DFL
-> +FPN / resolution
```

### Figure 6: Failure analysis

Yan yana:

- başarılı crowded/curve örnekleri,
- başarısız cross/no-line/night örnekleri.

---

## 22. Ablation’da kanıtlanması gerekenler

Bu mimarinin akademik olarak güçlü olması için şu sorulara cevap verilmelidir.

### 22.1 Structured decoder gerçekten işe yarıyor mu?

Karşılaştırma:

```text
Unstructured query baseline
vs
Instance + ordered row token decoder
```

Aynı backbone, aynı input, aynı schedule ile yapılmalıdır.

### 22.2 Row token katkısı var mı?

Ablation:

```text
instance token only
vs
instance + row token
```

### 22.3 DFL ne kadar katkı veriyor?

Ablation:

```text
L1 only
vs
L1 + DFL
```

### 22.4 Decoder layer sayısı ne kadar önemli?

Ablation:

```text
1 layer
2 layers
4 layers
6 layers
```

Beklenti:

- 1 -> 4 arası artış olabilir.
- 4 -> 6 küçük artış veya plateau olabilir.

### 22.5 Slot sayısı etkisi nedir?

Ablation:

```text
20 slots
32 slots
64 slots
```

Slot sayısı artarsa recall artabilir ama FP/duplicate riski de artar.

### 22.6 Resolution ve bin sayısı etkisi nedir?

Ablation:

```text
800x288 / 200 bins
1024x384 / 512 bins
1600x640 / 800 bins
```

Burada dikkat:

Kazanç sadece daha büyük inputtan mı geliyor, yoksa structured decoder aynı koşulda da kazandırıyor mu?

Bu ayrım net gösterilmelidir.

---

## 23. Son model konfigürasyonu

Şu anki güçlü model:

```yaml
model:
  input_h: 640
  input_w: 1600
  fpn_channels: 256
  dim: 256
  num_slots: 32
  num_rows: 160
  x_bins: 800
  structured_query:
    enabled: true
    num_instances: 32
    num_groups: 4
    num_layers: 4
    num_heads: 8
    ff_dim: 1024
    evidence_x_bins: 400
```

Training:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
  effective_batch_size: 16
  max_iters: 278000
  amp: true
  channels_last: true
```

Loss:

```yaml
loss:
  w_exist: 2.0
  w_point: 5.0
  w_range: 1.0
  w_smooth: 0.05
  w_line_iou: 2.0
  w_seg: 1.0
  w_centerline: 0.25
  w_quality: 0.5
  w_row_dfl: 0.5
```

Postprocess:

```yaml
postprocess:
  lane_nms_distance_thresh_px: 20.0
  lane_nms_min_overlap_points: 5
  top_k: 4
```

---

## 24. Mimariyi tek paragrafta anlatmak gerekirse

Model, input görüntüyü ResNet-34 ve FPN ile 256 kanallı feature map’e dönüştürür. Klasik lane query yaklaşımında her lane tek bir global vektörle temsil edilirken, bu model her lane adayını bir instance token ve ona bağlı sıralı row token dizisiyle temsil eder. Instance token lane kimliğini taşır; row tokenlar lane’in farklı y satırlarındaki geometrisini taşır. Her lane-row token, row-local cross-attention ile görüntünün aynı yatay satırındaki feature’lardan bilgi okur; aynı satırdaki farklı lane adayları inter-instance attention ile birbirinden haberdar olur; aynı lane’in farklı row’ları vertical self-attention ile süreklilik ve eğrilik bilgisini paylaşır. Decoder sonunda her row token 800 yatay bin üzerinde bir x dağılımı üretir; bu dağılımdan expected x koordinatı çıkarılır ve DFL ile doğru bin çevresinde keskinleşmesi sağlanır. Row tokenların global havuzlanmasıyla lane-level existence, range ve quality skorları üretilir. Eğitimde grouped matching ile prediction’lar GT lane’lerle eşleştirilir; point, LineIoU, range, existence, quality, DFL ve auxiliary segmentation/centerline loss’ları birlikte kullanılır.

---

## 25. En kısa teknik katkı cümlesi

```text
We propose a structured instance-to-row decoder for 2D lane detection, where each lane is represented by a global instance token and an ordered sequence of row-level geometry tokens. The row tokens perform row-local cross-attention over image features and predict row-wise x-coordinate distributions supervised by soft-bin DFL.
```

Türkçe:

```text
2D lane detection için her lane’i global bir instance token ve sıralı row-level geometri tokenlarıyla temsil eden structured instance-to-row decoder öneriyoruz. Row tokenlar görüntü feature’ları üzerinde row-local cross-attention yapar ve soft-bin DFL ile eğitilen row-wise x dağılımları üretir.
```

---

## 26. Danışman toplantısı için konuşma sırası

Hocaya anlatırken şu sırayı öneririm:

1. Önce lane representation’ı anlat:
   ```text
   Biz lane’i row-wise x koordinatları olarak temsil ediyoruz.
   ```

2. Sonra eski sorunu anlat:
   ```text
   Tek lane query bütün geometriyi tek vektöre sıkıştırıyor.
   ```

3. Sonra ana fikri söyle:
   ```text
   Lane kimliği ve row geometrisini ayırıyoruz.
   ```

4. Sonra toplama mantığını anlat:
   ```text
   instance token + row token = belirli lane’in belirli row’daki temsilcisi.
   ```

5. Sonra attention’ı basitleştir:
   ```text
   Row token görüntüye soru soruyor: bu row’da lane x nerede?
   ```

6. Sonra vertical attention’ı anlat:
   ```text
   Aynı lane’in row’ları birbirleriyle konuşuyor, böylece çizgi sürekliliği öğreniliyor.
   ```

7. Sonra output’u anlat:
   ```text
   Her row için 800-bin x dağılımı üretiyoruz; DFL ile doğru x etrafında keskinleştiriyoruz.
   ```

8. Sonra training’i anlat:
   ```text
   Matcher prediction’ları GT lane’lerle eşliyor; matched olanlar geometri loss’u, unmatched olanlar no-lane loss’u alıyor.
   ```

9. Son olarak contribution’ı netleştir:
   ```text
   Katkı DFL değil; lane’i instance + ordered row tokens olarak decoder içinde temsil etmek.
   ```

---

## 27. Dürüst sonuç

Bu mimari rastgele eklenmiş bir transformer bloğu değildir. Lane detection probleminin doğal row-wise yapısını transformer decoder içine açıkça taşıyan bir tasarımdır.

En savunulabilir argüman:

```text
Tek lane query temsili, lane geometrisi için fazla sıkıştırılmıştır.
Biz lane’i instance identity ve ordered row geometry olarak ayrıştırıyoruz.
Bu ayrıştırma hem mimari olarak açıklanabilir hem de deneysel olarak unstructured baseline’a göre daha iyi sonuç verir.
```

En zayıf argüman:

```text
DFL ekledik, yüksek çözünürlük kullandık, skor arttı.
```

Paper veya sunum bu ikinci argümana kayarsa çalışma zayıf görünür. Bu nedenle bütün anlatı structured instance-to-row representation etrafında kurulmalıdır.

