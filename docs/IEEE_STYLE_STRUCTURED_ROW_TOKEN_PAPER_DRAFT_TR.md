# Structured Row-Token Decoder for 2D Lane Detection

IEEE benzeri Türkçe makale / teknik rapor taslağı  
Son güncelleme: 2026-07-03

> Not: Bu doküman kod dokümanı değildir. Amaç, danışman hocaya mimari fikri, motivasyonu, model akışını, deney planını ve paper anlatısını açık şekilde sunmaktır. Kod seviyesine inmez; kopyalanabilir implementation detayı vermez. İstenirse bu metin daha sonra doğrudan İngilizce IEEE / LaTeX makale formatına çevrilebilir.

---

## Başlık önerileri

**Seçenek 1**

Structured Row-Token Decoding for Line-Anchor-Free 2D Lane Detection

**Seçenek 2**

Instance-to-Row Token Decoding for Structured Lane Geometry Modeling

**Seçenek 3**

Learning Lane Geometry with Ordered Row Tokens for 2D Lane Detection

Bu üç başlık içinde en güvenli olan birincisidir. Çünkü hem ana mimari fikri, hem de line-anchor-free konumlandırmayı taşır.

---

# Abstract

Lane detection, autonomous driving perception systems için temel bir problemdir. Mevcut yöntemlerin önemli bir bölümü lane geometrisini ya yoğun segmentation haritalarıyla, ya önceden tanımlanmış line anchor’larla, ya da tüm lane’i tek bir global query vektörüne sıkıştıran transformer tabanlı yapılarla modellemektedir. Ancak lane çizgileri uzun, ince, perspektiften etkilenen ve doğal olarak satır-satır tanımlanabilen geometrik yapılardır. Bu nedenle tüm lane geometrisini tek bir vektöre sıkıştırmak, özellikle eğimli, kalabalık, gece, gölge ve kısmen görünmeyen sahnelerde yetersiz bir temsil oluşturabilir.

Bu çalışmada, 2D lane detection için **structured row-token decoder** adı verilen yeni bir temsil öneriyoruz. Temel fikir, her lane adayını iki seviyede ayrıştırmaktır: global bir **instance token** lane kimliğini ve varlık bilgisini temsil ederken, sıralı **row token** dizisi aynı lane’in farklı yatay satırlardaki geometrisini temsil eder. Böylece model, lane’i tek bir bütüncül vektör olarak değil, lane kimliğiyle ilişkilendirilmiş satır-bazlı geometri dizisi olarak işler. Her row token görüntü özelliklerinden kendi satırına ait yatay kanıtları okur ve o satırdaki lane x-konumu için ayrık bir olasılık dağılımı üretir. Bu dağılım soft-bin distributional supervision ile eğitilerek lokalizasyon hassasiyeti artırılır.

Önerilen yapı, ResNet-34 backbone üzerinde CULane benchmark’ında güçlü ve rekabetçi sonuçlar üretmektedir. Çalışmanın ana katkısı yalnızca yüksek çözünürlük veya distributional loss kullanımı değil; lane kimliği ile row-wise geometriyi decoder seviyesinde açıkça ayrıştıran structured representation’dır. Kapsamlı ablation çalışmalarıyla instance token, ordered row token, distributional row-wise localization, decoder derinliği ve çözünürlük etkilerinin ayrı ayrı incelenmesi hedeflenmektedir.

---

# Index Terms

Lane Detection, Autonomous Driving, Transformer Decoder, Row-wise Localization, Structured Representation, Instance Token, Row Token, Distributional Regression, CULane.

---

# I. Introduction

Lane detection, sürücüsüz araçlarda yol yapısının anlaşılması, şerit takip, lokalizasyon ve karar verme modülleri için temel bir perception problemidir. Kamera tabanlı lane detection sistemleri, özellikle düşük maliyetli ve gerçek zamanlı uygulanabilir olmaları nedeniyle geniş kullanım alanına sahiptir. Ancak problem görünenden daha zordur. Lane çizgileri:

- uzun ve ince yapılardır,
- perspektif nedeniyle görüntünün farklı bölgelerinde farklı ölçeklerde görünür,
- gölge, gece, parlama, kalabalık trafik ve yol bozulmaları nedeniyle kısmen kaybolabilir,
- bazı sahnelerde fiziksel olarak hiç bulunmayabilir,
- bariyer, yol kenarı, ok işareti veya gölge gibi lane-benzeri yapılarla karışabilir.

Bu nedenle lane detection yalnızca “çizgi var mı?” problemi değildir. Aynı zamanda doğru lane kimliğini bulma, geometrik sürekliliği koruma, yanlış lane-benzeri yapıları bastırma ve satır-bazlı hassas lokalizasyon üretme problemidir.

## A. Mevcut temsil biçimleri

Literatürde lane detection için farklı temsil biçimleri kullanılmıştır:

1. **Segmentation tabanlı temsil**  
   Her piksel için lane / background sınıflandırması yapılır. Bu yöntemler spatial detay açısından güçlüdür; ancak instance ayrımı, post-process ve gerçek zamanlılık açısından maliyetli olabilir.

2. **Anchor tabanlı temsil**  
   Önceden tanımlanmış lane şablonları kullanılır ve model bu şablonları düzeltir. Bu yaklaşım güçlü performans verebilir; fakat anchor tasarımı ve heuristic ayarlar modele bağımlıdır.

3. **Row-wise classification/regression**  
   Görüntü yatay satırlara bölünür ve her satırda lane’in x-konumu tahmin edilir. Lane’in kamera görüntüsündeki doğal yapısına uygundur.

4. **Query tabanlı transformer temsil**  
   Her lane adayı bir query vektörüyle temsil edilir. Query, görüntü feature’larından bilgi toplayarak lane geometrisini tahmin eder.

Bu çalışmadaki çıkış noktamız şudur: Query tabanlı yöntemler güçlüdür, ancak çoğu yaklaşımda bir lane’in tüm geometrisi tek bir query vektörüne sıkıştırılır. Bu, lane gibi uzun ve satır-satır değişen bir yapı için fazla yoğun ve açıklaması zor bir temsildir.

## B. Temel gözlem

Lane çizgileri 2D görüntüde doğal olarak row-wise şekilde ifade edilebilir:

```text
lane = her yatay satırdaki x konumlarının sıralı dizisi
```

Bu durumda bir lane’i tek bir vektörle temsil etmek yerine, lane’i şu iki bileşene ayırmak daha doğal görünmektedir:

```text
lane identity  +  row-wise geometry
```

Bu çalışmanın ana fikri budur.

---

# II. Background and Motivation

Bu bölüm, modelin neden böyle tasarlandığını açıklamak için gerekli temel kavramları verir.

## A. Lane geometrisi neden row-wise düşünülür?

Bir kamera görüntüsünde lane çizgisi çoğu zaman yukarıdan aşağıya devam eden bir eğridir. Eğer görüntüde belirli yatay satırlar seçilirse, lane’in her satırdaki yatay konumu ile tüm lane çizgisi yeniden oluşturulabilir.

Şekil 1 bu fikri gösterir.

```text
Şekil 1. Row-wise lane representation

     y=0    ------------------------------------------------
            |                                              |
            |                         x_1                  |
     y=1    ------------------------------------------------
            |                       x_2                    |
     y=2    ------------------------------------------------
            |                    x_3                       |
            |                                              |
     ...    ------------------------------------------------
            |           x_R                                |
     y=R    ------------------------------------------------

Lane = [x_1, x_2, x_3, ..., x_R]
```

Bu formülasyonun avantajı:

- Lane çizgisi doğrudan nokta dizisi olarak elde edilir.
- Eğri, polinom veya spline gibi tek bir parametre ailesine zorlanmaz.
- Görünmeyen satırlar valid mask ile dışarıda bırakılabilir.
- CULane gibi benchmark’ların değerlendirme mantığına uygundur.

## B. Tek query temsili neden sınırlı?

Transformer tabanlı basit bir lane detector’da her lane adayı bir global query vektörüyle temsil edilebilir:

```text
lane query  --->  bütün lane koordinatları
```

Bu durumda tek vektör şunların tamamını taşımak zorundadır:

- lane var mı?
- hangi lane?
- nerede başlıyor?
- nerede bitiyor?
- her satırdaki x konumu nedir?
- eğrilik nasıl?
- hangi kısımlar görünür?
- lane kalitesi nedir?

Bu yapı çalışabilir; fakat temsil açısından sıkıştırılmıştır. Bir lane geometrisi 100’den fazla satırda değişen koordinatlardan oluşuyorsa, bunu tek bir vektörde tutmak modelin öğrenmesini zorlaştırabilir.

## C. Önerilen ayrıştırma

Biz bu problemi şu şekilde ayrıştırıyoruz:

```text
global lane identity  -> instance token
row-wise geometry     -> ordered row tokens
```

Bu sayede:

- lane’in kimliği ile satır-bazlı geometrisi ayrılır,
- her row için ayrı fakat aynı lane’e bağlı latent temsil oluşur,
- model hem lokal detay hem de global süreklilik öğrenebilir.

---

# III. Proposed Method

Bu bölümde önerilen structured row-token decoder mimarisi açıklanır.

## A. Genel mimari

Modelin genel akışı Şekil 2’de gösterilmiştir.

```text
Şekil 2. Önerilen modelin genel akışı

┌──────────────────┐
│ Input image       │
│ H x W             │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CNN backbone      │
│ ResNet-34         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feature pyramid   │
│ multi-scale cues  │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────┐
│ Structured row-token        │
│ decoder                     │
│                            │
│ instance tokens             │
│ ordered row tokens          │
└────────┬───────────────────┘
         │
         ▼
┌────────────────────────────┐
│ Lane predictions            │
│ existence / range / quality │
│ row-wise x distributions    │
└────────────────────────────┘
```

Model üç ana bölümden oluşur:

1. **Image encoder**: Görüntüden feature map çıkarır.
2. **Structured row-token decoder**: Lane adaylarını instance ve row tokenlarla işler.
3. **Prediction heads**: Lane varlığı, kalite, dikey geçerlilik aralığı ve row-wise x konumlarını üretir.

## B. Instance token nedir?

Instance token, bir lane adayının global kimliğini temsil eden öğrenilebilir vektördür.

Basit ifade:

```text
instance token = "bu lane adayı kim?"
```

Bir görüntüde model sabit sayıda lane adayı üretir. Örneğin son modelde 32 lane adayı vardır. Bu, görüntüde 32 gerçek lane olduğu anlamına gelmez. Bunlar potansiyel lane slotlarıdır. Inference sonunda yalnızca yüksek skorlu ve birbirinden farklı olan birkaç lane raporlanır.

Instance token’ın görevi:

- lane adaylarını birbirinden ayırmak,
- her lane slotuna global bir rol vermek,
- lane-level kararlar için temel temsil sağlamak,
- existence, quality ve range gibi global çıktılara bilgi taşımaktır.

Örneğin:

```text
instance token 1 -> bir lane adayı
instance token 2 -> başka bir lane adayı
...
instance token N -> başka bir lane adayı
```

Bu tokenlar training sırasında öğrenilir. Başlangıçta özel bir anlamları yoktur; model eğitim sırasında bu tokenları farklı lane adaylarını temsil etmek için kullanmayı öğrenir.

## C. Row token nedir?

Row token, görüntüdeki belirli bir yatay satırın geometri temsilidir.

Basit ifade:

```text
row token = "bu y satırında lane geometrisini nasıl aramalıyım?"
```

Lane detection’da her lane çizgisi farklı y satırlarındaki x koordinatlarıyla tanımlanabilir. Bu nedenle her satır için ayrı bir temsil kullanmak doğaldır.

Row token’ın görevi:

- satır pozisyonu bilgisini taşımak,
- o satırdaki lane konumunu tahmin etmeye yardımcı olmak,
- farklı satırların farklı perspektif davranışlarını öğrenmek,
- lane’in yukarıdan aşağıya düzenli yapısını modellemektir.

Örneğin:

```text
row token 1   -> görüntünün üst tarafına yakın satır
row token 80  -> orta bölge
row token 160 -> görüntünün alt bölgesi
```

Alt bölgedeki lane çizgileri genellikle daha geniş aralıklı ve daha görünürdür. Üst bölgedeki lane çizgileri daha küçük, daha gürültülü ve perspektif nedeniyle daha sıkışık olabilir. Row tokenlar bu farklılığı öğrenilebilir hale getirir.

## D. Instance token ve row token birlikte nasıl lane-row temsilini oluşturur?

Önerilen modelde her lane adayı için her row’da ayrı bir latent temsil vardır.

Kavramsal olarak:

```text
lane-row token(i, r)
  =
lane identity information(i)
  +
row position / row geometry information(r)
```

Yani:

```text
belirli lane adayı + belirli row pozisyonu
= o lane’in o row’daki geometri temsilcisi
```

Şekil 3 bunu gösterir.

```text
Şekil 3. Instance-to-row token factorization

                 row token 1      row token 2      ...      row token R
                    │                │                         │
                    ▼                ▼                         ▼
instance 1 ──►  token(1,1)      token(1,2)      ...      token(1,R)

instance 2 ──►  token(2,1)      token(2,2)      ...      token(2,R)

...

instance N ──►  token(N,1)      token(N,2)      ...      token(N,R)
```

Bu tablodaki her hücre, belirli bir lane adayının belirli bir satırdaki geometri temsilidir.

Bu nokta önemlidir: Model sadece “row token” üretmez. Her row token lane identity ile birleşerek lane’e özel hale gelir. Böylece “80. satır” tüm lane’ler için aynı değildir; “lane 3’ün 80. satırı” ve “lane 7’nin 80. satırı” ayrı tokenlardır.

## E. Neden bu iki bilgi birleştiriliyor?

Bir row token tek başına yalnızca “hangi satır?” bilgisini taşır. Bir instance token tek başına yalnızca “hangi lane adayı?” bilgisini taşır. Lane detection için ikisi de tek başına yetersizdir.

Gerekli olan temsil şudur:

```text
hangi lane adayının, hangi satırdaki noktası?
```

Bu nedenle instance ve row bilgileri aynı token içinde birleştirilir.

CNN benzetmesi:

- CNN feature map’te bir aktivasyon hem channel bilgisini hem spatial lokasyon bilgisini taşır.
- Burada lane-row token da hem lane identity hem row position bilgisini taşır.

Transformer benzetmesi:

- NLP’de kelime embedding’i ile pozisyon embedding’i birleştirilir.
- Burada lane instance embedding’i ile row embedding’i birleştirilir.

Bu tasarım, lane geometrisini grid gibi değil, lane’e bağlı sıralı temsil olarak işler.

---

# IV. Structured Decoder Design

Structured decoder, lane-row tokenlarını görüntü feature’larıyla etkileştirerek günceller.

Her decoder bloğunda üç temel etkileşim vardır:

1. row-local image attention,
2. same-row lane interaction,
3. same-lane vertical interaction.

## A. Row-local image attention

Her lane-row token, görüntünün kendi satırındaki yatay feature’lara bakar.

Amaç:

```text
Bu row’da lane x konumu nerede olabilir?
```

Şekil 4:

```text
Şekil 4. Row-local attention

                         image feature row r
          x=1      x=2      x=3      ...      x=M
           │        │        │                 │
           ▼        ▼        ▼                 ▼
      ┌────────────────────────────────────────────┐
      │ horizontal visual evidence for row r        │
      └────────────────────────────────────────────┘
                         ▲
                         │
              lane-row token(i,r)
```

Bu işlem, klasik transformer cross-attention’ın lane problemine uyarlanmış halidir. Row token, feature map’in aynı yatay satırındaki x konumlarından bilgi toplar.

Bu neden mantıklı?

- Lane’in bir row’daki bilinmeyeni çoğunlukla x konumudur.
- Row-local attention gereksiz global arama uzayını azaltır.
- Model satır satır daha kontrollü lokalizasyon yapar.

## B. Same-row lane interaction

Aynı row’da farklı lane adayları birbirine yakın olabilir. Model aynı lane’i iki kez üretmemeli ve paralel lane’leri ayırabilmelidir.

Bu yüzden aynı row’daki lane adayları birbirleriyle etkileşir:

```text
row r:

lane 1 token  <──>  lane 2 token  <──>  lane 3 token  <──> ...
```

Bu etkileşim şunlara yardımcı olur:

- duplicate tahminleri azaltmak,
- aynı row’daki lane adaylarını düzenlemek,
- sol/sağ/orta lane ayrımını öğrenmek.

## C. Same-lane vertical interaction

Lane çizgisi satırdan satıra süreklidir. Bir lane’in alt row’daki konumu, üst row’daki konumuyla ilişkilidir.

Bu yüzden aynı lane’in row tokenları birbirleriyle etkileşir:

```text
lane i:

row 1  <──>  row 2  <──>  row 3  <──>  ...  <──>  row R
```

Bu işlem:

- eğriliği modellemeye,
- occlusion durumunda eksik bölgeleri bağlamaya,
- row-wise jitter’ı azaltmaya,
- lane’in global geometrisini korumaya yardımcı olur.

Şekil 5:

```text
Şekil 5. Same-lane vertical information flow

lane i

row 1     ●
          │
row 2     ●
          │
row 3     ●
          │
...       │
          │
row R     ●

Her nokta ayrı row token; dikey etkileşim lane sürekliliğini öğretir.
```

## D. Decoder layer sayısı

Bir decoder layer, tokenlara görüntüden bilgi alma ve birbirleriyle etkileşme fırsatı verir. Layer sayısı arttıkça model lane geometrisini daha iteratif şekilde rafine edebilir.

Son modelde:

```text
structured decoder layer = 4
```

Bu değer deneysel olarak makul bir noktadır. Daha az layer model kapasitesini sınırlayabilir; daha fazla layer ise küçük artış karşılığında eğitim ve inference maliyetini artırabilir.

Paper’da önerilen ablation:

| Decoder Layer | Beklenen Davranış |
|---:|---|
| 1 | Zayıf global bağlam, düşük lokalizasyon kalitesi |
| 2 | Daha iyi, fakat karmaşık eğrilerde sınırlı |
| 4 | Ana model; iyi denge |
| 6 | Olası küçük artış, daha yüksek maliyet |

---

# V. Prediction Heads

Structured decoder sonunda her lane adayı için row-wise ve lane-level çıktılar üretilir.

## A. Row-wise x distribution

Her lane-row token, o satırdaki lane x konumu için bir olasılık dağılımı üretir.

Örneğin input genişliği 1600 px ve x-bin sayısı 800 ise:

```text
1 bin ≈ 2 px
```

Her row için model şunu üretir:

```text
P(x-bin = 1), P(x-bin = 2), ..., P(x-bin = 800)
```

Son x koordinatı bu dağılımdan elde edilir.

Şekil 6:

```text
Şekil 6. Row-wise x distribution

row r için:

x-bin:   1   2   3   ...  392  393  394  ...  800
prob:    .   .   .        .10  .65  .20        .

Tahmin edilen x, dağılımın merkezi / beklenen değeri olarak alınır.
```

Bu yapı, tek nokta regresyonuna göre daha açıklanabilirdir. Modelin hangi x bölgelerini olası gördüğü analiz edilebilir.

## B. Lane existence

Model sabit sayıda lane adayı üretir. Fakat gerçek görüntüde bunların hepsi lane değildir. Bu yüzden her lane adayı için existence skoru üretilir.

```text
existence score = bu aday gerçekten lane mi?
```

Bu skor inference sırasında düşük kaliteli adayları elemek için kullanılır.

## C. Vertical range

Her lane tüm görüntü boyunca görünmez. Model lane’in hangi dikey aralıkta geçerli olduğunu tahmin eder.

```text
range = [y_start, y_end]
```

Bu bilgi lane’in görünürlük aralığını modellemeye yardımcı olur.

## D. Quality score

Quality score, lane tahmininin geometrik güvenilirliğini ifade eder. Existence “lane var mı?” sorusuna daha yakındır; quality ise “bu lane ne kadar iyi lokalize edilmiş?” sorusuna daha yakındır.

Final skor şu tip bir birleşimle elde edilebilir:

```text
final confidence = existence confidence × quality confidence ağırlığı
```

Bu ağırlık final paper’da validation set üzerinden seçilmelidir.

---

# VI. Distributional Row-wise Localization

Bu bölüm DFL fikrini kod detayı vermeden açıklar.

## A. Neden doğrudan x regresyonu yeterli olmayabilir?

Bir row için modelin lane konumunu tek sayı olarak vermesi mümkündür. Fakat tek sayı, modelin belirsizliğini göstermez.

Örneğin:

```text
Gerçek x = 600
```

Model iki farklı şekilde aynı ortalamayı verebilir:

- x=600 civarında keskin ve emin dağılım,
- x=500 ve x=700 arasında kararsız geniş dağılım.

İkisi aynı beklenen değeri verebilir, ama aynı kalitede tahmin değildir.

## B. Soft-bin target

Lane noktası iki bin arasına düşebilir. Örneğin gerçek x konumu bin 300.25’e denk geliyorsa, hedef dağılım şöyle kurulabilir:

```text
bin 300 -> 0.75
bin 301 -> 0.25
```

Böylece model sadece doğru bin’i değil, sub-bin konumu da öğrenir.

Şekil 7:

```text
Şekil 7. Soft-bin supervision

GT position: bin 300.25

target distribution:

bin 299     bin 300     bin 301     bin 302
 0.00        0.75        0.25        0.00
```

Bu yaklaşım x lokalizasyonunu daha hassas hale getirir ve yayvan dağılımları cezalandırır.

## C. DFL’nin bu çalışmadaki rolü

DFL bu çalışmada ana mimari katkı değildir. Ana katkı structured instance-to-row decoder’dır.

DFL’nin rolü:

```text
structured row tokenların ürettiği x dağılımlarını daha doğru ve keskin hale getirmek
```

Bu ayrım paper’da net kurulmalıdır.

---

# VII. Training Objective

Modelin eğitimi birkaç loss bileşeninden oluşur. Bu bölüm loss’ları kavramsal olarak açıklar.

## A. Assignment

Model her görüntü için sabit sayıda lane adayı üretir. Ground truth lane sayısı ise değişkendir.

Bu nedenle önce tahminlerle ground truth lane’ler eşleştirilir.

Eşleştirme maliyeti genel olarak şu bilgilere dayanır:

- lane varlık skoru,
- row-wise x yakınlığı,
- dikey range yakınlığı,
- line IoU benzerliği.

Eşleştirme sonrası:

```text
matched predictions   -> lane olarak eğitilir
unmatched predictions -> no-lane olarak eğitilir
```

## B. Existence loss

Bu loss, hangi adayın gerçek lane olduğunu öğretir.

Amaç:

- gerçek lane adayına yüksek skor,
- boş / yanlış adaya düşük skor vermek.

## C. Point localization loss

Matched prediction ile ground truth lane arasında satır satır x farkı hesaplanır.

Amaç:

```text
her valid row’da predicted x, GT x’e yaklaşsın
```

## D. LineIoU loss

Lane yalnızca bağımsız noktalar değildir; çizgi olarak değerlendirilir. LineIoU loss, tahmin edilen çizginin GT çizgiyle örtüşmesini teşvik eder.

Bu loss, benchmark metriğiyle daha uyumlu bir sinyal sağlar.

## E. Range loss

Lane’in görünür olduğu dikey aralığı doğru tahmin etmeyi öğretir.

## F. Smoothness loss

Lane çizgilerinde aşırı zigzag davranışı istenmez. Smoothness loss, komşu row’lar arasında daha fiziksel ve sürekli çizgiler üretmeye yardımcı olur.

## G. Quality loss

Quality head, tahminin geometrik güvenilirliğini öğrenir. Bu skor inference sırasında aday sıralamaya yardımcı olur.

## H. Auxiliary segmentation / centerline supervision

Modelin final çıktısı segmentation değildir. Ancak eğitimde auxiliary segmentation ve centerline supervision kullanılabilir. Bunlar encoder feature’larının lane bölgelerine duyarlı hale gelmesini sağlar.

Bu yardımcı loss’lar final prediction formatını değiştirmez.

---

# VIII. Inference Procedure

Inference sırasında modelin ürettiği tüm lane adayları final sonuç olarak verilmez.

Adımlar:

1. Her lane adayı için row-wise x koordinatları çıkarılır.
2. Lane existence ve quality skorları hesaplanır.
3. Düşük skorlu adaylar threshold ile atılır.
4. Benzer lane’ler NMS ile temizlenir.
5. En iyi birkaç lane raporlanır.

Şekil 8:

```text
Şekil 8. Inference flow

32 lane candidates
       │
       ▼
score computation
       │
       ▼
threshold filtering
       │
       ▼
lane NMS
       │
       ▼
top-k final lanes
```

Önemli protokol notu:

```text
Threshold, quality ağırlığı ve checkpoint test set üzerinde seçilmemelidir.
Bu seçim validation set üzerinde yapılmalı; test set yalnızca final raporlama için kullanılmalıdır.
```

---

# IX. Method Comparison

Tablo I, önerilen yöntemi temel lane detection paradigmalarıyla karşılaştırır.

## Tablo I. Temsil biçimlerinin kavramsal karşılaştırması

| Yöntem Tipi | Temsil | Güçlü Yan | Zayıf Yan | Bizim Farkımız |
|---|---|---|---|---|
| Segmentation | Pixel-level mask | Spatial detay güçlü | Instance ayrımı ve postprocess zor | Biz doğrudan lane instance + row geometri üretiriz |
| Anchor-based | Önceden tanımlı line anchors | Güçlü performans | Anchor tasarımı heuristic | Biz line anchor kullanmadan learnable instance token kullanırız |
| Parametric curve | Polinom / spline parametreleri | Kompakt temsil | Karmaşık lokal eğrilerde sınırlı | Biz satır satır esnek geometri üretiriz |
| Row-wise classifier | Her row’da x sınıfı | Lane yapısına uygun | Instance ve global bağlam sınırlı olabilir | Biz row-wise yapıyı instance token ve attention ile birleştiririz |
| Holistic query transformer | Lane başına tek query | End-to-end ve esnek | Tüm geometri tek vektöre sıkışır | Biz lane’i instance + ordered row tokens olarak ayrıştırırız |

---

# X. Architecture Summary Table

Tablo II, son ana modelin yüksek seviyeli mimari özetini verir.

## Tablo II. Ana model konfigürasyonu

| Bileşen | Seçim | Gerekçe |
|---|---:|---|
| Backbone | ResNet-34 | Literatürle adil ve yaygın karşılaştırma |
| Input | 1600 × 640 | CondLSTR benzeri yüksek çözünürlük paritesi |
| FPN channels | 256 | İnce lane feature’ları için yeterli kapasite |
| Instance slots | 32 | CULane için yeterli aday sayısı; duplicate riskini sınırlama |
| Row count | 160 | Dikeyde yoğun lane örnekleme |
| X bins | 800 | Yaklaşık 2 px yatay lokalizasyon çözünürlüğü |
| Structured decoder layers | 4 | Kapasite / maliyet dengesi |
| Row-wise distribution | Var | Belirsizlik ve hassas lokalizasyon |
| DFL supervision | Var | Dağılımı doğru x çevresinde keskinleştirme |
| EMA / TTA | Yok | Şu anki skor saf model skoru olarak yorumlanabilir |

---

# XI. Experimental Protocol

Bu bölüm paper için önerilen deney protokolünü tanımlar.

## A. Dataset

Ana benchmark:

```text
CULane
```

Ek datasetler paper gücünü artırmak için önerilir:

```text
TuSimple
LLAMAS
CurveLanes
```

## B. Ana metrikler

Raporlanması gerekenler:

- Precision
- Recall
- F1
- kategori bazlı F1
- crossroad false positive sayısı
- tight IoU metrikleri, örneğin F1@0.7 / F1@0.75
- inference speed / FPS
- parametre sayısı
- FLOPs

## C. Threshold seçimi

Doğru protokol:

```text
Validation set:
  checkpoint seçimi
  confidence threshold seçimi
  quality power seçimi
  NMS parametresi seçimi

Test set:
  sadece final raporlama
```

Bu ayrım kritik önemdedir. Test üzerinde yapılan eşik araması paper review’da zayıf görünür.

---

# XII. Ablation Study Plan

Bu çalışmanın kabul edilebilirliği büyük ölçüde ablation kalitesine bağlıdır.

## A. Ana ablation soruları

1. Structured decoder gerçekten unstructured query baseline’dan iyi mi?
2. Instance token olmadan ne olur?
3. Ordered row token olmadan ne olur?
4. DFL ne kadar katkı veriyor?
5. Decoder layer sayısı nasıl etki ediyor?
6. Slot sayısı precision/recall dengesini nasıl değiştiriyor?
7. Çözünürlük artışı mı kazandırıyor, yoksa structured representation mı?

## Tablo III. Önerilen çekirdek ablation tablosu

| Deney | Instance Token | Ordered Row Token | DFL | Decoder Layer | Amaç |
|---|---:|---:|---:|---:|---|
| A. Unstructured baseline | Var | Yok | Yok | eşit | Eski temsil gücü |
| B. Instance only | Var | Yok | Var/Yok | eşit | Lane identity etkisi |
| C. Structured no DFL | Var | Var | Yok | eşit | Row token ana katkısı |
| D. Structured + DFL | Var | Var | Var | eşit | Distributional localization katkısı |
| E. Structured shallow | Var | Var | Var | 2 | Layer etkisi |
| F. Structured final | Var | Var | Var | 4 | Ana model |
| G. Structured deeper | Var | Var | Var | 6 | Ek layer katkısı |

## B. Resolution / capacity ablation

Tablo IV, çözünürlük ve kapasite etkisini ayırmak için önerilir.

## Tablo IV. Resolution ve kapasite kontrolü

| Input | Rows | X-bins | FPN | Decoder | Amaç |
|---:|---:|---:|---:|---:|---|
| 800×288 | 72 | 200 | 128 | 2/4 | Eski düşük çözünürlük |
| 1024×384 | 96 | 512 | 256 | 4 | Orta seviye |
| 1600×640 | 160 | 800 | 256 | 4 | Final high-res |

Bu tablo şu soruya cevap vermelidir:

```text
Kazanç yalnızca daha büyük inputtan mı geliyor,
yoksa aynı koşulda structured row-token decoder da belirgin katkı veriyor mu?
```

## C. Slot sayısı ablation

## Tablo V. Slot sayısı etkisi

| Slot Sayısı | Beklenen Etki |
|---:|---|
| 20 | Daha az FP, ama recall sınırlı olabilir |
| 32 | Dengeli final seçim |
| 64 | Recall artabilir, duplicate / FP riski artabilir |

## D. Decoder layer ablation

## Tablo VI. Decoder layer etkisi

| Layer | Beklenen Yorum |
|---:|---|
| 1 | Yetersiz interaction |
| 2 | Makul ama sınırlı |
| 4 | Ana denge noktası |
| 6 | Küçük ek kazanç veya plateau |

---

# XIII. Preliminary Results

Bu bölüm mevcut sonuçların paper’a nasıl konulacağını gösterir. Rakamlar final protokol tamamlanmadan “preliminary” olarak değerlendirilmelidir.

## Tablo VII. Mevcut ana sonuç özeti

| Model | Backbone | Input | EMA | TTA | F1 | Precision | Recall | Not |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Unstructured S0 baseline | ResNet-34 | düşük/orta | Yok | Yok | ~75.25 | TBD | TBD | Aynı protokolle yeniden doğrulanmalı |
| Structured S0 final | ResNet-34 | 1600×640 | Yok | Yok | ~79.9 | ~87.7 | ~73.4 | Threshold seçimi validation ile sabitlenmeli |

Önemli not:

```text
Bu tablo final paper tablosu değildir.
Final paper’da threshold/checkpoint validation set üzerinde seçilmeli ve test sonuçları tek sefer raporlanmalıdır.
```

## A. Beklenen yorum

Eğer aynı training recipe altında unstructured baseline yaklaşık 75.25 F1, structured final yaklaşık 79.9 F1 veriyorsa, bu yaklaşık +4.5 F1 civarında ciddi bir artıştır.

Ancak bu artışın tamamını structured row-token decoder’a bağlamak için ablation gereklidir. Aksi halde reviewer şu soruyu sorar:

```text
Kazanç representation’dan mı,
yoksa input çözünürlüğü / FPN kapasitesi / DFL / layer sayısı gibi faktörlerden mi geliyor?
```

Bu yüzden ana paper gücü sonuçtan değil, sonuç + temiz ablation kombinasyonundan gelir.

---

# XIV. Required Figures for the Paper

Hocanın istediği “grafikler ve tablolar” için önerilen figür listesi aşağıdadır.

## Figure 1. Overall architecture

Göstermeli:

```text
Input image -> ResNet/FPN -> structured row-token decoder -> lane outputs
```

Amaç:

Okuyucu modelin genel akışını tek bakışta anlamalı.

## Figure 2. Holistic query vs structured row-token representation

İki panel:

```text
(a) Holistic lane query:
    one vector -> full lane

(b) Proposed:
    instance token + ordered row tokens -> row-wise geometry
```

Bu figür paper’ın ana fikrini taşır.

## Figure 3. Instance-to-row token grid

Lane adayları satır, row tokenları sütun gibi gösterilebilir:

```text
            row1 row2 row3 ... rowR
instance1    ●    ●    ●       ●
instance2    ●    ●    ●       ●
...
instanceN    ●    ●    ●       ●
```

Her nokta lane-row token’dır.

## Figure 4. Decoder block

Üç attention türü gösterilmeli:

```text
row-local image attention
same-row lane interaction
same-lane vertical interaction
```

## Figure 5. Row-wise x distribution heatmap

Bir lane için:

```text
y-axis: row index
x-axis: x-bin
color: probability
```

Bu figür DFL ve distributional localization fikrini açıklar.

## Figure 6. Qualitative comparison

Unstructured baseline ve structured model yan yana gösterilmeli:

- crowded scene,
- curve scene,
- night scene,
- no-line/cross failure.

## Figure 7. Category-wise delta plot

Structured modelin baseline’a göre kategori bazlı kazanç/kayıpları:

```text
normal
crowd
hlight
shadow
noline
arrow
curve
night
cross FP
```

Bu grafik toplam F1’in arkasındaki davranışı gösterir.

## Figure 8. Precision-recall / threshold curve

Threshold değiştikçe precision, recall, F1 nasıl değişiyor?

Bu grafik scoring stabilitesini gösterir.

---

# XV. Failure Modes

Modelin güçlü olduğu ve zayıf olduğu durumlar açıkça yazılmalıdır.

## A. Güçlü olması beklenen durumlar

- normal lane çizgileri,
- kalabalık ama lane’in kısmen göründüğü sahneler,
- eğimli lane geometrileri,
- row-wise sürekliliğin işe yaradığı kısmi occlusion sahneleri.

## B. Zayıf olabilecek durumlar

- lane olmayan crossroad sahneleri,
- no-line görüntüler,
- gece ve aşırı düşük kontrast,
- bariyer / gölge / yol kenarı gibi lane-benzeri yapılar.

## C. Neden bu zayıflıklar oluşabilir?

Structured row-token decoder güçlü bir lane prior’ına sahiptir. Bu iyi bir şeydir; çünkü eksik veya zor lane’leri tamamlamaya yardımcı olabilir. Ancak aynı prior, lane olmayan bazı sahnelerde yanlış lane üretme eğilimi de yaratabilir.

Bu trade-off paper’da dürüstçe tartışılmalıdır.

## Tablo VIII. Failure mode analizi

| Failure Type | Muhtemel Sebep | Önerilen Analiz |
|---|---|---|
| Crossroad FP | Model lane prior’ı nedeniyle yapı arıyor | Cross FP sayısı ve confidence histogramı |
| No-line FP | Zayıf yol izlerini lane sanma | Qualitative examples |
| Night FN | Görsel kanıt zayıf | Category-wise recall |
| Curve error | Uzun menzilli geometri zor | Tight IoU / curve split |
| Duplicate lane | Slot sayısı / one-to-many assignment | NMS ve slot ablation |

---

# XVI. Discussion

## A. Çalışmanın ana katkısı nedir?

Ana katkı, lane detection için yeni bir structured representation önermesidir:

```text
lane = instance identity + ordered row geometry
```

Bu fikir, row-wise lane representation ile transformer query paradigmasını birleştirir.

## B. Çalışma ne değildir?

Bu çalışma yalnızca:

- yüksek çözünürlük denemesi,
- DFL uygulaması,
- FPN kanal artırımı,
- threshold tuning

olarak konumlandırılmamalıdır.

Bunlar destekleyici unsurlardır. Mimari iddia structured decoder’dır.

## C. Neden ResNet-34 önemli?

ResNet-34 birçok lane detection çalışmasında standart backbone olarak kullanılır. Bu nedenle ResNet-34 ile yapılan karşılaştırma, mimari farkı izole etmek için önemlidir.

Ancak paper’da farklı backbone’lar da eklenirse sonuç güçlenir:

```text
ResNet-18
ResNet-34
ResNet-50
opsiyonel DLA-34
```

## D. Q1 / CVPR açısından kritik nokta

Modelin yüksek skor alması tek başına yeterli değildir. Kabul edilebilirlik için şu üç şey gerekir:

1. Aynı koşulda unstructured baseline’a karşı net artış.
2. Artışın structured row-token decoder’dan geldiğini gösteren ablation.
3. Literatürdeki yakın yöntemlere karşı doğru ve dürüst konumlandırma.

---

# XVII. Limitations

Bu çalışmanın sınırlamaları şunlardır:

1. Row-wise representation, lane’in her row’da tek x konumuyla temsil edilebildiği varsayımına dayanır.
2. Çok karmaşık sahnelerde veya lane olmayan sahnelerde false positive riski vardır.
3. One-to-many assignment recall’ı artırırken duplicate/precision sorunları yaratabilir.
4. Yüksek çözünürlük ve fazla row/bin sayısı eğitim maliyetini artırır.
5. DFL ve quality score, doğru validation protokolü olmadan test set tuning’e açık hale gelebilir.

Bu sınırlamalar saklanmamalı; paper’da kontrollü analizlerle yönetilmelidir.

---

# XVIII. Conclusion

Bu çalışmada, 2D lane detection için structured row-token decoder yaklaşımı önerilmektedir. Önerilen model, lane’i tek bir global query vektörü olarak temsil etmek yerine, lane kimliğini instance token ile, satır-bazlı geometriyi ise ordered row token dizisiyle modellemektedir. Bu sayede lane detection probleminin doğal row-wise yapısı transformer decoder içine açıkça taşınmaktadır.

Her row token, görüntü feature’larından kendi satırına ait yatay kanıtları okuyarak x-konumu için ayrık bir dağılım üretir. Bu dağılım distributional supervision ile eğitilerek lokalizasyon hassasiyeti artırılır. Lane-level existence, range ve quality tahminleri ise row tokenların global özetinden elde edilir.

Ön sonuçlar, bu structured representation’ın unstructured query baseline’a göre belirgin performans artışı sağlayabileceğini göstermektedir. Ancak çalışmanın akademik gücü, final skorun yanında kapsamlı ablation, validation protokolü, kategori bazlı analiz ve literatürle adil karşılaştırma ile kurulmalıdır.

---

# Appendix A. Danışman hocaya sözlü anlatım için kısa versiyon

Hocaya 3 dakikada anlatılacak hali:

```text
Hocam, lane çizgisi aslında görüntüde satır satır x koordinatlarından oluşan uzun bir geometri. Klasik transformer query yaklaşımında bütün lane tek bir vektöre sıkıştırılıyor. Biz bunun yerine lane’i iki parçaya ayırıyoruz: instance token lane’in kimliğini temsil ediyor, row tokenlar ise aynı lane’in farklı yatay satırlardaki geometrisini temsil ediyor.

Her lane adayı için her row’da ayrı bir token var. Bu token görüntünün kendi satırındaki feature’lara bakıp o satırda lane’in x konumunu dağılım olarak tahmin ediyor. Aynı lane’in row tokenları da birbirleriyle konuşuyor, böylece çizginin sürekliliği ve eğriliği öğreniliyor.

Yani katkımız DFL veya yüksek çözünürlük değil; lane’i decoder içinde instance identity + ordered row geometry olarak yapılandırmak. DFL sadece her row’daki x dağılımını daha hassas eğitmek için kullandığımız yardımcı lokalizasyon loss’u.
```

---

# Appendix B. Paper’da kullanılabilecek contribution maddeleri

1. 2D lane detection için lane kimliği ve row-wise geometriyi ayrıştıran structured instance-to-row decoder öneriyoruz.

2. Her lane adayını global bir instance token ve ona bağlı ordered row token dizisiyle temsil ederek, whole-lane query temsiline göre daha güçlü bir geometrik inductive bias sağlıyoruz.

3. Row tokenların row-local visual evidence okuyarak x-konumu için ayrık dağılım üretmesini ve bu dağılımın soft-bin supervision ile eğitilmesini sağlıyoruz.

4. CULane üzerinde kapsamlı ablationlarla instance token, row token, distributional localization, decoder derinliği, slot sayısı ve çözünürlük etkilerini izole ediyoruz.

---

# Appendix C. Final paper checklist

Paper’a geçmeden önce tamamlanması gereken minimum liste:

- [ ] Unstructured baseline aynı protokolle yeniden eğitildi.
- [ ] Structured final aynı protokolle doğrulandı.
- [ ] Threshold ve checkpoint validation set üzerinde seçildi.
- [ ] Test set sadece final raporlama için kullanıldı.
- [ ] Instance-only ablation yapıldı.
- [ ] Row-token ablation yapıldı.
- [ ] DFL ablation yapıldı.
- [ ] Decoder layer ablation yapıldı.
- [ ] Slot sayısı ablation yapıldı.
- [ ] Resolution/capacity ablation yapıldı.
- [ ] Category-wise CULane sonuçları çıkarıldı.
- [ ] Cross FP analizi yapıldı.
- [ ] Qualitative success/failure görselleri hazırlandı.
- [ ] DFL probability heatmap görselleri hazırlandı.
- [ ] Speed / parameter / FLOP tablosu çıkarıldı.
- [ ] En yakın literatürle adil karşılaştırma tablosu hazırlandı.

