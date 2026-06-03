Evet, bunu **part part** yapalım. Bence en sağlıklısı şu: önce “implementation felsefesi ve MVP sırası”nı kilitleyelim, sonra dataset/target builder, tokenizer, matcher/loss, model blokları, training loop, debug checklist diye ilerleyelim.

Aşağıda **Part 1** başlıyor.

---

# DynLaneSeq-EG Implementation Plan — Part 1

## Ana kararlar, MVP sırası ve değişmeyecek teknik kabuller

Önce şunu netleştiriyorum: **Full DynLaneSeq-EG’yi direkt implement etmeyeceğiz.**
Bu modelin ana fikri doğru: önce lane-specific evidence çıkarıp sonra bu evidence’tan sequence üretmek. Senin dokümanda da modelin “image → lane-specific evidence → lane sequence” mantığında olması gerektiği açıkça tanımlanmıştı.  Fakat CondLSTR örneğinde gördüğümüz gibi, paper’daki temiz mimari çizgisi gerçek kodda birçok gizli detayla çalışıyor; CondLSTR paper’ı da feature map + positional embedding + lane query + transformer decoder + dynamic kernel + Hungarian matching akışını anlatıyor ama implementation seviyesinde loss, target, postprocess ve shape detayları asıl işi belirliyor. 

Bu yüzden planımızın amacı şu olacak:

> Önce küçük, türevlenebilir, debug edilebilir bir lane-slot modeli çıkaracağız. Sonra sequence, evidence sampler, low-rank bridge ve zoom-in refinement’ı sırayla ekleyeceğiz.

---

## 1. Ana implementation stratejisi

Full model şu an fazla karmaşık:

```text
Backbone
→ FPN
→ lane slots
→ positional prior
→ evidence extractor
→ coarse geometry
→ curve-aligned sampler
→ token decoder
→ soft geometry reconstruction
→ optional low-rank bridge
→ optional zoom-in refinement
```

Bunu tek seferde yazarsak hata geldiğinde hangi parçanın bozduğunu anlayamayız. O yüzden modeli **5 aşamada** kuracağız.

---

# 2. Model sürümleri

## Sürüm S0 — Geometry-only sanity model

Bu ilk çalışan sürüm olacak.

```text
Image
→ Backbone + FPN
→ lane slot queries
→ cross-attention
→ existence head
→ row-wise x head
→ lane points
```

Bu sürümde yok:

```text
token decoder yok
curve-aligned sampler yok
low-rank bridge yok
zoom-in yok
evidence consistency loss yok
confidence calibration yok
topology yok
```

Amaç:

```text
Model lane slot mantığını öğrenebiliyor mu?
Hungarian matching çalışıyor mu?
GT target builder doğru mu?
Postprocess görüntü üstüne doğru lane çiziyor mu?
```

Bu aşama çalışmadan sequence modeline geçmek hata olur.

---

## Sürüm S1 — Soft token decoder eklenmiş model

Bu aşamada token fikrini ekleyeceğiz ama geometri loss’unu koparmayacağız.

```text
Image
→ Backbone + FPN
→ lane slots
→ cross-attention
→ token decoder
→ soft expected coordinate decoding
→ lane points
```

Buradaki kritik nokta şu:

```text
argmax token → coordinate
```

yapmayacağız.

Onun yerine:

```text
token logits
→ softmax probability
→ weighted average over bin centers
→ continuous x coordinate
```

kullanacağız.

Çünkü daha önceki eleştiride de söylendiği gibi argmax ile token’dan koordinata geçersen geometry loss decoder’a gradient gönderemez. Bu yüzden continuous expected decoding zorunlu hale geliyor. 

---

## Sürüm S2 — Curve-aligned evidence sampler

Bu aşamada evidence-grounded kısmı başlıyor.

```text
Image
→ Backbone + FPN
→ lane slots
→ coarse geometry
→ curve-aligned sampler
→ evidence sequence
→ token decoder
→ lane points
```

Ama burada **cold-start problemi** var. Eğitim başında coarse geometry kötü olacağı için sampler yanlış yerden feature toplar. Bu yüzden ilk eğitimlerde predicted curve ile değil, **GT-guided curve** ile sampling yapacağız. Bu problem de daha önceki eleştiride açıkça belirtilmişti. 

Training schedule şöyle olacak:

```text
Epoch 0–E1:
  sample curve = GT curve + small noise

Epoch E1–E2:
  sample curve = mix(GT curve, predicted curve)

Epoch E2 sonrası:
  sample curve = predicted curve
```

Bu sayede sampler baştan gökyüzüne, kaldırıma veya görüntü dışına bakmaz.

---

## Sürüm S3 — Factorized low-rank bridge

Bu aşamada özgün katkı adayımızı ekleyeceğiz.

Ama önemli karar:

```text
Full dynamic kernel üretmeyeceğiz.
```

Yani şunu doğrudan materialize etmeyeceğiz:

```text
C_out × C_in × k × k
```

Çünkü her batch ve her lane slot için bunu üretmek VRAM’i patlatabilir. Önceki eleştiride de full kernel üretmenin dinamik tensör patlamasına yol açabileceği ve bunun yerine channel reduction → spatial filtering → channel expansion gibi factorized modulation yapılması gerektiği söylenmişti. 

Bizim bridge şu şekilde uygulanacak:

```text
F
→ lane-conditioned channel reduction
→ small spatial/depthwise filtering
→ lane-conditioned channel expansion
→ residual evidence map
```

Bu sürümde hâlâ zoom-in yok.

---

## Sürüm S4 — Optional zoom-in refinement

Bu en son gelecek.

```text
First pass:
  evidence → tokens → preliminary lane

Second pass:
  decoder hidden state → bridge → refined evidence → final tokens
```

Bunu MVP’ye koymuyoruz. Çünkü çok fazla hareketli parça ekliyor. Ancak S0, S1, S2, S3 stabil çalıştıktan sonra denenebilir.

---

# 3. İlk dataset kararı

İlk implementation için tek dataset seçmeliyiz.

Benim önerim:

```text
İlk dataset: CULane
```

Neden?

1. Lane detection literatüründe yaygın.
2. 2D lane output için uygun.
3. CondLSTR da CULane üzerinde rapor veriyor.
4. OpenLane gibi kategori/3D karmaşıklığı ilk aşamada gereksiz yük getirebilir.

İlk hedef:

```text
Full CULane değil.
Önce küçük subset.
```

Aşamalar:

```text
Step 1: 10 image overfit
Step 2: 100 image overfit
Step 3: 1k image train
Step 4: full train
```

10 image overfit olmadan full training’e geçilmeyecek.

---

# 4. Sabit input ve coordinate sistemi

Implementation’da en çok hata çıkaran şeylerden biri coordinate karışıklığıdır. O yüzden en baştan 4 coordinate sistemi tanımlıyoruz.

## 4.1 Original image coordinate

Dataset’ten gelen gerçek görüntü boyutu:

```text
W_orig × H_orig
```

CULane için görüntüler genelde geniş formatlıdır. Ama model içine direkt orijinal boyutta sokmayacağız.

---

## 4.2 Model input coordinate

Model input resize sonrası:

```text
W_in = 800
H_in = 288
```

Tüm GT lane noktaları bu coordinate sistemine çevrilecek.

Yani dataset’ten gelen lane noktası:

```text
(x_orig, y_orig)
```

şuna dönüşecek:

```text
x_in = x_orig * W_in / W_orig
y_in = y_orig * H_in / H_orig
```

Bundan sonra target builder model input coordinate üzerinden çalışacak.

---

## 4.3 Row-wise lane coordinate

Lane’i fixed row noktalarıyla temsil edeceğiz.

Başlangıç:

```text
P = 72 rows
```

Yani 288 yüksekliğinde her 4 pikselde bir row:

```text
y_rows = [0, 4, 8, ..., 284]
```

Her lane için target:

```text
x_rows      ∈ R^P
valid_mask  ∈ {0,1}^P
```

Eğer lane o row’da yoksa:

```text
valid_mask[p] = 0
x_rows[p] = -1
```

Eğer lane o row’da varsa:

```text
valid_mask[p] = 1
x_rows[p] = x coordinate
```

Bu representation ilk MVP için token’dan daha güvenli.

---

## 4.4 Grid sample coordinate

`grid_sample` kullanacağımız zaman coordinate sistemi farklıdır:

```text
x_grid ∈ [-1, 1]
y_grid ∈ [-1, 1]
```

Dönüşüm:

```text
x_grid = 2 * x_in / (W_in - 1) - 1
y_grid = 2 * y_in / (H_in - 1) - 1
```

Bunu tek bir utility fonksiyonda tutacağız. Her modülde tekrar yazılmayacak.

---

# 5. Lane slot sayısı kararı

İlk başta şu seçimi yapıyoruz:

```text
N_slots = 20
```

Daha önceki planda 8 veya 10 düşünülmüştü. Ama ben implementation için 20 öneriyorum.

Neden?

CondLSTR paper’ında lane query sayısı 20’den 80’e çıkarıldığında performans artıyor, 80’den 100’e geçince kazanç çok küçük kalıyor. Bu da query sayısının sadece “sahnede kaç lane var?” değil, “kaç aday lane prototipi arıyoruz?” meselesi olduğunu gösteriyor. 

İlk MVP için 80 ağır olabilir. 8/10 ise fazla kısıtlayıcı olabilir.

Bu yüzden başlangıç:

```text
N = 20
```

Sonra ablation:

```text
N = 10, 20, 40, 80
```

ama ilk çalışan sürüm 20.

---

# 6. Feature map kararı

Backbone + FPN sonrası tek ana feature map kullanacağız.

Başlangıç:

```text
Input image: B × 3 × 288 × 800
Feature map: B × C × Hf × Wf
C = 128
Hf = 72
Wf = 200
```

Yani input’un 1/4 resolution’ı.

Neden 1/4?

Lane ince yapı olduğu için 1/16 veya 1/32 feature map uzak ve ince lane çizgilerini kaybedebilir. İlk MVP’de yüksek çözünürlüklü feature map daha güvenli. VRAM yetmezse:

```text
Hf = 36
Wf = 100
```

seviyesine ineriz.

---

# 7. İlk backbone kararı

İlk MVP:

```text
Backbone = ResNet-34
Neck = simple FPN
Output channels = 128
```

Daha ağır backbone yok.

ResNet-50, ConvNeXt, Swin gibi şeyleri ilk implementation’a koymuyoruz. İlk hedef paper performance değil, **doğru çalışan training graph**.

---

# 8. S0 modelinin net tensor akışı

İlk yazılacak model budur.

```text
Input:
  images: B × 3 × 288 × 800

Backbone outputs:
  C2: B ×  64 × 72 × 200
  C3: B × 128 × 36 × 100
  C4: B × 256 × 18 × 50
  C5: B × 512 ×  9 × 25

FPN output:
  F: B × 128 × 72 × 200

2D positional encoding:
  F_pos: B × 128 × 72 × 200

Flatten:
  F_flat: B × 14400 × 128

Project to transformer dimension:
  F_mem: B × 14400 × 256

Lane queries:
  Q0: B × 20 × 256

Cross attention:
  Q1: B × 20 × 256

Heads:
  exist_logits: B × 20 × 2
  row_x_logits: B × 20 × P × X_bins
  range_pred: B × 20 × 2
```

Başlangıç değerleri:

```text
P = 72
X_bins = 200
```

Neden `X_bins=200`?

Çünkü feature width 200. Her bin yaklaşık input görüntüde 4 piksele denk gelir:

```text
800 / 200 = 4 px
```

Bu ilk MVP için makul.

---

# 9. S0’da token yok ama soft coordinate var

S0’da bile argmax yapmayacağız.

`row_x_logits` için:

```text
prob = softmax(row_x_logits, dim=-1)
bin_centers = [0, 1, 2, ..., 199]
x_feat = sum(prob * bin_centers)
x_input = x_feat * 4
```

Böylece:

```text
x_input ∈ [0, 796]
```

Bu differentiable olur. Geometry loss buradan akar.

Training sırasında:

```text
L_point = SmoothL1(x_input, gt_x_input)
```

sadece `valid_mask=1` olan row’larda hesaplanır.

---

# 10. S0 matching kararı

İlk MVP’de matching cost’a token CE koymuyoruz.

Matching cost:

```text
cost = λ_exist * object_cost
     + λ_point * masked_point_l1
     + λ_range * range_l1
```

Burada:

```text
object_cost = -log(P_lane)
masked_point_l1 = valid row’larda ortalama |pred_x - gt_x|
range_l1 = |pred_start_y - gt_start_y| + |pred_end_y - gt_end_y|
```

Neden token CE yok?

Çünkü daha önceki eleştiride de söylendiği gibi token CE’yi matching cost’a koymak hem pahalı hem de boş slotlar için anlamsız hale gelebilir.  Ayrıca CondLSTR tarafında da matching ana olarak object, heat/location, offset ve range gibi geometric/object maliyetler üzerinden kurgulanıyor. 

---

# 11. S0 loss kararı

İlk loss:

```text
L_total =
  2.0 * L_exist
+ 5.0 * L_point
+ 1.0 * L_range
+ 0.1 * L_smooth
```

İlk etapta yok:

```text
L_token yok
L_evidence yok
L_confidence yok
L_visibility yok
```

`L_exist`:

```text
matched slots     → lane class
unmatched slots   → no-lane class
```

`L_point`:

```text
sadece matched positive slotlarda
sadece valid rowlarda
```

`L_range`:

```text
matched positive slotlarda
```

`L_smooth`:

```text
predicted x rows üzerinde ikinci fark cezası
```

Ama `L_smooth` ilk 10 image overfit sırasında kapatılabilir. Çünkü önce modelin öğrenip öğrenmediğini görmek istiyoruz.

---

# 12. İlk debug hedefleri

S0 çalıştı demek için şu testleri geçmesi lazım.

## Test 1 — GT decode testi

Dataset’ten GT lane al:

```text
raw lane points
→ resize
→ fixed rows target
→ tekrar lane points olarak çiz
```

Eğer çizim orijinal lane üstüne düşmüyorsa model yazmaya başlamıyoruz. Önce target builder düzeltilir.

---

## Test 2 — 10 image overfit

10 görüntü seçilecek.

Hedef:

```text
training loss net düşmeli
predicted lanes görüntü üstünde GT’ye yaklaşmalı
empty slotlar boş kalmalı
```

Bu testte metric önemli değil. Görsel doğru mu ona bakacağız.

---

## Test 3 — matching visualization

Her görüntü için:

```text
GT lane 0 hangi slot ile eşleşti?
GT lane 1 hangi slot ile eşleşti?
unmatched slotlar no-lane oldu mu?
```

Bunu terminal log ve görsel olarak çıkaracağız.

---

## Test 4 — coordinate sanity

Her batch’te şunlar assert edilecek:

```text
0 <= pred_x <= W_in
0 <= gt_x <= W_in
valid_mask sum > 0 for existing lanes
range_start <= range_end
```

Bunlar yoksa loss sessizce bozulur.

---

# 13. İlk repo iskeleti

İlk aşamada minimal repo şöyle olmalı:

```text
dynlaneseq_eg/
│
├── configs/
│   └── culane_s0_res34.yaml
│
├── data/
│   ├── culane_dataset.py
│   ├── lane_target_builder.py
│   └── transforms.py
│
├── modeling/
│   ├── backbone_resnet.py
│   ├── fpn.py
│   ├── position_encoding.py
│   ├── lane_queries.py
│   ├── cross_attention.py
│   ├── heads_s0.py
│   └── dynlaneseq_s0.py
│
├── losses/
│   ├── matcher.py
│   ├── losses_s0.py
│   └── smoothness.py
│
├── engine/
│   ├── train_one_epoch.py
│   ├── validate.py
│   └── visualizer.py
│
├── tools/
│   ├── train.py
│   ├── debug_overfit.py
│   └── visualize_targets.py
│
└── README.md
```

Dikkat: Şimdilik `tokenizer.py`, `bridge.py`, `zoom_refinement.py` yok. Bunları sonra ekleyeceğiz.

---

# 14. Bu partın sonunda kilitlediğimiz kararlar

Şimdilik kararlarımız:

```text
Dataset: CULane
Input size: 288 × 800
Backbone: ResNet-34
FPN output: B × 128 × 72 × 200
Lane slots: N = 20
Fixed rows: P = 72
X bins: 200
First model: S0 geometry-only
Matching: object + point + range
No token CE in matching
No low-rank bridge yet
No zoom-in yet
No evidence consistency loss yet
Soft coordinate decoding mandatory
10 image overfit mandatory
```

Bu kararlar değişebilir ama ilk implementation için sabit tutacağız.

---

# DynLaneSeq-EG Implementation Plan — Part 2

## Dataset, annotation parser ve fixed-row target builder

Bu part modelden bile daha önemli. Çünkü **target builder yanlışsa model doğru olsa bile öğrenemez.** CondLSTR gibi modellerde paper’da birkaç formül gibi görünen şeyin kodda uzun target preprocessing’e dönüşmesinin sebebi bu. Biz de aynı tuzağa düşmemek için önce dataset tarafını çok net kilitleyeceğiz.

---

# 1. Part 2’nin amacı

Bu partın sonunda elimizde şu sistem olacak:

```text
CULane raw annotation
→ image resize
→ lane point cleaning
→ fixed-row interpolation
→ valid mask
→ lane range
→ training target
→ visualization check
```

Yani model daha yokken bile şunu test edebileceğiz:

```text
GT annotation oku
target'a çevir
tekrar lane olarak çiz
görüntünün üstüne doğru oturuyor mu?
```

Bu test geçmeden model kodlamaya başlamıyoruz.

---

# 2. CULane annotation formatı

CULane’da her görüntü için genelde bir `.lines.txt` annotation dosyası olur.

Bir satır bir lane’i temsil eder:

```text
x1 y1 x2 y2 x3 y3 ...
```

Örnek mantık:

```text
450 590 455 580 460 570 468 560 ...
```

Yani:

```text
Lane 0 = [(450,590), (455,580), (460,570), ...]
Lane 1 = [...]
Lane 2 = [...]
```

Bazı noktalar geçersiz olabilir veya görüntü dışına taşabilir. Bu yüzden parser doğrudan “her sayı doğru” diye kabul etmeyecek.

---

# 3. Dataset class ne döndürmeli?

İlk MVP S0 için dataset class şu formatta çıktı vermeli:

```text
sample = {
    "image": Tensor[3, H_in, W_in],
    "image_path": str,
    "orig_size": (H_orig, W_orig),
    "input_size": (H_in, W_in),

    "gt_lanes_raw": List[List[(x_orig, y_orig)]],
    "gt_lanes_resized": List[List[(x_in, y_in)]],

    "targets": {
        "x_rows": Tensor[M, P],
        "valid_mask": Tensor[M, P],
        "range_y": Tensor[M, 2],
        "exist": Tensor[M],
    }
}
```

Burada:

```text
M = görüntüdeki gerçek lane sayısı
P = fixed row sayısı
```

Başlangıçta:

```text
H_in = 288
W_in = 800
P = 72
```

---

# 4. Coordinate sistemi tekrar net

CULane annotation genelde original image coordinate sistemindedir.

Biz modeli şu input boyutuyla eğiteceğiz:

```text
W_in = 800
H_in = 288
```

Dönüşüm:

```text
x_in = x_orig * W_in / W_orig
y_in = y_orig * H_in / H_orig
```

Bu dönüşüm hem image resize ile hem annotation resize ile birebir aynı olmalı.

Çok kritik:

```text
Image resize hangi oranla yapılıyorsa,
GT lane noktaları da aynı oranla resize edilecek.
```

Eğer image crop edersen, GT’de de crop offset düşmelisin. İlk MVP’de bu yüzden crop kullanmıyoruz. Sadece resize.

---

# 5. İlk MVP augmentation politikası

İlk 10 image / 100 image overfit aşamasında augmentation kapalı olacak.

```text
augmentation = none
resize = yes
normalize = yes
```

Neden?

Çünkü ilk hedef model performansı değil, pipeline doğruluğu.

Sonra augmentation ekleyeceğiz:

```text
random horizontal flip
color jitter
random brightness
small affine
```

Ama ilk target builder debug aşamasında bunları kapatıyoruz.

---

# 6. Fixed-row representation

Lane’i her pixelde değil, sabit y satırlarında temsil edeceğiz.

Başlangıç:

```text
P = 72
H_in = 288
row_stride = 4
```

Yani:

```text
y_rows = [0, 4, 8, 12, ..., 284]
```

Her lane için:

```text
x_rows     ∈ R^72
valid_mask ∈ {0,1}^72
```

Eğer lane o row’da varsa:

```text
x_rows[p] = lane'in o y satırındaki x değeri
valid_mask[p] = 1
```

Yoksa:

```text
x_rows[p] = -1
valid_mask[p] = 0
```

---

# 7. Neden fixed-row kullanıyoruz?

Çünkü lane detection’da görüntü y ekseni boyunca ilerleyen ince bir yapı var. CULane/TuSimple tarzı datasetlerde lane’i birçok benchmark zaten satır satır değerlendiriyor.

Bu representation bize şunları sağlar:

```text
1. GT target basit olur.
2. Matching point distance kolay hesaplanır.
3. Smoothness loss kolay yazılır.
4. Tokenizer ileride row-wise offset tokenlarına kolay döner.
5. Visualization kolaydır.
```

Bu yüzden ilk modelde Bezier, spline, polygon mask gibi şeylerle başlamıyoruz.

---

# 8. Lane point cleaning

Annotation’dan gelen her lane için önce temizleme yapacağız.

Bir lane listesi:

```text
lane = [(x1,y1), (x2,y2), ...]
```

Şu adımlar uygulanacak:

## 8.1 Geçersiz noktaları at

Şunlar atılır:

```text
x < 0
y < 0
x >= W_orig
y >= H_orig
NaN
inf
```

Resize sonrası da tekrar clamp/check yapılır.

---

## 8.2 Duplicate y değerlerini çöz

Bazen aynı y satırında birden fazla nokta olabilir.

Örnek:

```text
(450, 200), (452, 200)
```

Bunu tek noktaya indirmek gerekir.

Basit çözüm:

```text
aynı y için x ortalaması al
```

Yani:

```text
x_mean = mean(x values at same y)
```

---

## 8.3 y’ye göre sırala

Interpolation için lane noktaları y’ye göre sıralı olmalı.

```text
lane_points = sorted(lane_points, key=y)
```

Dikkat: Görüntü coordinate sisteminde y aşağı doğru artar.

---

## 8.4 Çok kısa lane’i at

Eğer lane’de yeterli nokta yoksa training target’a koymayacağız.

Başlangıç kuralı:

```text
min_raw_points = 2
min_valid_rows = 5
```

Yani interpolation sonrası lane en az 5 fixed row’da görünmeli.

Aksi halde:

```text
discard lane
```

---

# 9. Interpolation nasıl yapılacak?

Elimizde resized lane points var:

```text
[(x1, y1), (x2, y2), ..., (xK, yK)]
```

Amacımız fixed y rows için x bulmak:

```text
for y in y_rows:
    x = interpolate_x_at_y(lane_points, y)
```

## 9.1 Hangi y aralığı valid?

Lane’in minimum ve maksimum y değeri:

```text
y_min = min(y_points)
y_max = max(y_points)
```

Bir fixed row valid sayılırsa:

```text
y_min <= y_row <= y_max
```

Ama bu tek başına yetmez. Çünkü lane noktaları arasında çok büyük boşluk varsa interpolation yanlış olabilir.

---

## 9.2 Segment bazlı interpolation

Daha sağlam yöntem:

Her iki ardışık nokta için segment oluştur:

```text
(xa, ya) → (xb, yb)
```

Eğer fixed row bu segmentin y aralığındaysa:

```text
t = (y_row - ya) / (yb - ya)
x = xa + t * (xb - xa)
```

Eğer birden fazla segment aynı row’u kapsarsa, ortalama alınabilir.

---

## 9.3 Horizontal segment problemi

Eğer:

```text
yb == ya
```

ise division by zero olur.

Bu segmenti interpolation için atacağız.

```text
if abs(yb - ya) < eps:
    skip
```

---

## 9.4 x görüntü dışına çıkarsa

Interpolation sonucu:

```text
x < 0 veya x >= W_in
```

olursa o row invalid yapılır.

```text
valid_mask[p] = 0
x_rows[p] = -1
```

Clamp yapmak ilk etapta riskli. Çünkü görüntü dışındaki tahmini zorla kenara yapıştırmak modelin yanlış öğrenmesine sebep olabilir.

---

# 10. Range target nasıl üretilecek?

Her lane için `valid_mask` çıktıktan sonra:

```text
valid_indices = where(valid_mask == 1)
```

Eğer valid row yoksa lane atılır.

Sonra:

```text
start_idx = min(valid_indices)
end_idx = max(valid_indices)
```

Bunları iki şekilde saklayabiliriz.

## Seçenek A — row index olarak

```text
range_idx = [start_idx, end_idx]
```

Örneğin:

```text
[20, 71]
```

## Seçenek B — input y coordinate olarak

```text
range_y = [y_rows[start_idx], y_rows[end_idx]]
```

Örneğin:

```text
[80, 284]
```

Ben S0 için `range_y` öneriyorum:

```text
range_y ∈ [0, H_in]
```

Çünkü model head doğrudan y coordinate regrese edebilir.

Target:

```text
range_y = Tensor[M, 2]
```

Normalize edersek:

```text
range_norm = range_y / H_in
```

Model için normalized range daha stabil olur.

Başlangıç kararı:

```text
range target normalized olacak.
range_pred sigmoid ile [0,1] aralığında çıkacak.
```

Sonra input coordinate’e çevrilir:

```text
range_y_pred = range_pred * H_in
```

---

# 11. x target normalize edilecek mi?

İki seçenek var.

## Seçenek A — x input pixel olarak

```text
x_rows ∈ [0, 800]
```

Loss:

```text
SmoothL1(pred_x_pixel, gt_x_pixel)
```

## Seçenek B — x normalized

```text
x_rows_norm = x_rows / W_in
```

Loss:

```text
SmoothL1(pred_x_norm, gt_x_norm)
```

Ben S0 için şunu öneriyorum:

```text
GT target x pixel olarak saklansın.
Model output soft bins üzerinden pixel’e çevrilsin.
Loss pixel coordinate üzerinden hesaplansın.
```

Neden?

Çünkü visualization ve debugging daha kolay.

Ama loss scale çok büyürse point loss’u normalize ederiz:

```text
L_point = SmoothL1(pred_x / W_in, gt_x / W_in)
```

Implementation’da ikisini de destekleyecek şekilde yazmak iyi olur.

İlk default:

```text
point loss normalized coordinate ile hesaplanacak.
```

Yani:

```text
pred_x_norm = pred_x / W_in
gt_x_norm = gt_x / W_in
```

---

# 12. Target tensor shape

Bir görüntüde M lane varsa:

```text
x_rows:      M × P
valid_mask:  M × P
range_y:     M × 2
exist:       M
```

Örnek:

```text
M = 4
P = 72

x_rows.shape      = [4, 72]
valid_mask.shape  = [4, 72]
range_y.shape     = [4, 2]
exist.shape       = [4]
```

`exist` aslında bütün GT lane’ler için 1’dir:

```text
exist = [1, 1, 1, 1]
```

No-lane sınıfı prediction slotları için loss aşamasında atanacak. Dataset target içine no-lane koymayacağız.

---

# 13. Batch collate problemi

Her görüntüde lane sayısı farklıdır.

Örnek:

```text
image 1: M=4
image 2: M=2
image 3: M=6
```

Bu yüzden targetları tek tensor olarak batchlemek zor.

İki seçenek var.

## Seçenek A — List target

PyTorch detection modelleri gibi:

```text
images: Tensor[B, 3, H, W]
targets: List[Dict]
```

Her target kendi M sayısını korur.

Bu daha kolay.

İlk MVP’de bunu kullanacağız.

```text
batch = {
    "images": Tensor[B, 3, H, W],
    "targets": List[Dict],
    "metas": List[Dict]
}
```

---

## Seçenek B — Pad target

Max lane sayısına kadar padlemek:

```text
x_rows: B × M_max × P
```

Ama matcher için list formatı daha temiz.

Karar:

```text
S0 için targets list-of-dict olacak.
```

---

# 14. Dataset parser pseudocode

Kod yazmayacağım ama mantık şöyle olmalı:

```text
load image
read annotation lines
for each line:
    parse floats
    pair as (x,y)
    remove invalid points
    resize points to input coordinate
    sort by y
    remove duplicate y
    interpolate to fixed y_rows
    if valid rows >= min_valid_rows:
        add lane target
return image, targets, meta
```

---

# 15. Horizontal flip target dönüşümü

Augmentation sonradan gelecek ama target builder bunu desteklemeli.

Eğer image horizontal flip yapılırsa:

```text
x_new = W_in - 1 - x_old
```

Bu hem raw resized lane points’e hem fixed-row x target’a uygulanabilir.

Dikkat:

```text
valid_mask değişmez
range_y değişmez
```

Ama lane sırası değişebilir. Biz set prediction kullandığımız için lane sırası önemli değil.

Yani:

```text
left-to-right reorder zorunlu değil
```

Ama visualization için istersen lane’leri bottom x değerine göre sıralayabilirsin. Training bunu gerektirmiyor.

---

# 16. Affine augmentation şimdilik yok

Random affine güzel ama ilk MVP’de kapalı.

Çünkü affine uygulayınca her lane point’e aynı transform matrix uygulanmalı. Sonra görüntü dışına çıkan noktalar temizlenmeli, lane yeniden interpolation’dan geçmeli.

Bu yapılabilir ama debug aşamasını zorlaştırır.

Karar:

```text
S0 debug: no affine
S0 full train: horizontal flip + color jitter
S1 sonrası: small affine denenebilir
```

---

# 17. Normalization

Image için klasik ImageNet normalization kullanılabilir:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Ama visualization yaparken unnormalize fonksiyonu şart.

`visualizer.py` içinde:

```text
unnormalize(image)
draw_gt_lanes()
draw_pred_lanes()
save
```

---

# 18. Target visualizer zorunlu

Modelden önce şu komutu yazacağız:

```text
python tools/visualize_targets.py --num 50
```

Bu script şunları kaydedecek:

```text
debug/targets/sample_000.jpg
debug/targets/sample_001.jpg
...
```

Her görselde:

```text
resized image
GT raw resized lane points
fixed-row interpolated lane points
range start/end noktaları
```

Farklı renklerle çizilecek.

Bu görselde şunları kontrol edeceğiz:

```text
Lane çizgileri doğru yerde mi?
Interpolation lane’i bozmuyor mu?
Yukarı/aşağı tersliği var mı?
x scale doğru mu?
range doğru mu?
çok kısa lane’ler atılıyor mu?
```

Bu script yoksa training’e geçmek yasak.

---

# 19. Target builder testleri

Unit test gibi düşüneceğiz.

## Test 1 — Shape test

Bir sample için:

```text
x_rows.shape[1] == P
valid_mask.shape == x_rows.shape
range_y.shape[1] == 2
```

---

## Test 2 — Valid x test

Valid olan yerlerde:

```text
0 <= x_rows[p] < W_in
```

Invalid olan yerlerde:

```text
x_rows[p] == -1
```

---

## Test 3 — Range test

Her lane için:

```text
range_y[0] <= range_y[1]
range_y[0] >= 0
range_y[1] < H_in
```

---

## Test 4 — Mask-range consistency

Valid rowların y aralığı ile range uyuşmalı:

```text
y_rows[first_valid] == range_start
y_rows[last_valid] == range_end
```

Küçük floating fark olabilir ama index bazında tutarlı olmalı.

---

## Test 5 — Decode consistency

Target’ı tekrar point listesine çevir:

```text
decoded_points = [(x_rows[p], y_rows[p]) for valid p]
```

Bunları image üstüne çiz. Lane raw annotation ile aynı hatta olmalı.

---

# 20. Target builder output örneği

Bir görüntüde 3 lane olduğunu düşünelim.

```text
targets = {
    "x_rows": Tensor[
        [ -1, -1, ..., 420.5, 421.3, 422.1, ...],
        [ -1, -1, ..., 510.7, 512.2, 514.1, ...],
        [180.2, 181.0, 182.5, ..., -1, -1]
    ],

    "valid_mask": Tensor[
        [0, 0, ..., 1, 1, 1, ...],
        [0, 0, ..., 1, 1, 1, ...],
        [1, 1, 1, ..., 0, 0]
    ],

    "range_y": Tensor[
        [96, 284],
        [112, 284],
        [0, 180]
    ],

    "exist": Tensor[
        1, 1, 1
    ]
}
```

Bu target S0 matcher/loss için yeterli.

---

# 21. Çok önemli karar: lane ordering kullanmayacağız

Dataset’te lane’ler belli bir sırada gelebilir. Ama biz buna güvenmeyeceğiz.

Training’de:

```text
prediction slots = unordered set
GT lanes = unordered set
Hungarian matching = hangi slot hangi GT'ye atanacak karar verir
```

Bu yüzden target builder lane’leri left-to-right sıralamak zorunda değil.

Ama debug görselleştirmede daha anlaşılır olsun diye opsiyonel sıralama yapılabilir:

```text
sort by x at bottom-most valid row
```

Bu sadece visualization veya logging için kullanılacak.

---

# 22. `min_valid_rows` kararı

Başlangıç:

```text
min_valid_rows = 5
```

Neden çok düşük değil?

Çünkü 1-2 noktadan oluşan lane, model için gürültü olabilir.

Neden çok yüksek değil?

Çünkü uzak lane’ler kısa görünebilir; onları tamamen atmak istemeyiz.

Ablation değil, pratik preprocessing kararı.

İlk debug’da eğer çok fazla lane atılıyorsa:

```text
min_valid_rows = 3
```

yapılabilir.

---

# 23. Row seçimi: yukarıdan aşağıya mı, aşağıdan yukarıya mı?

`y_rows` doğal olarak yukarıdan aşağıya olabilir:

```text
[0, 4, 8, ..., 284]
```

Ama lane modellerinde bazen aşağıdan yukarıya sıra daha sezgiseldir. Çünkü lane genelde aracın önünden başlar.

Tokenizer aşamasında aşağıdan yukarıya üretmek isteyebiliriz.

Karar:

Dataset target içinde:

```text
y_rows ascending: top → bottom
```

Decoder/tokenizer aşamasında gerekirse reverse edilir.

Neden?

Çünkü coordinate sistemiyle uyumlu ve interpolation daha temiz.

---

# 24. Range start/end anlamı

Bu karışmasın.

Bizim sistemde:

```text
range_start = en küçük y
range_end   = en büyük y
```

Yani görüntüde:

```text
range_start daha yukarıda
range_end daha aşağıda
```

Araç perspektifinde “lane başlangıcı” bazen bottom taraf demek olabilir. Ama coordinate sisteminde start/end kavramını böyle sabitleyeceğiz.

İsimleri daha açık yapmak için kodda şunu kullanmak daha iyi:

```text
y_min
y_max
```

Model head adı:

```text
range_pred = [y_min, y_max]
```

Böylece “start mı bottom mı?” karışmaz.

---

# 25. Dataset config

`configs/culane_s0_res34.yaml` içinde dataset kısmı şöyle olmalı:

```text
dataset:
  name: CULane
  root: /path/to/CULane
  train_list: list/train_gt.txt
  val_list: list/val.txt

input:
  height: 288
  width: 800

target:
  num_rows: 72
  row_stride: 4
  min_raw_points: 2
  min_valid_rows: 5
  invalid_x: -1.0

augmentation:
  train_debug: false
  horizontal_flip_prob: 0.5
  color_jitter: true
  affine: false
```

İlk debug config’te augmentation kapalı:

```text
augmentation:
  horizontal_flip_prob: 0.0
  color_jitter: false
  affine: false
```

---

# 26. Dataset output meta bilgisi

Her sample meta olarak şunları taşımalı:

```text
meta = {
    "image_path": "...",
    "anno_path": "...",
    "orig_h": H_orig,
    "orig_w": W_orig,
    "input_h": 288,
    "input_w": 800,
    "scale_x": W_in / W_orig,
    "scale_y": H_in / H_orig,
    "num_gt_lanes": M
}
```

Bunlar debugging için çok işe yarar.

Özellikle prediction’ı original image coordinate’e geri çevirmek istersek:

```text
x_orig = x_in / scale_x
y_orig = y_in / scale_y
```

---

# 27. Model prediction ile target uyumu

S0 model output:

```text
pred_x_rows: B × N × P
pred_exist_logits: B × N × 2
pred_range: B × N × 2
```

Dataset target:

```text
gt_x_rows: M × P
gt_valid_mask: M × P
gt_range: M × 2
```

Matcher her image için ayrı çalışır:

```text
for b in batch:
    cost_matrix: N × M
    Hungarian(cost_matrix)
```

Bu yüzden dataset target’ı batch içinde padlemiyoruz.

---

# 28. Çok önemli: valid mask olmadan point loss yazılmaz

Şu hatayı yapmayacağız:

```text
SmoothL1(pred_x_rows, gt_x_rows)
```

Çünkü invalid yerlerde `gt_x = -1`.

Doğru:

```text
loss = SmoothL1(pred_x[valid_mask], gt_x[valid_mask])
```

Eğer bir lane’de valid row yoksa zaten target builder onu atmış olmalı.

---

# 29. Target builder için minimal acceptance criteria

Part 2’nin sonunda “tamam bu çalışıyor” demek için:

```text
1. 50 target visualization doğru görünüyor.
2. invalid x değerleri loss’a girmiyor.
3. horizontal flip sonrası lane doğru aynalanıyor.
4. çok kısa lane’ler filtreleniyor.
5. batch collate farklı lane sayılarını bozmayıp list target döndürüyor.
```

Bunlar bitmeden model S0’a geçilmez.

---

# 30. Part 2 özeti

Bu partta kilitlediğimiz şeyler:

```text
Dataset: CULane
Input size: 288 × 800
Representation: fixed-row lane
Rows: 72 adet, stride 4
Target:
  x_rows: M × 72
  valid_mask: M × 72
  range_y: M × 2
  exist: M
Invalid x: -1
Min valid rows: 5
Batch target: list-of-dict
Ordering: unordered set
Augmentation debug’da kapalı
Target visualizer zorunlu
```

# DynLaneSeq-EG Implementation Plan — Part 3

## S0 Model Mimarisi: geometry-only sanity model

Bu partta ilk çalışan model olan **S0**’ı netleştiriyoruz. Bu model paper’daki final model olmayacak. Amacı sadece şunu kanıtlamak:

```text
Dataset target doğru mu?
Lane slot mantığı çalışıyor mu?
Hungarian matching çalışıyor mu?
Model 10 image overfit edebiliyor mu?
```

Yani S0, bizim “temel iskelet sağlam mı?” test modelimiz.

---

# 1. S0 modelinin görevi

S0 şunu yapacak:

```text
Image
→ feature map
→ lane slot queries
→ her slot için lane var/yok tahmini
→ her slot için row-wise x tahmini
→ her slot için y-range tahmini
→ final lane points
```

S0’da henüz şunlar yok:

```text
token decoder yok
Lane2Seq yok
curve-aligned sampler yok
dynamic low-rank bridge yok
zoom-in refinement yok
evidence consistency loss yok
visibility head yok
confidence calibration yok
```

Bu modelin çıktısı doğrudan row-wise lane point olacak.

---

# 2. S0 genel akış

```text
Input image
  ↓
ResNet-34 backbone
  ↓
FPN neck
  ↓
2D positional encoding
  ↓
Flatten feature map
  ↓
Lane slot queries
  ↓
Cross-attention
  ↓
Prediction heads
  ↓
Soft coordinate decoding
  ↓
Predicted lane points
```

Bunu çok sade tutuyoruz. Çünkü ilk amaç novelty değil, **çalışan training graph**.

---

# 3. Input shape

Model input:

```text
images: B × 3 × 288 × 800
```

Burada:

```text
B = batch size
H_in = 288
W_in = 800
```

İlk debug için batch size küçük olabilir:

```text
B = 2 veya 4
```

10 image overfit sırasında performans değil stabilite önemli.

---

# 4. Backbone: ResNet-34

İlk backbone:

```text
ResNet-34
```

Output olarak çok seviyeli feature alacağız:

```text
C2: B ×  64 × 72 × 200
C3: B × 128 × 36 × 100
C4: B × 256 × 18 × 50
C5: B × 512 ×  9 × 25
```

Neden C2 önemli?

Çünkü lane çizgileri ince. Eğer sadece C5 gibi düşük çözünürlüklü feature kullanırsak ince lane yapıları kaybolabilir.

---

# 5. FPN neck

FPN şu feature’ları birleştirecek:

```text
C2, C3, C4, C5
```

Tek ana feature map üretecek:

```text
F: B × 128 × 72 × 200
```

Burada:

```text
C_fpn = 128
Hf = 72
Wf = 200
```

Bu çözünürlük input’un 1/4’ü:

```text
288 / 4 = 72
800 / 4 = 200
```

Bu S0 için iyi bir denge: lane detayını koruyor ama VRAM’i çok patlatmıyor.

---

# 6. 2D positional encoding

FPN çıktısı tek başına spatial konumu tam temsil etmez. O yüzden positional encoding ekleyeceğiz.

Input:

```text
F: B × 128 × 72 × 200
```

Positional encoding:

```text
PE: 1 × 128 × 72 × 200
```

Output:

```text
F_pos = F + PE
F_pos: B × 128 × 72 × 200
```

Başlangıçta sinusoidal 2D positional encoding yeterli.

Bu katmanın amacı:

```text
Model sol/sağ/yukarı/aşağı konumunu bilsin.
```

Çünkü lane detection’da konum bilgisi çok önemli.

---

# 7. Feature projection

Transformer/cross-attention boyutu için FPN channel’ını 256’ya çıkaracağız.

Input:

```text
F_pos: B × 128 × 72 × 200
```

1×1 projection sonrası:

```text
F_proj: B × 256 × 72 × 200
```

Sonra flatten:

```text
F_mem: B × 14400 × 256
```

Çünkü:

```text
72 × 200 = 14400
```

Bu `F_mem`, cross-attention için memory olacak.

---

# 8. Lane slot queries

Learnable lane queries kullanacağız.

Başlangıç:

```text
N_slots = 20
D = 256
```

Learnable query table:

```text
Q_table: 20 × 256
```

Batch’e expand edilir:

```text
Q0: B × 20 × 256
```

Buradaki her query bir lane adayıdır.

Önemli: Query’lere “sol lane”, “sağ lane”, “orta lane” gibi sabit anlam vermiyoruz. Bunlar unordered lane slot’tur. Hangi slotun hangi GT lane ile eşleşeceğine Hungarian matching karar verecek.

---

# 9. Cross-attention block

S0’da decoder gibi ağır bir yapı değil, basit cross-attention bloğu kullanacağız.

Input:

```text
Queries: Q0     → B × 20 × 256
Memory:  F_mem  → B × 14400 × 256
```

Cross-attention:

```text
Q1 = CrossAttention(query=Q0, key=F_mem, value=F_mem)
```

Output:

```text
Q1: B × 20 × 256
```

Bu şu anlama gelir:

```text
Her lane slotu görüntüden kendisiyle ilgili bilgi toplar.
```

İlk S0 için cross-attention layer sayısı:

```text
num_layers = 2
```

Her layer:

```text
self-attention over lane slots
cross-attention to image feature
feed-forward network
layer norm
residual connection
```

Bu DETR decoder bloğuna benzer ama daha küçük tutulacak.

---

# 10. Neden lane slot self-attention var?

Lane slotlarının birbirinden haberi olsun istiyoruz.

Örneğin iki slot aynı lane’e bakmaya çalışıyorsa self-attention bunu azaltmaya yardım edebilir.

Layer içinde sıralama:

```text
Q = self_attn(Q)
Q = cross_attn(Q, F_mem)
Q = FFN(Q)
```

Output yine:

```text
B × 20 × 256
```

---

# 11. Prediction heads

Cross-attention sonrası elimizde:

```text
Q1: B × 20 × 256
```

Her slot için üç ana head olacak:

```text
existence head
row-wise x head
range head
```

---

## 11.1 Existence head

Görev:

```text
Bu slot lane mi, no-lane mi?
```

Input:

```text
Q1: B × 20 × 256
```

Output:

```text
exist_logits: B × 20 × 2
```

Class anlamı:

```text
class 0 = lane
class 1 = no-lane
```

Bunu baştan sabitliyoruz. Karışıklık olmasın.

Inference sırasında:

```text
lane_prob = softmax(exist_logits)[..., 0]
```

Eşik:

```text
lane_prob >= 0.5
```

İlk başta 0.5. Sonra validation’a göre 0.3/0.5/0.7 denenebilir.

---

## 11.2 Row-wise x head

Görev:

```text
Her lane slotu için 72 row’da x konumu tahmin et.
```

Input:

```text
Q1: B × 20 × 256
```

Output:

```text
row_x_logits: B × 20 × 72 × 200
```

Burada:

```text
P = 72
X_bins = 200
```

Her row için 200 x-bin var. Çünkü feature map width 200.

Bir row için:

```text
row_x_logits[b, slot, row, :] → 200 class-like bin logit
```

Ama bunu class olarak argmax yapmayacağız. Soft coordinate decoding yapacağız.

---

## 11.3 Range head

Görev:

```text
Lane hangi y aralığında geçerli?
```

Input:

```text
Q1: B × 20 × 256
```

Output:

```text
range_raw: B × 20 × 2
```

Sigmoid sonrası:

```text
range_norm: B × 20 × 2
```

Burada:

```text
range_norm[..., 0] = y_min_norm
range_norm[..., 1] = y_max_norm
```

Ama model bazen `y_min > y_max` üretebilir. Bunu engellemek için iki seçenek var.

---

# 12. Range head için doğru parametrizasyon

Direkt iki sayı üretmek yerine daha güvenli parametrizasyon kullanacağız.

Head şunu üretsin:

```text
range_center_raw: B × 20 × 1
range_length_raw: B × 20 × 1
```

Sonra:

```text
center = sigmoid(range_center_raw)
length = sigmoid(range_length_raw)

y_min = center - length / 2
y_max = center + length / 2
```

Sonra clamp:

```text
y_min = clamp(y_min, 0, 1)
y_max = clamp(y_max, 0, 1)
```

Ama clamp gradient’i kenarlarda zayıflatabilir. Daha basit başlangıç için şu da olabilir:

```text
range_pair = sigmoid(raw)
y_min = min(range_pair[0], range_pair[1])
y_max = max(range_pair[0], range_pair[1])
```

Fakat `min/max` da keskin operasyon.

Ben S0 için en sade ve yeterli çözümü seçiyorum:

```text
range_head doğrudan 2 sayı üretir.
sigmoid uygulanır.
loss hesaplamadan önce sort edilir.
```

Yani:

```text
r = sigmoid(range_raw)
y_min = min(r0, r1)
y_max = max(r0, r1)
```

Bu S0 için yeterli. Daha sonra center-length parametrizasyonuna geçilebilir.

---

# 13. Soft coordinate decoding

Bu kritik.

Row-wise x logits:

```text
row_x_logits: B × 20 × 72 × 200
```

Önce softmax:

```text
prob = softmax(row_x_logits, dim=-1)
```

Bin center:

```text
bin_centers_feat = [0, 1, 2, ..., 199]
```

Expected x feature coordinate:

```text
x_feat = Σ prob[k] * bin_centers_feat[k]
```

Shape:

```text
x_feat: B × 20 × 72
```

Input pixel coordinate’e çevir:

```text
x_input = x_feat * (W_in / Wf)
```

Çünkü:

```text
W_in / Wf = 800 / 200 = 4
```

Output:

```text
pred_x_rows: B × 20 × 72
```

Bu differentiable olur.

---

# 14. Neden argmax yok?

Çünkü argmax kullanırsak:

```text
logits → argmax bin → x coordinate
```

bu yoldan geometry loss gradient gönderemez.

Bizim yol:

```text
logits → softmax → expected coordinate → SmoothL1
```

Bu sayede `L_point` doğrudan row_x_logits’e gradient gönderir.

---

# 15. S0 forward output dictionary

Model forward sonunda şu dictionary dönecek:

```text
outputs = {
    "exist_logits": B × 20 × 2,
    "row_x_logits": B × 20 × 72 × 200,
    "pred_x_rows": B × 20 × 72,
    "range_raw": B × 20 × 2,
    "range_norm": B × 20 × 2,
    "features": optional,
    "queries": optional
}
```

Training için gerekli:

```text
exist_logits
pred_x_rows
range_norm
```

Debug için gerekli:

```text
row_x_logits
queries
features
```

---

# 16. Shape summary

Tek yerde görelim:

```text
images          : B × 3 × 288 × 800

C2              : B × 64  × 72 × 200
C3              : B × 128 × 36 × 100
C4              : B × 256 × 18 × 50
C5              : B × 512 × 9  × 25

FPN output F    : B × 128 × 72 × 200
F_proj          : B × 256 × 72 × 200
F_mem           : B × 14400 × 256

Q0              : B × 20 × 256
Q1              : B × 20 × 256

exist_logits    : B × 20 × 2
row_x_logits    : B × 20 × 72 × 200
pred_x_rows     : B × 20 × 72
range_norm      : B × 20 × 2
```

---

# 17. S0 model config

Config tarafında şu değerler olacak:

```text
model:
  name: DynLaneSeqS0

  input_height: 288
  input_width: 800

  backbone:
    name: resnet34
    pretrained: true
    out_indices: [1, 2, 3, 4]

  fpn:
    out_channels: 128
    output_stride: 4

  transformer:
    d_model: 256
    nhead: 8
    num_decoder_layers: 2
    dim_feedforward: 1024
    dropout: 0.1

  queries:
    num_slots: 20

  heads:
    num_rows: 72
    x_bins: 200
    exist_classes: 2
```

İlk debug’da `pretrained=true` kullanmak daha stabil olur. Ama backbone LR düşük olacak.

---

# 18. Positional encoding detayı

`F_proj` ile aynı channel boyutunda olmalı:

```text
PE: 1 × 256 × 72 × 200
```

Yani positional encoding’i FPN’den önce değil, projection’dan sonra eklemek daha temiz olabilir.

Akış şöyle olsun:

```text
F: B × 128 × 72 × 200
F_proj = Conv1x1(F) → B × 256 × 72 × 200
PE = PosEnc2D(256, 72, 200)
F_proj_pos = F_proj + PE
F_mem = flatten(F_proj_pos)
```

Bu daha net.

---

# 19. Mask/padding meselesi

İlk MVP’de tüm inputlar aynı boyuta resize edildiği için padding mask kullanmayacağız.

```text
padding_mask = None
```

Daha sonra farklı aspect ratio / padding kullanırsak padding mask eklenir.

Bu kararı özellikle veriyoruz çünkü CondLSTR gibi yapılarda padding mask önemli olabilir, ama bizim ilk MVP’de resize sabit olduğu için gereksiz karmaşıklık.

---

# 20. Attention memory maliyeti

Cross-attention şu boyutlarda çalışacak:

```text
query length = 20
memory length = 14400
d_model = 256
```

Attention map:

```text
B × nhead × 20 × 14400
```

Bu büyük ama yönetilebilir.

Eğer VRAM sorun çıkarsa iki çözüm var:

## Çözüm A — memory stride düşür

Feature map’i 1/8 yap:

```text
F: B × 128 × 36 × 100
memory length = 3600
```

## Çözüm B — deformable attention

Daha sonra eklenebilir ama S0’da kullanmıyoruz.

Başlangıç için full cross-attention ile debug daha kolay.

---

# 21. Prediction head mimarisi

Her head basit MLP olacak.

## Existence head

```text
Linear(256 → 256)
ReLU
Linear(256 → 2)
```

## Range head

```text
Linear(256 → 256)
ReLU
Linear(256 → 2)
Sigmoid later
```

## Row x head

Burada iki seçenek var.

### Seçenek A — direkt büyük MLP

```text
Linear(256 → 256)
ReLU
Linear(256 → P * X_bins)
reshape → P × X_bins
```

Yani:

```text
256 → 72*200 = 14400
```

Bu biraz büyük ama S0 için kabul edilebilir.

Parametre sayısı:

```text
256 × 14400 ≈ 3.7M
```

Bu çok korkunç değil.

### Seçenek B — row embedding ile daha düzenli head

Her row için ayrı embedding kullanıp query ile birleştirirsin. Daha temiz ama daha karmaşık.

S0 için karar:

```text
Direkt MLP kullan.
```

Çünkü debug daha kolay.

---

# 22. Row x head output reshape

Raw output:

```text
row_x_flat: B × 20 × 14400
```

Reshape:

```text
row_x_logits: B × 20 × 72 × 200
```

Bu reshape kesin assert ile kontrol edilecek.

---

# 23. Model initialization

Önemli başlangıçlar:

## Existence head bias

Başlangıçta çoğu slot no-lane olmalı. Çünkü her görüntüde 20 slot var ama genelde 2–6 lane var.

Bu yüzden existence head bias’ı no-lane tarafına hafif avantajlı başlatılabilir.

Örneğin:

```text
lane logit bias = -1.0
no-lane logit bias = +1.0
```

Ama 10 image overfit sırasında bu bazen öğrenmeyi yavaşlatabilir. İlk sade başlangıç:

```text
default initialization
```

Eğer çok fazla false positive çıkarsa bias ayarı eklenir.

## Row x head

Default Xavier/Kaiming yeterli.

## Range head

Range başlangıcı tüm görüntü olabilir:

```text
y_min ≈ 0
y_max ≈ 1
```

Ama default sigmoid genelde 0.5 civarı verir, range çok kısa olabilir. Bu sıkıntı çıkarırsa range loss başta zorlanır.

Alternatif initialization:

```text
range_raw bias = [-2, +2]
sigmoid(-2) ≈ 0.12
sigmoid(+2) ≈ 0.88
```

Bu iyi bir başlangıç olabilir.

Karar:

```text
Range head bias: [-2, 2]
```

Böylece model başta yaklaşık geniş bir y aralığı tahmin eder.

---

# 24. S0 inference output

Inference sırasında:

```text
lane_prob = softmax(exist_logits)[..., 0]
keep = lane_prob >= score_thresh
```

Başlangıç:

```text
score_thresh = 0.5
```

Her kept slot için:

```text
x_rows = pred_x_rows[slot]
y_rows = fixed y rows
range = predicted y_min, y_max
```

Range filtering:

```text
keep row p if y_min <= y_rows[p] <= y_max
```

Final lane points:

```text
[(x_rows[p], y_rows[p]) for p in kept_rows]
```

Minimum point filtresi:

```text
min_pred_points = 5
```

Eğer lane 5’ten az point içeriyorsa atılır.

---

# 25. S0’da valid mask prediction yok

Model her row için x tahmin ediyor. Ama lane her row’da geçerli olmayabilir. Bunu `range_head` ile filtreliyoruz.

Yani S0’da ayrı visibility head yok.

Bu basitleştirme şu demek:

```text
Lane’in y aralığı içindeki her row için x tahmin edilir.
```

Bu CULane için ilk MVP’de yeterli olabilir. Daha sonra visibility head eklenebilir.

---

# 26. S0 output visualizer

Predicted lane çizimi:

```text
for each kept slot:
    draw points after range filtering
```

GT çizimi:

```text
target x_rows + valid_mask
```

Görselde:

```text
GT = kalın çizgi
Prediction = ince çizgi
```

Ayrıca slot id ve score yazılmalı:

```text
slot 03 score=0.87
```

Bu matching debug için çok işe yarar.

---

# 27. S0 model acceptance criteria

S0 model “çalışıyor” demek için şu 5 kriteri geçmeli:

## Kriter 1 — Forward shape doğru

Bir batch forward sonrası:

```text
exist_logits.shape == [B, 20, 2]
row_x_logits.shape == [B, 20, 72, 200]
pred_x_rows.shape == [B, 20, 72]
range_norm.shape == [B, 20, 2]
```

---

## Kriter 2 — Loss finite

İlk batch’te:

```text
loss is not NaN
loss is not inf
```

---

## Kriter 3 — 10 image overfit

10 görüntüde:

```text
loss belirgin düşmeli
prediction GT’ye görsel olarak yaklaşmalı
```

---

## Kriter 4 — Empty slot öğrenimi

20 slotun hepsi lane üretmemeli.

Beklenen:

```text
görüntüde 4 lane varsa yaklaşık 4 slot lane, diğerleri no-lane
```

---

## Kriter 5 — Range mantıklı

Predicted range zamanla GT range’e yaklaşmalı.

Eğer range hep `[0, 1]` kalıyorsa range head öğrenmiyor demektir.
Eğer range çok daralıyorsa point loss’un etkisi azalabilir.

---

# 28. S0’da şimdilik yapmayacağımız şeyler

Bunları özellikle dışarıda bırakıyoruz:

```text
DETR encoder yok
multi-scale deformable attention yok
dynamic kernel yok
row visibility head yok
token CE yok
label smoothing yok
focal loss yok
NMS yok
polynomial smoothing yok
```

Sebep: İlk model küçük ve debug edilebilir olmalı.

---

# 29. S0 dosya/modül karşılığı

Bu partın repo karşılığı:

```text
modeling/
├── backbone_resnet.py
├── fpn.py
├── position_encoding.py
├── lane_queries.py
├── cross_attention_decoder.py
├── heads_s0.py
└── dynlaneseq_s0.py
```

Loss/matcher sonraki partta detaylanacak.

---

# 30. Part 3 özeti

Bu partta S0 mimarisini kilitledik:

```text
Input: 288 × 800
Backbone: ResNet-34
FPN output: B × 128 × 72 × 200
Projection: 128 → 256
Memory: B × 14400 × 256
Lane slots: 20
Decoder/cross-attention layers: 2
Existence output: B × 20 × 2
Row x logits: B × 20 × 72 × 200
Soft x output: B × 20 × 72
Range output: B × 20 × 2
No argmax during training
No token decoder yet
No bridge yet
No sampler yet
```

Bu model bizim ilk gerçek test alanımız olacak.

---

# DynLaneSeq-EG Implementation Plan — Part 4

## Hungarian Matching ve S0 Loss Sistemi

Bu part kritik. Çünkü S0 modelinin öğrenip öğrenmemesini büyük ölçüde **matching + loss tasarımı** belirleyecek.

Önce net karar:

```text
S0’da token CE yok.
S0’da evidence loss yok.
S0’da visibility loss yok.
S0’da confidence calibration yok.
```

Sadece şunu çözüyoruz:

```text
20 prediction slotu var.
Görüntüde M tane gerçek lane var.
Hangi prediction slotu hangi GT lane’e eşleşecek?
Eşleşmeyen slotlar no-lane olacak.
```

---

# 1. S0 output tekrar

Model forward çıktısı:

```text
outputs = {
    "exist_logits": B × N × 2,
    "pred_x_rows": B × N × P,
    "range_norm": B × N × 2,
    "row_x_logits": B × N × P × X_bins
}
```

Sabitler:

```text
B = batch size
N = 20 lane slots
P = 72 fixed rows
X_bins = 200
```

Class anlamı:

```text
exist class 0 = lane
exist class 1 = no-lane
```

Bu class mapping her yerde aynı olacak. CondLSTR kodunda foreground/background index karışıklığına benzer hata yaşamamak için bunu en baştan sabitliyoruz.

---

# 2. Target format tekrar

Her image için target:

```text
target = {
    "x_rows": M × P,
    "valid_mask": M × P,
    "range_y": M × 2,
    "exist": M
}
```

Burada:

```text
M = o görüntüdeki GT lane sayısı
```

`range_y` input pixel coordinate olabilir:

```text
range_y ∈ [0, H_in]
```

Loss/matching içinde normalize edeceğiz:

```text
range_norm = range_y / H_in
```

---

# 3. Matching neden gerekiyor?

Çünkü modelin slotları unordered.

Örnek:

```text
Slot 0 → sağ lane
Slot 1 → no-lane
Slot 2 → sol lane
Slot 3 → orta lane
...
```

Ama GT lane’lerin sırası dataset’te garanti değil. O yüzden “slot 0 = GT lane 0” diyemeyiz.

Bu yüzden her image için:

```text
prediction slots: N adet
GT lanes: M adet
cost matrix: N × M
Hungarian matching
```

kullanacağız.

---

# 4. Matching non-differentiable olacak, bu normal

Hungarian matching gradient almaz.

Akış:

```text
model prediction
→ cost matrix hesapla
→ cost matrix detach/cpu
→ scipy linear_sum_assignment
→ matched indexleri al
→ loss’u matched indexlere göre hesapla
```

Bu normal. DETR/CondLSTR tarzı set prediction sistemlerinde matching’in kendisi differentiable olmak zorunda değil.

Önemli olan:

```text
Matching sonucu seçilen prediction slotları üzerinden hesaplanan loss differentiable olacak.
```

---

# 5. Cost matrix bileşenleri

S0 matching cost:

```text
cost = λ_obj * cost_obj
     + λ_point * cost_point
     + λ_range * cost_range
```

Başlangıç ağırlıkları:

```text
λ_obj   = 2.0
λ_point = 5.0
λ_range = 1.0
```

Bunlar loss ağırlıklarıyla aynı olmak zorunda değil ama ilk MVP’de aynı mantığa yakın tutacağız.

---

# 6. Object cost

Prediction:

```text
exist_logits: N × 2
```

Softmax:

```text
exist_prob = softmax(exist_logits, dim=-1)
```

Lane olma olasılığı:

```text
p_lane = exist_prob[:, 0]
```

Object cost:

```text
cost_obj[i, j] = -log(p_lane[i] + eps)
```

Dikkat:

```text
cost_obj prediction slotuna bağlıdır, GT j’ye göre değişmez.
```

Yani N × M matrix’e broadcast edilir.

Amaç:

```text
lane olduğuna daha çok inanan slot,
GT lane’e eşleşmeye daha uygun olsun.
```

---

# 7. Point cost

Prediction:

```text
pred_x_rows: N × P
```

GT:

```text
gt_x_rows: M × P
gt_valid_mask: M × P
```

Her prediction slot `i` ve GT lane `j` için:

```text
valid = gt_valid_mask[j] == 1
```

Sadece valid rowlarda distance:

```text
diff = |pred_x_rows[i, valid] - gt_x_rows[j, valid]|
```

Normalize edilmiş point cost:

```text
cost_point[i, j] = mean(diff / W_in)
```

Yani pixel farkını direkt kullanmıyoruz; `W_in=800` ile bölüyoruz. Böylece cost scale daha stabil olur.

Örnek:

```text
80 px hata → 80 / 800 = 0.10
```

Bu, object cost ile daha dengeli olur.

---

# 8. GT valid row yoksa ne olacak?

Normalde target builder zaten `min_valid_rows=5` altında lane’i atacak.

Ama güvenlik için:

```text
if valid.sum() == 0:
    cost_point[i, j] = large_value
```

Örneğin:

```text
large_value = 1e6
```

Bu GT lane eşleşmeye uygun değil demektir.

Ama ideal durumda bu hiç olmayacak.

---

# 9. Range cost

Prediction:

```text
pred_range_norm: N × 2
```

GT:

```text
gt_range_y: M × 2
gt_range_norm = gt_range_y / H_in
```

Range sorting:

Prediction’da:

```text
pred_y_min = min(pred_range_norm[..., 0], pred_range_norm[..., 1])
pred_y_max = max(pred_range_norm[..., 0], pred_range_norm[..., 1])
```

GT zaten:

```text
gt_y_min <= gt_y_max
```

Cost:

```text
cost_range[i, j] =
    |pred_y_min[i] - gt_y_min[j]|
  + |pred_y_max[i] - gt_y_max[j]|
```

Bu normalized olduğu için 0–2 aralığında olur.

---

# 10. Cost scale örneği

Başlangıçta tipik değerler şöyle olabilir:

```text
cost_obj   ≈ 0.5 – 3.0
cost_point ≈ 0.05 – 0.40
cost_range ≈ 0.1 – 1.0
```

Ağırlıklarla:

```text
2.0 * cost_obj
5.0 * cost_point
1.0 * cost_range
```

Böylece point distance yeterince etkili olur ama object score tamamen ezilmez.

İlk debug’da cost değerlerini loglayacağız:

```text
mean cost_obj
mean cost_point
mean cost_range
mean total_cost
```

Eğer bir cost diğerlerini 100 kat eziyorsa ağırlıklar değiştirilecek.

---

# 11. Hungarian matching output

Her image için:

```text
cost_matrix: N × M
```

Hungarian sonucu:

```text
pred_indices
gt_indices
```

Örnek:

```text
pred_indices = [3, 7, 12, 18]
gt_indices   = [0, 1, 2, 3]
```

Anlamı:

```text
slot 3  → GT lane 0
slot 7  → GT lane 1
slot 12 → GT lane 2
slot 18 → GT lane 3
```

Eşleşmeyen diğer slotlar:

```text
no-lane
```

---

# 12. Eğer görüntüde hiç GT lane yoksa

Bazı datasetlerde no-lane image olabilir.

Eğer:

```text
M == 0
```

ise Hungarian matching yapılmaz.

Bu durumda:

```text
all slots → no-lane
```

Loss:

```text
L_exist hesaplanır
L_point = 0
L_range = 0
L_smooth = 0
```

Bu case’i açıkça handle edeceğiz.

---

# 13. Final training loss bileşenleri

S0 total loss:

```text
L_total =
  w_exist * L_exist
+ w_point * L_point
+ w_range * L_range
+ w_smooth * L_smooth
```

Başlangıç:

```text
w_exist = 2.0
w_point = 5.0
w_range = 1.0
w_smooth = 0.1
```

Ama ilk 10 image overfit sırasında:

```text
w_smooth = 0.0
```

olabilir. Çünkü önce modelin GT’ye yaklaşabildiğini görmek istiyoruz.

---

# 14. Existence loss

`exist_logits`:

```text
B × N × 2
```

Her image için target class oluşturacağız:

Başlangıçta tüm slotlar:

```text
exist_target = no-lane class = 1
```

Matched prediction slotları:

```text
exist_target[pred_indices] = lane class = 0
```

Sonra cross entropy:

```text
L_exist = CE(exist_logits.reshape(B*N, 2), exist_target.reshape(B*N))
```

---

# 15. No-lane class weighting

20 slot var ama çoğu no-lane olacak. Bu yüzden no-lane class sayısı lane class’tan fazla.

Başlangıçta iki seçenek var.

## Seçenek A — düz CE

```text
class_weight = [1.0, 1.0]
```

Bu debug için daha sade.

## Seçenek B — no-lane weight azalt

```text
class_weight = [1.0, 0.4]
```

CondLSTR tarzı sistemlerde no-object/eos weight genelde azaltılır. Bunun amacı modelin sürekli “no-lane” demeye kaçmasını engellemek.

Karar:

```text
İlk 10 image overfit: class_weight = [1.0, 1.0]
Full train başlangıcı: class_weight = [1.0, 0.4]
```

Eğer model her şeye no-lane diyorsa no-lane weight düşürülür.

---

# 16. Point loss

Point loss sadece matched positive slotlarda hesaplanır.

Her matched pair için:

```text
pred = pred_x_rows[pred_idx]      # P
gt   = gt_x_rows[gt_idx]          # P
mask = gt_valid_mask[gt_idx]      # P
```

Normalize:

```text
pred_norm = pred / W_in
gt_norm   = gt / W_in
```

Loss:

```text
SmoothL1(pred_norm[mask], gt_norm[mask])
```

Tüm matched lane’ler üzerinde ortalama alınır.

Önemli:

```text
invalid rowlar loss’a girmez.
```

Bu kural değişmeyecek.

---

# 17. SmoothL1 beta

SmoothL1 için beta seçimi önemli.

Normalized coordinate kullandığımız için hata 0–1 aralığında.

Başlangıç:

```text
beta = 1.0 / Wf
```

Çünkü x bin resolution 1/200 civarıdır:

```text
1 / 200 = 0.005
```

Pratik başlangıç:

```text
beta = 0.01
```

Bu şu demek:

```text
yaklaşık 8 px altı hata quadratic,
üstü L1 gibi davranır.
```

Eğer implementation framework default SmoothL1 kullanıyorsa beta default olabilir ama config’e koymak daha iyi.

---

# 18. Range loss

Range loss sadece matched positive slotlarda.

Prediction:

```text
pred_range_norm[pred_idx] = [r0, r1]
```

Sort:

```text
pred_y_min = min(r0, r1)
pred_y_max = max(r0, r1)
```

GT:

```text
gt_range_norm = gt_range_y / H_in
```

Loss:

```text
L_range = SmoothL1(pred_y_min, gt_y_min)
        + SmoothL1(pred_y_max, gt_y_max)
```

Tüm matched lane’ler üzerinde ortalama.

---

# 19. Smoothness loss

Smoothness loss lane’in zikzak yapmasını engeller.

Prediction:

```text
pred_x_rows[pred_idx] : P
```

Sadece GT valid rows üzerinde kullanacağız.

Ama ikinci türev için ardışık valid rowlar gerekiyor.

Basit yaklaşım:

```text
x_valid = pred_x_rows[valid]
```

Second difference:

```text
d2 = x_valid[2:] - 2*x_valid[1:-1] + x_valid[:-2]
```

Normalize:

```text
d2_norm = d2 / W_in
```

Loss:

```text
L_smooth = mean(abs(d2_norm))
```

Bu sadece matched positive slotlarda hesaplanır.

---

# 20. Smoothness loss tehlikesi

Bu loss fazla güçlü olursa gerçek curved lane’leri düzleştirir.

O yüzden:

```text
w_smooth = 0.0 first debug
w_smooth = 0.05 veya 0.1 sonra
```

Ayrıca çok keskin curve sahnelerinde smoothness loss’u azaltmak gerekebilir.

---

# 21. Point loss ile smoothness loss ilişkisi

Point loss GT’ye yaklaştırır.

Smoothness loss çizgiyi düzenler.

Ama smoothness, GT’ye karşı değil prediction’ın kendi şekline karşı çalışır.

Bu yüzden asıl loss:

```text
L_point
```

olmalı.

Smoothness sadece yardımcı regularization.

---

# 22. Row range ve point loss ilişkisi

Şu soruya dikkat:

> Point loss sadece GT valid rows’da hesaplanıyorsa, range head ne işe yarıyor?

Range head inference sırasında hangi rows’un lane’e ait olduğunu belirliyor.

Training’de:

```text
point loss → x koordinatlarını öğrenir
range loss → lane’in geçerli y aralığını öğrenir
exist loss → slot lane mi değil mi öğrenir
```

Yani range head, point loss’un maskesini üretmiyor. Mask GT’den geliyor. Range head sadece kendi target’ına karşı eğitiliyor.

---

# 23. Loss normalization

Loss’ları normalize etmek önemli.

## L_exist

Tüm `B × N` slot üzerinde ortalama:

```text
mean over all slots
```

## L_point

Tüm matched valid rowlar üzerinde ortalama:

```text
sum point loss / total_valid_points
```

Bu en doğru yöntem.

Yani her lane eşit değil, her valid point eşit ağırlık alır.

Alternatif lane başına ortalama alıp sonra lane ortalaması yapılabilir. İlk sürümde:

```text
point-level average
```

kullanacağız.

## L_range

Matched lane sayısı üzerinden ortalama:

```text
sum / num_matched_lanes
```

## L_smooth

Matched lane sayısı veya valid d2 count üzerinden ortalama.

Başlangıç:

```text
sum / total_smooth_terms
```

---

# 24. Hiç matched lane yoksa

Eğer batch’te hiçbir GT lane yoksa:

```text
num_matched = 0
```

Bu durumda:

```text
L_point = 0
L_range = 0
L_smooth = 0
```

Ama `L_exist` yine hesaplanır.

Loss tensor olarak 0 olmalı, Python float değil. Yoksa backward sorun çıkarabilir.

---

# 25. Matching cost için prediction detach edilecek mi?

Cost matrix hesaplanırken prediction tensorlarından değer kullanıyoruz. Ama Hungarian non-differentiable olduğu için cost’u detach etmek normal.

Akış:

```text
with torch.no_grad():
    cost_matrix = compute_cost(outputs, targets)
    pred_idx, gt_idx = hungarian(cost_matrix)
```

Sonra loss hesaplama kısmı `no_grad` dışında yapılır:

```text
loss = compute_loss(outputs, targets, matched_indices)
```

Böylece loss gradient verir.

---

# 26. CPU/GPU konusu

`scipy.optimize.linear_sum_assignment` CPU’da çalışır.

Bu yüzden:

```text
cost_matrix.detach().cpu().numpy()
```

yapacağız.

N küçük:

```text
N=20
M genelde 2–8
```

Bu yüzden overhead çok sorun olmaz.

Full train’de hız problemi olursa torch tabanlı matcher düşünülebilir ama gerek yok.

---

# 27. Matching visualization

Debug için her birkaç iterasyonda şunu kaydedeceğiz:

```text
image
GT lanes
pred lanes
matched lines
slot id
GT id
cost value
lane probability
```

Görsel üzerine örnek yazı:

```text
slot 07 → gt 02 | cost=0.31 | p=0.84
slot 12 → gt 00 | cost=0.42 | p=0.73
```

Bu çok önemli. Çünkü model overfit etmiyorsa önce matching yanlış mı ona bakacağız.

---

# 28. Cost matrix logging

Her batch için log:

```text
num_gt_lanes
num_matched
mean_p_lane_matched
mean_p_lane_unmatched
mean_cost_obj
mean_cost_point
mean_cost_range
total_loss
L_exist
L_point
L_range
L_smooth
```

Bu loglar olmadan training debug etmek çok zor.

---

# 29. Expected training behavior

İlk 10 image overfit sırasında beklenen davranış:

## İlk iterasyonlar

```text
p_lane slotlarda karışık
pred_x_rows çoğu ortalara yakın
range belirsiz
loss yüksek
```

## Birkaç yüz iterasyon sonra

```text
bazı slotlar sürekli belirli GT lane’lere eşleşir
p_lane matched slotlarda artar
unmatched slotlarda düşer
pred_x_rows GT’ye yaklaşır
```

## Overfit başarılıysa

```text
GT lane sayısı kadar slot lane olur
diğer slotlar no-lane olur
prediction çizgileri GT’ye oturur
```

---

# 30. Olası hata 1: Model her şeye no-lane diyor

Belirti:

```text
p_lane çok düşük
matched slotlar bile no-lane
L_exist düşüyor ama L_point iyi düşmüyor
```

Çözümler:

```text
no-lane class weight düşür: [1.0, 0.2]
w_point artır
exist head biasını lane’e biraz daha nötr yap
pretrained backbone kullan
learning rate kontrol et
```

---

# 31. Olası hata 2: Tüm slotlar lane üretmeye çalışıyor

Belirti:

```text
p_lane çoğu slotta yüksek
çok fazla duplicate prediction
unmatched slotlar no-lane öğrenmiyor
```

Çözümler:

```text
L_exist weight artır
no-lane class weight artır
score threshold yükselt
matching visualization kontrol et
```

---

# 32. Olası hata 3: Matching sürekli slot değiştiriyor

Belirti:

```text
aynı GT lane her iterasyonda farklı slota eşleşiyor
training noisy
slot specialization oluşmuyor
```

Bu ilk aşamada normal olabilir. Ama çok uzun sürerse:

```text
cost_point weight artır
object cost weight azalt
lane query self-attention kontrol et
learning rate düşür
```

İleride assignment stabilitesi için denoising queries veya one-to-many pretraining denenebilir ama S0’da yok.

---

# 33. Olası hata 4: x koordinatları ortada kalıyor

Belirti:

```text
pred_x_rows ≈ 400 civarında
lane şekli öğrenilmiyor
```

Muhtemel sebepler:

```text
row_x_head gradient almıyor
soft coordinate decoding yanlış
point loss valid mask yanlış
gt_x_rows normalize/scale hatalı
```

Kontrol:

```text
pred_x_rows min/max logla
gt_x_rows min/max logla
valid_mask sum logla
```

---

# 34. Olası hata 5: Range head çökmüş

Belirti:

```text
range_pred çok dar
hiç point çıkmıyor
veya hep full range kalıyor
```

Çözüm:

```text
range loss weight artır
range head bias [-2, 2] kontrol et
inference’da range filtering’i debug sırasında kapatıp prediction x doğru mu bak
```

İlk overfit debug’da predictionları hem range filtreli hem filtresiz çizmek iyi olur.

---

# 35. Olası hata 6: Loss NaN oluyor

Kontrol listesi:

```text
valid_mask sum zero mu?
log(0) var mı? eps ekledin mi?
softmax overflow var mı?
range sort sonrası NaN var mı?
gt_x invalid -1 loss’a girmiş mi?
learning rate fazla mı?
```

Özellikle:

```text
-log(p_lane + 1e-6)
```

kullanılmalı.

---

# 36. Implementation file yapısı

Bu partın dosyaları:

```text
losses/
├── matcher_s0.py
├── loss_s0.py
└── smoothness.py
```

`matcher_s0.py`:

```text
compute_cost_matrix
hungarian_match_single_image
match_batch
```

`loss_s0.py`:

```text
build_exist_targets
compute_exist_loss
compute_point_loss
compute_range_loss
compute_smoothness_loss
compute_total_loss
```

`smoothness.py`:

```text
second_difference_loss
```

---

# 37. Matcher input/output contract

Matcher input:

```text
outputs for one image:
  exist_logits: N × 2
  pred_x_rows: N × P
  range_norm: N × 2

target for one image:
  x_rows: M × P
  valid_mask: M × P
  range_y: M × 2
```

Matcher output:

```text
{
  "pred_indices": LongTensor[K],
  "gt_indices": LongTensor[K],
  "num_gt": M,
  "num_matched": K
}
```

Burada:

```text
K = min(N, M)
```

Normalde `N >= M`, dolayısıyla:

```text
K = M
```

---

# 38. Loss input/output contract

Loss module input:

```text
outputs
targets
matches
```

Loss output:

```text
loss_dict = {
    "loss_total": Tensor,
    "loss_exist": Tensor,
    "loss_point": Tensor,
    "loss_range": Tensor,
    "loss_smooth": Tensor,

    "num_matched": int,
    "num_gt": int
}
```

Training loop sadece:

```text
loss_dict["loss_total"].backward()
```

kullanacak.

---

# 39. Inference threshold training loss’a dahil değil

`score_thresh=0.5` sadece inference/postprocess için.

Training sırasında:

```text
tüm slotlar loss’a girer
matched slotlar lane
unmatched slotlar no-lane
```

Score threshold training loss içinde kullanılmaz.

Bu önemli. Yoksa düşük score alan slotlar hiç öğrenemez.

---

# 40. NMS şimdilik yok

S0’da duplicate prediction çıkabilir.

Ama ilk aşamada NMS eklemiyoruz.

Neden?

Çünkü duplicate varsa bunu training/matching/existence loss ile gözlemlemek istiyoruz. NMS bunu gizler.

Postprocess’te sadece:

```text
score threshold
range filtering
min points
```

olacak.

NMS daha sonra validation metric için gerekirse eklenir.

---

# 41. Part 4 acceptance criteria

Part 4 tamamlandı demek için:

```text
1. Cost matrix N × M doğru hesaplanıyor.
2. M=0 image case çalışıyor.
3. Hungarian matched indexleri doğru dönüyor.
4. Exist target matched/unmatched slotlara doğru atanıyor.
5. Point loss sadece valid rows’da hesaplanıyor.
6. Invalid x=-1 hiçbir loss’a girmiyor.
7. Loss NaN/inf üretmiyor.
8. 10 image overfit’te loss düşüyor.
9. Matching visualization mantıklı görünüyor.
```

Bunlar olmadan S1’e geçilmeyecek.

---

# 42. Part 4 özeti

Bu partta S0 training sistemini kilitledik:

```text
Matching:
  object cost + point cost + range cost

Loss:
  L_exist + L_point + L_range + optional L_smooth

Important:
  token CE yok
  evidence loss yok
  visibility yok
  matching no_grad
  loss differentiable
  invalid rows maskeli
  no-lane slots CE ile eğitiliyor
```

Bu yapı sade ama güçlü. En önemlisi debug edilebilir.

---

# DynLaneSeq-EG Implementation Plan — Part 5

## Training Loop, Optimizer, Scheduler, Overfit Protocol ve Debug Sistemi

Bu partta artık modelin “nasıl eğitileceğini” kilitliyoruz. S0 mimarisi ve loss doğru olsa bile training loop kötü kurulursa sonuç yine çöp olur. O yüzden burada hedefimiz paper seviyesinde değil, **repo seviyesinde çalışacak eğitim planı** yazmak.

---

# 1. Part 5’in amacı

Bu part sonunda şu şeyler net olacak:

```text
train.py nasıl çalışacak?
optimizer nasıl kurulacak?
backbone learning rate ayrı mı olacak?
scheduler ne olacak?
AMP kullanılacak mı?
gradient clipping olacak mı?
10 image overfit nasıl yapılacak?
hangi loglar tutulacak?
hangi görseller kaydedilecek?
hangi durumda eğitim durdurulacak?
```

Bu part özellikle önemli çünkü S0’ın görevi SOTA almak değil, **pipeline’ın öğrenebildiğini kanıtlamak**.

---

# 2. Training aşamaları

S0 için eğitim 4 aşamada ilerleyecek.

```text
Stage 0 — Target visualization
Stage 1 — 10 image overfit
Stage 2 — 100 image overfit
Stage 3 — small subset train
Stage 4 — full CULane train
```

Bunların hiçbirini atlamıyoruz.

---

## Stage 0 — Target visualization

Model eğitimi yok.

Amaç:

```text
Dataset parser ve target builder doğru mu?
```

Komut:

```text
python tools/visualize_targets.py --config configs/culane_s0_res34.yaml --num 50
```

Output:

```text
debug/targets/sample_000.jpg
debug/targets/sample_001.jpg
...
```

Kontrol:

```text
GT lane doğru yerde mi?
Fixed-row interpolation doğru mu?
Range start/end doğru mu?
Invalid rowlar çizilmiyor mu?
Horizontal flip doğru mu?
```

Bu aşama geçmeden training yasak.

---

## Stage 1 — 10 image overfit

Amaç:

```text
Model küçük veriyi ezberleyebiliyor mu?
```

Config:

```text
dataset:
  mode: overfit
  num_samples: 10
  shuffle: true

augmentation:
  horizontal_flip_prob: 0.0
  color_jitter: false
  affine: false

training:
  batch_size: 2
  max_iters: 3000
  eval_interval: 100
  vis_interval: 100
```

Beklenen:

```text
loss düşecek
predicted lanes GT’ye yaklaşacak
matched slot p_lane artacak
unmatched slotlar no-lane olacak
```

Bu geçmeden 100 image’a geçilmeyecek.

---

## Stage 2 — 100 image overfit

Amaç:

```text
Model sadece 10 görüntüyü değil, biraz daha çeşitli küçük seti öğrenebiliyor mu?
```

Config:

```text
num_samples: 100
batch_size: 4
max_iters: 5000–10000
augmentation: kapalı veya sadece çok hafif color jitter
```

Burada artık modelin duplicate lane üretip üretmediğine, range head’in stabil olup olmadığına ve no-lane class’ın öğrenilip öğrenilmediğine bakacağız.

---

## Stage 3 — Small subset train

Amaç:

```text
Pipeline gerçek training davranışında çöküyor mu?
```

Subset:

```text
1000–5000 image
```

Bu aşamada ilk kez hafif augmentation açılabilir:

```text
horizontal_flip_prob: 0.5
color_jitter: true
affine: false
```

Affine hâlâ kapalı.

---

## Stage 4 — Full CULane train

Bu en son.

Bu aşamaya sadece şunlar sağlandıysa geçilir:

```text
target visualization doğru
10 image overfit başarılı
100 image overfit başarılı
small subset loss stabil
validation görselleri makul
```

---

# 3. Optimizer kararı

Optimizer:

```text
AdamW
```

Başlangıç değerleri:

```text
base_lr = 1e-4
backbone_lr = 1e-5
weight_decay = 1e-4
betas = (0.9, 0.999)
```

Neden backbone LR düşük?

Backbone pretrained olacak. Başta çok bozmak istemiyoruz.

Param group mantığı:

```text
backbone params       → lr = 1e-5
new model params      → lr = 1e-4
norm/bias params      → weight_decay = 0
other params          → weight_decay = 1e-4
```

Bu CondLSTR tarzı pratiklerde de görülen önemli bir engineering detayı: backbone ile yeni head’lerin learning rate’i aynı tutulmamalı.

---

# 4. Param group tasarımı

Parametreleri 4 gruba ayıracağız:

```text
1. backbone_decay
2. backbone_no_decay
3. model_decay
4. model_no_decay
```

Kurallar:

```text
if param belongs to backbone:
    lr = backbone_lr
else:
    lr = base_lr

if param name contains bias or norm/bn/ln:
    weight_decay = 0
else:
    weight_decay = weight_decay
```

Örnek config:

```text
optimizer:
  name: AdamW
  base_lr: 0.0001
  backbone_lr: 0.00001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  no_decay_keywords: ["bias", "bn", "norm", "ln"]
```

---

# 5. Scheduler kararı

İlk debug aşamasında scheduler istemiyoruz.

## 10 image overfit

```text
scheduler: none
constant lr
```

Neden?

Çünkü loss düşmüyorsa scheduler yüzünden mi model yüzünden mi anlamak istemiyoruz.

## Small/full train

Başlangıç için:

```text
warmup + cosine decay
```

Config:

```text
scheduler:
  name: cosine
  warmup_iters: 1000
  min_lr_ratio: 0.1
```

Yani LR:

```text
ilk 1000 iter warmup
sonra cosine decay
minimum lr = base_lr * 0.1
```

Alternatif paper’a daha yakın step decay:

```text
lr 1e-4 başla
40 epoch sonra 1e-5
```

Ama bizim implementation debug için cosine daha yumuşak.

---

# 6. AMP kullanımı

Mixed precision:

```text
AMP = true
```

Ama ilk 10 image overfit sırasında iki seçenek var:

```text
İlk 100 iter AMP kapalı debug
sonra AMP açık
```

Neden?

NaN veya shape hatası varsa FP32 daha kolay debug edilir.

Config:

```text
training:
  amp: true
```

Debug config:

```text
training:
  amp: false
```

---

# 7. Gradient clipping

Kesin olacak.

Başlangıç:

```text
clip_grad_norm = 1.0
```

Neden?

Transformer/cross-attention + soft coordinate decoding + Hungarian sonrası loss bazen gradient spike üretebilir.

Config:

```text
training:
  grad_clip_norm: 1.0
```

Log:

```text
grad_norm before clipping
grad_norm after clipping
```

İlk debug’da en azından before clipping loglanmalı.

---

# 8. Batch size

İlk değerler:

```text
10 image overfit: batch_size = 2
100 image overfit: batch_size = 4
small subset: batch_size = 4 veya 8
full train: GPU memory’e göre 8–16
```

RTX 3090 için muhtemel:

```text
batch_size = 8
AMP açık
feature map 72×200
N=20
```

Eğer OOM olursa:

```text
batch_size düşür
gradient accumulation kullan
feature map 36×100 dene
```

---

# 9. Gradient accumulation

Full train’de batch küçük kalırsa:

```text
grad_accum_steps = 2 veya 4
```

Effective batch:

```text
effective_batch = batch_size * grad_accum_steps
```

Başlangıçta kapalı:

```text
grad_accum_steps = 1
```

Çünkü debug kolay olmalı.

---

# 10. Train loop ana akışı

Training loop şu sırada çalışacak:

```text
for iteration, batch in train_loader:

    images, targets, metas = batch

    images = images.to(device)
    targets = move_targets_to_device(targets)

    with autocast(enabled=amp):
        outputs = model(images)
        matches = matcher(outputs, targets)
        loss_dict = criterion(outputs, targets, matches)
        loss = loss_dict["loss_total"] / grad_accum_steps

    scaler.scale(loss).backward()

    if iteration % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step_if_needed()

    log losses
    maybe visualize
    maybe save checkpoint
```

Önemli:

```text
matcher no_grad içinde çalışacak
loss no_grad dışında çalışacak
```

---

# 11. `targets` device handling

Target list-of-dict olduğu için `.to(device)` doğrudan çalışmaz.

Utility:

```text
move_targets_to_device(targets, device)
```

Şunları taşıyacak:

```text
x_rows
valid_mask
range_y
exist
```

Meta string/path gibi şeyler CPU’da kalacak.

---

# 12. Checkpoint sistemi

Checkpoint içinde şunlar olacak:

```text
model_state
optimizer_state
scheduler_state
scaler_state
iteration
epoch
best_metric
config
```

Dosyalar:

```text
outputs/s0_res34/checkpoints/latest.pth
outputs/s0_res34/checkpoints/iter_001000.pth
outputs/s0_res34/checkpoints/best.pth
```

10 image overfit sırasında:

```text
latest.pth yeterli
```

Full train’de:

```text
latest + best + periodic
```

---

# 13. Resume training

`train.py` resume desteklemeli:

```text
python tools/train.py --config configs/culane_s0_res34.yaml --resume outputs/.../latest.pth
```

Resume ederken:

```text
model
optimizer
scheduler
scaler
iteration
```

geri yüklenecek.

Bu olmazsa uzun training’de ciddi zaman kaybedilir.

---

# 14. Logging sistemi

Başlangıçta TensorBoard veya WandB olabilir. En basit:

```text
TensorBoard + console log
```

Loglanacak ana değerler:

```text
loss_total
loss_exist
loss_point
loss_range
loss_smooth

lr_backbone
lr_model
grad_norm

num_gt
num_matched
mean_p_lane_matched
mean_p_lane_unmatched

mean_cost_obj
mean_cost_point
mean_cost_range
mean_total_cost
```

Bunlar kritik.

Sadece total loss loglamak yetmez.

---

# 15. Console log formatı

Örnek:

```text
iter 000420 | loss 2.384 | exist 0.521 | point 0.312 | range 0.143 | smooth 0.000 |
gt 4.1 | matched 4.1 | p+ 0.72 | p- 0.18 |
cost obj 0.61 | point 0.08 | range 0.21 |
lr 1.0e-4 | grad 0.84
```

Bu logdan training’in nasıl gittiği anlaşılmalı.

---

# 16. Visualization interval

Debug’da görselleştirme çok sık olacak:

```text
10 image overfit: every 100 iters
100 image overfit: every 250 iters
small/full train: every 1000 iters
```

Kaydedilecek:

```text
outputs/s0_res34/visualizations/train_iter_000100/
```

Her görselde:

```text
image
GT lanes
pred lanes
matched slot ids
scores
range
```

Ek olarak iki versiyon çizilecek:

```text
pred_range_filtered.jpg
pred_no_range_filter.jpg
```

Neden?

Eğer range head kötü ise x tahmini doğru olsa bile lane görünmez. No-range çizim bunu anlamamızı sağlar.

---

# 17. Validation loop

S0 için ilk validation metric çok karmaşık olmayacak.

Başlangıç validation:

```text
visual validation + simple point distance
```

Resmi CULane metric daha sonra.

S0 validation’da hesaplanacak basit metrikler:

```text
mean matched point error px
mean range error
exist precision/recall rough
num predictions per image
```

Resmi F1 için CULane evaluation tool sonra entegre edilir.

---

# 18. Simple validation matching

Validation’da GT ile prediction’ı yine matching ile eşleştirebiliriz.

Ama inference sonrası prediction sayısı değişir.

Akış:

```text
postprocess predictions
GT targets
cost = point distance
match predicted lanes to GT lanes
mean point error hesapla
```

Bu sadece debug metriği. Paper metriği değil.

---

# 19. Overfit başarı kriterleri

## 10 image overfit başarılı sayılırsa:

```text
loss_total net düşmüş olmalı
loss_point ciddi düşmüş olmalı
matched p_lane > 0.8 civarına çıkmalı
unmatched p_lane < 0.2 civarına inmeli
görsellerde lane çizgileri GT üstüne oturmalı
```

Kesin sayısal eşik koymak zor ama görsel çok belirleyici.

## Başarısızlık işaretleri:

```text
loss_point düşmüyor
tüm slotlar no-lane
tüm slotlar lane
x tahminleri sürekli ortada
range çok dar
NaN/inf
matching sürekli saçma
```

Bu durumda S1’e geçilmez.

---

# 20. İlk LR denemeleri

Eğer 10 image overfit çalışmazsa sırayla denenecek LR’ler:

```text
base_lr = 1e-4
base_lr = 3e-4
base_lr = 5e-5
```

Backbone LR:

```text
backbone_lr = base_lr * 0.1
```

Overfit için gerekirse backbone freeze denenebilir:

```text
freeze_backbone_first_iters = 500
```

Ama ilk default:

```text
backbone trainable
```

---

# 21. Backbone freeze opsiyonu

Eğer training çok kararsızsa:

```text
first 500 iter:
  backbone frozen
  sadece FPN + decoder + heads train

sonra:
  backbone unfreeze
```

Bu opsiyon config’te olacak ama default kapalı.

Config:

```text
training:
  freeze_backbone_iters: 0
```

---

# 22. Determinism ve seed

Debug için seed sabitlenecek:

```text
seed = 42
```

Ayarlar:

```text
torch.manual_seed
numpy seed
random seed
cudnn benchmark false debug’da
```

Full train’de:

```text
cudnn benchmark true
```

olabilir.

---

# 23. DataLoader ayarları

Debug:

```text
num_workers = 0 veya 2
persistent_workers = false
shuffle = true
```

Full train:

```text
num_workers = 4–8
persistent_workers = true
pin_memory = true
shuffle = true
```

Debug’da `num_workers=0` hata mesajlarını daha net verir.

---

# 24. Config yapısı

`configs/culane_s0_res34.yaml` içinde training kısmı:

```text
training:
  output_dir: outputs/s0_res34
  seed: 42
  device: cuda

  batch_size: 4
  max_epochs: 50
  max_iters: null

  amp: true
  grad_clip_norm: 1.0
  grad_accum_steps: 1

  log_interval: 20
  vis_interval: 500
  val_interval: 1000
  save_interval: 1000

optimizer:
  name: AdamW
  base_lr: 0.0001
  backbone_lr: 0.00001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

scheduler:
  name: cosine
  warmup_iters: 1000
  min_lr_ratio: 0.1
```

Overfit config ayrı olabilir:

```text
configs/debug/culane_s0_10img_overfit.yaml
```

---

# 25. Debug overfit config

```text
debug:
  mode: overfit
  num_samples: 10

training:
  batch_size: 2
  max_iters: 3000
  amp: false
  grad_clip_norm: 1.0
  log_interval: 10
  vis_interval: 100
  val_interval: 500
  save_interval: 500

augmentation:
  horizontal_flip_prob: 0.0
  color_jitter: false
  affine: false

scheduler:
  name: none

loss:
  w_exist: 2.0
  w_point: 5.0
  w_range: 1.0
  w_smooth: 0.0
  no_lane_weight: 1.0
```

Bu config’in amacı performance değil, öğrenme testi.

---

# 26. Loss scale debug protokolü

İlk 100 iter boyunca loglanacak:

```text
raw loss_exist
raw loss_point
raw loss_range
weighted loss_exist
weighted loss_point
weighted loss_range
```

Eğer weighted loss’lardan biri total loss’un %95’ini kaplıyorsa ayar bozuk olabilir.

Örnek sorun:

```text
loss_exist = 0.1
loss_point = 0.002
loss_range = 0.001
```

Bu durumda point gradient zayıf kalabilir.

Tam tersi:

```text
loss_point = 50
```

ise coordinate normalization yanlış olabilir.

---

# 27. Gradient flow debug

İlk debug’da şu parametrelerin gradient normları loglanabilir:

```text
backbone grad norm
fpn grad norm
cross_attention grad norm
exist_head grad norm
row_x_head grad norm
range_head grad norm
```

Eğer:

```text
row_x_head grad norm = 0
```

ise soft coordinate/loss bağlantısı kopmuştur.

Eğer:

```text
backbone grad norm = 0
```

ve backbone frozen değilse bir yerde detach vardır.

---

# 28. Soft coordinate sanity test

Training’den bağımsız küçük bir test olacak.

Amaç:

```text
row_x_logits gradient alıyor mu?
```

Test:

```text
rastgele row_x_logits oluştur
soft coordinate decode et
gt_x ile SmoothL1 loss hesapla
backward yap
row_x_logits.grad non-zero mı?
```

Bu test geçmeden model training’e girmemeli.

---

# 29. Matcher sanity test

Synthetic küçük örnek:

```text
N = 3 prediction
M = 2 GT
pred_x_rows:
  slot 0 GT0'a yakın
  slot 1 GT1'e yakın
  slot 2 uzak

expected matching:
  slot 0 → GT0
  slot 1 → GT1
```

Bu unit test matcher için yazılmalı.

Eğer bu geçmiyorsa gerçek training’de matching’e güvenemeyiz.

---

# 30. Target builder + model integration test

Bir batch al:

```text
images, targets = next(loader)
outputs = model(images)
matches = matcher(outputs, targets)
loss = criterion(outputs, targets, matches)
loss.backward()
```

Beklenen:

```text
shape error yok
NaN yok
grad var
```

Bu test `tools/debug_one_batch.py` olarak ayrı yazılabilir.

---

# 31. Training durdurma koşulları

Debug sırasında şu durumlarda hemen dur:

```text
loss NaN/inf
gt_valid_points = 0
pred_x min/max çok saçma
grad_norm NaN
all p_lane < 0.001 uzun süre
all p_lane > 0.999 uzun süre
```

Bunlar silent bug olabilir.

---

# 32. Prediction coordinate clamp

Training sırasında `pred_x_rows` soft expectation ile zaten 0–796 civarında olacak. Clamp gerekmez.

Inference’da güvenlik için:

```text
x = clamp(x, 0, W_in - 1)
```

Ama training loss öncesinde clamp yapmayacağız. Çünkü soft expectation zaten aralıkta.

---

# 33. Range clamp

Range sigmoid çıktısı 0–1 aralığında olduğu için clamp gerekmez.

Sort sonrası:

```text
y_min <= y_max
```

Inference’da:

```text
y_min_px = y_min * H_in
y_max_px = y_max * H_in
```

---

# 34. Checkpoint selection

S0’da best checkpoint resmi metric’e göre değil, debug validation point error’a göre seçilecek.

Best criterion:

```text
lowest val_mean_point_error
```

Full CULane metric entegre edilince:

```text
best F1
```

kullanılır.

---

# 35. İlk full training tahmini

İlk S0 full train için:

```text
epochs = 30–50
batch_size = GPU’ya göre
base_lr = 1e-4
backbone_lr = 1e-5
warmup = 1000 iter
cosine decay
```

Ama tekrar: full training’e geçmek için önce overfit şart.

---

# 36. S0’dan S1’e geçiş kriterleri

S1 token decoder’a geçmeden önce S0’da şunlar sağlanmalı:

```text
10 image overfit başarılı
100 image overfit başarılı
small subset’te validation point error düşüyor
prediction visualization mantıklı
matcher stabil
empty slotlar öğreniliyor
```

Eğer S0 bu kriterleri geçmiyorsa S1’e geçmek sadece karmaşıklığı artırır.

---

# 37. Part 5 dosya karşılığı

```text
engine/
├── train_one_epoch.py
├── validate_s0.py
├── logger.py
├── checkpoint.py
└── visualizer.py

tools/
├── train.py
├── debug_overfit.py
├── debug_one_batch.py
├── visualize_targets.py
└── visualize_predictions.py
```

---

# 38. Part 5 özeti

Bu partta training sistemini kilitledik:

```text
Optimizer: AdamW
base_lr: 1e-4
backbone_lr: 1e-5
weight_decay: 1e-4
scheduler: debug’da none, full train’de warmup+cosine
AMP: debug’da kapalı olabilir, full train’de açık
grad clipping: 1.0
first target: 10 image overfit
logging: loss, cost, p_lane, grad, matching
visualization: zorunlu
checkpoint/resume: zorunlu
```

Bu aşamada artık S0 gerçekten kodlanabilir hale geliyor.

---

# DynLaneSeq-EG Implementation Plan — Part 6

## S1: Token Decoder ve Soft Expected Token Decoding

Bu partta S0’daki basit `row_x_head` yapısını daha sequence/token tabanlı hale getiriyoruz. Ama yine kontrollü gidiyoruz. Full Lane2Seq tarzı karmaşık vocabulary’ye hemen geçmiyoruz.

Önce net karar:

```text
S1’in amacı:
S0’daki direkt MLP row_x_head yerine,
row-wise token decoder kullanmak.
```

Yani S1 hâlâ final DynLaneSeq-EG değil. Ama sequence tarafına ilk güvenli geçişimiz olacak.

---

# 1. S0’dan S1’e geçiş mantığı

S0’da şu vardı:

```text
Q1: B × N × 256
↓
MLP
↓
row_x_logits: B × N × P × X_bins
```

Bu çalışırsa S1’de bunu şu hale getiriyoruz:

```text
Q1: B × N × 256
+
row embeddings
↓
row-token decoder
↓
row_x_logits: B × N × P × X_bins
```

Yani output shape aynı kalıyor:

```text
B × N × 72 × 200
```

Ama üretim şekli değişiyor.

S0’da her slot tek vektörden 72 row’u tek seferde MLP ile çıkarıyordu.
S1’de her row ayrı bir token gibi temsil edilecek ve decoder row’lar arası ilişkiyi öğrenebilecek.

---

# 2. S1’de hâlâ yapmayacağımız şeyler

Bunları özellikle koymuyoruz:

```text
<EXISTS> <START_X> <ANGLE> <CURVE> <END> gibi full vocabulary yok
autoregressive decoder yok
beam search yok
language-model tarzı generation yok
curve-aligned sampler yok
low-rank bridge yok
zoom-in yok
evidence consistency loss yok
```

Neden?

Çünkü önce şu soruya cevap veriyoruz:

> Row-wise tokenization modeli bozuyor mu, yoksa S0’dan daha düzenli lane üretmeye başlıyor mu?

---

# 3. S1’in temel output’u

S1 hâlâ fixed-row lane representation üretecek.

Her slot için:

```text
P = 72 row
X_bins = 200 x-token
```

Output:

```text
row_x_logits: B × N × P × X_bins
```

Buradaki her row için model bir x-token dağılımı üretir.

Örnek:

```text
slot 3, row 50:
x-bin 103 daha olası
x-bin 104 ikinci olası
...
```

Training’de argmax yok.

---

# 4. Token ne demek?

S1’de token şudur:

```text
x coordinate bin token
```

Yani vocabulary ilk aşamada sadece x-binlerden oluşur:

```text
X_0, X_1, X_2, ..., X_199
```

Bunlar input görüntüde yaklaşık şu pixel aralıklarına denk gelir:

```text
X_0   ≈ x 0–4 px
X_1   ≈ x 4–8 px
...
X_199 ≈ x 796–800 px
```

Çünkü:

```text
W_in = 800
X_bins = 200
bin_width = 4 px
```

---

# 5. Peki `<EMPTY>`, `<EXISTS>`, `<END>` yok mu?

S1’de yok.

Çünkü existence zaten ayrı head ile çözülüyor:

```text
exist_logits: B × N × 2
```

Lane’in hangi rowlarda geçerli olduğu da range head ile çözülüyor:

```text
range_norm: B × N × 2
```

Yani S1’de token decoder sadece şunu yapıyor:

```text
Bu lane slotu için her fixed row’da x nerede?
```

Boş slotların x tokenları loss’a sokulmayacak. Bu yüzden `<EMPTY>` tokenına gerek yok.

---

# 6. S1 vocabulary

Başlangıç vocabulary:

```text
vocab_size = X_bins = 200
```

İleride S2/S3 tarafında vocabulary genişleyebilir:

```text
<EMPTY>
<EXISTS>
<END>
<OFFSET_BIN>
<VIS_0>
<VIS_1>
<ANGLE_BIN>
<CURVE_BIN>
```

Ama S1 için:

```text
sadece x-bin tokenları
```

Bu karar implementation’ı çok sadeleştirir.

---

# 7. GT x-bin target nasıl üretilecek?

Part 2’de target builder şunu üretiyordu:

```text
gt_x_rows: M × P
gt_valid_mask: M × P
```

S1 için ek target:

```text
gt_x_bins: M × P
```

Dönüşüm:

```text
x_bin = round(gt_x / bin_width)
```

Ama `round` yerine daha stabil olarak:

```text
x_bin = floor(gt_x / bin_width)
```

kullanacağız.

Yani:

```text
bin_width = W_in / X_bins = 800 / 200 = 4
x_bin = floor(gt_x / 4)
```

Sonra clamp:

```text
x_bin = clamp(x_bin, 0, X_bins - 1)
```

Invalid rowlarda:

```text
x_bin = ignore_index
```

Örneğin:

```text
ignore_index = -100
```

PyTorch CE için bu uygun.

---

# 8. Soft expected decoding neden hâlâ gerekli?

Token CE tek başına modelin doğru bin’i seçmesini öğretir. Ama bizim geometry loss da istiyoruz.

Yanlış yöntem:

```text
argmax(row_x_logits) → x_bin → x_pixel → SmoothL1
```

Doğru yöntem:

```text
softmax(row_x_logits)
→ expected x_bin
→ expected x_pixel
→ SmoothL1
```

Yani:

```text
prob = softmax(logits, dim=-1)
x_expected_bin = Σ prob[k] * bin_center[k]
x_expected_pixel = x_expected_bin * bin_width
```

Bu sayede geometry loss token logits’e gradient gönderir.

---

# 9. S1 model akışı

S1 genel akış:

```text
Image
↓
Backbone + FPN
↓
F_proj + 2D positional encoding
↓
Flatten image memory
↓
Lane slot cross-attention
↓
Q1 lane slot features
↓
Row-token decoder
↓
row_x_logits
↓
soft expected x
↓
pred_x_rows
```

Existence ve range head aynı kalır:

```text
Q1 → existence head
Q1 → range head
```

---

# 10. Row-token decoder input’u

Elimizde lane slot feature var:

```text
Q1: B × N × D
```

Her slot için 72 row token oluşturacağız.

Learnable row embedding:

```text
E_row: P × D
```

Burada:

```text
P = 72
D = 256
```

Her slot için row token:

```text
R_i,p = Q1_i + E_row_p
```

Shape:

```text
row_tokens: B × N × P × D
```

Sonra bunu decoder’a uygun hale getireceğiz.

---

# 11. Row-token decoder shape

Bütün slotları batch dimension’a katabiliriz.

```text
row_tokens: B × N × P × D
```

reshape:

```text
row_tokens_flat: (B*N) × P × D
```

Yani her lane slot için ayrı 72 tokenlık mini sequence var.

Örnek:

```text
B = 4
N = 20
P = 72

row_tokens_flat = 80 × 72 × 256
```

Bu gayet yönetilebilir.

---

# 12. Row-token decoder tipi

S1 için küçük Transformer encoder kullanacağız.

Neden encoder?

Çünkü row tokenları aynı anda üretiyoruz, autoregressive değil. Row’lar birbirini görebilir.

Akış:

```text
row_tokens_flat
↓
TransformerEncoder over P rows
↓
row_hidden
↓
Linear to X_bins
```

Output:

```text
row_hidden: (B*N) × P × D
row_x_logits_flat: (B*N) × P × X_bins
```

Reshape:

```text
row_x_logits: B × N × P × X_bins
```

---

# 13. Row-token decoder config

Başlangıç:

```text
num_layers = 2
nhead = 8
d_model = 256
dim_feedforward = 512 veya 1024
dropout = 0.1
```

S0 zaten image cross-attention yapıyordu. Bu row decoder sadece row içi geometri düzenini öğreniyor. Çok büyük yapmaya gerek yok.

İlk config:

```text
row_decoder:
  type: transformer_encoder
  num_layers: 2
  d_model: 256
  nhead: 8
  dim_feedforward: 512
  dropout: 0.1
```

---

# 14. Row position bilgisi

Row embedding zaten position bilgisi veriyor.

Ama daha net olsun diye row embedding iki parçadan oluşabilir:

```text
learnable row embedding
+
sinusoidal y positional embedding
```

S1 için sade karar:

```text
learnable row embedding yeterli
```

Çünkü P sabit 72.

---

# 15. S1 output dictionary

S1 forward output:

```text
outputs = {
    "exist_logits": B × N × 2,
    "row_x_logits": B × N × P × X_bins,
    "pred_x_rows": B × N × P,
    "range_raw": B × N × 2,
    "range_norm": B × N × 2,
    "row_hidden": optional,
    "queries": optional
}
```

S0 ile aynı ana outputları koruyoruz. Bu çok önemli.

Böylece S0 matcher ve S0 point/range/exist loss direkt kullanılabilir.

Ek olarak token CE loss eklenir.

---

# 16. S1 matching değişecek mi?

Hayır, ilk S1’de matching aynı kalacak.

Matching cost:

```text
cost = λ_obj * cost_obj
     + λ_point * cost_point
     + λ_range * cost_range
```

Token CE matching cost’a eklenmeyecek.

Neden?

Çünkü daha önce konuştuğumuz gibi token CE’yi matching cost’a eklemek pahalı ve dengesiz olabilir. Ayrıca boş slotların token CE’si anlamsız.

Yani:

```text
Matching önce geometry/object/range ile yapılır.
Token CE sadece matched positive slotlarda hesaplanır.
```

Bu karar değişmeyecek.

---

# 17. S1 loss sistemi

S1 total loss:

```text
L_total =
  w_exist  * L_exist
+ w_point  * L_point
+ w_range  * L_range
+ w_token  * L_token
+ w_smooth * L_smooth
```

Başlangıç ağırlıkları:

```text
w_exist  = 2.0
w_point  = 5.0
w_range  = 1.0
w_token  = 1.0
w_smooth = 0.0 debug / 0.05 later
```

İlk debug’da smooth kapalı kalabilir.

---

# 18. Token CE loss nasıl hesaplanacak?

Token CE sadece matched positive slotlarda ve valid rowlarda.

Her matched pair:

```text
pred_idx → gt_idx
```

Prediction:

```text
row_x_logits[pred_idx]: P × X_bins
```

Target:

```text
gt_x_bins[gt_idx]: P
gt_valid_mask[gt_idx]: P
```

Invalid rowlarda target:

```text
ignore_index = -100
```

Loss:

```text
CE(row_x_logits[pred_idx], gt_x_bins[gt_idx], ignore_index=-100)
```

Tüm matched pairler üzerinde ortalama.

---

# 19. Token target invalid row handling

Target builder `gt_x_bins` üretirken:

```text
if valid_mask[p] == 1:
    gt_x_bins[p] = floor(gt_x_rows[p] / bin_width)
else:
    gt_x_bins[p] = ignore_index
```

CE böylece invalid rowları otomatik yok sayar.

Bu çok kritik. Yoksa model invalid rowlarda da rastgele x öğrenmeye zorlanır.

---

# 20. Geometry loss hâlâ kalacak mı?

Evet, kesin kalacak.

S1’de token CE var diye `L_point` kaldırılmayacak.

Neden?

Çünkü token CE bin doğruluğunu öğretir. Ama bin merkezine yakınlık ve gerçek pixel hatası için geometry loss daha doğrudan sinyal verir.

Örneğin GT x = 401 px ise:

```text
bin_width = 4
gt_bin = 100
```

Model bin 99 veya 101 seçerse CE sert cezalandırır. Ama geometry loss bunun sadece 4 px hata olduğunu bilir.

Bu yüzden ikisi beraber daha iyi:

```text
L_token = discrete classification signal
L_point = continuous geometry signal
```

---

# 21. Token CE ve geometry loss ilişkisi

Başlangıçta token CE yüksek olabilir. Geometry loss ise expected coordinate üzerinden daha yumuşak sinyal verir.

Bu yüzden ilk S1 debug’da:

```text
w_token = 0.5
w_point = 5.0
```

bile denenebilir.

Benim önerim:

```text
S1 first debug:
w_token = 0.5

S1 stable:
w_token = 1.0
```

Çünkü CE çok baskın olursa model bin classification’a fazla kilitlenebilir.

---

# 22. Label smoothing kullanacak mıyız?

İlk S1’de hayır.

```text
label_smoothing = 0.0
```

Neden?

Çünkü önce net bug var mı görmek istiyoruz.

Daha sonra:

```text
label_smoothing = 0.05
```

denenebilir.

---

# 23. Soft bin target alternatifi

İleride daha iyi bir token loss için hard CE yerine soft label kullanılabilir.

Örneğin GT x iki bin arasındaysa:

```text
x = 401 px
bin_width = 4
x_bin_float = 100.25
```

Target dağılımı:

```text
bin 100 → 0.75
bin 101 → 0.25
```

Bu daha smooth olur.

Ama S1 ilk implementation’da bunu yapmıyoruz.

Başlangıç:

```text
hard x-bin CE
```

Sonra ablation:

```text
soft bin KL loss
```

---

# 24. S1 row-token decoder neden faydalı?

S0’da row_x_head tek MLP idi:

```text
lane slot vector → 72 row prediction
```

Bu MLP row’lar arasında explicit ilişki kurmuyor. S1’de row decoder:

```text
row 20, row 21, row 22...
```

arasında attention kurabilir.

Bu şu açıdan faydalı:

```text
lane çizgisi rowlar boyunca tutarlı ilerlemeli
ani zigzag olmamalı
uzaktaki rowlar yakın rowlardan bilgi almalı
```

Yani S1, sequence fikrine güvenli bir geçiş.

---

# 25. S1 hâlâ Lane2Seq mi?

Tam anlamıyla hayır.

Bu daha çok:

```text
row-wise non-autoregressive lane token decoder
```

Ama Lane2Seq fikrine doğru ilk adım.

Full Lane2Seq gibi tek bir “lane cümlesi” üretmiyoruz. Çünkü o daha riskli.

Bizim aşamalı yolumuz:

```text
S0: continuous row geometry
S1: row-wise x-token decoder
S2: evidence-grounded row-token decoder
S3: hybrid geometry + offset tokenization
```

Yani S1 kontrollü ara basamak.

---

# 26. S1 inference

Inference akışı S0 ile aynı.

```text
exist_logits → lane_prob
row_x_logits → soft expected x
range_norm → y_min/y_max
score threshold
range filtering
min point filtering
```

Inference’da argmax kullanabilir miyiz?

İki seçenek var.

## Seçenek A — soft expected x

```text
x = expected coordinate
```

Bu daha smooth.

## Seçenek B — argmax x-bin

```text
x = argmax bin center
```

Bu daha keskin ama basamaklı olabilir.

Başlangıç inference:

```text
soft expected x
```

Debug için argmax versiyonu da çizilebilir.

---

# 27. S1 postprocess aynı mı?

Evet, S0 postprocess kullanılacak:

```text
lane_prob threshold
range filtering
min_pred_points
x clamp
draw lane
```

NMS hâlâ yok.

---

# 28. S1 training protocol

S1’e geçmeden önce S0 başarılı olmalı.

S1 training aşamaları:

```text
1. S0 checkpoint yükle
2. row_x_head yerine row-token decoder koy
3. backbone + FPN + slot decoder ağırlıklarını S0’dan al
4. yeni row-token decoder random init
5. 10 image overfit
6. 100 image overfit
7. small subset
```

Burada iki seçenek var.

## Seçenek A — S1 sıfırdan train

Basit ama daha zor.

## Seçenek B — S0’dan initialize

Daha mantıklı.

Benim kararım:

```text
S1, S0 checkpoint’ten initialize edilecek.
```

Ama row_x_head farklı olduğu için onun ağırlığı yüklenmez.

---

# 29. S1 fine-tuning ayarı

S0 checkpoint’ten başlarken ilk 500 iter:

```text
backbone + FPN + cross-attention düşük LR
row-token decoder daha yüksek LR
```

Örnek:

```text
backbone_lr = 1e-5
model_lr = 5e-5
row_decoder_lr = 1e-4
```

Ama bunu çok karmaşıklaştırmadan ilk config:

```text
base_lr = 1e-4
backbone_lr = 1e-5
```

yeterli.

Eğer S1 S0 bilgisini bozarsa:

```text
first 500 iter backbone/fpn/cross-attention freeze
only row decoder + heads train
```

opsiyonu eklenir.

---

# 30. S1 config

Yeni config:

```text
model:
  name: DynLaneSeqS1

  input_height: 288
  input_width: 800

  backbone:
    name: resnet34
    pretrained: true

  fpn:
    out_channels: 128
    output_stride: 4

  transformer:
    d_model: 256
    nhead: 8
    num_decoder_layers: 2

  queries:
    num_slots: 20

  row_decoder:
    enabled: true
    num_rows: 72
    x_bins: 200
    d_model: 256
    nhead: 8
    num_layers: 2
    dim_feedforward: 512
    dropout: 0.1

loss:
  w_exist: 2.0
  w_point: 5.0
  w_range: 1.0
  w_token: 0.5
  w_smooth: 0.0
  ignore_index: -100
```

---

# 31. S1 output compatibility

S1’in en önemli engineering avantajı:

```text
S0 matcher değişmeyecek.
S0 postprocess değişmeyecek.
S0 visualizer büyük ölçüde değişmeyecek.
```

Sadece loss’a şu eklenecek:

```text
L_token
```

Bu sayede bug alanı küçük kalır.

---

# 32. Token CE için target builder güncellemesi

Part 2’deki target builder’a şu alan eklenecek:

```text
"x_bins": Tensor[M, P]
```

Target artık:

```text
target = {
    "x_rows": M × P,
    "x_bins": M × P,
    "valid_mask": M × P,
    "range_y": M × 2,
    "exist": M
}
```

Invalid rowlarda:

```text
x_bins = -100
```

Bu target S0 için de kullanılabilir ama S0 loss bunu ignore eder.

---

# 33. S1 unit testleri

S1’e özel testler:

## Test 1 — x_bin conversion

```text
gt_x = 0      → bin 0
gt_x = 3.9    → bin 0
gt_x = 4.0    → bin 1
gt_x = 799.9  → bin 199
invalid row   → -100
```

---

## Test 2 — token CE ignore

Invalid rowlar CE’ye girmiyor mu?

Synthetic test:

```text
logits random
target bazı rowlarda -100
loss backward
NaN yok
```

---

## Test 3 — soft expected decoding gradient

```text
row_x_logits → soft expected x → L_point → backward
```

Gradient non-zero olmalı.

---

## Test 4 — row decoder shape

```text
Q1: B × N × D
row_hidden: B × N × P × D
row_x_logits: B × N × P × X_bins
```

---

# 34. S1 acceptance criteria

S1 başarılı sayılması için:

```text
S0’dan kötü şekilde çökmez.
10 image overfit eder.
L_token düşer.
L_point düşer.
Prediction görselleri S0’a benzer veya daha düzenli olur.
Rowlar arası zigzag azalır.
```

Eğer S1, S0’dan belirgin kötü ise:

```text
row decoder fazla ağır olabilir
token CE fazla baskın olabilir
S0 checkpoint yüklemesi bozuk olabilir
x_bin target yanlış olabilir
```

---

# 35. S1’de beklenen olası sorunlar

## Sorun 1 — Token CE düşüyor ama geometry kötü

Sebep:

```text
x_bin target doğru ama expected coordinate dağılımı fazla yaygın
```

Çözüm:

```text
temperature düşür
w_point artır
softmax distribution entropy logla
```

---

## Sorun 2 — Geometry iyi ama token CE düşmüyor

Sebep:

```text
model doğru x’e yakın expected coordinate üretiyor ama dağılım keskin değil
```

Bu aslında çok kötü olmayabilir. Ama token decoder istiyorsak dağılımın da doğru bin’e yoğunlaşması gerekir.

Çözüm:

```text
w_token artır
label smoothing kapalı tut
temperature kontrol et
```

---

## Sorun 3 — Output çok basamaklı

Argmax inference kullanılıyorsa normal.

Çözüm:

```text
soft expected inference kullan
veya X_bins artır
```

Mesela:

```text
X_bins = 400
```

daha hassas olur ama memory artar.

---

## Sorun 4 — Row decoder overfit etmiyor

Kontrol:

```text
row embeddings öğreniyor mu?
row_decoder grad norm var mı?
Q1 doğru geliyor mu?
S0 checkpoint yüklenmiş mi?
```

Gerekirse S1’i önce S0 gibi direkt MLP ile kıyasla.

---

# 36. S1’de temperature kullanımı

Soft expected decoding’de softmax temperature ekleyebiliriz:

```text
prob = softmax(logits / τ)
```

Başlangıç:

```text
τ = 1.0
```

Eğer dağılım çok yaygınsa:

```text
τ = 0.7
```

Eğer training kararsızsa:

```text
τ = 1.5
```

İlk default:

```text
temperature = 1.0
```

Ablation olarak sonra denenir.

---

# 37. Entropy logging

S1’de row token dağılımının ne kadar keskin olduğunu loglamak faydalı.

Entropy:

```text
H = -Σ p log p
```

Log:

```text
mean_token_entropy_matched
mean_token_entropy_unmatched
```

Beklenen:

```text
training ilerledikçe matched valid rowlarda entropy azalır
```

Ama çok hızlı sıfıra inerse model overconfident olabilir.

---

# 38. S1 visualization ekleri

S1 visualizer’da sadece lane çizmek yetmez. Ek olarak birkaç row için x dağılımı çizilebilir.

Örneğin:

```text
slot 3, row 60:
x-bin probability histogram
GT bin işaretli
expected x işaretli
argmax x işaretli
```

Bu debug çok işe yarar.

Kaydedilecek:

```text
debug/token_distributions/iter_000500_slot03_row60.png
```

---

# 39. S1 neden final değildir?

Çünkü S1 hâlâ görüntü evidence’ını row sequence’e doğrudan grounded etmiyor.

Şu anda:

```text
Q1 lane slot feature
→ row decoder
→ x tokens
```

Yani evidence hâlâ tek slot vektöründe sıkışıyor.

Bizim final iddiamız ise:

```text
lane-specific evidence map / sequence
→ token decoder
```

Bu S2’de gelecek.

---

# 40. Part 6 özeti

Bu partta S1’i kilitledik:

```text
S1 = S0 + row-wise token decoder

Vocabulary:
  sadece X_bins = 200

No:
  <EMPTY>, <END>, AR decoder, full Lane2Seq vocabulary

Training:
  matching S0 ile aynı
  token CE sadece matched valid rows
  geometry loss soft expected coordinate üzerinden
  argmax training’de yok

Target:
  x_bins eklenecek
  invalid rows = ignore_index -100

Acceptance:
  10 image overfit
  L_token ve L_point düşmeli
  görseller S0’dan kötü olmamalı
```

---

# DynLaneSeq-EG Implementation Plan — Part 7

## S2: Curve-Aligned Evidence Sampler

Bu partta artık modelin “evidence-grounded” tarafına ilk gerçek adımı atıyoruz.

S1’de model şunu yapıyordu:

```text
Image
→ Backbone + FPN
→ lane slot query
→ Q1
→ row-token decoder
→ x token / x coordinate
```

Ama burada hâlâ büyük bir eksik vardı:

```text
Row decoder, lane üzerindeki görsel kanıtı doğrudan okumuyordu.
```

S2’de bunu değiştiriyoruz.

Yeni hedef:

```text
Her lane slotu için kaba bir lane eğrisi çıkar.
Bu eğri üzerindeki feature noktalarını oku.
Bu feature dizisini row-token decoder’a ver.
```

Yani model artık sadece `Q1` vektöründen lane yazmayacak; lane’in geçtiği yerlerden görsel kanıt okuyacak.

---

# 1. S2’nin ana fikri

S2 akışı:

```text
Image
↓
Backbone + FPN
↓
F_proj image feature map
↓
Lane slot queries Q1
↓
Coarse geometry / coarse x rows
↓
Curve-aligned sampler
↓
Lane-specific evidence sequence
↓
Evidence-to-sequence adapter
↓
Row-token decoder
↓
Final x rows
```

S2’nin farkı şu:

```text
S1:
  Q1 + row embedding → row decoder

S2:
  Q1 + row embedding + sampled visual evidence → row decoder
```

Bu küçük gibi görünüyor ama modelin felsefesini değiştiriyor.

S1 daha çok:

```text
slot feature’dan lane tahmini
```

S2 ise:

```text
lane boyunca görüntüden kanıt okuyup lane tahmini
```

haline geliyor.

---

# 2. S2’de hâlâ yapmayacağımız şeyler

S2’de şunlar hâlâ yok:

```text
low-rank bridge yok
dynamic full kernel yok
zoom-in refinement yok
topology head yok
evidence consistency loss yok
full hybrid vocabulary yok
autoregressive decoder yok
```

Yani S2 sadece şunu ekliyor:

```text
curve-aligned visual evidence sampling
```

Bu yüzden debug edilebilir kalıyor.

---

# 3. S2 neden gerekli?

S1’de her lane slotu için tek vektör vardı:

```text
Q1_i ∈ R^256
```

Bu vektör tüm lane’i temsil etmeye çalışıyordu.

Problem:

```text
Lane uzun ve ince bir yapı.
Tek vektör tüm rowlardaki lokal görüntü bilgisini taşıyamayabilir.
```

Özellikle:

```text
occlusion
uzak lane noktaları
gölge
düşük kontrast
fork / merge
çok kıvrımlı lane
```

durumlarında row bazlı lokal evidence gerekir.

S2 bunu yapar:

```text
Lane’in üstüne küçük kameralar koyar.
Her row civarından feature okur.
Decoder bu feature dizisini kullanarak x token üretir.
```

---

# 4. S2’de hangi feature map sample edilecek?

S0/S1’de FPN output vardı:

```text
F: B × 128 × 72 × 200
```

Projection sonrası:

```text
F_proj: B × 256 × 72 × 200
```

S2’de sampler için şunu kullanacağız:

```text
F_sample = F_proj
```

Shape:

```text
F_sample: B × 256 × 72 × 200
```

Neden `F_proj`?

Çünkü row decoder ve lane query dimension’ı da 256. Böylece evidence feature doğrudan aynı embedding uzayında olur.

---

# 5. S2’de coarse curve nereden gelecek?

Bu çok kritik.

S2 sampler’ın çalışması için önce bir kaba lane tahmini gerekir.

İki kaynak var:

## Kaynak A — S1/S0 predicted x rows

Model zaten şunu üretiyor:

```text
pred_x_rows_coarse: B × N × P
```

Bu coarse lane olarak kullanılabilir.

## Kaynak B — GT-guided curve

Training başında predicted x rows kötü olacağı için GT lane kullanılacak.

Bu cold-start problemini çözmek için şart.

---

# 6. Cold-start problemi

Eğitimin başında modelin predicted curve’ü rastgeledir.

Eğer sampler bunu kullanırsa:

```text
yanlış yerde feature sample eder
decoder yanlış evidence okur
loss gürültülü olur
training zorlaşır
```

Bu yüzden S2 training’de sampler input’u curriculum ile seçilecek.

---

# 7. Sampler curriculum

S2 training’de üç aşama olacak.

## Aşama 1 — GT-guided sampling

İlk başta:

```text
sample_x_rows = gt_x_rows + noise
```

Yani sampler, gerçek lane’in yakınından feature okur.

Ama doğrudan kusursuz GT kullanmayacağız; küçük noise ekleyeceğiz ki model biraz tolerans öğrensin.

Örnek:

```text
noise ~ Normal(0, σ)
σ = 2–4 px
```

Bu aşamada:

```text
decoder doğru lane civarındaki evidence’ı görür
row decoder evidence kullanmayı öğrenir
```

---

## Aşama 2 — Mixed sampling

Sonra predicted ve GT karıştırılır:

```text
sample_x_rows = α * gt_x_rows + (1 - α) * pred_x_rows_coarse
```

Başta:

```text
α = 1.0
```

Zamanla:

```text
α → 0.0
```

Örnek schedule:

```text
epoch 0–3:
  α = 1.0

epoch 4–8:
  α linearly 1.0 → 0.0

epoch 9+:
  α = 0.0
```

---

## Aşama 3 — Predicted sampling

Son aşamada:

```text
sample_x_rows = pred_x_rows_coarse
```

Inference’ta zaten GT yok. Bu yüzden training’in sonunda model kendi tahminiyle sample etmeyi öğrenmeli.

---

# 8. S2’de matched slot problemi

GT-guided sampling sadece matched positive slotlar için mümkün.

Çünkü hangi prediction slotunun hangi GT lane’e karşılık geldiğini bilmek gerekir.

Bu yüzden S2 training akışı şöyle olacak:

```text
1. Model first-pass coarse prediction üretir.
2. Matcher çalışır.
3. Matched positive slotlar için sample curve seçilir.
4. Row decoder evidence ile final prediction üretir.
5. Loss hesaplanır.
```

Bu önemli bir değişiklik.

S0/S1’de matcher doğrudan final output üzerinden çalışıyordu.
S2’de iki aşamalı output olacak.

---

# 9. S2 forward iki modda çalışacak

S2 model forward’u training ve inference’ta biraz farklı davranacak.

## Training mode

```text
coarse_outputs = first_pass(images)
matches = matcher(coarse_outputs, targets)

sample curves:
  matched slots → GT/pred mix
  unmatched slots → predicted curve veya dummy curve

final_outputs = second_pass_with_sampler(images, coarse_outputs, sample_curves)
loss = final loss
```

Ama model class içinde matcher koymak temiz değil. Bu yüzden daha iyi tasarım:

```text
model.forward(images, sample_curves=None)
```

Eğer `sample_curves=None` ise model kendi predicted curve’ünü kullanır.

Training loop:

```text
coarse_outputs = model.forward_coarse(images)
matches = matcher(coarse_outputs, targets)
sample_curves = build_sample_curves(coarse_outputs, targets, matches, curriculum)
final_outputs = model.forward_refine(images, coarse_outputs, sample_curves)
loss = criterion(final_outputs, targets, matches)
```

Bu daha debug edilebilir.

---

# 10. Coarse output ve final output ayrımı

S2’de iki output tutacağız.

```text
coarse_outputs:
  exist_logits_coarse
  pred_x_rows_coarse
  range_norm_coarse

final_outputs:
  exist_logits
  row_x_logits
  pred_x_rows
  range_norm
```

Başlangıçta existence/range finalda yeniden üretilmeyebilir. En sade çözüm:

```text
exist_logits = exist_logits_coarse
range_norm = range_norm_coarse
```

S2 sadece `row_x_logits/pred_x_rows` kısmını evidence ile refine eder.

Bu çok daha güvenli.

---

# 11. S2 minimum değişiklik prensibi

S2’de şunu yapıyoruz:

```text
exist head aynı
range head aynı
matching aynı
loss aynı + token loss aynı
sadece row decoder input’u evidence ile zenginleşiyor
```

Yani S1’de çalışan çoğu şey korunuyor.

Bu yüzden bug çıkarsa büyük ihtimalle sampler/adaptor tarafındadır.

---

# 12. Curve-aligned sampler inputları

Sampler inputları:

```text
F_sample: B × C × Hf × Wf
sample_x_rows: B × N × P
y_rows: P
valid_or_range_mask: B × N × P
```

Burada:

```text
C = 256
Hf = 72
Wf = 200
P = 72
```

`sample_x_rows` input pixel coordinate sistemindedir:

```text
x ∈ [0, 800]
```

`y_rows` da input pixel coordinate:

```text
y ∈ [0, 288]
```

---

# 13. Grid sample coordinate dönüşümü

`torch.nn.functional.grid_sample` koordinatları `[-1, 1]` ister.

Ama dikkat: `F_sample` 72×200 feature map’tir. Biz input pixel coordinate’ten feature map grid coordinate’e geçmeliyiz.

Input coordinate:

```text
x_in ∈ [0, W_in - 1]
y_in ∈ [0, H_in - 1]
```

Grid coordinate:

```text
x_grid = 2 * x_in / (W_in - 1) - 1
y_grid = 2 * y_in / (H_in - 1) - 1
```

Bu doğrudan `align_corners=True` varsayımıyla uyumludur.

Karar:

```text
grid_sample align_corners=True
```

Bunu her yerde sabit tutacağız.

---

# 14. Sampler grid shape

Her lane slot için P row sample edilecek.

Grid shape için pratik yol:

```text
grid: B × N × P × 1 × 2
```

Ama `grid_sample` inputu:

```text
input: B × C × Hf × Wf
grid : B × H_out × W_out × 2
```

Her slot için ayrı grid sample yapmak pahalı olabilir.

Daha pratik yöntem:

F_sample’ı slot sayısı kadar repeat et:

```text
F_rep: (B*N) × C × Hf × Wf
grid_flat: (B*N) × P × 1 × 2
```

Sonra:

```text
sampled = grid_sample(F_rep, grid_flat)
```

Output:

```text
sampled: (B*N) × C × P × 1
```

Squeeze/permute:

```text
E_seq: B × N × P × C
```

---

# 15. Memory uyarısı

`F_rep` fiziksel olarak repeat edilirse memory artar.

Başlangıçta N=20, B küçük olduğu için tolere edilebilir. Ama daha iyi yöntem:

```text
expand kullan, mümkünse view ile çalış
```

Eğer memory patlarsa slot loop kullanılabilir:

```text
for slot in N:
    sample F with grid for that slot
```

Bu daha yavaş ama debug için güvenli.

Karar:

```text
S2 debug:
  slot loop kabul edilebilir

S2 optimized:
  batched grid_sample
```

İlk implementation’da doğru çalışan versiyon öncelikli.

---

# 16. Evidence sequence output

Sampler output:

```text
E_seq: B × N × P × C
```

Başlangıç:

```text
C = 256
```

Bu şu demek:

```text
Her lane slotu için 72 row boyunca 256-dim visual evidence var.
```

Örnek:

```text
E_seq[b, slot, row]
```

şu bilgiyi taşır:

```text
Bu slotun lane eğrisi üzerinde, bu row civarında görüntü feature’ı nedir?
```

---

# 17. Sadece tek nokta mı sample edeceğiz?

İlk S2’de evet:

```text
her row için 1 nokta
```

Yani:

```text
(x_row, y_row)
```

Ama bu kırılgan olabilir. Çünkü coarse x birkaç pixel yanlışsa tam lane çizgisinin yanından feature alınabilir.

Bu yüzden S2.1’de lokal window sampling ekleyebiliriz.

---

# 18. Local window sampling opsiyonu

Daha sağlam versiyon:

Her row’da sadece merkez değil, çevresinden birkaç nokta sample edilir.

Örneğin horizontal offsets:

```text
offsets = [-8, -4, 0, +4, +8] px
```

Bu durumda:

```text
K = 5 sample point per row
```

Output:

```text
E_local: B × N × P × K × C
```

Sonra reduce:

```text
E_seq = attention_pool(E_local)
```

veya basit:

```text
E_seq = mean(E_local)
```

İlk S2’de bunu kapalı tutabiliriz.

Karar:

```text
S2 initial:
  K = 1

S2 robust ablation:
  K = 5 horizontal local sampling
```

---

# 19. Evidence-to-sequence adapter

Sampler output:

```text
E_seq: B × N × P × 256
```

Row decoder input’u S1’de şöyleydi:

```text
row_tokens = Q1_i + row_embedding_p
```

S2’de:

```text
row_tokens = Q1_i + row_embedding_p + Adapter(E_seq_i,p)
```

Adapter basit MLP/Linear olacak.

Input:

```text
E_seq: B × N × P × 256
```

Adapter:

```text
Linear(256 → 256)
LayerNorm
ReLU/GELU
Linear(256 → 256)
```

Output:

```text
E_adapted: B × N × P × 256
```

Son row token:

```text
row_tokens = Q1.unsqueeze(2) + E_row.unsqueeze(0,0) + E_adapted
```

Shape:

```text
B × N × P × 256
```

---

# 20. Gating ekleyecek miyiz?

Evidence bazen kötü olabilir. Özellikle predicted sampler yanlış yere bakarsa.

Bu yüzden adapter’a gate eklemek mantıklı.

Gate:

```text
g = sigmoid(Linear([Q1_i, E_seq_i,p]))
```

Ama ilk implementation’da daha basit:

```text
row_tokens = Q1 + row_emb + γ * E_adapted
```

Burada `γ` learnable scalar olabilir.

Başlangıç:

```text
γ = 0.0 veya 0.1
```

Neden?

Eğer S2’yi S1 checkpoint’ten başlatıyorsak, evidence birden modeli bozmasın.

Karar:

```text
learnable evidence_scale γ
initial γ = 0.1
```

Böylece model evidence’ı yavaş yavaş kullanmayı öğrenir.

---

# 21. S2 row decoder

S1’deki row decoder aynen kalır.

Input:

```text
row_tokens: B × N × P × D
```

Flatten:

```text
(B*N) × P × D
```

Transformer encoder:

```text
row_hidden: (B*N) × P × D
```

Output head:

```text
row_x_logits: B × N × P × X_bins
```

Soft expected decoding:

```text
pred_x_rows: B × N × P
```

Yani S2’nin output contract’ı S1 ile aynı kalır.

---

# 22. S2 loss değişecek mi?

İlk S2’de loss değişmeyecek.

```text
L_total =
  w_exist  * L_exist
+ w_point  * L_point
+ w_range  * L_range
+ w_token  * L_token
+ w_smooth * L_smooth
```

Evidence consistency loss yok.

Neden?

Çünkü evidence loss eklersek iki şey aynı anda değişmiş olur:

```text
sampler değişti
loss değişti
```

Bug çıkarsa sebebi anlaşılmaz.

Önce evidence input fayda veriyor mu bakacağız.

---

# 23. S2’de matching hangi output ile yapılacak?

Başlangıç kararı:

```text
matching coarse_outputs ile yapılacak
```

Çünkü sampler için matched slotları bilmemiz gerekiyor.

Akış:

```text
coarse_outputs = S1-like first pass
matches = matcher(coarse_outputs, targets)
sample_curves = build_sample_curves(matches)
final_outputs = evidence refine
loss = criterion(final_outputs, targets, matches)
```

Burada final outputs ile yeniden matching yapmıyoruz.

Neden?

Çünkü aynı iterasyonda iki matching yaparsak sistem karışır. İlk S2 için tek matching yeterli.

Daha sonra ablation:

```text
coarse matching vs final matching
```

denenebilir.

---

# 24. Coarse branch nereden gelecek?

S2, S1 checkpoint’ten başlatılacak.

Coarse branch aslında S1’in outputudur:

```text
Q1 → row decoder without evidence → coarse row_x_logits
```

Sonra evidence branch final row decoder’a gider.

Ama aynı row decoder’ı iki kez kullanmak karışık olabilir.

Daha temiz tasarım:

## Coarse row head

S0’daki basit MLP head’i tekrar ekle:

```text
Q1 → coarse_row_x_logits
```

Bu hızlı coarse prediction verir.

## Final row decoder

Evidence ile çalışan S1 row decoder:

```text
Q1 + row_emb + evidence → final_row_x_logits
```

Karar:

```text
S2’de coarse için basit MLP row head kullanılacak.
Final için row-token decoder kullanılacak.
```

Bu daha stabil.

---

# 25. S2 output dictionary

```text
outputs = {
    "coarse": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "final": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "evidence": {
        "sample_x_rows": B × N × P,
        "E_seq": B × N × P × D,
        "evidence_scale": scalar
    }
}
```

Training loss ana olarak `final` üzerinden hesaplanır.

Opsiyonel auxiliary coarse loss eklenebilir.

---

# 26. Coarse auxiliary loss

S2’de coarse branch’in de düzgün kalması lazım. Çünkü sampler onun tahminine güvenecek.

Bu yüzden küçük auxiliary loss eklemek mantıklı.

```text
L_total =
  L_final
+ λ_coarse * L_coarse
```

Başlangıç:

```text
λ_coarse = 0.5
```

Coarse loss S0/S1 loss sisteminin aynısı olabilir ama token CE olmadan veya daha düşük ağırlıkla.

Basit karar:

```text
L_coarse = L_exist + L_point + L_range
```

Final loss:

```text
L_final = L_exist + L_point + L_range + L_token
```

Ama `exist/range` aynı ise iki kere CE yazmak gereksiz olabilir.

İlk S2 için:

```text
coarse auxiliary point loss = 1.0 * L_coarse_point
```

yeterli.

---

# 27. Matched slots dışında evidence ne olacak?

Training’de matched positive slots için GT-guided sampling var.

Unmatched slots için üç seçenek var:

## Seçenek A — predicted curve ile sample et

Unmatched slotlar kendi predicted curve’lerinden sample eder.

## Seçenek B — evidence zero yap

Unmatched slotlar no-lane olduğu için evidence gereksiz.

## Seçenek C — tüm slotları predicted curve ile sample et, matched olanları GT mix ile değiştir

En pratik olan C.

Akış:

```text
sample_x_rows = pred_x_rows_coarse.clone()

for each matched pred_idx, gt_idx:
    sample_x_rows[pred_idx] = mix(gt_x_rows[gt_idx], pred_x_rows_coarse[pred_idx])
```

Unmatched slotlar predicted curve kullanır.

Bu sayede tensor shape bozulmaz.

---

# 28. GT valid mask ve sample curve

GT lane her row’da valid değildir.

Matched slot için GT-guided sample yaparken invalid rowlarda ne olacak?

Seçenekler:

## Seçenek A — invalid rowlarda predicted x kullan

```text
if gt_valid_mask[p] == 1:
    sample_x = mix(gt_x, pred_x)
else:
    sample_x = pred_x
```

Bu iyi.

Çünkü GT’nin olmadığı row’da lane yoktur; orada GT x yok.

Karar:

```text
valid rowlarda GT/pred mix
invalid rowlarda pred_x
```

---

# 29. Range ile sampler mask

Sampler tüm 72 row için feature alacak mı?

İlk S2’de evet.

```text
E_seq: B × N × 72 × D
```

Ama row decoder hangi rowların lane’e ait olduğunu range head ile bilebilir mi?

Şu anda row decoder’a range embedding vermiyoruz. Bunu ekleyebiliriz.

Basit çözüm:

```text
range mask embedding ekle
```

Ama ilk implementation’da gerek yok. Çünkü loss zaten valid rowlarda hesaplanır.

Daha sonra:

```text
range-aware row embedding
```

eklenebilir.

---

# 30. Sampler out-of-bound davranışı

Eğer sample_x_rows görüntü dışına çıkarsa:

```text
x < 0 veya x > W_in - 1
```

grid_sample padding kullanır.

Karar:

```text
sample_x_rows clamp edilecek
```

Yani:

```text
x = clamp(x, 0, W_in - 1)
```

Training sampler için bu kabul edilebilir.

---

# 31. Grid sample padding mode

`grid_sample` parametreleri:

```text
mode = "bilinear"
padding_mode = "border"
align_corners = True
```

Neden `border`?

Eğer x biraz dışarı taşarsa sıfır feature almak yerine kenar feature’ı almak daha stabil olabilir.

Alternatif:

```text
padding_mode = "zeros"
```

Ama zeros bazı durumlarda yapay sinyal olabilir.

Karar:

```text
padding_mode = "border"
```

---

# 32. Evidence dropout

Evidence branch’e aşırı güvenmesin diye dropout eklenebilir.

Başlangıçta hayır.

```text
evidence_dropout = 0.0
```

Sonra:

```text
0.1
```

denenebilir.

İlk debug’da kapalı.

---

# 33. S2 training schedule

S2, S1 checkpoint’ten başlatılacak.

Aşamalar:

```text
1. S1 checkpoint load
2. coarse MLP head init veya S0 row head transfer
3. evidence adapter init
4. evidence_scale γ = 0.1
5. 10 image overfit
6. 100 image overfit
7. small subset
```

Sampler curriculum:

```text
debug overfit:
  first 1000 iter α = 1.0
  next 1000 iter α 1.0 → 0.0
  final 1000 iter α = 0.0

full train:
  first 3 epochs α = 1.0
  epoch 4–8 α linearly 1.0 → 0.0
  epoch 9+ α = 0.0
```

---

# 34. S2 config

```text
model:
  name: DynLaneSeqS2

  evidence_sampler:
    enabled: true
    feature_source: F_proj
    mode: curve_aligned
    num_sample_points: 72
    local_window:
      enabled: false
      offsets_px: [-8, -4, 0, 4, 8]
    grid_sample:
      mode: bilinear
      padding_mode: border
      align_corners: true

  evidence_adapter:
    d_in: 256
    d_out: 256
    hidden_dim: 256
    use_layernorm: true
    evidence_scale_init: 0.1

  coarse_branch:
    enabled: true
    type: mlp_row_head

curriculum:
  sampler:
    gt_alpha_start: 1.0
    gt_alpha_end: 0.0
    warmup_epochs: 3
    decay_epochs: 5
    noise_std_px: 3.0
```

---

# 35. S2 loss config

```text
loss:
  final:
    w_exist: 2.0
    w_point: 5.0
    w_range: 1.0
    w_token: 0.5
    w_smooth: 0.0

  coarse_aux:
    enabled: true
    w_coarse_point: 1.0
    w_coarse_range: 0.5
```

İlk debug’da coarse aux gerekirse açılır. Eğer karışıklık yaparsa kapatılır.

---

# 36. S2 visualizer

S2’de görselleştirme çok önemli.

Her sample için kaydedilecek:

```text
1. GT lanes
2. coarse prediction
3. sample curve
4. final prediction
```

Renkler ayrı olmalı:

```text
GT = yeşil
coarse = mavi
sample curve = sarı
final = kırmızı
```

Bu görselde şunu anlamalıyız:

```text
Sampler doğru yerden feature okuyor mu?
GT-guided schedule doğru uygulanıyor mu?
Final prediction coarse’dan daha iyi mi?
Evidence modeli bozuyor mu?
```

---

# 37. Evidence debug heatmap

İlk S2’de evidence map yok, sadece sampled sequence var. Ama sampled noktaları görüntü üstünde çizmek şart.

Her matched slot için:

```text
sampled points
GT lane
pred lane
```

yan yana çizilecek.

Eğer sampled points GT’den kopuksa curriculum bug var demektir.

---

# 38. S2 logging

Ek loglar:

```text
sampler_alpha
evidence_scale_gamma
mean_sample_x_error_to_gt
coarse_point_loss
final_point_loss
final_minus_coarse_point_error
```

Özellikle:

```text
final_point_loss < coarse_point_loss
```

olmasını bekleriz.

Her zaman olmayabilir ama trend olarak final daha iyi olmalı.

---

# 39. S2 acceptance criteria

S2 başarılı sayılması için:

```text
1. S1 checkpoint’ten başlayıp 10 image overfit edebilmeli.
2. Sampler görselleri doğru lane civarından feature aldığını göstermeli.
3. final prediction coarse prediction’dan kötü olmamalı.
4. L_point final düşmeli.
5. Evidence scale gamma sıfırda kalmamalı.
6. Predicted-sampling aşamasına geçince model tamamen çökmemeli.
```

Eğer `γ` hep 0’a yakın kalıyorsa model evidence kullanmıyor demektir.

Eğer evidence açılınca performans düşüyorsa:

```text
sample curve yanlış
adapter fazla güçlü
γ init fazla büyük
GT/pred curriculum hızlı geçiyor
```

olabilir.

---

# 40. Olası hata 1 — GT-guided sampler çalışıyor ama predicted sampler çöküyor

Sebep:

```text
coarse branch yeterince iyi değil
curriculum çok hızlı
pred_x_rows çok noisy
```

Çözüm:

```text
GT alpha decay’i yavaşlat
coarse auxiliary loss artır
pred_x_rows’u detach ederek sampler’a ver
local window sampling aç
```

---

# 41. Olası hata 2 — Evidence hiç kullanılmıyor

Belirti:

```text
S2 ≈ S1 performansı
gamma düşük
adapter gradient düşük
```

Çözüm:

```text
evidence_scale init 0.1 → 0.5
adapter LR artır
row token içinde Q1 ağırlığını azalt
evidence dropout kapalı tut
```

Ama dikkat: evidence’ı zorla kullandırmak modeli bozabilir.

---

# 42. Olası hata 3 — Final prediction coarse’dan kötü

Sebep:

```text
sampled feature noise
adapter kötü init
row decoder S1 bilgisini kaybetti
```

Çözüm:

```text
S1 row decoder checkpoint doğru yüklendi mi kontrol et
evidence_scale küçük başlat
adapter output residual olsun
ilk 500 iter sadece adapter train et
```

---

# 43. Olası hata 4 — Memory patlıyor

Sebep:

```text
F_sample slot sayısı kadar repeat edildi
B*N büyük
```

Çözüm:

```text
slot loop grid_sample
batch_size düşür
AMP aç
F_sample channel 256 → 128 yap
P = 72 yerine P = 36 dene
```

Ama P=36 lane hassasiyetini azaltır. Önce batch/loop ile çöz.

---

# 44. S2 dosya yapısı

Yeni dosyalar:

```text
modeling/
├── evidence/
│   ├── curve_aligned_sampler.py
│   ├── evidence_adapter.py
│   └── sampler_curriculum.py
│
├── heads/
│   └── coarse_row_head.py
│
└── dynlaneseq_s2.py
```

Training tarafında:

```text
engine/
├── build_sample_curves.py
└── train_s2.py
```

Ama mümkünse `train.py` generic kalmalı, model type’a göre branch seçmeli.

---

# 45. S2’nin önemli implementation contract’ı

`CurveAlignedSampler` input:

```text
F_sample: B × C × Hf × Wf
x_rows: B × N × P
y_rows: P
```

Output:

```text
E_seq: B × N × P × C
```

`EvidenceAdapter` input:

```text
E_seq: B × N × P × C
```

Output:

```text
E_adapted: B × N × P × D
```

`RowDecoder` input:

```text
Q1: B × N × D
E_adapted: B × N × P × D
row_emb: P × D
```

Output:

```text
row_x_logits: B × N × P × X_bins
```

---

# 46. S2’de detach kararı

Sampler’a verilen `sample_x_rows` predicted curve’den geliyorsa bu yol differentiable olabilir mi?

`grid_sample` coordinate’e gradient verebilir. Yani teorik olarak final loss, sample_x_rows üzerinden coarse branch’e de gradient gönderebilir.

Ama ilk implementation’da bu karışıklık yaratabilir.

Karar:

```text
S2 initial:
  sample_x_rows = sample_x_rows.detach()
```

Yani sampler koordinatları üzerinden coarse branch’e gradient göndermiyoruz.

Neden?

Çünkü önce evidence’ın decoder’a faydasını görmek istiyoruz. Coarse branch ayrı loss ile eğitilecek.

Sonra ablation:

```text
detach_sample_coords = false
```

denenebilir.

---

# 47. Neden detach mantıklı?

Eğer detach etmezsek:

```text
final loss → grid_sample coords → coarse pred_x_rows
```

gradyanı akar. Bu iyi görünebilir ama erken aşamada coordinate gradient gürültülü olabilir.

İlk S2’de daha stabil yaklaşım:

```text
coarse branch kendi point/range loss’u ile öğrenir
final branch sampled evidence ile öğrenir
```

Sonra daha end-to-end yapılabilir.

---

# 48. S2 inference

Inference’ta GT yok.

Akış:

```text
1. images → backbone/FPN/Q1
2. coarse branch → pred_x_rows_coarse
3. sample_x_rows = pred_x_rows_coarse
4. curve sampler → E_seq
5. evidence adapter + row decoder → final pred_x_rows
6. exist/range filtering
7. final lane output
```

Yani inference tek pass gibi görünse de içeride coarse → refine var.

---

# 49. S2 hız etkisi

S2, S1’den yavaş olacak çünkü grid_sample ve ikinci row decoder var.

Bunun için ölçülecek:

```text
FPS S1
FPS S2
GPU memory S1
GPU memory S2
```

Eğer S2 çok yavaşsa:

```text
local window kapalı tut
P azaltmayı dene
row decoder layer azalt
F_sample channel azalt
```

Ama ilk amaç hız değil, doğruluk ve çalışabilirlik.

---

# 50. Part 7 özeti

Bu partta S2’yi kilitledik:

```text
S2 = S1 + curve-aligned evidence sampler

Ana fikir:
  coarse lane curve çıkar
  lane boyunca feature sample et
  evidence sequence’i row decoder’a ekle

Cold-start çözümü:
  GT-guided sampling
  mixed sampling
  predicted sampling

Important:
  matching coarse output ile yapılır
  token CE matching’e eklenmez
  sample coords ilk başta detach edilir
  evidence_scale γ = 0.1 başlar
  local window ilk sürümde kapalı
  loss büyük ölçüde S1 ile aynı
```

S2, final modelin asıl “evidence-grounded” kimliğini başlatan versiyon olacak. Ama hâlâ low-rank bridge yok. Low-rank bridge’i bundan sonra, yani S3’te ekleyeceğiz.

---

# DynLaneSeq-EG Implementation Plan — Part 8

## S3: Factorized Low-Rank Dynamic Evidence Bridge

Bu partta modelin en özgün ama en riskli kısmına geliyoruz:

```text
S3 = S2 + factorized low-rank dynamic evidence bridge
```

Ama en baştan kritik karar:

```text
Full dynamic convolution kernel üretmeyeceğiz.
```

Yani şunu doğrudan üretmek yok:

```text
B × N × C_out × C_in × k × k
```

Çünkü bu hem memory hem hız açısından çok tehlikeli. Bunun yerine **factorized modulation** kullanacağız.

---

# 1. S3’ün amacı

S2’de evidence şu şekilde geliyordu:

```text
F_sample
→ coarse curve boyunca grid_sample
→ E_seq
→ row decoder
```

Yani sampler, feature map’ten doğrudan lane üzerindeki noktaları okuyordu.

S3’te bundan önce feature map’i lane slotuna göre hafifçe modüle edeceğiz:

```text
F_sample
+ lane slot Q1
→ lane-conditioned feature modulation
→ lane-specific refined feature
→ curve-aligned sampler
→ E_seq
```

Ama bunu full dynamic conv ile değil, düşük maliyetli bridge ile yapacağız.

---

# 2. S3 neden gerekli?

S2’de her slot aynı feature map’ten sample alıyordu:

```text
F_sample aynı
sample curve farklı
```

Bu iyi ama eksik.

Çünkü iki farklı lane slotu aynı x/y civarından geçebilir veya fork/merge gibi durumlarda birbirine yakın olabilir. Böyle durumlarda sadece koordinat üzerinden sample etmek yetmeyebilir.

S3’ün amacı:

```text
Her lane slotu, feature map’e kendi “ne arıyorum?” bilgisini uygulasın.
```

Yani:

```text
slot 3: sol kıvrılan lane’e duyarlı feature
slot 7: sağdaki kesik lane’e duyarlı feature
slot 12: uzak merge bölgesine duyarlı feature
```

Ama bunu maliyeti patlatmadan yapacağız.

---

# 3. Full dynamic kernel neden yasak?

Teorik fikir şuydu:

```text
Ω_i = (V_i U_i^T) ⊗ S_i
```

Eğer bunu açık kernel olarak üretirsek:

```text
C_out × C_in × k × k
```

kadar ağırlık gerekir.

Örnek:

```text
C_in = 256
C_out = 256
k = 3
```

Tek lane slot için:

```text
256 × 256 × 3 × 3 = 589,824 weight
```

Eğer:

```text
B = 4
N = 20
```

ise:

```text
4 × 20 × 589,824 ≈ 47 milyon dynamic weight
```

Bu sadece kernel ağırlığı. Intermediate activation, gradient, optimizer falan dahil değil.

Bu yüzden full dynamic conv implementation için kötü fikir.

---

# 4. S3’ün doğru fikri: factorized dynamic bridge

Full kernel yerine şu sırayı kullanacağız:

```text
F
→ lane-conditioned channel reduction
→ lightweight spatial filtering
→ lane-conditioned channel expansion
→ residual refined feature
```

Yani:

```text
U: input channel modulation / reduction
S: small spatial filter
V: output channel modulation / expansion
```

Ama bunları explicit büyük kernel haline getirmiyoruz.

---

# 5. Bridge’in temel input/output’u

Inputlar:

```text
F_sample: B × C × Hf × Wf
Q1:       B × N × D
```

Başlangıç değerleri:

```text
C = 256
D = 256
Hf = 72
Wf = 200
N = 20
```

Output:

```text
F_bridge: B × N × C × Hf × Wf
```

Ama dikkat: Bu output’u her zaman tam materialize etmek istemiyoruz. İlk debug’da materialize edebiliriz. Optimized sürümde sampler ile birleşik çalıştıracağız.

---

# 6. İlk S3 debug versiyonu

İlk implementation’da sade ve açık yazacağız:

```text
for each slot:
    F_i = apply_bridge(F_sample, Q1_i)
    E_seq_i = sampler(F_i, sample_curve_i)
```

Bu daha yavaş ama debug edilebilir.

Sonra batched/optimized hale getirilir.

İlk hedef hız değil:

```text
Doğru çalışıyor mu?
Gradient akıyor mu?
Evidence gerçekten iyileşiyor mu?
```

---

# 7. Rank seçimi

Low-rank bridge için rank:

```text
r = 16 veya 32
```

Başlangıç:

```text
r = 16
```

Neden?

Çünkü:

```text
C = 256
r = 16
```

channel reduction maliyeti makul olur.

Ablation:

```text
r = 8, 16, 32, 64
```

Ama ilk S3:

```text
rank = 16
```

---

# 8. Bridge varyantları

S3 için iki uygulama seçeneği var.

## Varyant A — FiLM-style channel modulation

En basit ve güvenli bridge:

```text
Q1_i → gamma_i, beta_i
F_i = gamma_i * F + beta_i
```

Shape:

```text
gamma_i: B × N × C
beta_i:  B × N × C
```

Bu çok hafif. Ama spatial filter yok.

## Varyant B — Factorized low-rank bridge

Daha özgün versiyon:

```text
F
→ dynamic channel reduction
→ spatial depthwise conv
→ dynamic channel expansion
→ residual
```

Bizim ana S3 hedefimiz B.

Ama implementation sırası olarak önce A denenebilir.

---

# 9. Tavsiye edilen sıra

S3’e direkt complex low-rank bridge ile başlama.

Önce:

```text
S3-A: FiLM bridge
```

Sonra:

```text
S3-B: Factorized low-rank bridge
```

Neden?

Çünkü FiLM bile S2’den iyi yapıyorsa lane-conditioned modulation işe yarıyor demektir. Eğer FiLM bile kötü yapıyorsa low-rank bridge eklemek problemi büyütür.

---

# 10. S3-A: FiLM bridge

## Input

```text
F_sample: B × C × Hf × Wf
Q1:       B × N × D
```

## Query’den modulation üret

```text
gamma_beta = MLP(Q1)
```

Output:

```text
gamma_beta: B × N × 2C
```

Split:

```text
gamma: B × N × C
beta:  B × N × C
```

Apply:

```text
F_i = F_sample * (1 + gamma_i) + beta_i
```

Burada `1 + gamma` kullanmak önemli. Çünkü başlangıçta bridge identity’ye yakın olsun.

---

# 11. FiLM initialization

Başlangıçta bridge modeli bozmasın.

Bu yüzden MLP son layer bias/weight şöyle başlatılabilir:

```text
gamma ≈ 0
beta ≈ 0
```

Böylece:

```text
F_i ≈ F_sample
```

Yani S3-A, S2’den başlarken modeli bozmaz.

---

# 12. S3-A output

FiLM sonrası:

```text
F_film_i: B × N × C × Hf × Wf
```

Ama memory için slot loop daha iyi:

```text
for i in slots:
    F_i = F * (1 + gamma_i) + beta_i
    E_seq_i = sampler(F_i, sample_curve_i)
```

Böylece tam `B×N×C×H×W` tensor’ını tutmak zorunda kalmayabiliriz.

---

# 13. S3-A acceptance

FiLM bridge başarılı sayılması için:

```text
S2 checkpoint’ten başlar.
10 image overfit bozulmaz.
small subset’te S2’ye eşit veya biraz daha iyi olur.
gamma/beta sıfırda kalmaz.
```

Eğer gamma/beta hep sıfır kalıyorsa model bridge kullanmıyor.

Eğer performance düşerse:

```text
FiLM scale fazla
LR fazla
bridge initialization bozuk
sampler zaten yeterli, modulation noise ekliyor
```

olabilir.

---

# 14. S3-B: Factorized low-rank bridge

FiLM’den sonra ana bridge’e geçiyoruz.

Amaç:

```text
Lane query Q1_i, feature map’i hem channel hem spatial olarak modüle etsin.
```

Bridge akışı:

```text
F
→ U_i ile channel reduction
→ S_i ile spatial filtering
→ V_i ile channel expansion
→ residual add
```

---

# 15. S3-B matematiksel akış

Input feature:

```text
F ∈ R^{B × C × H × W}
```

Her slot için query:

```text
q_i ∈ R^D
```

Bridge query’den şunları üretir:

```text
U_i ∈ R^{C × r}
V_i ∈ R^{r × C}
S_i ∈ R^{r × k × k}
```

Ama implementation’da bu matrisleri dikkatli üreteceğiz.

Akış:

```text
Z_i = U_i^T F          # channel reduction, C → r
Z_i = depthwise_conv(Z_i, S_i)
ΔF_i = V_i^T Z_i       # channel expansion, r → C
F_i = F + bridge_scale * ΔF_i
```

Buradaki `F_i`, lane slotuna özel modüle edilmiş feature’dır.

---

# 16. Neden residual?

Çünkü bridge kötü başlarsa feature map’i bozmasın.

```text
F_i = F + γ * ΔF_i
```

Başlangıç:

```text
γ = 0.1
```

veya learnable:

```text
γ initialized to 0.0 / 0.1
```

Benim önerim:

```text
bridge_scale γ = 0.1
```

Çünkü γ=0 başlarsa bazen bridge gradient çok yavaş öğrenebilir. 0.1 daha iyi başlangıç.

---

# 17. U ve V nasıl üretilecek?

Query’den doğrudan büyük `C×r` ve `r×C` üretmek bile maliyetli olabilir.

Örnek:

```text
C = 256
r = 16
C*r = 4096
```

U için 4096, V için 4096 param çıkışı gerekir. Toplam 8192 per slot. Bu kabul edilebilir.

MLP:

```text
q_i → Linear(D → hidden)
→ GELU
→ Linear(hidden → 2*C*r)
```

Output:

```text
uv_params: B × N × (2*C*r)
```

Split:

```text
U_params: B × N × C × r
V_params: B × N × r × C
```

Başlangıç için bu yapılabilir.

---

# 18. S nasıl üretilecek?

Spatial filter:

```text
S_i ∈ R^{r × k × k}
```

Başlangıç:

```text
k = 3
r = 16
```

Param sayısı:

```text
16 × 3 × 3 = 144
```

Çok küçük.

MLP’den ayrıca üretilebilir:

```text
q_i → S_params: B × N × r × k × k
```

Toplam dynamic param per slot:

```text
U: 4096
V: 4096
S: 144
Total ≈ 8336
```

B=4, N=20 için yaklaşık:

```text
4 × 20 × 8336 ≈ 666k dynamic params
```

Bu full kernel’e göre çok daha hafif.

---

# 19. U/V normalizasyonu

Dynamic U/V çok büyük değerler üretirse feature patlayabilir.

Bu yüzden U/V output’una normalization veya scale gerekir.

Seçenekler:

```text
tanh ile sınırla
LayerNorm query üzerinde kullan
param scale factor uygula
```

Başlangıç önerisi:

```text
U = 0.1 * tanh(U_raw)
V = 0.1 * tanh(V_raw)
S = normalized spatial filter
```

Ama çok kısıtlarsak öğrenme zayıflar.

Daha pratik:

```text
q_i önce LayerNorm
MLP son layer küçük init
```

Son layer weight std küçük:

```text
std = 1e-3
```

Böylece başlangıçta bridge residual küçük olur.

---

# 20. Spatial filter normalizasyonu

S için iki seçenek var.

## Seçenek A — normal conv weight

```text
S_i raw kullanılır
```

## Seçenek B — softmax spatial filter

Her rank channel için 3×3 filtre softmax yapılır:

```text
S_i[r] = softmax(S_raw[r])
```

Bu filtreyi attention-like yapar.

Başlangıç için daha stabil olan:

```text
softmax over k×k
```

Ama bu sadece pozitif ağırlıklar verir. Edge/contrast gibi negatif filtre öğrenmek zorlaşır.

O yüzden ilk S3-B’de:

```text
S raw conv weight
```

kullanılır, ama küçük init ile.

---

# 21. Implementation zorluğu: dynamic depthwise conv

Her sample ve slot için farklı `S_i` var.

Bunu PyTorch’ta en kolay slot loop ile yaparsın:

```text
for b in B:
  for n in N:
    Z = reduce(F[b], U[b,n])
    Z = depthwise_conv(Z, S[b,n])
    ΔF = expand(Z, V[b,n])
```

Bu yavaş ama debug için net.

Optimized versiyon grouped conv ile yapılabilir ama ilk implementation’da gerek yok.

---

# 22. Debug versiyonunda loop kabul ediliyor

S3-B ilk implementation:

```text
loop over B and N
```

kabul.

Çünkü S3-B zaten final optimize edilmiş repo aşaması değil. Önce matematik doğru mu bakacağız.

Sonra hız için:

```text
B*N grouped operation
einsum
batched matmul
grouped conv
```

optimize edilir.

---

# 23. Channel reduction implementation

F single sample:

```text
F_b: C × H × W
```

Flatten spatial:

```text
F_flat: C × HW
```

U:

```text
U: C × r
```

Reduction:

```text
Z = U^T @ F_flat
```

Output:

```text
Z: r × HW
```

Reshape:

```text
Z: r × H × W
```

PyTorch karşılığı:

```text
einsum("cr,chw->rhw", U, F)
```

---

# 24. Channel expansion implementation

After spatial filtering:

```text
Z_filtered: r × H × W
```

V:

```text
V: r × C
```

Expansion:

```text
ΔF = V^T @ Z
```

Output:

```text
ΔF: C × H × W
```

PyTorch:

```text
einsum("rc,rhw->chw", V, Z_filtered)
```

---

# 25. Depthwise spatial filtering

Z:

```text
Z: r × H × W
```

S:

```text
S: r × 1 × k × k
```

Apply depthwise conv:

```text
conv2d(Z.unsqueeze(0), weight=S, groups=r, padding=k//2)
```

Output:

```text
1 × r × H × W
```

Squeeze:

```text
r × H × W
```

---

# 26. Bridge output’u sampler’a nasıl bağlanacak?

S2’de sampler input:

```text
F_sample
```

S3’te:

```text
F_i = Bridge(F_sample, Q1_i)
E_seq_i = CurveSampler(F_i, sample_x_rows_i)
```

Yani her slot için ayrı bridge feature’ı sample edilir.

---

# 27. Full feature map yerine sadece sample noktalarında bridge yapılabilir mi?

Evet, daha optimize fikir:

```text
Önce F_sample’dan curve noktalarını sample et.
Sonra U/S/V modulation’ı sadece E_seq üzerinde uygula.
```

Bu çok daha ucuz olur.

Ama spatial filtering S_i 3×3 istiyorsak çevre bilgisi gerekir. Yine de local window sampling ile yapılabilir.

İki seçenek:

## Option 1 — feature-map bridge

```text
F → bridge → sample
```

Daha güçlü, daha pahalı.

## Option 2 — sampled-sequence bridge

```text
sample → bridge over sequence
```

Daha ucuz, daha az spatial.

İlk S3-B için:

```text
feature-map bridge
```

ama memory kötü olursa sampled-sequence bridge’e dönülür.

---

# 28. Daha güvenli S3-B alternatif: sampled evidence modulation

Memory riskinden dolayı ciddi alternatif şu:

S2’den gelen:

```text
E_seq: B × N × P × C
```

bunun üzerinde low-rank modulation yap:

```text
E_reduced = E_seq @ U_i
E_filtered = 1D depthwise conv along P
E_out = E_filtered @ V_i
E_seq_refined = E_seq + γ * E_out
```

Bu, 2D feature map yerine lane sequence üzerinde çalışır.

Maliyet çok daha düşük.

Bu aslında “curve-aligned low-rank bridge” olur.

Benim dürüst önerim:

```text
S3-B1: sequence-level low-rank bridge
S3-B2: feature-map-level low-rank bridge
```

Önce B1 yapılmalı.

---

# 29. S3-B1: Sequence-level low-rank bridge

Input:

```text
E_seq: B × N × P × C
Q1:    B × N × D
```

Query’den üret:

```text
U_i: C × r
V_i: r × C
S_i: r × k_1d
```

Burada `S_i` artık 1D conv filtresi:

```text
k_1d = 3 veya 5
```

Akış:

```text
E_reduced = E_seq @ U_i      # C → r
E_filtered = depthwise_conv1d over P
E_out = E_filtered @ V_i     # r → C
E_refined = E_seq + γ * E_out
```

Output:

```text
E_refined: B × N × P × C
```

Sonra adapter/row decoder’a gider.

Bu çok daha implementable.

---

# 30. S3 için benim net önerim

Doğrudan feature-map-level bridge’e atlama.

Şu sırayı kullan:

```text
S3-A: FiLM feature modulation
S3-B1: sequence-level low-rank bridge
S3-B2: feature-map-level low-rank bridge
```

Paper katkısı için B1 bile yeterli olabilir, çünkü model zaten curve-aligned evidence üzerinde lane-conditioned dynamic modulation yapıyor.

B2 daha iddialı ama daha riskli.

---

# 31. S3-B1 tensor shape

E_seq:

```text
B × N × P × C
```

Başlangıç:

```text
B = 4
N = 20
P = 72
C = 256
r = 16
```

Reduction:

```text
E_red: B × N × P × r
```

1D depthwise conv için reshape:

```text
(B*N) × r × P
```

Filter:

```text
S: B × N × r × k
```

Dynamic depthwise conv1d ilk debug’da loop ile yapılır.

Output:

```text
E_filt: B × N × P × r
```

Expansion:

```text
E_out: B × N × P × C
```

Residual:

```text
E_refined = E_seq + γ * E_out
```

---

# 32. S3-B1 neden daha güvenli?

Çünkü S2 sampler zaten feature map’ten lane boyunca kanıtı çıkardı.

Artık bridge’in görevi:

```text
Bu lane evidence sequence’i, lane query’ye göre yeniden ağırlıklandırmak.
```

Bu, full feature-map modulation’dan daha az riskli.

Ayrıca memory:

```text
B × N × P × C
```

Örnek:

```text
4 × 20 × 72 × 256 ≈ 1.47M float
```

Bu çok makul.

Feature-map bridge ise:

```text
B × N × C × H × W
```

Örnek:

```text
4 × 20 × 256 × 72 × 200 ≈ 294M float
```

Çok daha ağır.

Bu yüzden implementation için B1 daha mantıklı.

---

# 33. S3-B1 query-conditioned U/V/S üretimi

MLP:

```text
q_i → params
```

Output dim:

```text
U: C*r = 256*16 = 4096
V: r*C = 16*256 = 4096
S: r*k = 16*3 = 48
total = 8240
```

Bu makul.

MLP:

```text
LayerNorm(D)
Linear(D → 512)
GELU
Linear(512 → 8240)
```

Son layer küçük init:

```text
std = 1e-3
bias = 0
```

---

# 34. S3-B1 apply işlemi

Her slot için:

```text
E_i: P × C
U_i: C × r
V_i: r × C
S_i: r × k
```

Reduction:

```text
Z = E_i @ U_i
```

Shape:

```text
P × r
```

Conv1D için:

```text
Z_t = Z.transpose → r × P
```

Depthwise conv1d:

```text
Z_f = conv1d(Z_t, S_i, groups=r, padding=k//2)
```

Back:

```text
P × r
```

Expansion:

```text
ΔE = Z_f @ V_i
```

Residual:

```text
E_refined = E_i + γ * ΔE
```

---

# 35. S3-B1 normalization

Bridge output çok büyümesin diye:

```text
E_refined = LayerNorm(E_seq + γ * ΔE)
```

Ama LayerNorm residual etkisini değiştirebilir.

Başlangıç:

```text
E_out = E_seq + γ * ΔE
E_out = LayerNorm(E_out)
```

Bu row decoder’a daha stabil input verir.

Config:

```text
use_layernorm = true
```

---

# 36. S3-B1 bridge scale

`γ` learnable scalar:

```text
bridge_scale = nn.Parameter(torch.tensor(0.1))
```

Alternatif per-layer/per-slot scale yok. İlk sürümde tek scalar yeter.

Loglanacak:

```text
bridge_scale value
```

---

# 37. S3 loss değişecek mi?

İlk S3’te loss değişmeyecek.

```text
L_final = L_exist + L_point + L_range + L_token + L_smooth
```

Bridge’e özel loss yok.

Neden?

Çünkü önce bridge’in doğrudan performansa etkisini görmek istiyoruz.

Evidence consistency loss eklemek yine debug zorluğu yaratır.

---

# 38. S3 matching değişecek mi?

Hayır.

S2’deki gibi:

```text
coarse output ile matching
sample curve build edilir
final output loss alır
```

Bridge sadece evidence sequence’i veya feature map’i modüle eder.

---

# 39. S3 training protocol

S3, S2 checkpoint’ten başlatılır.

Aşamalar:

```text
1. S2 checkpoint load
2. Bridge module eklenir
3. bridge params random small init
4. bridge_scale = 0.1
5. 10 image overfit
6. 100 image overfit
7. small subset
8. S2 vs S3 comparison
```

İlk 500 iter opsiyonel:

```text
only bridge + adapter + row decoder train
backbone/FPN frozen
```

Ama default:

```text
all trainable, düşük LR
```

---

# 40. S3 learning rate

Bridge yeni olduğu için biraz daha yüksek LR verilebilir.

Param groups:

```text
backbone_lr = 1e-5
base_lr = 1e-4
bridge_lr = 1e-4
```

Yani bridge LR base ile aynı.

Eğer S3 bridge öğrenmiyorsa:

```text
bridge_lr = 3e-4
```

denenebilir.

---

# 41. S3 config

```text
model:
  name: DynLaneSeqS3

  bridge:
    enabled: true
    type: sequence_low_rank   # film | sequence_low_rank | feature_low_rank
    rank: 16
    kernel_size_1d: 3
    d_model: 256
    hidden_dim: 512
    bridge_scale_init: 0.1
    use_layernorm: true
    small_init_std: 0.001

  evidence_sampler:
    enabled: true
    mode: curve_aligned

loss:
  same_as_s2: true
```

---

# 42. S3-B2 feature-map bridge config

Bunu hemen kullanmayacağız ama plan içinde dursun:

```text
bridge:
  type: feature_low_rank
  rank: 16
  kernel_size_2d: 3
  implementation: slot_loop
  bridge_scale_init: 0.05
```

B2 için `bridge_scale_init` daha küçük olmalı çünkü feature map düzeyinde etkisi daha büyük olabilir.

---

# 43. S3 visualizer

S3’te görsel olarak bridge’in ne yaptığını anlamak zor.

Bu yüzden şu debug görselleri eklenecek:

```text
S2 final prediction
S3 final prediction
GT
sample curve
```

Ayrıca numerical comparison:

```text
coarse point error
S2 final point error
S3 final point error
```

Aynı image üzerinde kıyaslanmalı.

---

# 44. Bridge activation logging

Loglanacak:

```text
bridge_scale
mean_abs_delta_E
mean_abs_E_seq
ratio = mean_abs_delta_E / mean_abs_E_seq
```

Beklenen:

```text
ratio çok küçükse bridge etkisiz
ratio çok büyükse bridge feature’ı bozuyor
```

Kabaca:

```text
ratio ≈ 0.05–0.3
```

makul olabilir.

---

# 45. Gradient logging

Bridge için:

```text
bridge_param_grad_norm
adapter_grad_norm
row_decoder_grad_norm
```

Eğer bridge grad 0 ise:

```text
bridge output row decoder’a bağlanmıyor
bridge_scale 0’da kilitli
detach yanlış yerde
```

olabilir.

---

# 46. Olası hata 1 — Bridge performansı düşürüyor

Sebep:

```text
bridge_scale büyük
U/V/S init büyük
LayerNorm yanlış
S2 checkpoint düzgün yüklenmedi
```

Çözüm:

```text
bridge_scale 0.1 → 0.01
small_init_std 1e-3 → 1e-4
ilk 500 iter sadece bridge train
S3-A FiLM’e geri dön
```

---

# 47. Olası hata 2 — Bridge hiçbir şey katmıyor

Belirti:

```text
S3 ≈ S2
bridge_scale küçük kalıyor
delta_E çok düşük
```

Çözüm:

```text
bridge_lr artır
bridge_scale init 0.1 → 0.2
rank 16 → 32
kernel_size 3 → 5
evidence adapter içinde Q1 ağırlığını azalt
```

Ama önce S2 zaten çok iyiyse küçük katkı normal olabilir.

---

# 48. Olası hata 3 — NaN/inf

Muhtemel kaynak:

```text
U/V dynamic params patlıyor
conv output çok büyüyor
LayerNorm input NaN
LR fazla
```

Çözüm:

```text
gradient clipping
small init
tanh param bound
bridge output clamp değil ama norm kontrolü
AMP kapalı debug
```

Gerekirse:

```text
U = tanh(U_raw)
V = tanh(V_raw)
```

eklenir.

---

# 49. Olası hata 4 — Çok yavaş

Özellikle feature-map bridge loop yavaş olur.

Çözüm sırası:

```text
sequence-level bridge kullan
B2 feature-map bridge’i kapat
rank azalt
slot loop yerine batched einsum
AMP aç
```

Paper için B1 yeterli katkı verirse B2’ye gerek yok.

---

# 50. S3 ablation planı

S3 için ablation çok önemli.

Minimum ablation:

```text
S2 baseline: no bridge
S3-A: FiLM bridge
S3-B1: sequence low-rank bridge r=16
S3-B1: sequence low-rank bridge r=32
```

Opsiyonel:

```text
feature-map low-rank bridge
bridge without spatial conv
bridge without U/V, only FiLM
bridge without LayerNorm
```

---

# 51. S3’ün paper’daki novelty karşılığı

S3-B1 için novelty cümlesi şöyle olabilir:

```text
We introduce a lane-conditioned low-rank evidence bridge that refines curve-aligned visual evidence before sequence decoding, avoiding the explicit generation of full dynamic convolution kernels.
```

Türkçesi:

```text
Her lane için curve-aligned görsel kanıtı sequence decoding öncesinde lane-conditioned düşük-rütbeli bir bridge ile rafine ediyoruz; bunu full dynamic kernel üretmeden yapıyoruz.
```

Bu daha savunulabilir. Çünkü hem CondLSTR tarafındaki dynamic kernel fikrine bağlı, hem de Lane2Seq tarafındaki sequence decoder’a hizmet ediyor.

---

# 52. S3 implementasyonunda en önemli karar

Benim net önerim:

```text
Feature-map-level low-rank bridge’i ana implementation olarak başlatma.
```

Önce:

```text
S3-A FiLM
S3-B1 sequence-level low-rank
```

yap.

Çünkü B1 daha az memory ister, debug daha kolaydır ve modelin asıl evidence sequence mantığıyla daha uyumludur.

B2 ancak B1 çalıştıktan sonra denenir.

---

# 53. Part 8 özeti

Bu partta S3’ü kilitledik:

```text
S3 = S2 + lane-conditioned bridge

Full dynamic kernel:
  yasak / ilk implementation’da yok

Sıra:
  S3-A FiLM bridge
  S3-B1 sequence-level low-rank bridge
  S3-B2 feature-map-level low-rank bridge optional

Önerilen ana bridge:
  sequence-level low-rank

Input:
  E_seq: B × N × P × C
  Q1: B × N × D

Bridge:
  Q1 → U, S, V
  E_seq → C→r reduction
  1D depthwise filtering over rows
  r→C expansion
  residual + LayerNorm

Başlangıç:
  rank = 16
  kernel_size = 3
  bridge_scale = 0.1
  small init
  loss değişmiyor
  matching değişmiyor
```

---

# DynLaneSeq-EG Implementation Plan — Part 9

## S4: Optional Zoom-In Refinement

Bu partta full modelin en son ve en riskli parçasına geliyoruz:

```text
S4 = S3 + one-step zoom-in refinement
```

Ama en baştan net karar:

```text
S4 ana MVP değildir.
```

S4 sadece şu durumda denenir:

```text
S0 çalıştı
S1 çalıştı
S2 evidence sampler çalıştı
S3 bridge en azından modeli bozmadan çalıştı
```

Bunlar olmadan S4’e geçmek hata olur.

---

# 1. S4’ün ana fikri

S3’e kadar model şu akıştaydı:

```text
Image
→ Backbone/FPN
→ lane slots
→ coarse prediction
→ curve-aligned evidence
→ bridge
→ row decoder
→ final lane
```

S4’te bir refinement turu daha ekliyoruz:

```text
First prediction
→ predicted lane daha net hale gelir
→ bu tahmine göre evidence tekrar okunur
→ decoder ikinci kez çalışır
→ refined final lane çıkar
```

Yani model önce geniş bakıyor, sonra kendi tahminini kullanarak daha odaklı bakıyor.

Basit ifade:

```text
coarse look → focused look
```

---

# 2. S4 neden gerekli olabilir?

S2/S3’te sampler coarse prediction’a dayanıyor.

Ama coarse prediction bazen şöyle olabilir:

```text
lane’in biraz yanında
occlusion bölgesinde kopuk
far-range tarafında hatalı
fork/merge noktasında kararsız
```

İlk decoder tahmini, coarse prediction’dan daha iyi olabilir. O zaman ikinci sampling turunda bu daha iyi tahmini kullanmak mantıklı olur.

Akış şöyle:

```text
coarse_x_rows
→ first refined x_rows
→ second sampling bu first refined x_rows üzerinden
→ final refined x_rows
```

Böylece sampler giderek lane’e yaklaşabilir.

---

# 3. S4’te kaç refinement olacak?

İlk implementation:

```text
1 refinement step
```

Yani toplam iki prediction:

```text
prediction_0 = first refined prediction
prediction_1 = zoom-in refined prediction
```

Daha fazla yok.

Neden?

Çünkü 2 veya 3 refinement:

```text
daha yavaş
daha fazla memory
daha zor training
daha zor debug
```

İlk S4:

```text
one-step zoom-in
```

---

# 4. S4’te hangi outputlar olacak?

S4 output dictionary şöyle olabilir:

```text
outputs = {
    "coarse": {...},

    "stage1": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "stage2": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "evidence": {
        "E_seq_stage1": B × N × P × D,
        "E_seq_stage2": B × N × P × D,
        "sample_x_stage1": B × N × P,
        "sample_x_stage2": B × N × P
    }
}
```

Burada final prediction:

```text
outputs["stage2"]
```

olacak.

---

# 5. S4 training akışı

Training sırasında akış şöyle:

```text
1. Backbone/FPN/Q1 hesapla
2. Coarse branch pred_x_rows_coarse üretir
3. Coarse output ile Hungarian matching yapılır
4. Curriculum ile stage1 sample curve hazırlanır
5. Stage1 evidence sample edilir
6. Stage1 decoder prediction üretir
7. Stage2 sample curve hazırlanır
8. Stage2 evidence sample edilir
9. Stage2 decoder final prediction üretir
10. Loss hesaplanır
```

Daha açık:

```text
coarse_x
→ sample_x_stage1
→ E_stage1
→ pred_x_stage1
→ sample_x_stage2
→ E_stage2
→ pred_x_stage2
```

---

# 6. Stage1 sample curve nasıl seçilecek?

S2’deki curriculum aynı şekilde kullanılır.

Matched positive slotlarda:

```text
sample_x_stage1 = α * gt_x + (1 - α) * coarse_x
```

Invalid GT rowlarda:

```text
sample_x_stage1 = coarse_x
```

Unmatched slotlarda:

```text
sample_x_stage1 = coarse_x
```

Başlangıçta:

```text
α = 1.0
```

Sonra:

```text
α → 0.0
```

Yani stage1 S2 ile aynı mantıkta.

---

# 7. Stage2 sample curve nasıl seçilecek?

S4’ün asıl farkı burada.

Stage2 için kaynak:

```text
pred_x_stage1
```

Ama training başında stage1 de kötü olabilir. Bu yüzden yine curriculum gerekir.

Stage2 için:

```text
sample_x_stage2 = β * gt_x + (1 - β) * pred_x_stage1
```

Başlangıçta:

```text
β = 1.0
```

Sonra:

```text
β → 0.0
```

Ancak `β` schedule’ı `α`dan biraz daha yavaş düşebilir. Çünkü stage2’nin tamamen predicted stage1’e güvenmesi daha riskli.

Örnek:

```text
epoch 0–3:
  α = 1.0
  β = 1.0

epoch 4–8:
  α: 1.0 → 0.0
  β: 1.0 → 0.5

epoch 9–13:
  α = 0.0
  β: 0.5 → 0.0

epoch 14+:
  α = 0.0
  β = 0.0
```

Bu daha güvenli.

---

# 8. Stage2’de detach kararı

Stage2 sample curve `pred_x_stage1`ten geliyor.

Şimdi kritik soru:

```text
stage2 loss, pred_x_stage1 üzerinden stage1’e gradient göndersin mi?
```

İlk implementation’da:

```text
sample_x_stage2 = sample_x_stage2.detach()
```

Yani stage2 sampling koordinatları üzerinden stage1’e gradient göndermiyoruz.

Neden?

Çünkü grid_sample coordinate gradient’i erken aşamada noisy olabilir. İlk S4’te stage1 kendi loss’u ile, stage2 kendi loss’u ile eğitilecek.

Sonra ablation:

```text
detach_stage2_coords = false
```

denenebilir.

---

# 9. Aynı decoder mı, ayrı decoder mı?

İki seçenek var.

## Seçenek A — Stage1 ve Stage2 aynı decoder’ı paylaşır

Avantaj:

```text
daha az parametre
daha hızlı
daha az overfit
```

Dezavantaj:

```text
stage1 ve stage2 farklı görevler yapıyor olabilir
```

## Seçenek B — Stage2 ayrı refinement decoder kullanır

Avantaj:

```text
stage2 özellikle correction/refinement öğrenir
```

Dezavantaj:

```text
daha fazla parametre
daha fazla memory
debug daha zor
```

İlk implementation kararı:

```text
aynı row decoder paylaşılacak
```

Yani:

```text
stage1 decoder = stage2 decoder
```

Ama adapter ayrı olabilir.

---

# 10. Adapter aynı mı, ayrı mı?

S4’te iki evidence sequence var:

```text
E_seq_stage1
E_seq_stage2
```

Bunları aynı adapter’dan geçirmek daha sade:

```text
E_adapted = EvidenceAdapter(E_seq)
```

Karar:

```text
stage1 ve stage2 aynı evidence adapter’ı paylaşacak
```

Böylece S4, S3’ten çok kopmaz.

İleride ablation:

```text
separate stage2 adapter
```

denenebilir.

---

# 11. Bridge aynı mı, ayrı mı?

S3 bridge kullanılıyorsa:

```text
E_seq → bridge → E_refined
```

S4’te stage1 ve stage2 için bridge paylaşılacak.

Karar:

```text
same bridge shared across stages
```

Neden?

Çünkü bridge’in görevi evidence sequence’i lane query’ye göre modüle etmek. Stage1/stage2 için tamamen ayrı bridge ilk implementation’da gereksiz.

---

# 12. Stage2 row token input’u

S3’te row token şöyleydi:

```text
row_tokens = Q1 + row_emb + evidence
```

S4 stage2’de ek olarak stage1 hidden state de kullanılabilir.

İki seçenek var.

## Seçenek A — Stage2 sadece yeni evidence kullanır

```text
row_tokens_stage2 = Q1 + row_emb + E_stage2
```

## Seçenek B — Stage2 stage1 hidden state’i de kullanır

```text
row_tokens_stage2 = Q1 + row_emb + E_stage2 + H_stage1
```

Burada:

```text
H_stage1: B × N × P × D
```

Bu daha güçlü ama daha karmaşık.

İlk S4 için karar:

```text
Stage2, H_stage1 kullanacak ama gated şekilde.
```

Çünkü “zoom-in refinement” fikrinin asıl anlamı decoder’ın ilk tahmin bağlamından yararlanmak.

---

# 13. Stage1 hidden gate

Stage2 input:

```text
row_tokens_stage2 =
    Q1
  + row_emb
  + E_stage2_adapted
  + δ * H_stage1
```

Burada:

```text
δ = learnable scalar
```

Başlangıç:

```text
δ = 0.1
```

Neden 0.1?

Stage1 hidden state birden stage2’yi domine etmesin. Eğer faydalıysa model scale’i büyütür.

Config:

```text
zoom_refine:
  use_stage1_hidden: true
  hidden_scale_init: 0.1
```

---

# 14. Stage1 hidden detach edilecek mi?

Soru:

```text
Stage2 loss, H_stage1 üzerinden stage1 decoder’a gradient göndersin mi?
```

İlk implementation’da daha stabil seçenek:

```text
H_stage1_detached = H_stage1.detach()
```

Yani stage2, stage1 hidden’ı feature gibi kullanır ama stage2 loss stage1 decoder’ı ikinci yoldan zorlamaz.

Stage1 zaten kendi auxiliary loss’u ile eğitilir.

Karar:

```text
detach_stage1_hidden = true
```

Ablation:

```text
detach_stage1_hidden = false
```

daha sonra denenir.

---

# 15. Stage1 ve Stage2 loss

S4’te final loss stage2 üzerinden olacak.

Ama stage1’in de düzgün kalması gerekir.

Loss:

```text
L_total =
  L_stage2
+ λ_stage1 * L_stage1
+ λ_coarse * L_coarse
```

Başlangıç:

```text
λ_stage1 = 0.5
λ_coarse = 0.25
```

Yani final stage daha önemli.

Stage1 loss, S3 final loss ile aynı olabilir:

```text
L_stage1 =
  L_exist
+ L_point
+ L_range
+ L_token
```

Stage2 loss:

```text
L_stage2 =
  L_exist
+ L_point
+ L_range
+ L_token
```

Ama existence/range stage1/stage2’de aynı kullanılıyorsa bunları iki kere yazmaya gerek yok.

---

# 16. Existence/range stage2’de yeniden üretilecek mi?

İlk S4’te existence ve range tekrar üretilmeyecek.

Karar:

```text
exist_logits = coarse/S3 output
range_norm = coarse/S3 output
```

Stage2 sadece row x prediction’ı refine eder.

Neden?

Çünkü S4’ün amacı lane konumlarını zoom-in ile düzeltmek. Existence/range’i de değiştirmek ekstra karmaşıklık.

Yani loss:

```text
L_exist ve L_range coarse/final shared output üzerinden
L_point ve L_token stage1/stage2 üzerinden
```

Daha net:

```text
L_total =
  w_exist * L_exist
+ w_range * L_range
+ w_point * L_point_stage2
+ w_token * L_token_stage2
+ λ_stage1 * (w_point * L_point_stage1 + w_token * L_token_stage1)
+ λ_coarse * L_point_coarse
```

Bu sade ve yeterli.

---

# 17. Matching hangi output ile yapılacak?

S4’te matching yine coarse output ile yapılacak.

Neden?

Çünkü stage1 ve stage2 sample curve üretmek için matched GT gerekiyor. Matching’i stage2’den sonra yapmak mantıksal olarak geç kalır.

Akış:

```text
coarse_outputs → matcher → matches
matches → build stage1 sample curves
stage1 → build stage2 sample curves
stage2 → loss with same matches
```

Bu karar S2/S3 ile tutarlı.

---

# 18. S4 inference

Inference’ta GT yok.

Akış:

```text
1. image → backbone/FPN/Q1
2. coarse_x üret
3. sample_x_stage1 = coarse_x
4. stage1 evidence sample et
5. stage1 pred_x üret
6. sample_x_stage2 = stage1 pred_x
7. stage2 evidence sample et
8. stage2 pred_x üret
9. exist/range filtering
10. final lanes
```

Yani inference tamamen self-refinement.

---

# 19. S4 inference maliyeti

S4, S3’e göre yaklaşık şu kadar yavaşlar:

```text
+1 extra sampler
+1 extra row decoder forward
+1 extra adapter/bridge forward
```

Yani FPS düşebilir.

Bu yüzden S4’ün gerçekten değip değmediği ölçülmeli:

```text
S3 F1 / FPS
S4 F1 / FPS
```

Eğer S4 sadece çok küçük F1 getirip FPS’i ciddi düşürüyorsa ana modelde kullanılmayabilir.

---

# 20. S4 training schedule

S4, S3 checkpoint’ten başlatılacak.

Aşamalar:

```text
1. S3 checkpoint load
2. zoom refinement modülleri açılır
3. hidden scale δ = 0.1
4. stage2 sampling curriculum başlar
5. 10 image overfit
6. 100 image overfit
7. small subset
8. S3 vs S4 comparison
```

İlk 500 iter:

```text
stage2-related params daha yüksek LR alabilir
```

Ama shared decoder kullanıyorsak param ayırmak zor olabilir.

Başlangıçta:

```text
base_lr = 5e-5 veya 1e-4
```

S3 checkpoint’i bozmamak için `5e-5` daha güvenli olabilir.

---

# 21. S4 config

```text
model:
  name: DynLaneSeqS4

  zoom_refine:
    enabled: true
    num_steps: 1
    share_decoder: true
    share_adapter: true
    share_bridge: true

    use_stage1_hidden: true
    hidden_scale_init: 0.1
    detach_stage1_hidden: true
    detach_stage2_sample_coords: true

  sampler_curriculum:
    stage1:
      alpha_start: 1.0
      alpha_end: 0.0
      warmup_epochs: 3
      decay_epochs: 5
      noise_std_px: 3.0

    stage2:
      beta_start: 1.0
      beta_end: 0.0
      warmup_epochs: 3
      decay_epochs: 10
      noise_std_px: 2.0

loss:
  w_exist: 2.0
  w_range: 1.0
  w_point: 5.0
  w_token: 0.5

  aux:
    stage1_weight: 0.5
    coarse_weight: 0.25
```

---

# 22. Stage2 sample noise

Stage1 sample’da noise vardı:

```text
GT + noise
```

Stage2’de noise daha küçük olmalı.

Çünkü stage2 daha odaklı refinement yapıyor.

Öneri:

```text
stage1 noise_std_px = 3.0
stage2 noise_std_px = 2.0
```

Full predicted sampling aşamasında noise yok.

---

# 23. Stage2 local window sampling

S4 için local window sampling daha faydalı olabilir.

Çünkü zoom-in aşamasında prediction lane’e yakın olur; küçük bir local window ile çevre kanıtı alınabilir.

Opsiyon:

```text
stage2 local_window:
  enabled: true
  offsets_px: [-4, 0, +4]
```

Ama ilk S4’te local window kapalı tutulabilir.

Eğer stage2 predicted sampling çöküyorsa:

```text
local_window aç
```

Bu, küçük hatalara tolerans sağlar.

---

# 24. S4 visualizer

S4’te görselleştirme çok önemli.

Her görüntü için yan yana:

```text
GT
coarse prediction
stage1 prediction
stage2 prediction
```

Ayrıca sample curve’ler:

```text
stage1 sample curve
stage2 sample curve
```

Renk önerisi:

```text
GT = yeşil
coarse = mavi
stage1 = turuncu
stage2 = kırmızı
stage1 sample = sarı nokta
stage2 sample = mor nokta
```

Bu görselden şunu anlamalıyız:

```text
Stage2 gerçekten stage1’den daha iyi mi?
Yoksa sadece farklı ama daha kötü mü?
```

---

# 25. S4 logging

Yeni loglar:

```text
point_error_coarse
point_error_stage1
point_error_stage2

token_loss_stage1
token_loss_stage2

stage1_to_stage2_improvement
stage2_sample_error_to_gt

hidden_scale_delta
sampler_alpha
sampler_beta
```

Özellikle şu metrik kritik:

```text
stage1_to_stage2_improvement =
  point_error_stage1 - point_error_stage2
```

Beklenen:

```text
pozitif olmalı
```

Yani stage2 error daha düşük olmalı.

Eğer sürekli negatifse S4 zarar veriyor.

---

# 26. S4 acceptance criteria

S4 başarılı sayılması için:

```text
1. 10 image overfit bozulmamalı.
2. Stage2 prediction, stage1’den görsel olarak daha kötü olmamalı.
3. Small subset’te stage2 point error stage1’den düşük olmalı.
4. FPS düşüşü kabul edilebilir olmalı.
5. hidden_scale δ tamamen sıfırda kalmamalı.
6. predicted-only stage2 sampling’e geçince model çökmemeli.
```

Eğer bu kriterler sağlanmazsa S4 ana modelden çıkarılır.

---

# 27. S4 başarısız olursa ne yapacağız?

S4 başarısızsa bu modelin çöpe gittiği anlamına gelmez.

Ana model olarak S3 kullanılır:

```text
DynLaneSeq-EG-B = S3
```

S4 sadece full/large variant olur:

```text
DynLaneSeq-EG-Z = S4
```

Paper’da şöyle denebilir:

```text
iterative refinement gives marginal gains but increases latency
```

Eğer sonuç buysa S4 opsiyonel kalır.

---

# 28. Olası hata 1 — Stage2 stage1’den kötü

Sebep:

```text
stage2 sampler yanlış yere bakıyor
stage1 hidden state fazla baskın
hidden scale δ büyük
stage2 curriculum hızlı
```

Çözüm:

```text
β decay’i yavaşlat
hidden_scale 0.1 → 0.01
detach_stage1_hidden true kontrol et
stage2 local window aç
stage2 loss weight artırma, azalt
```

---

# 29. Olası hata 2 — Stage2 hiçbir şey değiştirmiyor

Belirti:

```text
stage2 ≈ stage1
improvement ≈ 0
hidden_scale düşük
E_stage2 ≈ E_stage1
```

Çözüm:

```text
stage2 sample curve gerçekten stage1 pred ile kuruluyor mu kontrol et
hidden_scale artır
separate stage2 adapter dene
stage2 local window aç
bridge stage2’de aktif mi kontrol et
```

---

# 30. Olası hata 3 — Memory/FPS çok kötü

Sebep:

```text
sampler + decoder iki kez çalışıyor
bridge iki kez çalışıyor
```

Çözüm:

```text
share decoder/adapter/bridge
stage1 hidden saklama precision kontrol et
P=72 yerine P=36 ablation
row_decoder layer 2 → 1
S4’ü sadece large variant yap
```

---

# 31. Olası hata 4 — Training kararsız

Sebep:

```text
stage1/stage2 losses birbirini çekiştiriyor
sample coord detach yanlış
curriculum çok hızlı
LR fazla
```

Çözüm:

```text
base_lr 1e-4 → 5e-5
stage1 aux weight 0.5 → 0.25
stage2 beta daha yavaş decay
AMP kapalı debug
grad clip 1.0 → 0.5
```

---

# 32. Stage1 hidden kullanmadan S4 ablation

Mutlaka denenmeli:

```text
S4-no-hidden:
  stage2 row_tokens = Q1 + row_emb + E_stage2
```

Buna karşı:

```text
S4-hidden:
  stage2 row_tokens = Q1 + row_emb + E_stage2 + δH_stage1
```

Eğer hidden kullanımı katkı vermiyorsa çıkarılır. Çünkü gereksiz karmaşıklık.

---

# 33. Stage2 shared decoder vs separate decoder ablation

Minimum ablation:

```text
shared decoder
separate stage2 decoder
```

Beklenti:

```text
shared decoder daha hafif
separate decoder belki daha iyi ama overfit riski yüksek
```

İlk paper için shared decoder daha savunulabilir.

---

# 34. S4’ün paper’daki rolü

S4’ü ana contribution gibi sunmak riskli olabilir. Çünkü iterative refinement birçok alanda bilinen bir fikir.

Daha doğru sunum:

```text
optional zoom-in refinement module
```

Ana contribution hâlâ:

```text
evidence-grounded lane sequence decoding
curve-aligned visual evidence
lane-conditioned low-rank evidence bridge
```

S4 ise:

```text
additional refinement for difficult cases
```

olmalı.

---

# 35. S4 hangi durumlarda işe yarayabilir?

Beklenen fayda alanları:

```text
occlusion
far-range lane
fork/merge
düşük kontrast
coarse prediction’ın birkaç pixel kaydığı durumlar
```

Bu yüzden sadece overall F1 değil, subset metrikleri de bakılmalı:

```text
curve subset
crowded/occlusion subset
far-range error
night/shadow subset
```

Eğer sadece normal sahnelerde fark yok ama zor sahnelerde iyileşme varsa S4 değerli olabilir.

---

# 36. S4’te dikkat: refinement hallucination yapabilir

Stage2 kendi stage1 tahminine güvenerek yanlış lane’i daha da güçlendirebilir.

Örnek:

```text
stage1 yanlış çizgiye kaydı
stage2 o yanlış çizgiden evidence aldı
yanlış prediction daha da keskinleşti
```

Buna karşı:

```text
local window
GT-guided curriculum
confidence gating
stage2 hidden scale düşük
```

yardımcı olur.

İleride confidence düşükse stage2 refinement azaltılabilir:

```text
refine_gate = sigmoid(confidence)
```

Ama ilk S4’te yok.

---

# 37. Confidence-gated refinement opsiyonu

Opsiyonel fikir:

```text
if stage1 confidence high:
    stage2 evidence daha fazla kullan
else:
    stage2 refinement zayıf olsun
```

Formül:

```text
row_tokens_stage2 =
  base_tokens + g_conf * E_stage2 + δ * H_stage1
```

Burada:

```text
g_conf = sigmoid(MLP(Q1))
```

Ama bu ekstra head demek. İlk S4’te yok.

Ablation olarak ileride denenebilir.

---

# 38. S4 implementation file yapısı

Yeni dosyalar:

```text
modeling/
├── refinement/
│   ├── zoom_refinement.py
│   └── refinement_scheduler.py
│
└── dynlaneseq_s4.py
```

Training tarafı:

```text
engine/
├── build_stage1_sample_curves.py
├── build_stage2_sample_curves.py
└── train_s4.py
```

Ama ideal olarak sample curve builder generic yazılır:

```text
build_sample_curves(stage="stage1" or "stage2")
```

---

# 39. S4 output contract

S4 model forward contract:

```text
forward_features(images)
  → F_sample, Q1, coarse_outputs

forward_stage1(F_sample, Q1, sample_x_stage1)
  → stage1_outputs, H_stage1, E_stage1

forward_stage2(F_sample, Q1, sample_x_stage2, H_stage1=None)
  → stage2_outputs, H_stage2, E_stage2
```

Böyle modüler olursa debug kolay olur.

---

# 40. S4 training contract

Training loop tarafında:

```text
features = model.forward_features(images)

coarse_outputs = features["coarse"]
matches = matcher(coarse_outputs, targets)

sample_x_stage1 = build_sample_curves(
    source="coarse",
    coarse_outputs,
    targets,
    matches,
    alpha
)

stage1_outputs = model.forward_stage1(features, sample_x_stage1)

sample_x_stage2 = build_sample_curves(
    source="stage1",
    stage1_outputs,
    targets,
    matches,
    beta
)

stage2_outputs = model.forward_stage2(features, sample_x_stage2, stage1_hidden)

loss = criterion_s4(coarse, stage1, stage2, targets, matches)
```

Bu net olmalı. Model içinde matcher saklanmayacak.

---

# 41. S4 criterion output

Loss dict:

```text
loss_dict = {
    "loss_total": ...,

    "loss_exist": ...,
    "loss_range": ...,

    "loss_point_stage1": ...,
    "loss_token_stage1": ...,

    "loss_point_stage2": ...,
    "loss_token_stage2": ...,

    "loss_point_coarse": ...,

    "stage1_weight": 0.5,
    "coarse_weight": 0.25
}
```

Böyle loglar net olur.

---

# 42. S4 sonrası model isimleri

Net isimlendirme:

```text
DynLaneSeq-EG-S0:
  geometry-only sanity

DynLaneSeq-EG-S1:
  row token decoder

DynLaneSeq-EG-S2:
  curve-aligned evidence sampler

DynLaneSeq-EG-S3:
  low-rank evidence bridge

DynLaneSeq-EG-S4:
  zoom-in refinement
```

Paper varyantları:

```text
DynLaneSeq-EG-S:
  S2 veya hafif S3

DynLaneSeq-EG-B:
  S3-B1 sequence low-rank bridge

DynLaneSeq-EG-Z:
  S4 zoom-in refinement
```

Ana paper modeli için benim önerim:

```text
DynLaneSeq-EG-B = S3-B1
```

S4 sadece large/full variant.

---

# 43. Part 9 özeti

Bu partta S4’ü kilitledik:

```text
S4 = optional one-step zoom-in refinement

Akış:
  coarse → stage1 sample → stage1 pred
         → stage2 sample → stage2 pred

Training:
  stage1 sample: GT/pred curriculum with alpha
  stage2 sample: GT/stage1 curriculum with beta
  beta daha yavaş düşer

Stability:
  stage2 sample coords detach
  stage1 hidden detach
  hidden scale δ = 0.1
  shared decoder/adapter/bridge

Loss:
  final stage2 ana loss
  stage1 auxiliary loss
  coarse auxiliary point loss

Inference:
  GT yok
  coarse → stage1 → stage2

Decision:
  S4 ana MVP değil
  S4 sadece S3 başarılı olduktan sonra
```

---

# DynLaneSeq-EG Implementation Plan — Part 10

## Ablation Planı, Evaluation Stratejisi, Benchmark Yol Haritası ve Repo Milestone Planı

Bu partta artık modelin parçalarını nasıl raporlayacağımızı ve hangi sırayla deney yapacağımızı netleştiriyoruz.

Buradaki amaç şu:

```text
Sadece modeli kodlamak değil,
modelin gerçekten ne katkı verdiğini izole şekilde kanıtlamak.
```

Çünkü bu tarz bir modelde en büyük risk şu:

```text
Model çalışsa bile hangi parçanın işe yaradığını gösteremezsen,
paper zayıf görünür.
```

O yüzden ablation planı en az mimari kadar önemli.

---

# 1. Final model ailesi

Artık model sürümlerini net isimlendirelim.

```text
S0 = Geometry-only sanity model
S1 = Row-wise token decoder
S2 = Curve-aligned evidence sampler
S3-A = FiLM evidence modulation
S3-B1 = Sequence-level low-rank evidence bridge
S3-B2 = Feature-map-level low-rank bridge, optional
S4 = Optional one-step zoom-in refinement
```

Paper için ana model önerim:

```text
DynLaneSeq-EG-B = S3-B1
```

Yani:

```text
Backbone + FPN
→ lane slots
→ coarse prediction
→ curve-aligned evidence sampler
→ sequence-level low-rank bridge
→ row-wise token decoder
→ final lane points
```

S4’ü ana model yapmazdım. S4 daha çok “large / refined variant” gibi durmalı.

---

# 2. Neden ana model S3-B1 olmalı?

Çünkü S3-B1 şu dengeyi veriyor:

```text
novelty var
implementation yapılabilir
memory makul
debug edilebilir
FPS tamamen ölmez
paper’da savunulabilir
```

S4 daha iddialı ama daha riskli:

```text
daha yavaş
daha karmaşık
hakem “iterative refinement zaten biliniyor” diyebilir
debug maliyeti yüksek
```

Bu yüzden ana paper contribution şunlar olmalı:

```text
1. Evidence-grounded lane sequence decoding
2. Curve-aligned visual evidence sampling
3. Lane-conditioned low-rank evidence bridge
```

Zoom-in refinement varsa:

```text
optional refinement module
```

olarak sunulur.

---

# 3. Ana benchmark sırası

İlk benchmark:

```text
CULane
```

Sebep:

```text
2D lane detection için standart
çok fazla karşılaştırma var
normal / crowded / curve / shadow / night gibi alt senaryolar var
```

İkinci benchmark:

```text
TuSimple
```

Sebep:

```text
daha basit highway senaryosu
accuracy metric’i yaygın
modelin kolay sahnelerde de bozulmadığını gösterir
```

Üçüncü benchmark:

```text
CurveLanes
```

Sebep:

```text
curved, forked, dense lane yapıları için önemli
bizim evidence-grounded iddiamızı test eder
```

Opsiyonel daha sonra:

```text
LLAMAS veya OpenLane
```

Ama OpenLane’i ilk aşamada zorlamazdım. Çünkü class/category ve daha karmaşık annotation tarafı implementation yükünü büyütür.

---

# 4. Evaluation metrikleri

## CULane için

Resmi metrik:

```text
F1
Precision
Recall
```

Alt senaryolar:

```text
Normal
Crowded
Dazzle light
Shadow
No line
Arrow
Curve
Cross
Night
```

Özellikle bakılacaklar:

```text
Curve
Crowded
Night
Shadow
```

Çünkü bizim modelin iddiası şu:

```text
lane-specific evidence zor ve karışık sahnelerde daha iyi çalışmalı.
```

---

## TuSimple için

Raporlanacak:

```text
Accuracy
FPR
FNR
F1
```

TuSimple daha kolay olduğu için burada büyük fark beklememek lazım.

Burada amaç:

```text
Model basit highway senaryosunda da stabil mi?
False positive artıyor mu?
```

---

## CurveLanes için

Raporlanacak:

```text
F1
Precision
Recall
```

Özellikle niteliksel görseller önemli:

```text
curved lanes
forked lanes
dense lanes
blocked lanes
```

Çünkü modelin “curve-aligned evidence” iddiası burada görsel olarak daha iyi anlatılır.

---

# 5. Ek debug metrikleri

Resmi benchmark metrikleri dışında development sırasında şu metrikleri tutacağız:

```text
mean point error
median point error
range error
exist precision
exist recall
average number of predicted lanes
duplicate lane count
matched p_lane
unmatched p_lane
```

Bunlar paper metric olmayabilir ama training debug için çok kritik.

---

# 6. Zorunlu ablation tablosu

Paper için minimum ablation tablosu şöyle olmalı:

```text
Baseline S0
S1: + row-wise token decoder
S2: + curve-aligned evidence sampler
S3-A: + FiLM bridge
S3-B1: + low-rank evidence bridge
S4: + zoom-in refinement
```

Tablo formatı:

```text
Model Variant | CULane F1 | Curve F1 | Night F1 | FPS | Params | Memory
```

Bence en kritik sütunlar:

```text
Overall F1
Curve subset F1
Night/Shadow subset F1
FPS
GPU memory
```

Çünkü modelin katkısı sadece overall F1’de değil, zor durumlarda görünmeli.

---

# 7. Ablation 1 — Evidence etkisi

Soru:

```text
Lane-specific evidence gerçekten işe yarıyor mu?
```

Karşılaştırma:

```text
S1: Q1 + row decoder
S2: Q1 + curve-aligned evidence + row decoder
```

Beklenen:

```text
S2, S1’den özellikle curve/crowded/occlusion tarzı sahnelerde daha iyi olmalı.
```

Eğer S2 sadece overall’da çok az artıyor ama zor subsetlerde artıyorsa bu yine değerli.

---

# 8. Ablation 2 — Curve-aligned sampling etkisi

Soru:

```text
Evidence’ı lane boyunca okumak gerçekten global pooling’den iyi mi?
```

Karşılaştırma:

```text
Global pooled feature
Flattened cross-attention feature
Curve-aligned sampled feature
```

Üç varyant:

```text
A. Q1 only
B. Q1 + global pooled F
C. Q1 + curve-aligned E_seq
```

Beklenen:

```text
C en iyi olmalı.
```

Bu ablation çok önemli çünkü modelin ana iddiası burada.

---

# 9. Ablation 3 — GT-guided curriculum etkisi

Soru:

```text
Sampler training için curriculum gerekli mi?
```

Karşılaştırma:

```text
No curriculum: always predicted curve
GT-only warmup
GT→pred mixed curriculum
```

Beklenen:

```text
No curriculum daha kararsız olur.
GT→pred curriculum en stabil sonuç verir.
```

Bu ablation implementation açısından da önemli. Çünkü cold-start problemini paper’da savunmak için iyi kanıt olur.

---

# 10. Ablation 4 — Bridge etkisi

Soru:

```text
Lane-conditioned bridge gerçekten S2’ye katkı veriyor mu?
```

Karşılaştırma:

```text
S2: no bridge
S3-A: FiLM bridge
S3-B1: sequence low-rank bridge
S3-B1 r=32
```

Beklenen:

```text
FiLM küçük katkı verir.
Low-rank bridge daha iyi verir.
Rank 32, 16’dan biraz iyi olabilir ama daha pahalıdır.
```

Eğer FiLM, low-rank kadar iyi çıkarsa bu da dürüstçe raporlanmalı. O zaman daha sade model tercih edilebilir.

---

# 11. Ablation 5 — Low-rank rank seçimi

Rank değerleri:

```text
r = 8
r = 16
r = 32
r = 64
```

Tablo:

```text
Rank | F1 | Curve F1 | FPS | Memory
```

Beklenen:

```text
r=8 yetersiz olabilir.
r=16 iyi denge olabilir.
r=32 biraz daha iyi ama daha pahalı olabilir.
r=64 gereksiz olabilir.
```

Ana model için pratik seçim:

```text
r = 16
```

---

# 12. Ablation 6 — Token decoder etkisi

Soru:

```text
Row-token decoder, direkt MLP row head’den daha iyi mi?
```

Karşılaştırma:

```text
S0 direct MLP row head
S1 row-wise token decoder
```

Beklenen:

```text
S1 daha smooth ve tutarlı lane üretmeli.
```

Ama dikkat: S1 overall F1’de çok büyük fark getirmeyebilir. Onun katkısı daha çok sequence-based framework’e geçiş ve row consistency olabilir.

---

# 13. Ablation 7 — Token CE etkisi

Soru:

```text
Token CE gerçekten gerekli mi, yoksa geometry loss yeterli mi?
```

Karşılaştırma:

```text
L_point only
L_token only
L_point + L_token
```

Beklenen:

```text
L_point + L_token en iyi ve en stabil sonuç vermeli.
```

Ama `L_token only` kötü olabilir çünkü discrete bin loss gerçek pixel mesafesini tam yansıtmaz.

---

# 14. Ablation 8 — Soft expected decoding

Soru:

```text
Soft expected decoding gerekli mi?
```

Karşılaştırma training için değil, analiz için yapılır:

```text
Argmax coordinate with no geometry gradient
Soft expected coordinate with geometry gradient
```

Argmax versiyonun kötü veya daha zor train olması beklenir.

Bu ablation paper’da küçük bir tablo olabilir ya da appendix’e konabilir.

---

# 15. Ablation 9 — Zoom-in refinement

Soru:

```text
S4 gerçekten değer mi?
```

Karşılaştırma:

```text
S3-B1
S4 one-step zoom-in
S4 without stage1 hidden
S4 with stage1 hidden
```

Tablo:

```text
Variant | F1 | Curve F1 | FPS | Improvement over S3
```

Beklenen:

```text
S4 zor sahnelerde küçük katkı verebilir.
FPS düşecektir.
```

Eğer katkı düşükse S4 ana model yapılmamalı.

---

# 16. Ablation 10 — Number of lane slots

Değerler:

```text
N = 10
N = 20
N = 40
N = 80
```

Başlangıç implementation N=20 idi.

Paper için bu ablation önemli. Çünkü CondLSTR tarafında query sayısı performansı etkiliyor; bizim modelde de slot sayısı candidate kapasitesini belirliyor.

Beklenen:

```text
N=10 yetersiz olabilir.
N=20 iyi MVP.
N=40 daha iyi olabilir.
N=80 pahalı ama belki en iyi.
```

Ana modelde denge:

```text
N = 20 veya 40
```

FPS/Memory’ye göre karar verilir.

---

# 17. Ablation 11 — Row resolution

Değerler:

```text
P = 36
P = 72
P = 144
```

Input yüksekliği 288 ise:

```text
P=36  → 8 px stride
P=72  → 4 px stride
P=144 → 2 px stride
```

Başlangıç:

```text
P = 72
```

Beklenen:

```text
P=36 hızlı ama detay kaybeder.
P=72 iyi denge.
P=144 daha hassas ama daha pahalı.
```

Ana model için muhtemelen P=72 yeterli.

---

# 18. Ablation 12 — X bins

Değerler:

```text
X_bins = 100
X_bins = 200
X_bins = 400
```

Input width 800 ise:

```text
100 bins → 8 px/bin
200 bins → 4 px/bin
400 bins → 2 px/bin
```

Başlangıç:

```text
X_bins = 200
```

Beklenen:

```text
100 kaba kalabilir.
200 iyi denge.
400 daha hassas ama token CE ve memory artar.
```

---

# 19. Model complexity tablosu

Paper için mutlaka şu tablo olmalı:

```text
Model | Params | FLOPs | FPS | GPU Memory | F1
```

Karşılaştırılacak varyantlar:

```text
S1
S2
S3-B1
S4
```

Bu tablo olmadan hakem şunu sorar:

```text
Bu kadar karmaşıklık performansa değiyor mu?
```

---

# 20. FPS ölçüm protokolü

FPS ölçerken net protokol olmalı:

```text
batch size = 1
input size = 288×800
GPU modeli belirtilir
warmup = 100 iter
measure = 500 iter
torch.cuda.synchronize kullanılır
postprocess dahil / hariç ayrı raporlanır
```

İki değer verilebilir:

```text
model forward FPS
end-to-end FPS
```

End-to-end içine postprocess dahil olur.

---

# 21. Memory ölçüm protokolü

GPU memory:

```text
torch.cuda.max_memory_allocated()
```

Aynı koşulda ölç:

```text
batch size = 1
batch size = 4
AMP on/off
```

Özellikle S3/S4 için memory farkı raporlanmalı.

---

# 22. Resmi CULane metric entegrasyonu

Development sırasında simple point error yeterliydi ama paper için resmi CULane metric şart.

Milestone:

```text
S0/S1 debug sırasında resmi metric şart değil.
S2 small subset sonrası resmi CULane metric entegre edilir.
S3 ana modelden önce resmi metric kesin çalışmalı.
```

Yani resmi metric’i çok sona bırakma. Çünkü output formatı yanlışsa bunu erken görmek gerekir.

---

# 23. Output format dönüşümü

Model output:

```text
x_rows: P fixed row
y_rows: P fixed row
```

CULane evaluation genelde belirli format ister.

Dönüşüm:

```text
range filtering
valid points
input coordinate → original coordinate
write lane points to txt
run official evaluation
```

Coordinate geri dönüş:

```text
x_orig = x_in / scale_x
y_orig = y_in / scale_y
```

Bu Part 2’deki meta bilgilerle yapılacak.

---

# 24. Postprocess ablation

Postprocess de sonucu etkileyebilir.

Minimum postprocess:

```text
score threshold
range filtering
min points
```

Opsiyonel:

```text
lane NMS
polynomial smoothing
moving average smoothing
```

Paper’da ana sonuç için postprocess çok ağır olmamalı. Çünkü modelin katkısını gizleyebilir.

Ablation:

```text
without smoothing
with smoothing
with lane NMS
```

Ama ilk ana sonuç:

```text
no heavy smoothing
```

olmalı.

---

# 25. Score threshold tuning

Score threshold:

```text
0.3
0.5
0.7
```

Validation set üzerinde seçilir.

Ama test set için threshold değiştirilmez.

Config:

```text
score_thresh = 0.5 default
```

Validation sonrası:

```text
best val threshold
```

raporlanır.

---

# 26. Lane NMS gerekli mi?

İdeal hedef:

```text
NMS-free
```

Ama duplicate lane çıkarsa hafif NMS gerekebilir.

Lane NMS distance:

```text
iki lane’in ortak valid rowlarındaki average |x1-x2|
```

Eğer:

```text
avg distance < threshold
```

düşük score olan atılır.

Başlangıç threshold:

```text
15–20 px
```

Ama paper’da NMS kullanıyorsan açıkça söylemek gerekir. “NMS-free” iddiası yapılmaz.

---

# 27. Failure case analizi

Paper için sadece iyi görseller değil, failure case de lazım.

Raporlanacak failure tipleri:

```text
çok yoğun lane
ağır occlusion
gece düşük kontrast
çok keskin fork
yanlış road marking
gölgeyi lane sanma
```

Her failure için kısa açıklama:

```text
coarse sampler wrong evidence region’a kaydı
existence head false positive verdi
token decoder row consistency korudu ama yanlış lane’e oturdu
```

Bu dürüstlük paper’ı güçlendirir.

---

# 28. Görsel analiz figürleri

Paper için önerilen figürler:

## Figure 1 — Pipeline

```text
image
→ lane slots
→ coarse curve
→ curve-aligned evidence
→ low-rank bridge
→ row-token decoder
→ final lanes
```

## Figure 2 — Evidence sampling visualization

Her lane için sampled points göster.

```text
coarse curve noktaları
final prediction
GT
```

## Figure 3 — Bridge effect

S2 vs S3 aynı görüntüde:

```text
S2 prediction
S3 prediction
GT
```

Özellikle zor case seç.

## Figure 4 — Failure cases

Modelin nerede hata yaptığını göster.

---

# 29. Repo milestone planı

Repo development’ı 10 milestone’a bölelim.

---

## Milestone 1 — Dataset and target builder

Çıktılar:

```text
CULane parser
resize target conversion
fixed-row interpolation
valid mask
range target
target visualizer
```

Acceptance:

```text
50 target visualization doğru
unit testler geçiyor
```

---

## Milestone 2 — S0 model forward

Çıktılar:

```text
ResNet34 backbone
FPN
positional encoding
lane queries
cross-attention
exist/range/row heads
soft coordinate decoding
```

Acceptance:

```text
one batch forward works
all shapes correct
soft decode gradient test passes
```

---

## Milestone 3 — S0 matcher/loss

Çıktılar:

```text
Hungarian matcher
exist loss
point loss
range loss
smoothness optional
matching visualization
```

Acceptance:

```text
synthetic matcher test geçiyor
one batch loss backward çalışıyor
NaN yok
```

---

## Milestone 4 — S0 overfit

Çıktılar:

```text
10 image overfit
100 image overfit
visual predictions
debug logs
```

Acceptance:

```text
10 image overfit başarılı
empty slots öğreniliyor
prediction GT’ye yaklaşıyor
```

---

## Milestone 5 — S1 token decoder

Çıktılar:

```text
x-bin target
row-token decoder
token CE
soft expected token decoding
S1 visualizer
```

Acceptance:

```text
S1 10 image overfit
L_token düşüyor
L_point düşüyor
S0’dan kötü çöküş yok
```

---

## Milestone 6 — S2 evidence sampler

Çıktılar:

```text
curve-aligned sampler
GT/pred curriculum
evidence adapter
S2 training loop
sampler visualization
```

Acceptance:

```text
GT-guided sampler doğru çalışıyor
predicted sampler’a geçince model çökmüyor
S2, S1’den kötü değil
```

---

## Milestone 7 — S3 bridge

Çıktılar:

```text
FiLM bridge
sequence low-rank bridge
bridge logging
S2 vs S3 comparison
```

Acceptance:

```text
S3 10 image overfit
bridge delta non-zero
S3 small subset S2’ye eşit veya daha iyi
```

---

## Milestone 8 — Official evaluation

Çıktılar:

```text
CULane output writer
official CULane metric integration
TuSimple converter optional
CurveLanes converter optional
```

Acceptance:

```text
validation metric çalışıyor
coordinate conversion doğru
```

---

## Milestone 9 — Full training and ablations

Çıktılar:

```text
S1/S2/S3 full train
ablation tables
FPS/memory profiling
```

Acceptance:

```text
ana ablation tablosu tamam
model complexity tablosu tamam
```

---

## Milestone 10 — Paper artifacts

Çıktılar:

```text
pipeline figure
qualitative results
failure cases
ablation tables
method description
implementation details
```

Acceptance:

```text
paper draft için yeterli sonuç ve görsel var
```

---

# 30. Deney sırası

Deneyleri şu sırada yap:

```text
1. S0 10 image
2. S0 100 image
3. S0 small subset
4. S1 10 image
5. S1 small subset
6. S2 10 image
7. S2 small subset
8. S3-A small subset
9. S3-B1 small subset
10. Official CULane validation
11. Full train S1/S2/S3-B1
12. Ablations
13. S4 optional
```

Full training’i en başta yapma. Small subset sonuçları kötü ise full train zaman kaybı olur.

---

# 31. En kritik “go / no-go” kararları

## S0 no-go

S0 10 image overfit edemiyorsa:

```text
S1/S2/S3’e geçme.
```

Sorun dataset, target, matching veya loss’tadır.

---

## S1 no-go

S1, S0’dan sürekli kötü ve overfit edemiyorsa:

```text
token decoder tasarımı yanlış olabilir.
```

Önce S1’i düzelt.

---

## S2 no-go

S2 evidence eklenince sürekli bozuyorsa:

```text
sampler yanlış
curriculum yanlış
evidence adapter fazla güçlü
coarse branch kötü
```

S3’e geçme.

---

## S3 no-go

S3 bridge fayda vermiyorsa:

```text
ana model S2 olabilir.
```

Bridge’i paper contribution yapmak riskli olur.

---

## S4 no-go

S4 sadece küçük katkı veriyor ama çok yavaşsa:

```text
S4’ü optional large variant yap.
Ana model S3-B1 kalır.
```

---

# 32. Paper’da “implementation details” kısmı nasıl yazılmalı?

Bu projede implementation details çok önemli. Orada mutlaka şunlar verilmeli:

```text
input resolution
number of lane slots
number of fixed rows
x bins
backbone
FPN output stride
row decoder depth
sampler curriculum
bridge rank
loss weights
optimizer
learning rate
batch size
epochs
score threshold
min points
whether NMS used
```

CondLSTR örneğinde gördüğümüz problem tekrar yaşanmasın. Paper’da bu detaylar açık verilmeli.

---

# 33. Final implementation checklist

Kodlamaya başlamadan önce son checklist:

```text
Dataset target builder spec tamam mı?
Coordinate systems net mi?
S0 tensor shapes net mi?
Soft expected decoding net mi?
Matcher cost net mi?
Loss weights başlangıç değerleri net mi?
Training/debug configs ayrı mı?
Visualizer planı var mı?
Official metric ne zaman entegre edilecek belli mi?
Ablation sırası belli mi?
```

Bunlar evet olmadan kodlamaya başlamak riskli.

---

# 34. En önemli dürüst değerlendirme

Bu planın tam hali artık implementation için önceki versiyondan çok daha iyi.

Ama yine de şunu unutmamak lazım:

```text
S0-S1-S2-S3 sırasına sadık kalmazsan,
model debug edilemez hale gelir.
```

Bence bu projenin başarı ihtimali şu sıraya bağlı:

```text
%40 target builder + matching doğru mu?
%25 S0/S1 training stabil mi?
%20 S2 sampler gerçekten evidence sağlıyor mu?
%10 bridge katkı veriyor mu?
%5 zoom-in refinement
```

Yani en kritik yer hâlâ novelty kısmı değil:

```text
target builder
matching
loss
visual debug
```

Bu dördü sağlam olursa modelin üst parçalarını denemek mantıklı hale gelir.

---

# 35. Part 10 özeti

Bu partta şunları kilitledik:

```text
Ana model adayı: S3-B1
S4 optional
İlk benchmark: CULane
Sonra TuSimple / CurveLanes
Ana ablationlar:
  S0 → S1 → S2 → S3 → S4
  evidence
  sampler curriculum
  bridge type
  rank
  token loss
  slot count
  row resolution
  x bins
Metrikler:
  F1, precision, recall, FPS, memory, subset F1
Repo milestones:
  dataset → S0 → loss → overfit → S1 → S2 → S3 → official eval → ablation → paper artifacts
```

---

# DynLaneSeq-EG Implementation Specification Document — Part 11

## Final Doküman Formatına Dönüştürme: Scope, Versioning, Sabit Kararlar ve Repo Yapısı

Bu parttan itibaren önceki bütün parçaları artık dağınık plan gibi değil, **resmi implementation specification document** gibi yazacağım. Yani bunu doğrudan kendi proje notuna, README taslağına veya development dokümanına çevirebilirsin.

---

# 1. Project Scope

## 1.1 Project name

```text
DynLaneSeq-EG
Evidence-Grounded Dynamic Lane Sequence Model
```

## 1.2 Main objective

Bu projenin amacı, 2D lane detection için **lane-specific visual evidence** çıkaran ve bu evidence üzerinden **row-wise lane token sequence** üreten bir model geliştirmektir.

Modelin temel fikri:

```text
Global image feature’dan doğrudan lane üretme.
Önce her lane slotu için görsel kanıt çıkar.
Sonra bu kanıtı row-wise token sequence’e dönüştür.
```

Daha sade:

```text
image
→ lane slots
→ lane-specific evidence
→ row-wise lane tokens
→ lane points
```

---

# 2. Non-Goals

Bu proje ilk implementation aşamasında şunları hedeflemez:

```text
3D lane detection
OpenLane category/class prediction
topology prediction
multi-camera fusion
BEV lane detection
autoregressive full language generation
diffusion-style iterative generation
real-time embedded deployment
```

İlk hedef sadece:

```text
CULane üzerinde çalışan, debug edilebilir, 2D lane detection modeli
```

olacak.

---

# 3. Main Research Hypothesis

Bu modelin ana hipotezi şudur:

```text
Lane sequence generation, global image feature’dan kör şekilde yapılırsa zayıf kalır.
Her lane için instance-aligned visual evidence çıkarılırsa,
row-wise token decoder daha doğru ve daha tutarlı lane üretebilir.
```

Yani paper novelty şu üç bileşene dayanır:

```text
1. Evidence-grounded lane sequence decoding
2. Curve-aligned visual evidence sampling
3. Lane-conditioned low-rank evidence bridge
```

Optional ek katkı:

```text
4. One-step zoom-in refinement
```

Ama zoom-in refinement ana MVP değildir.

---

# 4. Model Family

Model tek seferde full haliyle yazılmayacak. Beş ana sürüm olacak.

```text
S0 — Geometry-only sanity model
S1 — Row-wise token decoder
S2 — Curve-aligned evidence sampler
S3 — Low-rank evidence bridge
S4 — Optional zoom-in refinement
```

---

## 4.1 S0 — Geometry-only sanity model

Amaç:

```text
Dataset, target builder, matching ve loss sistemi doğru mu?
```

Akış:

```text
Image
→ ResNet-34 + FPN
→ lane slot queries
→ cross-attention
→ existence head
→ row-wise x head
→ range head
→ lane points
```

S0’da yok:

```text
token decoder
curve-aligned sampler
low-rank bridge
zoom-in refinement
evidence consistency loss
visibility head
topology head
```

S0 geçmeden S1’e geçilmez.

---

## 4.2 S1 — Row-wise token decoder

Amaç:

```text
Direkt MLP row head yerine row-wise token decoder kullanmak.
```

Akış:

```text
Image
→ backbone/FPN
→ lane slots
→ Q1
→ row embeddings
→ row-token decoder
→ x-bin logits
→ soft expected x coordinate
```

S1’de vocabulary sadece şudur:

```text
X_0, X_1, ..., X_199
```

Yani S1’de henüz:

```text
<EMPTY>
<EXISTS>
<END>
<ANGLE>
<CURVE>
<VIS>
```

yoktur.

Existence ayrı head ile, range ayrı head ile çözülür.

---

## 4.3 S2 — Curve-aligned evidence sampler

Amaç:

```text
Decoder’ın sadece Q1 vektöründen değil,
lane eğrisi boyunca sample edilmiş visual evidence’tan prediction yapması.
```

Akış:

```text
Image
→ backbone/FPN
→ coarse lane prediction
→ curve-aligned sampler
→ E_seq
→ evidence adapter
→ row-token decoder
→ final x rows
```

Cold-start çözümü:

```text
Başta GT-guided sampling
Sonra GT/pred mixed sampling
En son predicted sampling
```

S2, modelin gerçek **evidence-grounded** hale geldiği ilk sürümdür.

---

## 4.4 S3 — Low-rank evidence bridge

Amaç:

```text
Curve-aligned evidence sequence’i lane query’ye göre dinamik olarak refine etmek.
```

Ana önerilen bridge:

```text
sequence-level low-rank bridge
```

Akış:

```text
E_seq
+ Q1
→ query-conditioned U, S, V
→ channel reduction
→ 1D depthwise filtering over rows
→ channel expansion
→ residual refined evidence
→ row decoder
```

Full dynamic kernel üretmek yoktur.

Yasaklanan yaklaşım:

```text
B × N × C_out × C_in × k × k dynamic kernel materialization
```

Ana paper modeli için önerilen sürüm:

```text
DynLaneSeq-EG-B = S3-B1
```

---

## 4.5 S4 — Optional zoom-in refinement

Amaç:

```text
İlk prediction sonrası evidence’ı ikinci kez daha odaklı sample etmek.
```

Akış:

```text
coarse prediction
→ stage1 evidence
→ stage1 prediction
→ stage2 evidence from stage1 prediction
→ stage2 final prediction
```

S4 ana MVP değildir.

Sadece şu durumda denenir:

```text
S0 başarılı
S1 başarılı
S2 başarılı
S3 başarılı
```

Ana modelin üstüne optional large variant olabilir:

```text
DynLaneSeq-EG-Z = S4
```

---

# 5. Global Fixed Constants

İlk implementation boyunca bazı değerler sabit tutulacak. Bunlar proje boyunca karışıklığı azaltmak için erken kilitlenir.

```text
Dataset: CULane
Input height: 288
Input width: 800
Backbone: ResNet-34
FPN output stride: 4
FPN output shape: 72 × 200
FPN output channels: 128
Transformer dimension: 256
Number of lane slots: 20
Number of fixed rows: 72
X bins: 200
Bin width: 4 px
Exist classes: 2
Exist class 0: lane
Exist class 1: no-lane
Ignore index for token CE: -100
```

---

# 6. Coordinate Systems

Bu projede coordinate hatası en büyük risklerden biridir. Bu yüzden dört coordinate sistemi net ayrılır.

---

## 6.1 Original image coordinate

Dataset’ten gelen ham görüntü boyutu:

```text
W_orig × H_orig
```

Annotation noktaları başlangıçta bu coordinate sistemindedir.

Örnek:

```text
(x_orig, y_orig)
```

---

## 6.2 Model input coordinate

Model input boyutu:

```text
W_in = 800
H_in = 288
```

Annotation dönüşümü:

```text
x_in = x_orig * W_in / W_orig
y_in = y_orig * H_in / H_orig
```

Bütün target builder işlemleri bu sistemde yapılır.

---

## 6.3 Fixed-row coordinate

Model lane’i sabit y satırlarında temsil eder.

```text
P = 72
row_stride = 4
y_rows = [0, 4, 8, ..., 284]
```

Her lane için:

```text
x_rows: P
valid_mask: P
```

Invalid row:

```text
x_rows[p] = -1
valid_mask[p] = 0
```

Valid row:

```text
x_rows[p] = x coordinate
valid_mask[p] = 1
```

---

## 6.4 Grid sample coordinate

`grid_sample` için coordinate aralığı:

```text
[-1, 1]
```

Dönüşüm:

```text
x_grid = 2 * x_in / (W_in - 1) - 1
y_grid = 2 * y_in / (H_in - 1) - 1
```

Karar:

```text
align_corners = True
padding_mode = border
mode = bilinear
```

Bu karar tüm sampler modüllerinde sabit kalır.

---

# 7. Dataset Specification

İlk dataset:

```text
CULane
```

Dataset class output:

```text
sample = {
    "image": Tensor[3, 288, 800],
    "targets": {
        "x_rows": Tensor[M, 72],
        "x_bins": Tensor[M, 72],
        "valid_mask": Tensor[M, 72],
        "range_y": Tensor[M, 2],
        "exist": Tensor[M]
    },
    "meta": {
        "image_path": str,
        "anno_path": str,
        "orig_h": int,
        "orig_w": int,
        "input_h": 288,
        "input_w": 800,
        "scale_x": float,
        "scale_y": float,
        "num_gt_lanes": M
    }
}
```

Burada:

```text
M = image içindeki GT lane sayısı
```

Batch format:

```text
images: Tensor[B, 3, 288, 800]
targets: List[Dict]
metas: List[Dict]
```

Targetlar padlenmez. Her image kendi lane sayısını korur.

---

# 8. Target Builder Rules

## 8.1 Raw lane point cleaning

Her lane için:

```text
1. Geçersiz noktaları at.
2. x/y görüntü dışında ise at.
3. duplicate y değerlerini ortalama x ile birleştir.
4. y’ye göre sırala.
5. en az 2 raw point yoksa lane’i at.
```

---

## 8.2 Fixed-row interpolation

Her lane için segment-based interpolation yapılır.

Bir segment:

```text
(xa, ya) → (xb, yb)
```

Eğer:

```text
ya <= y_row <= yb
```

veya ters sıralı ise aynı mantıkla:

```text
t = (y_row - ya) / (yb - ya)
x = xa + t * (xb - xa)
```

Eğer:

```text
abs(yb - ya) < eps
```

ise segment interpolation için kullanılmaz.

---

## 8.3 Invalid x handling

Interpolation sonrası:

```text
x < 0
x >= W_in
```

ise:

```text
valid_mask[p] = 0
x_rows[p] = -1
```

Clamp yapılmaz.

---

## 8.4 Minimum valid rows

Lane target olarak tutulmak için:

```text
min_valid_rows = 5
```

olmalı.

Aksi halde lane training target’tan çıkarılır.

---

## 8.5 Range target

Valid rowlardan:

```text
first_valid = min(valid_indices)
last_valid = max(valid_indices)
```

Range:

```text
range_y = [y_rows[first_valid], y_rows[last_valid]]
```

Kodda isim olarak daha net:

```text
y_min
y_max
```

kullanılmalıdır.

---

## 8.6 X-bin target

S1 ve sonrası için:

```text
bin_width = W_in / X_bins = 800 / 200 = 4
x_bin = floor(x / bin_width)
x_bin = clamp(x_bin, 0, X_bins - 1)
```

Invalid rowlarda:

```text
x_bin = -100
```

Bu token CE için ignore index’tir.

---

# 9. Data Augmentation Policy

## 9.1 Debug stage

İlk debug aşamasında augmentation kapalıdır.

```text
resize: yes
normalize: yes
horizontal flip: no
color jitter: no
affine: no
```

Sebep:

```text
Önce target builder ve model pipeline doğru mu görmek.
```

---

## 9.2 Small/full training stage

Sonra açılabilecek augmentation:

```text
horizontal_flip_prob = 0.5
color_jitter = true
affine = false initially
```

Affine augmentation daha sonra eklenir. Çünkü lane noktalarına aynı transform matrix uygulanmalıdır ve bu bug riskini artırır.

---

# 10. Target Visualization Requirement

Model training’e başlamadan önce mutlaka şu script çalışmalıdır:

```text
python tools/visualize_targets.py --config configs/culane_s0_res34.yaml --num 50
```

Her output görselinde:

```text
resized image
raw resized GT points
fixed-row interpolated points
range y_min/y_max
invalid rows excluded
```

gösterilir.

Bu aşama geçmeden training yapılmaz.

Acceptance criteria:

```text
50 görselde lane çizgileri doğru yere oturmalı.
Interpolation lane şeklini bozmamalı.
x/y scale hatası olmamalı.
Range doğru görünmeli.
```

---

# 11. Repository Structure

Başlangıç repo yapısı:

```text
dynlaneseq_eg/
│
├── configs/
│   ├── culane_s0_res34.yaml
│   ├── culane_s1_res34.yaml
│   ├── culane_s2_res34.yaml
│   ├── culane_s3_res34.yaml
│   ├── culane_s4_res34.yaml
│   └── debug/
│       ├── culane_s0_10img_overfit.yaml
│       ├── culane_s1_10img_overfit.yaml
│       ├── culane_s2_10img_overfit.yaml
│       └── culane_s3_10img_overfit.yaml
│
├── data/
│   ├── culane_dataset.py
│   ├── lane_target_builder.py
│   ├── transforms.py
│   ├── collate.py
│   └── visualization.py
│
├── modeling/
│   ├── backbone_resnet.py
│   ├── fpn.py
│   ├── position_encoding.py
│   ├── lane_queries.py
│   ├── cross_attention_decoder.py
│   ├── heads_s0.py
│   ├── row_token_decoder.py
│   ├── dynlaneseq_s0.py
│   ├── dynlaneseq_s1.py
│   ├── dynlaneseq_s2.py
│   ├── dynlaneseq_s3.py
│   └── dynlaneseq_s4.py
│
├── modeling/evidence/
│   ├── curve_aligned_sampler.py
│   ├── evidence_adapter.py
│   ├── sampler_curriculum.py
│   ├── film_bridge.py
│   └── low_rank_bridge.py
│
├── modeling/refinement/
│   └── zoom_refinement.py
│
├── losses/
│   ├── matcher_s0.py
│   ├── loss_s0.py
│   ├── loss_s1.py
│   ├── loss_s2.py
│   ├── loss_s3.py
│   ├── loss_s4.py
│   └── smoothness.py
│
├── engine/
│   ├── train_one_epoch.py
│   ├── validate_s0.py
│   ├── validate_official.py
│   ├── checkpoint.py
│   ├── logger.py
│   └── visualizer.py
│
├── evaluation/
│   ├── culane_writer.py
│   ├── culane_metric.py
│   ├── tusimple_writer.py
│   ├── curve_lanes_writer.py
│   └── profiler.py
│
├── tools/
│   ├── train.py
│   ├── debug_one_batch.py
│   ├── debug_overfit.py
│   ├── visualize_targets.py
│   ├── visualize_predictions.py
│   └── benchmark_latency.py
│
└── README.md
```

---

# 12. Development Rule

Bu projenin ana development kuralı:

```text
Bir sürüm 10 image overfit geçmeden sonraki sürüme geçilmez.
```

Kesin sıralama:

```text
Target builder
→ S0 forward
→ S0 matcher/loss
→ S0 overfit
→ S1 token decoder
→ S1 overfit
→ S2 sampler
→ S2 overfit
→ S3 bridge
→ S3 overfit
→ official eval
→ full train
→ S4 optional
```

Bu sırayı bozmak debug maliyetini çok artırır.

---

# 13. Go / No-Go Summary

## S0 Go Criteria

```text
Target visualizer doğru.
One batch forward çalışıyor.
Loss NaN üretmiyor.
10 image overfit başarılı.
Matched slot p_lane artıyor.
Unmatched slot p_lane düşüyor.
Prediction GT’ye yaklaşıyor.
```

## S1 Go Criteria

```text
x-bin target doğru.
Token CE invalid rows’u ignore ediyor.
Soft expected decoding gradient veriyor.
10 image overfit başarılı.
L_token ve L_point düşüyor.
```

## S2 Go Criteria

```text
Sampler doğru lane civarından feature okuyor.
GT-guided sampling çalışıyor.
Predicted sampling aşamasına geçince model çökmüyor.
Final prediction coarse’dan kötü değil.
```

## S3 Go Criteria

```text
Bridge delta non-zero.
Bridge modeli bozmuyor.
S3, S2’ye eşit veya daha iyi.
Memory/FPS kabul edilebilir.
```

## S4 Go Criteria

```text
Stage2, stage1’den kötü değil.
One-step refinement zor sahnelerde katkı veriyor.
FPS düşüşü kabul edilebilir.
```

---

# 14. Part 11 Özeti

Bu partta önceki planı artık resmi doküman formatına çevirmeye başladık.

Kilitlediğimiz başlıklar:

```text
Project scope
Non-goals
Main hypothesis
Model family
Global constants
Coordinate systems
Dataset specification
Target builder rules
Augmentation policy
Target visualization requirement
Repository structure
Development rule
Go / No-Go criteria
```

Bu artık gerçek implementation dokümanının omurgası.

---

# DynLaneSeq-EG Implementation Specification Document — Part 12

## S0/S1 Architecture, Matcher, Loss ve Training Specification

Bu partta artık dokümanın model/loss/training kısmını resmi hale getiriyoruz. Burada amaç “nasıl kodlanacak?” sorusuna net cevap vermek.

---

# 15. S0 Architecture Specification

## 15.1 S0 amacı

S0, final model değildir. S0’ın görevi şudur:

```text
Dataset target builder doğru mu?
Lane slot prediction çalışıyor mu?
Hungarian matching doğru mu?
Model küçük veriyi overfit edebiliyor mu?
```

S0 başarıyla çalışmadan S1/S2/S3’e geçilmez.

---

## 15.2 S0 input

```text
images: Tensor[B, 3, 288, 800]
targets: List[Dict]
```

Target her image için:

```text
target["x_rows"]      : Tensor[M, 72]
target["x_bins"]      : Tensor[M, 72]
target["valid_mask"]  : Tensor[M, 72]
target["range_y"]     : Tensor[M, 2]
target["exist"]       : Tensor[M]
```

S0 `x_bins` kullanmaz ama target içinde bulunabilir.

---

## 15.3 Backbone

Backbone:

```text
ResNet-34
pretrained = true
```

Backbone output feature’ları:

```text
C2: B ×  64 × 72 × 200
C3: B × 128 × 36 × 100
C4: B × 256 × 18 × 50
C5: B × 512 ×  9 × 25
```

Backbone’un amacı:

```text
C2: ince lane çizgileri
C3/C4: orta seviye yol yapısı
C5: global road context
```

---

## 15.4 FPN

FPN input:

```text
C2, C3, C4, C5
```

FPN output:

```text
F: Tensor[B, 128, 72, 200]
```

Bu output modelin ana image feature map’idir.

---

## 15.5 Projection + 2D positional encoding

FPN output 256 dimension’a çıkarılır:

```text
F_proj = Conv1x1(F)

F_proj: B × 256 × 72 × 200
```

Sonra 2D positional encoding eklenir:

```text
PE: 1 × 256 × 72 × 200

F_pos = F_proj + PE
```

Flatten:

```text
F_mem = flatten(F_pos)

F_mem: B × 14400 × 256
```

Çünkü:

```text
72 × 200 = 14400
```

---

## 15.6 Lane slot queries

Learnable query table:

```text
Q_table: 20 × 256
```

Batch’e expand edilir:

```text
Q0: B × 20 × 256
```

Her query bir lane adayıdır. Query’lerin sabit “sol lane / sağ lane” anlamı yoktur. Eşleşmeyi Hungarian matcher belirler.

---

## 15.7 Cross-attention decoder

S0’da küçük DETR-benzeri decoder kullanılır.

Input:

```text
Q0:    B × 20 × 256
F_mem: B × 14400 × 256
```

Decoder layer sayısı:

```text
num_layers = 2
```

Her layer:

```text
lane slot self-attention
image cross-attention
feed-forward network
residual connection
layer normalization
```

Output:

```text
Q1: B × 20 × 256
```

---

## 15.8 Prediction heads

S0’da üç head vardır:

```text
existence head
row-wise x head
range head
```

---

### 15.8.1 Existence head

Input:

```text
Q1: B × 20 × 256
```

Output:

```text
exist_logits: B × 20 × 2
```

Class mapping:

```text
class 0 = lane
class 1 = no-lane
```

Bu mapping kesinlikle değişmeyecek.

Inference’ta:

```text
p_lane = softmax(exist_logits)[..., 0]
```

---

### 15.8.2 Row-wise x head

Input:

```text
Q1: B × 20 × 256
```

MLP output:

```text
row_x_logits: B × 20 × 72 × 200
```

Burada:

```text
72 = fixed row sayısı
200 = x-bin sayısı
```

S0’da bu head direkt MLP olabilir:

```text
Linear(256 → 256)
ReLU/GELU
Linear(256 → 72 * 200)
reshape → 72 × 200
```

---

### 15.8.3 Range head

Input:

```text
Q1: B × 20 × 256
```

Output:

```text
range_raw: B × 20 × 2
```

Sigmoid sonrası:

```text
range_norm = sigmoid(range_raw)

range_norm: B × 20 × 2
```

Loss/inference öncesi sıralama yapılır:

```text
y_min = min(range_norm[..., 0], range_norm[..., 1])
y_max = max(range_norm[..., 0], range_norm[..., 1])
```

Range normalized coordinate sistemindedir:

```text
0.0 → image top
1.0 → image bottom
```

Range head bias başlangıcı:

```text
bias = [-2, 2]
```

Böylece başlangıçta yaklaşık geniş bir y aralığı tahmin edilir.

---

## 15.9 Soft coordinate decoding

Training’de argmax kullanılmaz.

`row_x_logits`:

```text
B × 20 × 72 × 200
```

Softmax:

```text
prob = softmax(row_x_logits, dim=-1)
```

Bin merkezleri:

```text
bin_centers = [0, 1, 2, ..., 199]
```

Expected x bin:

```text
x_bin_expected = Σ prob[k] * bin_centers[k]
```

Pixel coordinate:

```text
bin_width = 800 / 200 = 4

pred_x_rows = x_bin_expected * 4
```

Output:

```text
pred_x_rows: B × 20 × 72
```

Bu yol differentiable olduğu için `L_point` doğrudan `row_x_logits` tarafına gradient gönderir.

---

## 15.10 S0 forward output

S0 forward çıktısı:

```text
outputs = {
    "exist_logits": Tensor[B, 20, 2],
    "row_x_logits": Tensor[B, 20, 72, 200],
    "pred_x_rows": Tensor[B, 20, 72],
    "range_raw": Tensor[B, 20, 2],
    "range_norm": Tensor[B, 20, 2],
    "queries": Optional[Tensor[B, 20, 256]],
    "features": Optional[Tensor[B, 256, 72, 200]]
}
```

Training için zorunlu olanlar:

```text
exist_logits
pred_x_rows
range_norm
row_x_logits
```

---

# 16. S0 Matcher Specification

## 16.1 Matcher amacı

Model 20 lane slotu üretir. Bir görüntüde `M` tane GT lane vardır.

Matcher’ın görevi:

```text
N=20 prediction slotu ile M GT lane arasında en iyi eşleşmeyi bulmak.
```

Her image için ayrı çalışır.

---

## 16.2 Matcher input

Bir image için prediction:

```text
exist_logits: N × 2
pred_x_rows: N × P
range_norm: N × 2
```

Bir image için target:

```text
gt_x_rows: M × P
gt_valid_mask: M × P
gt_range_y: M × 2
```

Sabitler:

```text
N = 20
P = 72
W_in = 800
H_in = 288
```

---

## 16.3 Cost matrix

Cost matrix:

```text
cost_matrix: N × M
```

Toplam cost:

```text
cost = λ_obj * cost_obj
     + λ_point * cost_point
     + λ_range * cost_range
```

Başlangıç ağırlıkları:

```text
λ_obj   = 2.0
λ_point = 5.0
λ_range = 1.0
```

---

## 16.4 Object cost

```text
p_lane = softmax(exist_logits, dim=-1)[:, 0]
```

Cost:

```text
cost_obj[i, j] = -log(p_lane[i] + eps)
```

`eps`:

```text
eps = 1e-6
```

Bu cost GT lane indexine bağlı değildir, prediction slotuna bağlıdır ve M boyunca broadcast edilir.

---

## 16.5 Point cost

Her prediction `i` ve GT lane `j` için:

```text
valid = gt_valid_mask[j] == 1
```

Sadece valid rowlarda:

```text
diff = abs(pred_x_rows[i, valid] - gt_x_rows[j, valid])
```

Normalize:

```text
cost_point[i, j] = mean(diff / W_in)
```

Eğer valid row yoksa:

```text
cost_point[i, j] = 1e6
```

Ama normalde target builder bunu engellemelidir.

---

## 16.6 Range cost

GT range normalize edilir:

```text
gt_range_norm = gt_range_y / H_in
```

Prediction range sort edilir:

```text
pred_y_min = min(pred_range[0], pred_range[1])
pred_y_max = max(pred_range[0], pred_range[1])
```

Cost:

```text
cost_range[i, j] =
    abs(pred_y_min[i] - gt_y_min[j])
  + abs(pred_y_max[i] - gt_y_max[j])
```

---

## 16.7 Hungarian matching

Cost matrix `no_grad` içinde hesaplanır.

```text
with torch.no_grad():
    cost_matrix = compute_cost(...)
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
```

Output:

```text
match = {
    "pred_indices": LongTensor[K],
    "gt_indices": LongTensor[K],
    "num_gt": M,
    "num_matched": K
}
```

Normal durumda:

```text
K = M
```

Çünkü:

```text
N >= M
```

Eğer `M == 0` ise:

```text
pred_indices = []
gt_indices = []
```

ve tüm slotlar no-lane kabul edilir.

---

# 17. S0 Loss Specification

## 17.1 Total loss

S0 total loss:

```text
L_total =
  w_exist * L_exist
+ w_point * L_point
+ w_range * L_range
+ w_smooth * L_smooth
```

Başlangıç:

```text
w_exist = 2.0
w_point = 5.0
w_range = 1.0
w_smooth = 0.0 debug
w_smooth = 0.05 / 0.1 later
```

İlk 10 image overfit sırasında:

```text
w_smooth = 0.0
```

---

## 17.2 Existence loss

Tüm slotlar için target önce no-lane yapılır:

```text
exist_target[:] = 1
```

Matched slotlar lane yapılır:

```text
exist_target[pred_indices] = 0
```

Loss:

```text
L_exist = CrossEntropy(exist_logits, exist_target)
```

Shape:

```text
exist_logits: B × N × 2
exist_target: B × N
```

Debug’da class weight:

```text
class_weight = [1.0, 1.0]
```

Full train’de gerekirse:

```text
class_weight = [1.0, 0.4]
```

Yani no-lane class ağırlığı düşürülebilir.

---

## 17.3 Point loss

Point loss sadece matched positive slotlarda hesaplanır.

Her matched pair için:

```text
pred = pred_x_rows[pred_idx]    # P
gt = gt_x_rows[gt_idx]          # P
mask = gt_valid_mask[gt_idx]    # P
```

Normalize:

```text
pred_norm = pred / W_in
gt_norm = gt / W_in
```

Loss:

```text
SmoothL1(pred_norm[mask], gt_norm[mask])
```

Tüm valid pointler üzerinden ortalama alınır.

Önemli kural:

```text
invalid x = -1 asla loss’a girmeyecek.
```

---

## 17.4 SmoothL1 beta

Normalized coordinate kullanıldığı için beta:

```text
beta = 0.01
```

Bu yaklaşık 8 piksel altındaki hataları quadratic bölgeye alır.

Config’te tutulur:

```text
loss:
  smooth_l1_beta: 0.01
```

---

## 17.5 Range loss

Sadece matched positive slotlarda hesaplanır.

Prediction:

```text
pred_range_norm[pred_idx]
```

Sort:

```text
pred_y_min = min(r0, r1)
pred_y_max = max(r0, r1)
```

GT:

```text
gt_range_norm = gt_range_y / H_in
```

Loss:

```text
L_range =
  SmoothL1(pred_y_min, gt_y_min)
+ SmoothL1(pred_y_max, gt_y_max)
```

Matched lane sayısına göre ortalama alınır.

---

## 17.6 Smoothness loss

Matched positive slotlarda hesaplanır.

Prediction:

```text
x_valid = pred_x_rows[pred_idx][valid_mask]
```

Second difference:

```text
d2 = x_valid[2:] - 2*x_valid[1:-1] + x_valid[:-2]
```

Normalize:

```text
d2_norm = d2 / W_in
```

Loss:

```text
L_smooth = mean(abs(d2_norm))
```

Bu loss fazla güçlü olursa curved lane’leri düzleştirebilir. O yüzden düşük ağırlıkla kullanılır.

---

## 17.7 Empty GT case

Eğer bir batch/image içinde GT lane yoksa:

```text
L_exist hesaplanır.
L_point = 0
L_range = 0
L_smooth = 0
```

Bu 0 değerleri tensor olmalıdır, Python float olmamalıdır.

---

# 18. S0 Training Protocol

## 18.1 Training aşamaları

S0 şu sırayla eğitilir:

```text
Stage 0: target visualization
Stage 1: 10 image overfit
Stage 2: 100 image overfit
Stage 3: small subset train
Stage 4: full CULane train
```

---

## 18.2 Stage 0 — target visualization

Komut:

```text
python tools/visualize_targets.py --config configs/culane_s0_res34.yaml --num 50
```

Bu geçmeden training yapılmaz.

---

## 18.3 Stage 1 — 10 image overfit

Config:

```text
dataset:
  mode: overfit
  num_samples: 10

augmentation:
  horizontal_flip_prob: 0.0
  color_jitter: false
  affine: false

training:
  batch_size: 2
  max_iters: 3000
  amp: false
  scheduler: none
  vis_interval: 100
  log_interval: 10
```

Başarı kriteri:

```text
loss düşmeli
L_point düşmeli
matched p_lane artmalı
unmatched p_lane düşmeli
prediction GT’ye görsel olarak yaklaşmalı
```

---

## 18.4 Optimizer

Optimizer:

```text
AdamW
```

Param group:

```text
backbone_decay
backbone_no_decay
model_decay
model_no_decay
```

Learning rates:

```text
base_lr = 1e-4
backbone_lr = 1e-5
```

Weight decay:

```text
weight_decay = 1e-4
bias/norm weight_decay = 0
```

Betas:

```text
betas = (0.9, 0.999)
```

---

## 18.5 Scheduler

Debug/overfit:

```text
scheduler = none
```

Small/full train:

```text
warmup + cosine
warmup_iters = 1000
min_lr_ratio = 0.1
```

---

## 18.6 AMP

Debug:

```text
amp = false
```

Full train:

```text
amp = true
```

---

## 18.7 Gradient clipping

Her training aşamasında:

```text
clip_grad_norm = 1.0
```

Loglanacak:

```text
grad_norm
```

---

## 18.8 Logging

Her log interval’da:

```text
loss_total
loss_exist
loss_point
loss_range
loss_smooth

mean_cost_obj
mean_cost_point
mean_cost_range

num_gt
num_matched
mean_p_lane_matched
mean_p_lane_unmatched

lr_backbone
lr_model
grad_norm
```

Sadece total loss loglamak yasaktır. Debug için alt losslar zorunludur.

---

## 18.9 Visualization

Her visualization interval’da kaydedilecek:

```text
GT lanes
pred lanes with range filtering
pred lanes without range filtering
matched slot ids
lane probabilities
range predictions
```

Bu iki görsel özellikle önemlidir:

```text
range_filtered.jpg
no_range_filter.jpg
```

Çünkü range head kötü ise x prediction doğru olsa bile lane kaybolabilir.

---

# 19. S1 Token Decoder Specification

## 19.1 S1 amacı

S1, S0’daki direkt MLP row head yerine row-wise token decoder getirir.

S0:

```text
Q1 → MLP → row_x_logits
```

S1:

```text
Q1 + row embeddings
→ row-token decoder
→ row_x_logits
```

Output shape aynı kalır:

```text
row_x_logits: B × 20 × 72 × 200
```

Bu sayede matcher ve postprocess büyük ölçüde değişmez.

---

## 19.2 S1 vocabulary

S1 vocabulary sadece x-binlerden oluşur:

```text
X_0, X_1, ..., X_199
```

Yani:

```text
vocab_size = 200
```

S1’de yok:

```text
<EMPTY>
<EXISTS>
<END>
<ANGLE>
<CURVE>
<VIS>
```

Existence ayrı head ile, range ayrı head ile çözülür.

---

## 19.3 Row embeddings

Learnable row embedding:

```text
E_row: 72 × 256
```

Lane slot feature:

```text
Q1: B × 20 × 256
```

Row token oluşturma:

```text
row_tokens[b, n, p] = Q1[b, n] + E_row[p]
```

Shape:

```text
row_tokens: B × 20 × 72 × 256
```

Flatten:

```text
row_tokens_flat: (B * 20) × 72 × 256
```

---

## 19.4 Row-token decoder

Decoder tipi:

```text
TransformerEncoder over rows
```

Config:

```text
num_layers = 2
d_model = 256
nhead = 8
dim_feedforward = 512
dropout = 0.1
```

Input:

```text
(B * 20) × 72 × 256
```

Output:

```text
row_hidden: (B * 20) × 72 × 256
```

Linear head:

```text
Linear(256 → 200)
```

Output:

```text
row_x_logits: B × 20 × 72 × 200
```

---

## 19.5 Soft expected token decoding

S1’de de argmax yok.

```text
prob = softmax(row_x_logits / temperature, dim=-1)
```

Default:

```text
temperature = 1.0
```

Expected x:

```text
x_bin_expected = Σ prob[k] * k
pred_x_rows = x_bin_expected * 4
```

Output:

```text
pred_x_rows: B × 20 × 72
```

---

## 19.6 S1 forward output

S1 output:

```text
outputs = {
    "exist_logits": Tensor[B, 20, 2],
    "row_x_logits": Tensor[B, 20, 72, 200],
    "pred_x_rows": Tensor[B, 20, 72],
    "range_raw": Tensor[B, 20, 2],
    "range_norm": Tensor[B, 20, 2],
    "row_hidden": Optional[Tensor[B, 20, 72, 256]],
    "queries": Optional[Tensor[B, 20, 256]]
}
```

---

# 20. S1 Loss Additions

## 20.1 S1 total loss

S1 total loss:

```text
L_total =
  w_exist  * L_exist
+ w_point  * L_point
+ w_range  * L_range
+ w_token  * L_token
+ w_smooth * L_smooth
```

Başlangıç:

```text
w_exist = 2.0
w_point = 5.0
w_range = 1.0
w_token = 0.5 debug
w_token = 1.0 stable
w_smooth = 0.0 debug
```

---

## 20.2 Matching değişmez

S1 matcher, S0 matcher ile aynıdır.

Matching cost içine token CE eklenmez.

```text
cost = object + point + range
```

Token CE sadece matched positive slotlarda loss olarak hesaplanır.

---

## 20.3 Token CE loss

Her matched pair için:

```text
pred_logits = row_x_logits[pred_idx]   # 72 × 200
target_bins = gt_x_bins[gt_idx]        # 72
```

Invalid rowlarda:

```text
target_bins[p] = -100
```

Loss:

```text
L_token = CrossEntropy(
    pred_logits,
    target_bins,
    ignore_index = -100
)
```

Sadece matched positive slotlarda hesaplanır.

Unmatched/no-lane slotlarda token CE hesaplanmaz.

---

## 20.4 Token CE + geometry loss birlikte kullanılır

S1’de `L_point` kaldırılmaz.

Sebep:

```text
L_token discrete bin classification verir.
L_point continuous geometry correction verir.
```

İkisi birlikte kullanılacak.

---

## 20.5 S1 training protocol

S1, S0 checkpoint’ten initialize edilir.

Akış:

```text
1. S0 checkpoint yükle.
2. Backbone/FPN/cross-attention/existence/range kısımlarını aktar.
3. Row-token decoder random initialize edilir.
4. 10 image overfit yapılır.
5. 100 image overfit yapılır.
6. Small subset train yapılır.
```

İlk S1 debug config:

```text
batch_size = 2
max_iters = 3000
amp = false
scheduler = none
w_token = 0.5
w_smooth = 0.0
```

Başarı kriteri:

```text
L_token düşmeli.
L_point düşmeli.
Prediction S0’dan kötü şekilde çökmemeli.
10 image overfit başarılı olmalı.
```

---

# 21. Part 12 Özeti

Bu partta resmi specification dokümanına şu bölümleri ekledik:

```text
15. S0 Architecture Specification
16. S0 Matcher Specification
17. S0 Loss Specification
18. S0 Training Protocol
19. S1 Token Decoder Specification
20. S1 Loss Additions
```

Bu noktadan sonra S0 ve S1 artık doğrudan kodlanabilecek seviyede tanımlandı.

---

# DynLaneSeq-EG Implementation Specification Document — Part 13

## S2/S3 Evidence Sampler, Curriculum ve Low-Rank Bridge Specification

Bu partta resmi dokümana modelin asıl “evidence-grounded” kısmını ekliyoruz. S0 ve S1 modeli çalıştırmak içindi; S2 ve S3 artık modelin araştırma katkısını oluşturan kısımlar.

---

# 21. S2 Curve-Aligned Evidence Sampler Specification

## 21.1 S2 amacı

S2’nin amacı, S1’deki row-token decoder’ı sadece lane slot vektörüyle çalıştırmak yerine, **lane eğrisi boyunca görüntüden alınmış görsel kanıtla** beslemektir.

S1:

```text
Q1 + row_embedding
→ row-token decoder
→ x prediction
```

S2:

```text
Q1 + row_embedding + curve-aligned visual evidence
→ row-token decoder
→ x prediction
```

Yani S2’de decoder artık “kör” tahmin yapmaz. Lane’in geçtiği yerlerden feature okur.

---

## 21.2 S2 genel akış

```text
Image
→ Backbone + FPN
→ F_proj
→ lane slot queries Q1
→ coarse row prediction
→ curve-aligned sampler
→ E_seq
→ evidence adapter
→ row-token decoder
→ final row_x_logits
→ soft expected x
```

S2’de hâlâ yok:

```text
low-rank bridge
zoom-in refinement
topology head
evidence consistency loss
full autoregressive sequence generation
```

S2 sadece **curve-aligned evidence sampling** ekler.

---

## 21.3 S2 input feature map

Sampler’ın kullanacağı feature map:

```text
F_sample = F_proj
```

Shape:

```text
F_sample: Tensor[B, 256, 72, 200]
```

Burada:

```text
256 = feature channel
72  = feature height
200 = feature width
```

Bu feature map input görüntünün 1/4 çözünürlüğündedir.

---

## 21.4 Coarse branch

S2’de sampler için önce kaba lane tahmini gerekir.

Coarse branch output:

```text
coarse_row_x_logits: B × N × P × X_bins
coarse_pred_x_rows: B × N × P
```

Başlangıç değerleri:

```text
N = 20
P = 72
X_bins = 200
```

Coarse branch basit MLP row head olabilir:

```text
Q1
→ Linear/ReLU/Linear
→ coarse_row_x_logits
→ soft expected decoding
→ coarse_pred_x_rows
```

Bu branch’in amacı final prediction üretmek değil, sampler’a ilk eğriyi vermektir.

---

## 21.5 Final branch

Final branch, sampled evidence ile çalışır.

Inputlar:

```text
Q1: B × N × 256
E_seq: B × N × P × 256
row_embedding: P × 256
```

Row token oluşturma:

```text
row_tokens = Q1.unsqueeze(2) + row_embedding + evidence_adapter(E_seq)
```

Shape:

```text
row_tokens: B × N × P × 256
```

Sonra S1’deki row-token decoder kullanılır:

```text
row_tokens
→ row-token decoder
→ final_row_x_logits
→ soft expected decoding
→ final_pred_x_rows
```

---

# 22. Curve-Aligned Sampler Specification

## 22.1 Sampler input

```text
F_sample: B × C × Hf × Wf
sample_x_rows: B × N × P
y_rows: P
```

Sabitler:

```text
C = 256
Hf = 72
Wf = 200
W_in = 800
H_in = 288
P = 72
N = 20
```

`sample_x_rows` input coordinate sistemindedir:

```text
x ∈ [0, 799]
```

`y_rows`:

```text
[0, 4, 8, ..., 284]
```

---

## 22.2 Sampler output

```text
E_seq: B × N × P × C
```

Yani:

```text
E_seq[b, n, p]
```

şunu temsil eder:

```text
b. görüntüde,
n. lane slotu için,
p. fixed row üzerinde,
sample edilen visual evidence feature.
```

---

## 22.3 Coordinate dönüşümü

`grid_sample` için input coordinate’ten normalized grid coordinate’e dönüşüm:

```text
x_grid = 2 * x_in / (W_in - 1) - 1
y_grid = 2 * y_in / (H_in - 1) - 1
```

Parametreler:

```text
mode = bilinear
padding_mode = border
align_corners = True
```

Bu ayarlar bütün sampler modüllerinde sabit kalır.

---

## 22.4 Sampling yöntemi

İlk implementation’da her row için tek nokta sample edilir:

```text
sample point = (sample_x_rows[b,n,p], y_rows[p])
```

Yani:

```text
K = 1
```

Output:

```text
E_seq: B × N × P × C
```

Daha sonra local window sampling denenebilir:

```text
offsets_px = [-8, -4, 0, +4, +8]
K = 5
```

Ama ilk S2’de local window kapalıdır.

---

## 22.5 Local window sampling opsiyonu

Opsiyonel robust sampling:

```text
for each row:
    sample x + offsets_px
```

Shape:

```text
E_local: B × N × P × K × C
```

Basit reduce:

```text
E_seq = mean(E_local, dim=K)
```

Daha sonra attention pooling denenebilir ama ilk implementation’da yoktur.

Config:

```text
local_window:
  enabled: false
  offsets_px: [-8, -4, 0, 4, 8]
```

---

## 22.6 Out-of-bound handling

Sampling öncesi:

```text
sample_x_rows = clamp(sample_x_rows, 0, W_in - 1)
```

Böylece grid_sample görüntü dışına çok fazla taşmaz.

Padding mode yine de:

```text
border
```

olarak kalır.

---

# 23. S2 Curriculum Specification

## 23.1 Problem

Training başında coarse prediction rastgeledir. Eğer sampler doğrudan bu prediction üzerinden feature okursa yanlış bölgelerden evidence alır.

Bu yüzden S2’de sampler doğrudan predicted curve ile başlatılmaz.

---

## 23.2 Curriculum aşamaları

S2’de üç aşamalı sampling curriculum kullanılır.

### Aşama 1 — GT-guided sampling

```text
sample_x = gt_x + noise
```

Noise:

```text
noise ~ Normal(0, σ)
σ = 3 px
```

Sadece GT valid rowlarda uygulanır.

Invalid GT rowlarda:

```text
sample_x = coarse_pred_x
```

---

### Aşama 2 — Mixed sampling

```text
sample_x = α * gt_x + (1 - α) * coarse_pred_x
```

Burada:

```text
α: 1.0 → 0.0
```

zamanla azalır.

---

### Aşama 3 — Predicted sampling

```text
sample_x = coarse_pred_x
```

Inference’ta GT olmadığı için final training aşamasında model kendi coarse prediction’ıyla sampling yapmayı öğrenmelidir.

---

## 23.3 Curriculum schedule

Full train için önerilen schedule:

```text
epoch 0–3:
  alpha = 1.0

epoch 4–8:
  alpha linearly decays 1.0 → 0.0

epoch 9+:
  alpha = 0.0
```

Overfit/debug için iterasyon bazlı schedule:

```text
iter 0–1000:
  alpha = 1.0

iter 1000–2000:
  alpha linearly decays 1.0 → 0.0

iter 2000+:
  alpha = 0.0
```

---

## 23.4 Matched slot handling

GT-guided sampling sadece matched positive slotlar için yapılır.

Training akışı:

```text
1. coarse_outputs üret
2. matcher çalıştır
3. matched pred_idx ↔ gt_idx bilgisi al
4. matched slotlarda GT/pred mix uygula
5. unmatched slotlarda coarse_pred_x kullan
```

Unmatched slots:

```text
sample_x = coarse_pred_x
```

---

## 23.5 Invalid GT row handling

Matched slot için:

```text
if gt_valid_mask[p] == 1:
    sample_x[p] = α * gt_x[p] + (1 - α) * coarse_x[p]
else:
    sample_x[p] = coarse_x[p]
```

Bu kural değişmeyecek.

---

## 23.6 Sample coordinate detach

İlk S2 implementation’da:

```text
sample_x_rows = sample_x_rows.detach()
```

Yani final loss, grid_sample coordinate yolu üzerinden coarse branch’e gradient göndermeyecek.

Coarse branch kendi auxiliary loss’u ile eğitilecek.

Daha sonra ablation:

```text
detach_sample_coords = false
```

denenebilir.

---

# 24. Evidence Adapter Specification

## 24.1 Adapter input/output

Input:

```text
E_seq: B × N × P × 256
```

Output:

```text
E_adapted: B × N × P × 256
```

Adapter:

```text
Linear(256 → 256)
LayerNorm
GELU
Linear(256 → 256)
```

---

## 24.2 Evidence scale

Evidence birden row decoder’ı bozmasın diye learnable scale kullanılır.

```text
row_tokens =
    Q1.unsqueeze(2)
  + row_embedding
  + γ * E_adapted
```

Başlangıç:

```text
γ = 0.1
```

`γ` learnable scalar’dır.

Loglanacak:

```text
evidence_scale_gamma
```

Eğer gamma hep 0’a yakın kalırsa model evidence kullanmıyor demektir.

---

# 25. S2 Output Specification

S2 output dictionary:

```text
outputs = {
    "coarse": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "final": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "evidence": {
        "sample_x_rows": B × N × P,
        "E_seq": B × N × P × 256,
        "evidence_scale": scalar
    }
}
```

İlk S2’de:

```text
final.exist_logits = coarse.exist_logits
final.range_norm = coarse.range_norm
```

Yani final branch sadece row x tahminini refine eder.

---

# 26. S2 Loss and Training Specification

## 26.1 Matching

S2’de matching coarse output ile yapılır:

```text
matches = matcher(outputs["coarse"], targets)
```

Final output ile ikinci kez matching yapılmaz.

Sebep:

```text
sample curve üretmek için matching sonucu gerekir.
Tek iterasyonda iki matching debug’ı zorlaştırır.
```

---

## 26.2 Final loss

Ana loss final output üzerinden hesaplanır:

```text
L_final =
  w_exist  * L_exist
+ w_point  * L_point_final
+ w_range  * L_range
+ w_token  * L_token_final
+ w_smooth * L_smooth_final
```

Başlangıç:

```text
w_exist = 2.0
w_point = 5.0
w_range = 1.0
w_token = 0.5
w_smooth = 0.0 debug
```

---

## 26.3 Coarse auxiliary loss

Coarse branch’in düzgün kalması için auxiliary point loss kullanılır.

```text
L_total = L_final + λ_coarse * L_coarse
```

Başlangıç:

```text
λ_coarse = 0.5
```

Coarse loss:

```text
L_coarse = L_point_coarse + 0.5 * L_range_coarse
```

İlk debug’da gerekirse sadece:

```text
L_coarse = L_point_coarse
```

kullanılabilir.

---

## 26.4 S2 training initialization

S2, S1 checkpoint’ten başlatılır.

Aktarılan modüller:

```text
backbone
FPN
positional encoding
lane queries
cross-attention decoder
existence head
range head
row-token decoder
```

Yeni modüller:

```text
coarse row head
curve-aligned sampler
evidence adapter
evidence scale gamma
```

---

## 26.5 S2 training stages

```text
1. S1 checkpoint load
2. S2 new modules initialize
3. target visualization tekrar kontrol edilir
4. 10 image overfit
5. 100 image overfit
6. small subset training
7. S1 vs S2 comparison
```

S2’ye geçmeden önce S1 başarılı olmalıdır.

---

## 26.6 S2 acceptance criteria

S2 başarılı sayılması için:

```text
1. GT-guided sampling görsel olarak doğru çalışmalı.
2. 10 image overfit bozulmamalı.
3. Predicted sampling aşamasına geçince model çökmemeli.
4. Final prediction coarse prediction’dan kötü olmamalı.
5. Evidence scale gamma sıfırda kalmamalı.
6. Small subset’te S2, S1’den en azından kötü olmamalı.
```

---

# 27. S3 Bridge Specification

## 27.1 S3 amacı

S3’ün amacı, S2’de çıkarılan curve-aligned evidence sequence’i lane query’ye göre dinamik biçimde refine etmektir.

S2:

```text
E_seq → adapter → row decoder
```

S3:

```text
E_seq + Q1
→ lane-conditioned bridge
→ E_refined
→ adapter
→ row decoder
```

---

## 27.2 Yasaklanan full dynamic kernel

İlk implementation’da şu yapılmaz:

```text
B × N × C_out × C_in × k × k
```

şeklinde full dynamic convolution kernel üretmek.

Sebep:

```text
memory patlaması
yavaşlık
debug zorluğu
```

Bunun yerine sequence-level low-rank bridge kullanılır.

---

# 28. S3 Bridge Variants

S3 için üç varyant tanımlanır.

```text
S3-A: FiLM bridge
S3-B1: sequence-level low-rank bridge
S3-B2: feature-map-level low-rank bridge, optional
```

Ana önerilen model:

```text
S3-B1
```

---

## 28.1 S3-A FiLM bridge

Input:

```text
E_seq: B × N × P × C
Q1: B × N × D
```

Query’den modulation üret:

```text
gamma, beta = MLP(Q1)
```

Shape:

```text
gamma: B × N × C
beta: B × N × C
```

Apply:

```text
E_film = E_seq * (1 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
```

Başlangıçta gamma/beta sıfıra yakın initialize edilir.

Amaç:

```text
S2’ye çok düşük riskli lane-conditioned modulation eklemek.
```

S3-A, S3-B1’den önce test edilebilir.

---

## 28.2 S3-B1 Sequence-level low-rank bridge

Bu ana bridge’tir.

Input:

```text
E_seq: B × N × P × C
Q1: B × N × D
```

Sabitler:

```text
C = 256
D = 256
P = 72
rank r = 16
kernel_size_1d = 3
```

Query’den dynamic parametreler üretilir:

```text
U_i: C × r
V_i: r × C
S_i: r × k
```

Her lane slot için:

```text
Z = E_i @ U_i
Z = depthwise_conv1d(Z, S_i)
ΔE = Z @ V_i
E_refined = LayerNorm(E_i + γ * ΔE)
```

Burada:

```text
E_i: P × C
Z: P × r
ΔE: P × C
```

Bridge scale:

```text
γ = learnable scalar
initial γ = 0.1
```

---

## 28.3 S3-B1 dynamic parameter generator

MLP:

```text
LayerNorm(256)
Linear(256 → 512)
GELU
Linear(512 → param_dim)
```

Param dim:

```text
U: C*r = 256*16 = 4096
V: r*C = 16*256 = 4096
S: r*k = 16*3 = 48

param_dim = 4096 + 4096 + 48 = 8240
```

Final linear küçük init edilir:

```text
weight std = 1e-3
bias = 0
```

Bu başlangıçta bridge’in feature’ı bozmasını engeller.

---

## 28.4 S3-B1 normalization

Bridge output:

```text
E_refined = LayerNorm(E_seq + γ * ΔE)
```

LayerNorm son feature dimension üzerinde uygulanır:

```text
dim = C
```

Config:

```text
use_layernorm = true
```

---

## 28.5 S3-B2 Feature-map-level low-rank bridge

Bu opsiyoneldir ve ilk implementation’da ana yol değildir.

Input:

```text
F_sample: B × C × Hf × Wf
Q1: B × N × D
```

Her slot için:

```text
F_i = F + γ * ΔF_i
E_seq_i = sampler(F_i, sample_curve_i)
```

Bu daha güçlü ama çok daha pahalıdır.

İlk implementation kararı:

```text
S3-B2 sadece S3-B1 başarılı olduktan sonra denenir.
```

---

# 29. S3 Loss and Training Specification

## 29.1 Loss değişmez

S3’te yeni bridge için özel loss eklenmez.

Loss S2 ile aynıdır:

```text
L_total =
  L_final
+ λ_coarse * L_coarse
```

Ana fark sadece final branch’in input evidence’ının bridge ile refine edilmesidir.

---

## 29.2 Matching değişmez

S3 matching yine coarse output ile yapılır:

```text
matches = matcher(outputs["coarse"], targets)
```

Token CE matching cost’a eklenmez.

---

## 29.3 S3 initialization

S3, S2 checkpoint’ten başlatılır.

Aktarılanlar:

```text
S2’deki bütün modüller
```

Yeni modüller:

```text
FiLM bridge veya low-rank bridge
bridge_scale
```

Bridge küçük initialize edilir.

---

## 29.4 S3 training stages

```text
1. S2 checkpoint load
2. Bridge ekle
3. 10 image overfit
4. 100 image overfit
5. small subset train
6. S2 vs S3 comparison
```

S3’e geçmeden önce S2 başarılı olmalıdır.

---

## 29.5 S3 acceptance criteria

S3 başarılı sayılması için:

```text
1. 10 image overfit bozulmamalı.
2. bridge output delta non-zero olmalı.
3. bridge_scale sıfırda kalmamalı.
4. S3 small subset sonucu S2’den kötü olmamalı.
5. Memory/FPS kabul edilebilir olmalı.
```

---

# 30. S3 Logging and Debugging

S3’te ek loglar:

```text
bridge_scale
mean_abs_delta_E
mean_abs_E_seq
delta_ratio = mean_abs_delta_E / mean_abs_E_seq
bridge_grad_norm
```

Beklenen:

```text
delta_ratio çok küçükse bridge etkisiz.
delta_ratio çok büyükse bridge feature’ı bozuyor.
```

Kabaca makul aralık:

```text
0.05 – 0.30
```

Bu sadece debug sezgisidir, kesin kural değildir.

---

# 31. S3 Ablation Protocol

Minimum ablation:

```text
S2: no bridge
S3-A: FiLM bridge
S3-B1: low-rank bridge r=16
S3-B1: low-rank bridge r=32
```

Opsiyonel:

```text
without LayerNorm
without 1D spatial filter
rank = 8 / 16 / 32 / 64
feature-map-level bridge
```

Bu ablationlar modelin gerçek katkısını göstermek için önemlidir.

---

# 32. Part 13 Özeti

Bu partta resmi specification dokümanına şu bölümleri ekledik:

```text
21. S2 Curve-Aligned Evidence Sampler Specification
22. Curve-Aligned Sampler Specification
23. S2 Curriculum Specification
24. Evidence Adapter Specification
25. S2 Output Specification
26. S2 Loss and Training Specification
27. S3 Bridge Specification
28. S3 Bridge Variants
29. S3 Loss and Training Specification
30. S3 Logging and Debugging
31. S3 Ablation Protocol
```

Bu noktada S0, S1, S2 ve S3 implementation açısından netleşti.

---

# DynLaneSeq-EG Implementation Specification Document — Part 14

## S4 Zoom-In Refinement, Inference/Postprocess, Evaluation, Ablation ve Final Checklist

Bu partta resmi dokümanın son büyük teknik kısmını tamamlıyoruz. Burada artık modelin optional refinement aşaması, inference akışı, postprocess kuralları, evaluation protokolü, ablation planı ve milestone listesi netleşecek.

---

# 32. S4 Zoom-In Refinement Specification

## 32.1 S4 amacı

S4, S3 üstüne eklenen optional refinement modülüdür.

S3 akışı:

```text
coarse prediction
→ curve-aligned evidence
→ low-rank bridge
→ row-token decoder
→ final prediction
```

S4 akışı:

```text
coarse prediction
→ stage1 evidence
→ stage1 prediction
→ stage2 evidence from stage1 prediction
→ stage2 prediction
```

Yani S4 şunu yapar:

```text
İlk prediction ile lane yaklaşık bulunur.
Sonra bu prediction kullanılarak evidence daha odaklı yeniden okunur.
Son prediction bu ikinci evidence üzerinden yapılır.
```

Kısa ifade:

```text
coarse look → focused look
```

---

## 32.2 S4 ana MVP değildir

S4 sadece şu şartlar sağlanırsa denenir:

```text
S0 overfit başarılı
S1 token decoder başarılı
S2 evidence sampler başarılı
S3 bridge modeli bozmadan çalışıyor
```

S4 başarısız olursa ana model çöpe gitmez. Ana model olarak S3-B1 kullanılabilir.

Önerilen model isimleri:

```text
DynLaneSeq-EG-B = S3-B1
DynLaneSeq-EG-Z = S4 optional zoom-in refinement
```

---

## 32.3 S4 output structure

S4 forward çıktısı:

```text
outputs = {
    "coarse": {
        "exist_logits": B × N × 2,
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "range_norm": B × N × 2
    },

    "stage1": {
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "row_hidden": B × N × P × D
    },

    "stage2": {
        "row_x_logits": B × N × P × X_bins,
        "pred_x_rows": B × N × P,
        "row_hidden": B × N × P × D
    },

    "evidence": {
        "sample_x_stage1": B × N × P,
        "sample_x_stage2": B × N × P,
        "E_seq_stage1": B × N × P × D,
        "E_seq_stage2": B × N × P × D
    }
}
```

İlk S4 implementation’da:

```text
exist_logits = coarse.exist_logits
range_norm = coarse.range_norm
```

Yani S4 sadece row-wise x prediction’ı refine eder. Existence ve range tekrar üretilmez.

---

## 32.4 S4 training akışı

Training sırasında:

```text
1. images → backbone/FPN/Q1
2. coarse branch → coarse prediction
3. coarse output ile Hungarian matching
4. stage1 sample curve hazırlanır
5. stage1 evidence sample edilir
6. stage1 decoder prediction üretir
7. stage2 sample curve hazırlanır
8. stage2 evidence sample edilir
9. stage2 decoder final prediction üretir
10. loss hesaplanır
```

Daha net:

```text
coarse_x
→ sample_x_stage1
→ E_stage1
→ pred_x_stage1
→ sample_x_stage2
→ E_stage2
→ pred_x_stage2
```

---

## 32.5 Stage1 sample curve

Stage1 sample curve, S2’deki curriculum ile aynıdır.

Matched slotlarda:

```text
sample_x_stage1 = α * gt_x + (1 - α) * coarse_x
```

Invalid GT rowlarda:

```text
sample_x_stage1 = coarse_x
```

Unmatched slotlarda:

```text
sample_x_stage1 = coarse_x
```

Schedule:

```text
early training:
  α = 1.0

middle training:
  α linearly decays 1.0 → 0.0

late training:
  α = 0.0
```

---

## 32.6 Stage2 sample curve

Stage2 sample curve, stage1 prediction’dan gelir.

Matched slotlarda:

```text
sample_x_stage2 = β * gt_x + (1 - β) * pred_x_stage1
```

Invalid GT rowlarda:

```text
sample_x_stage2 = pred_x_stage1
```

Unmatched slotlarda:

```text
sample_x_stage2 = pred_x_stage1
```

Stage2 için `β` daha yavaş azaltılır. Çünkü stage2’nin stage1 prediction’a tamamen güvenmesi daha risklidir.

Önerilen schedule:

```text
epoch 0–3:
  α = 1.0
  β = 1.0

epoch 4–8:
  α: 1.0 → 0.0
  β: 1.0 → 0.5

epoch 9–13:
  α = 0.0
  β: 0.5 → 0.0

epoch 14+:
  α = 0.0
  β = 0.0
```

---

## 32.7 Detach kararları

İlk S4 implementation’da stabilite için şu kararlar alınır:

```text
sample_x_stage1 = sample_x_stage1.detach()
sample_x_stage2 = sample_x_stage2.detach()
H_stage1_for_stage2 = H_stage1.detach()
```

Yani stage2 loss, grid_sample coordinate yolu üzerinden stage1’e karmaşık gradient göndermeyecek.

Stage1 zaten kendi auxiliary loss’u ile eğitilir.

Ablation olarak daha sonra:

```text
detach_stage2_sample_coords = false
detach_stage1_hidden = false
```

denenebilir.

---

## 32.8 Decoder sharing

İlk S4’te stage1 ve stage2 aynı row-token decoder’ı paylaşır.

```text
share_decoder = true
share_adapter = true
share_bridge = true
```

Sebep:

```text
daha az parametre
daha az overfit riski
daha kolay debug
```

Ablation:

```text
shared decoder
separate stage2 decoder
```

şeklinde yapılabilir.

---

## 32.9 Stage1 hidden state kullanımı

Stage2 input’una stage1 hidden state eklenebilir.

Stage2 row token:

```text
row_tokens_stage2 =
    Q1
  + row_embedding
  + E_stage2_adapted
  + δ * H_stage1_detached
```

Burada:

```text
δ = learnable scalar
initial δ = 0.1
```

Bu sayede stage2 sadece yeni evidence’a değil, stage1’in öğrendiği row-wise bağlama da bakar.

Ablation:

```text
S4 without H_stage1
S4 with H_stage1
```

mutlaka yapılmalıdır.

---

## 32.10 S4 loss

S4 loss:

```text
L_total =
  w_exist * L_exist
+ w_range * L_range
+ w_point * L_point_stage2
+ w_token * L_token_stage2
+ λ_stage1 * (w_point * L_point_stage1 + w_token * L_token_stage1)
+ λ_coarse * L_point_coarse
```

Başlangıç:

```text
w_exist = 2.0
w_range = 1.0
w_point = 5.0
w_token = 0.5

λ_stage1 = 0.5
λ_coarse = 0.25
```

S4’te ana prediction stage2’dir.

---

## 32.11 S4 acceptance criteria

S4 başarılı sayılması için:

```text
1. 10 image overfit bozulmamalı.
2. stage2 prediction, stage1’den kötü olmamalı.
3. small subset’te stage2 point error stage1’den düşük olmalı.
4. predicted-only sampling aşamasında model çökmemeli.
5. FPS düşüşü kabul edilebilir olmalı.
6. hidden scale δ tamamen sıfırda kalmamalı.
```

Eğer S4 çok küçük katkı verip ciddi yavaşlatıyorsa ana modelden çıkarılır ve sadece optional variant olarak kalır.

---

# 33. Inference and Postprocess Specification

## 33.1 S0/S1 inference

S0 ve S1 inference akışı:

```text
1. image preprocess
2. model forward
3. exist_logits → p_lane
4. row_x_logits → soft expected x
5. range_norm → y_min/y_max
6. score threshold
7. range filtering
8. min points filtering
9. final lane points
```

---

## 33.2 S2/S3 inference

S2/S3 inference akışı:

```text
1. image preprocess
2. backbone/FPN/Q1
3. coarse branch → coarse_x
4. sample_x = coarse_x
5. curve-aligned sampler → E_seq
6. bridge/adaptor/row decoder → final_x
7. existence/range filtering
8. final lane points
```

Training’de GT-guided sampling vardır ama inference’ta yoktur.

---

## 33.3 S4 inference

S4 inference:

```text
1. image preprocess
2. backbone/FPN/Q1
3. coarse branch → coarse_x
4. sample_x_stage1 = coarse_x
5. stage1 decoder → pred_x_stage1
6. sample_x_stage2 = pred_x_stage1
7. stage2 decoder → pred_x_stage2
8. existence/range filtering
9. final lane points
```

Final output:

```text
pred_x_stage2
```

---

## 33.4 Score threshold

Existence probability:

```text
p_lane = softmax(exist_logits)[..., 0]
```

Default threshold:

```text
score_thresh = 0.5
```

Validation’da denenir:

```text
0.3
0.5
0.7
```

Test set için validation’da seçilen threshold sabit kullanılır.

---

## 33.5 Range filtering

Prediction range:

```text
range_norm: [r0, r1]
```

Sort:

```text
y_min_norm = min(r0, r1)
y_max_norm = max(r0, r1)
```

Pixel coordinate:

```text
y_min = y_min_norm * H_in
y_max = y_max_norm * H_in
```

Final lane pointleri:

```text
keep row p if y_min <= y_rows[p] <= y_max
```

---

## 33.6 Min point filtering

Bir prediction lane olarak tutulmak için:

```text
num_points >= min_pred_points
```

Default:

```text
min_pred_points = 5
```

Bu değer target builder’daki `min_valid_rows = 5` ile uyumludur.

---

## 33.7 Coordinate clamp

Inference sırasında:

```text
x = clamp(x, 0, W_in - 1)
```

Training sırasında soft expected x zaten aralık içinde olduğundan clamp gerekmez.

---

## 33.8 Soft expected vs argmax inference

Default inference:

```text
soft expected x
```

Alternatif debug:

```text
argmax x-bin
```

İlk ana sonuçlarda soft expected kullanılmalıdır. Argmax daha basamaklı lane üretebilir.

---

## 33.9 Optional lane NMS

İdeal hedef NMS-free’dir. Ancak duplicate lane çıkarsa hafif lane NMS eklenebilir.

Lane distance:

```text
common valid rows üzerindeki average |x1 - x2|
```

Duplicate condition:

```text
avg_distance < threshold
```

Default threshold:

```text
15–20 px
```

Düşük score’lu lane atılır.

Önemli:

```text
NMS kullanılırsa paper’da açıkça belirtilmelidir.
NMS-free iddiası yapılmamalıdır.
```

---

## 33.10 Optional smoothing

Opsiyonel postprocess:

```text
moving average smoothing
polynomial smoothing
```

Ama ana model sonucunda ağır smoothing kullanılmamalıdır. Çünkü modelin gerçek katkısını gizleyebilir.

Ablation:

```text
without smoothing
with moving average
with polynomial smoothing
```

yapılabilir.

---

# 34. Evaluation Protocol

## 34.1 İlk benchmark

İlk benchmark:

```text
CULane
```

Sebep:

```text
2D lane detection için yaygın
normal/crowded/curve/night/shadow gibi alt senaryolar var
```

---

## 34.2 Sonraki benchmarklar

İkinci:

```text
TuSimple
```

Üçüncü:

```text
CurveLanes
```

Opsiyonel:

```text
LLAMAS
OpenLane
```

OpenLane ilk aşamada önerilmez çünkü category/class ve annotation karmaşıklığı implementation yükünü artırır.

---

## 34.3 CULane metrics

Raporlanacak:

```text
F1
Precision
Recall
```

Alt senaryolar:

```text
Normal
Crowded
Dazzle light
Shadow
No line
Arrow
Curve
Cross
Night
```

Özellikle odak:

```text
Curve
Crowded
Night
Shadow
```

Çünkü evidence-grounded yaklaşımın zor sahnelerde avantaj göstermesi beklenir.

---

## 34.4 TuSimple metrics

Raporlanacak:

```text
Accuracy
FPR
FNR
F1
```

TuSimple daha kolay highway senaryosudur. Burada büyük fark beklemek doğru olmaz. Ama modelin basit sahnelerde bozulmadığını göstermesi önemlidir.

---

## 34.5 CurveLanes metrics

Raporlanacak:

```text
F1
Precision
Recall
```

Niteliksel analiz önemli:

```text
curved lane
forked lane
dense lane
blocked lane
```

Curve-aligned sampler iddiası burada daha iyi gösterilebilir.

---

## 34.6 Development metrics

Resmi metric dışında development sırasında loglanacak:

```text
mean point error
median point error
range error
exist precision
exist recall
average predicted lanes per image
duplicate lane count
matched p_lane
unmatched p_lane
coarse point error
final point error
```

Bunlar paper metric olmayabilir ama debug için kritiktir.

---

## 34.7 Official CULane metric entegrasyonu

Development’ın başında basit point error yeterlidir. Ancak S2 küçük subset sonrası resmi metric entegre edilmelidir.

Milestone:

```text
S0/S1 debug:
  official metric şart değil

S2 small subset:
  official metric entegre edilmeli

S3 full train:
  official metric kesin çalışmalı
```

Output writer, model outputlarını resmi formatta kaydetmelidir.

---

## 34.8 Output coordinate conversion

Model output input coordinate sistemindedir:

```text
x_in, y_in
```

Official evaluation için original coordinate gerekirse:

```text
x_orig = x_in / scale_x
y_orig = y_in / scale_y
```

Bu değerler `meta` içinden alınır.

---

## 34.9 FPS protocol

FPS ölçümü net protokolle yapılır:

```text
batch size = 1
input size = 288×800
warmup = 100 iterations
measure = 500 iterations
torch.cuda.synchronize kullanılır
```

İki değer raporlanabilir:

```text
model forward FPS
end-to-end FPS
```

End-to-end FPS postprocess’i de içerir.

GPU modeli açıkça belirtilmelidir.

---

## 34.10 Memory protocol

GPU memory:

```text
torch.cuda.max_memory_allocated()
```

Ölçümler:

```text
batch size = 1
batch size = 4
AMP on/off
```

S2/S3/S4 memory farkı özellikle raporlanmalıdır.

---

# 35. Ablation Protocol

## 35.1 Main progression ablation

Ana ablation tablosu:

```text
S0
S1
S2
S3-A
S3-B1
S4
```

Tablo sütunları:

```text
Model
Overall F1
Curve F1
Night F1
Crowded F1
FPS
Params
GPU Memory
```

---

## 35.2 Evidence ablation

Soru:

```text
Curve-aligned evidence gerçekten işe yarıyor mu?
```

Karşılaştırma:

```text
S1: Q1 + row decoder
S2: Q1 + curve-aligned evidence + row decoder
```

Beklenti:

```text
S2 özellikle curve/crowded/occlusion senaryolarında daha iyi olmalı.
```

---

## 35.3 Sampler type ablation

Karşılaştırma:

```text
Q1 only
Q1 + global pooled feature
Q1 + curve-aligned E_seq
```

Amaç:

```text
Evidence’ı lane boyunca okumanın global pooling’den daha iyi olduğunu göstermek.
```

---

## 35.4 Curriculum ablation

Karşılaştırma:

```text
always predicted sampling
GT-only warmup
GT→pred mixed curriculum
```

Beklenti:

```text
always predicted sampling training başında daha kararsız olur.
GT→pred curriculum en stabil seçenektir.
```

---

## 35.5 Bridge ablation

Karşılaştırma:

```text
S2 no bridge
S3-A FiLM bridge
S3-B1 low-rank bridge r=16
S3-B1 low-rank bridge r=32
```

Beklenti:

```text
FiLM küçük katkı verebilir.
Low-rank bridge daha güçlü ama daha pahalı olabilir.
```

Eğer FiLM, low-rank kadar iyi çıkarsa bunu dürüstçe raporlamak gerekir.

---

## 35.6 Rank ablation

Rank değerleri:

```text
r = 8
r = 16
r = 32
r = 64
```

Tablo:

```text
Rank
F1
Curve F1
FPS
Memory
```

Ana model için başlangıç önerisi:

```text
r = 16
```

---

## 35.7 Token decoder ablation

Karşılaştırma:

```text
S0 direct MLP row head
S1 row-token decoder
```

Amaç:

```text
Row-token decoder row consistency sağlıyor mu?
```

---

## 35.8 Token CE ablation

Karşılaştırma:

```text
L_point only
L_token only
L_point + L_token
```

Beklenti:

```text
L_point + L_token en stabil sonuç verir.
```

---

## 35.9 Soft expected decoding ablation

Karşılaştırma:

```text
argmax coordinate training
soft expected coordinate training
```

Beklenti:

```text
soft expected coordinate training daha stabil olur çünkü geometry loss gradient verir.
```

Bu ablation ana tabloda değil, appendix veya küçük analiz olarak verilebilir.

---

## 35.10 Slot count ablation

Değerler:

```text
N = 10
N = 20
N = 40
N = 80
```

Başlangıç implementation:

```text
N = 20
```

Paper için bu ablation önemli olabilir.

---

## 35.11 Row resolution ablation

Değerler:

```text
P = 36
P = 72
P = 144
```

Karşılığı:

```text
P=36  → 8 px stride
P=72  → 4 px stride
P=144 → 2 px stride
```

Ana model:

```text
P = 72
```

---

## 35.12 X-bin ablation

Değerler:

```text
X_bins = 100
X_bins = 200
X_bins = 400
```

Karşılığı:

```text
100 bins → 8 px/bin
200 bins → 4 px/bin
400 bins → 2 px/bin
```

Ana model:

```text
X_bins = 200
```

---

## 35.13 Zoom-in ablation

Karşılaştırma:

```text
S3-B1
S4 one-step zoom-in
S4 without stage1 hidden
S4 with stage1 hidden
```

Eğer S4 az katkı verip çok yavaşlatırsa ana model yapılmaz.

---

# 36. Milestone Plan

## Milestone 1 — Dataset and target builder

Çıktılar:

```text
CULane parser
fixed-row interpolation
valid mask
range target
x-bin target
target visualizer
```

Acceptance:

```text
50 target visualization doğru
unit testler geçiyor
```

---

## Milestone 2 — S0 forward

Çıktılar:

```text
ResNet34 backbone
FPN
positional encoding
lane queries
cross-attention decoder
exist/range/row heads
soft coordinate decoding
```

Acceptance:

```text
one batch forward works
all shapes correct
soft decode gradient test passes
```

---

## Milestone 3 — S0 matcher/loss

Çıktılar:

```text
Hungarian matcher
exist loss
point loss
range loss
smoothness optional
matching visualization
```

Acceptance:

```text
synthetic matcher test geçiyor
one batch backward çalışıyor
NaN yok
```

---

## Milestone 4 — S0 overfit

Çıktılar:

```text
10 image overfit
100 image overfit
visual predictions
debug logs
```

Acceptance:

```text
prediction GT’ye yaklaşır
empty slots öğrenilir
matched p_lane artar
unmatched p_lane düşer
```

---

## Milestone 5 — S1 token decoder

Çıktılar:

```text
row-token decoder
token CE
soft expected token decoding
token distribution visualizer
```

Acceptance:

```text
10 image overfit başarılı
L_token düşüyor
L_point düşüyor
S0’dan kötü çöküş yok
```

---

## Milestone 6 — S2 evidence sampler

Çıktılar:

```text
curve-aligned sampler
GT/pred curriculum
evidence adapter
sampler visualizer
```

Acceptance:

```text
GT-guided sampler doğru
predicted sampler’a geçince model çökmüyor
S2, S1’den kötü değil
```

---

## Milestone 7 — S3 bridge

Çıktılar:

```text
FiLM bridge
sequence low-rank bridge
bridge logging
S2 vs S3 comparison
```

Acceptance:

```text
bridge delta non-zero
S3, S2’den kötü değil
memory/FPS kabul edilebilir
```

---

## Milestone 8 — Official evaluation

Çıktılar:

```text
CULane output writer
official metric integration
coordinate conversion check
```

Acceptance:

```text
validation metric çalışıyor
prediction output format doğru
```

---

## Milestone 9 — Full training and ablations

Çıktılar:

```text
S1/S2/S3 full train
main ablation table
FPS/memory profiling
```

Acceptance:

```text
ana model ve ablation sonuçları hazır
```

---

## Milestone 10 — Optional S4

Çıktılar:

```text
one-step zoom-in refinement
stage1/stage2 visualizer
S3 vs S4 comparison
```

Acceptance:

```text
stage2, stage1’den kötü değil
FPS düşüşü kabul edilebilir
```

---

## Milestone 11 — Paper artifacts

Çıktılar:

```text
pipeline figure
qualitative results
failure cases
ablation tables
implementation details
```

Acceptance:

```text
paper draft için yeterli sonuç ve görsel var
```

---

# 37. Final Implementation Checklist

Kodlamaya başlamadan önce:

```text
[ ] Coordinate systems net mi?
[ ] Target builder specification tamam mı?
[ ] Target visualizer zorunlu mu?
[ ] S0 tensor shapes net mi?
[ ] Soft expected decoding net mi?
[ ] Matcher cost net mi?
[ ] Loss weights başlangıç değerleri net mi?
[ ] Debug config ayrı mı?
[ ] 10 image overfit protokolü net mi?
[ ] Official metric ne zaman entegre edilecek belli mi?
[ ] Ablation sırası belli mi?
```

S0 bitmeden:

```text
[ ] Target visualizer doğru
[ ] One batch forward doğru
[ ] One batch backward doğru
[ ] Synthetic matcher test doğru
[ ] 10 image overfit başarılı
```

S1 bitmeden:

```text
[ ] x-bin target doğru
[ ] Token CE invalid rows’u ignore ediyor
[ ] Soft expected token gradient veriyor
[ ] L_token düşüyor
[ ] S1, S0’dan kötü çökmedi
```

S2 bitmeden:

```text
[ ] GT-guided sampler doğru lane üstünde sample alıyor
[ ] Mixed curriculum çalışıyor
[ ] Predicted sampling aşamasında model çökmüyor
[ ] Evidence scale gamma sıfırda kalmıyor
[ ] Final prediction coarse’dan kötü değil
```

S3 bitmeden:

```text
[ ] Bridge delta non-zero
[ ] Bridge scale öğreniliyor
[ ] S3, S2’den kötü değil
[ ] Memory/FPS kabul edilebilir
[ ] Bridge ablationları yapılabilir durumda
```

S4 bitmeden:

```text
[ ] Stage2 prediction stage1’den kötü değil
[ ] Stage2 predicted sampling çalışıyor
[ ] Hidden state ablation yapılmış
[ ] FPS düşüşü raporlanmış
```

---

# 38. Final karar

Bu specification’a göre ana development yolu şu olmalı:

```text
S0 → S1 → S2 → S3-B1
```

Ana paper modeli:

```text
DynLaneSeq-EG-B = S3-B1
```

Optional large/refined model:

```text
DynLaneSeq-EG-Z = S4
```

En kritik başarı koşulu:

```text
Target builder + matching + loss + visualization sağlam olacak.
```

Modelin novelty kısmı ancak bu temel sağlam olduktan sonra anlamlı hale gelir.


