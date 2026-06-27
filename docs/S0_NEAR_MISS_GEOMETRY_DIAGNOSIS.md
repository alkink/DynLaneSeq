# S0 Near-Miss Geometry Diagnosis

Tarih: 2026-06-26  
Kapsam: `culane_s0_structured_query_res34_b16_50ep / iter_0175000` S0 modeli, sonrasında yapılan affine oracle, frozen probe, no-harm gate, visual hypothesis, IGAR ve dynamic row evidence deneyleri.

Bu dökümanın amacı tek cümleyle şudur: S0’daki ana performans fırsatının ne olduğunu, hangi deneylerin bunu ispatladığını, hangi çözüm denemelerinin neden yeterli olmadığını ve şu an gerçek eksik mekanizmanın ne olduğunu netleştirmek.

## 1. Kısa hüküm

S0’daki büyük fırsat hâlâ **near-miss geometri hatasıdır**. Bu teşhis değişmedi.

Ancak son deneylerin öğrettiği kritik nokta şudur:

```text
Oracle correction fırsatı büyük.
Learned safe/no-harm correction henüz yok.
```

Yani problem “affine oracle yanlışmış” değil. Oracle doğru. Problem, modelin inference sırasında oracle’ın bildiği şu bilgileri güvenilir şekilde öğrenememesi:

```text
- Hangi aday near-miss?
- Hangi GT’ye ait?
- Ne kadar / hangi yönde düzeltilmeli?
- Düzeltme yapılırsa TP olur mu?
- Zaten iyi olan TP bozulur mu?
- Background veya yanlış lane adayı yanlışlıkla diriltilir mi?
```

Bugünkü en dürüst teşhis:

```text
S0 birçok lane’i kabaca buluyor ama poliline geometrisini yeterince hassas oturtamıyor.
Bu hataların büyük kısmı düşük serbestlikli shift/tilt/affine sapma gibi görünüyor.
Fakat bu sapmayı güvenli biçimde düzeltecek learned action-selection mekanizması henüz çalışmadı.
```

Bu yüzden son haftalardaki sonuçların doğru yorumu:

```text
1. Near-miss oracle hâlâ güçlü.
2. Correction capacity teorik olarak yeterli.
3. Learned no-harm correction / gating / visual verification zayıf.
4. Dynamic row evidence küçük ama gerçek bir high-IoU geometri sinyali verdi.
5. Bu sinyal official F1’e güçlü yansımadı; scoring/selection hâlâ uyumsuz.
```

## 2. Önemli kavram ayrımları

### 2.1 `all_raw recall` ne demek?

`all_raw R@0.5`, postprocess öncesindeki tüm ham aday havuzunda, her GT lane için IoU `>= 0.5` olan en az bir aday var mı sorusunu ölçer.

Bu şu demek değildir:

```text
Yanlış: all_raw R düşükse model o lane için hiçbir şey üretmemiştir.
```

Doğru yorum:

```text
all_raw R@0.5 düşükse, bazı GT lane’ler için IoU>=0.5 aday yoktur.
Ama bu lane için hiç aday yok anlamına gelmez.
En iyi aday IoU=0.48 ise de all_raw R@0.5 içinde başarısız sayılır.
```

Bu ayrım kritik. Çünkü affine oracle’ın gösterdiği şey tam olarak şu:

```text
Birçok başarısız lane için aday “var”, ama yanlış konumda.
Küçük/orta bir global düzeltme ile IoU eşiğini geçebiliyor.
```

### 2.2 Near-miss ne demek?

Bu çalışma içinde near-miss çoğunlukla şu rejimi ifade ediyor:

```text
Seçilmiş aday FP sayılıyor,
ama resmi IoU 0.3–0.5 bandında.
```

Yani model tamamen saçmalamamış. Lane’e benzer bir poliline üretmiş, fakat resmi IoU eşiğinin altında kalmış.

### 2.3 Oracle correction neyi kanıtlar, neyi kanıtlamaz?

Oracle correction şunu kanıtlar:

```text
Eğer doğru adaya doğru düzeltme uygulanabilirse, büyük F1 rezervi var.
```

Oracle correction şunu kanıtlamaz:

```text
Model bu düzeltmeyi görüntüden güvenli şekilde öğrenebilir.
```

Bizim deney zincirinde asıl kopuş burada oldu. Oracle potansiyel büyük; learned safe correction zayıf.

## 3. Baseline S0 durumu

Ana baseline:

```text
config: dynlaneseq_eg/configs/culane_s0_structured_query_res34_b16_50ep.yaml
checkpoint: outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt
```

Test üzerinde doğrulanmış iyi S0 sonucu:

```text
score_thresh = 0.40
quality_score_power = 0.25
F1 = 77.33
Precision = 80.40
Recall = 74.50
Night F1 = 73.39
```

Val üzerinde aynı ayarlarla temel official postprocess:

```text
TP = 25225
FP = 4944
FN = 7457
Precision = 0.8361
Recall = 0.7718
F1 ≈ 0.8027
```

Val oracle/top-k tanısı:

| Metrik | Hit | Recall | Mean best IoU |
|---|---:|---:|---:|
| all_raw @0.5 | 26278 | 0.8041 | 0.6800 |
| all_raw @0.7 | 20751 | 0.6349 | 0.6800 |
| model_topk_nms @0.5, q=0.25, thr=0.40 | 25225 | 0.7718 | 0.6476 |
| oracle_topk @0.5 | 26278 | 0.8041 | 0.6351 |
| oracle_topk @0.7 | 20751 | 0.6349 | 0.5311 |

Buradan çıkan ilk gerçek:

```text
Top-K/NMS ana problem değil.
Oracle top-k, all_raw ile aynı seviyede.
Yani doğru aday ham havuzda varsa, top-k kapasitesi çoğunlukla yeterli.
```

İkinci gerçek:

```text
R@0.5 = 0.8041 ama R@0.7 = 0.6349.
Bu büyük düşüş geometri hassasiyetinin zayıf olduğunu gösterir.
```

Bu nedenle ana sorun sadece “lane var mı yok mu?” değil. Model lane’i çoğu zaman kabaca buluyor, ama yüksek IoU için yeterince doğru hizalayamıyor.

## 4. Duplicate / NMS / Top-K hipotezinin durumu

Eski aşamalarda duplicate oranı çok yüksek görünüyordu. Bu ciddi bir verimsizlik sinyaliydi. Ancak daha sonra official postprocess ve transition analizleri şunu gösterdi:

```text
NMS büyük duplicate ordusunu çoğunlukla temizliyor.
Kalan resmi FP’lerin ana kısmı “aynı lane’in birebir klonu” değil.
Near-miss veya background/yanlış geometri adayı.
```

Bu nedenle bugünkü teşhis:

```text
Duplicate önemli bir mimari kirlilik olabilir.
Ama mevcut S0 F1 tavanını kırmak için birincil darboğaz değil.
```

## 5. Affine oracle: ana ispat

Near-miss oracle analizinde baseline seçili adaylar:

```text
images = 9675
selected = 30169
selected_tp_official = 25225
selected_fp_official = 4944
near_miss_fp = 2269
```

Low-rank exact oracle sonuçları:

| Oracle | Exact rescued / 2269 | Oran | Yorum |
|---|---:|---:|---|
| constant_shift | 1793 | 79.0% | Sadece global yatay shift bile çok güçlü |
| constant_shift_then_row_bound_4px | 2169 | 95.6% | Shift sonrası küçük row düzeltme neredeyse oracle’a yaklaşıyor |
| affine_deformation | 2244 | 98.9% | 2 parametreli düşük-rank düzeltme near-miss’lerin neredeyse tamamını kurtarıyor |
| affine_deformation_plus_range | 2255 | 99.4% | Range ek getirisi çok küçük |
| quadratic_deformation | 2255 | 99.4% | Quadratic ek getirisi çok küçük |
| quadratic_deformation_plus_range | 2265 | 99.8% | Teorik maksimuma yakın, ama ek karmaşıklık getirisi marjinal |

Bu tablo çalışmanın en güçlü analitik kanıtlarından biridir.

Kabaca F1 etkisi:

```text
Baseline val:
TP=25225 FP=4944 FN=7457 F1≈80.27

Eğer affine oracle 2244 near-miss FP’yi TP’ye dönüştürüp hiç TP öldürmezse:
TP=27469
FP=2700
FN=5213
F1≈87.4
```

Bu yüzden “near-miss geometri fırsatı büyük mü?” sorusunun cevabı nettir:

```text
Evet, çok büyük.
```

Fakat bu oracle şunu da söylemez:

```text
Bu düzeltme learned model tarafından no-harm şekilde yapılabilir.
```

Zincirin geri kalanındaki başarısızlıklar tam olarak bu ikinci soruya aittir.

## 6. Affine decodability probe: bilgi feature’da kısmen var ama yeterli değil

Frozen checkpoint üzerinde affine hedefleri decode etmeyi denedik. Bu test, modelin mevcut representation’larından `a0/a1` affine düzeltmesini çıkarıp çıkaramayacağını ölçtü.

Kaynak:

```text
outputs/culane_s0_structured_query_res34_b16_50ep/failure_diagnostics_val_175k/affine_decodability_probe_v2.json
```

Ana setup:

```text
source near-misses = 2269
train samples = 1599
test samples = 670
oracle rescue rate on test subset ≈ 0.9925
```

Önemli probe sonuçları:

| Feature / probe | Direction acc a0 | Direction acc a1 | Predicted rescue @0.5 | Yorum |
|---|---:|---:|---:|---|
| geometry_linear | 0.616 | 0.611 | 0.387 | Sadece geometrik prior zayıf |
| geometry_mlp | 0.659 | 0.676 | 0.522 | Dataset/shape prior bir miktar bilgi taşıyor |
| q_ins_linear | 0.676 | 0.670 | 0.490 | q_ins lineer olarak sınırlı |
| q_ins_mlp | 0.724 | 0.743 | 0.577 | q_ins MLP ile kısmi bilgi var |
| pooled_mlp | 0.735 | 0.755 | 0.585 | pooled representation kısmi bilgi taşıyor |
| segmented_mlp | 0.757 | 0.756 | 0.610 | En iyi frozen probe ama oracle’dan çok uzak |

Bu deneyden çıkan doğru sonuç:

```text
Frozen representation tamamen kör değil.
Ama oracle affine action’ı güvenli uygulayacak kadar güçlü değil.
```

Özellikle dikkat:

```text
Oracle rescue ≈ 0.9925
En iyi probe rescue ≈ 0.610
```

Bu fark çok büyük. Yani “2 parametre öğrenmesi kolay olmalı” varsayımı pratikte tutmadı. Hedef düşük boyutlu olsa bile doğru action selection zor.

## 7. No-harm gate sequence CV: safe action selection başarısız

Affine düzeltmenin en kritik şartı no-harm:

```text
Sadece düzeltildiğinde TP olacak near-miss’e müdahale et.
Zaten iyi olan TP’ye dokunma.
Background/yanlış lane’i diriltme.
```

Sequence CV no-harm probe sonucu:

```text
total runs = 15
positive_net_runs = 0
negative_net_runs = 3
zero_net_runs = 12
abstain_runs = 11
official_net_tp mean = -0.33
official_f1_delta mean ≈ -0.00006
test AP mean ≈ 0.165
```

Bu sert bir sonuçtur.

Yorum:

```text
Gate modeli near-miss düzeltmesini güvenli seçemedi.
Çoğu koşuda ya abstain etti ya da küçük negatif net TP üretti.
```

Bu, affine oracle’ın yanlış olduğu anlamına gelmez. Bu, oracle action’ı learned gate ile seçmenin zor olduğunu gösterir.

## 8. Visual hypothesis bank / visual verifier denemeleri

Affine action’ı sadece token’dan değil, görsel evidence’dan seçtirmeyi denedik.

Visual bank summary:

```text
positive_net_runs = 2 / 15
negative_net_runs = 4 / 15
zero_net_runs = 9 / 15
abstain_runs = 8 / 15
official_net_tp mean = +2.27
official_f1_delta mean ≈ +0.00038
```

Visual bank + geometry summary:

```text
positive_net_runs = 3 / 15
negative_net_runs = 5 / 15
zero_net_runs = 7 / 15
abstain_runs = 6 / 15
official_net_tp mean = +3.87
official_f1_delta mean ≈ +0.00066
```

Bu sonuçlar “tamamen ölü” değil ama deploy edilebilir çözüm de değil.

Özellikle split 2022 pozitifti; diğer splitlerde negatif/abstain davranışı vardı. Bu da sequence generalization’ın zayıf olduğunu gösterir.

Doğru yorum:

```text
Visual evidence tarafında zayıf sinyal var.
Ama güvenilir no-harm action selector yok.
```

## 9. IGAR geometry-only deneyleri

IGAR fikri:

```text
S0 coarse_x üretir.
Curve-aligned FPN evidence toplanır.
2-DoF affine head a0/a1 üretir.
final_x = coarse_x + a0 + a1*y
```

Deney sonuçları:

### 9.1 Joint IGAR 2.5k

```text
final val:
TP=25196 FP=5451 FN=7486 F1=79.57

coarse val:
TP=25200 FP=5447 FN=7482 F1=79.58
```

Bu model S0’dan drift etti ve kötüleşti.

### 9.2 Head-only IGAR 2.5k

```text
final val:
TP=25214 FP=4955 FN=7468 F1=80.23

coarse val:
TP=25215 FP=4954 FN=7467 F1=80.24
```

Yorum:

```text
IGAR final, coarse’dan iyi değil.
Affine correction head resmi metric’e anlamlı katkı vermedi.
```

Bu da yine aynı yere işaret eder:

```text
2-DoF correction capacity var.
Ama learned correction/no-harm uygulaması çalışmıyor.
```

## 10. Dynamic row evidence: küçük high-IoU geometri sinyali var

Dynamic row evidence fikri:

```text
row token -> dinamik kernel
FPN row feature -> evidence feature
kernel · feature -> row-wise x-bin evidence logits
final logits = base_row_x_logits + gamma * evidence_logits
```

Önemli fark:

```text
Bu branch explicit affine correction değildir.
No-harm gate değildir.
Yeni lane proposal generator değildir.
Sadece mevcut row_x logits’e image-grounded residual evidence ekler.
```

### 10.1 Joint dynamic evidence 2.5k

Sonuç:

```text
F1 = 80.03
all_raw R@0.5 = 0.8022
all_raw R@0.7 = 0.6362
gamma ≈ 0.0063
```

Yorum:

```text
Joint training branch’i neredeyse kullanmadı.
S0 drift’i baskın.
```

### 10.2 Head-only dynamic evidence 2.5k

Sonuç:

```text
F1 = 80.28
all_raw R@0.5 = 0.8049
all_raw R@0.7 = 0.6404
gamma: 0.05 -> 0.092
```

Yorum:

```text
Branch gerçekten öğreniyor.
R@0.7’de küçük ama gerçek sinyal var.
```

### 10.3 Head-only dynamic evidence 10k

Sonuç:

```text
F1 = 80.29
TP=25231 FP=4938 FN=7451
all_raw R@0.5 = 0.8057
all_raw R@0.7 = 0.6393
gamma: 0.092 -> 0.178
```

Baseline’a göre:

```text
all_raw R@0.5: 0.8041 -> 0.8057
all_raw R@0.7: 0.6349 -> 0.6393
F1: ~80.27 -> ~80.29
```

Bu sonuç önemli ama sınırlı:

```text
Dynamic evidence high-IoU geometry’yi biraz iyileştiriyor.
Fakat official F1’e neredeyse hiç yansımıyor.
```

## 11. Dynamic head-only transition analizi

S0 baseline ve dynamic head-only 10k arasında stage transition analizi yapıldı.

Kaynak:

```text
outputs/culane_s0_dynamic_row_evidence_headonly_25k/transition_vs_s0_val_10k/transitions_iou0p5.json
outputs/culane_s0_dynamic_row_evidence_headonly_25k/transition_vs_s0_val_10k/transitions_iou0p7.json
```

### 11.1 IoU@0.5 transition

```text
S0:
selected_tp = 25225
selected_fp = 4944

Dynamic head-only:
selected_tp = 25214
selected_fp = 4955
```

Net:

```text
TP: -11
FP: +11
```

Transition summary:

```text
rescued_tp = 132
killed_by_geometry = 120
killed_by_score = 58
killed_by_nms = 5
```

En büyük geçişler:

```text
selected_tp -> selected_tp: 25082
absent_raw -> selected_tp: 110
selected_tp -> absent_raw: 80
selected_tp -> below_threshold: 58
below_threshold -> selected_tp: 15
nms_removed -> selected_tp: 7
selected_tp -> nms_removed: 5
```

Yorum:

```text
IoU@0.5 seviyesinde dynamic head güvenli değil.
Kurtardığı kadar bozuyor.
Official F1 bu yüzden artmıyor.
```

### 11.2 IoU@0.7 transition

```text
S0:
selected_tp = 20019
selected_fp = 10150

Dynamic head-only:
selected_tp = 20172
selected_fp = 9997
```

Net:

```text
TP: +153
FP: -153
```

Transition summary:

```text
rescued_tp = 390
killed_by_geometry = 272
killed_by_score = 46
killed_by_nms = 16
```

En büyük geçişler:

```text
selected_tp -> selected_tp: 19782
absent_raw -> selected_tp: 335
selected_tp -> absent_raw: 175
below_threshold -> selected_tp: 39
nms_removed -> selected_tp: 16
selected_tp -> nms_removed: 16
selected_tp -> below_threshold: 46
```

Yorum:

```text
Dynamic head high-IoU geometriyi gerçekten iyileştiriyor.
Ama IoU@0.5 official metric için no-harm değil.
```

Bu sonuç bugünkü en önemli yeni kanıttır:

```text
Görsel evidence branch tamamen boşa değil.
Ama oracle correction fırsatının sadece küçük bir kısmını kullanıyor.
```

## 12. “Model kör mü?” sorusunun doğru cevabı

İlk probe’lar nedeniyle “model kör” ifadesi kullanıldı. Bugünkü daha doğru cevap:

```text
Model tamamen kör değil.
FPN/representation içinde bazı düzeltme sinyalleri var.
Ama mevcut representation ve loss, oracle action’ı güvenli çıkaracak kadar organize değil.
```

Bunu gösteren kanıtlar:

```text
1. Frozen decodability probe en iyi MLP ile rescue ≈ 0.61 üretti.
2. Dynamic row evidence head-only, frozen base üzerinde IoU@0.7 net +153 TP verdi.
```

Yani “feature’da hiçbir bilgi yok” iddiası yanlış.

Ama şu da doğru:

```text
Feature sinyali oracle kadar güçlü değil.
Action selection güvenilir değil.
```

## 13. “Sorun lane başlangıç noktası mı?” sorusunun cevabı

Hayır, tek başına başlangıç noktası değil.

`x=400 yerine x=410` örneği öğretici bir basitleştirmeydi. Gerçek sorun daha genel:

```text
Tüm lane poliline’i düşük serbestlikli şekilde yanlış hizalanıyor.
Bu bazen global shift, bazen tilt, bazen range/visibility hatası, bazen curve altında süreklilik hatası.
```

Affine oracle’ın güçlü olması şunu gösterir:

```text
Hataların büyük kısmı 72 bağımsız row offset gerektirmiyor.
2 parametreli lane-level deformation bile near-miss’lerin neredeyse tamamını kurtarıyor.
```

Ama bu, başlangıç noktasının tek problem olduğu anlamına gelmez.

## 14. “LineIoU kapalıydı, açınca çözülür” iddiasının durumu

Bu iddia mevcut config için yanlıştır.

S0 config’te zaten:

```yaml
matcher:
  lambda_line_iou: 1.0

loss:
  w_line_iou: 2.0
  line_iou_radius: 7.5
```

Kod tarafında da matcher LineIoU/GIoU maliyeti hesaplıyor:

```text
dynlaneseq_eg/losses/matcher_s0.py
compute_line_iou_cost(...)
```

Loss tarafında da LineIoU benzeri kayıp var:

```text
dynlaneseq_eg/losses/loss_s0.py
compute_line_iou_loss(...)
```

Bu yüzden “matcher/loss IoU-aware değilmiş, bir satır açalım” çözüm değildir. Zaten açık.

Geçerli kalan eleştiri:

```text
Loss explicit olarak resmi IoU@0.5 threshold margin’i optimize etmiyor.
Near-miss örneklerini özel ağırlıklandırmıyor.
```

Bu denenebilir bir ablation’dır ama ana çözüm olduğu kanıtlanmış değildir.

## 15. Mevcut teşhisin son hali

Bugünkü en doğru problem tanımı:

```text
S0’un büyük performans rezervi near-miss geometri hatalarından geliyor.
Bu hatalar oracle affine ile neredeyse tamamen kurtarılabiliyor.
Ancak model, bu correction action’ı güvenli ve genel geçer biçimde öğrenemiyor.
```

Bunu daha mekanik yazarsak:

```text
1. Proposal havuzunda birçok lane için kabaca doğru ama IoU eşiği altında aday var.
2. Bu adaylar düşük-rank shift/tilt correction ile TP’ye dönüştürülebilir.
3. Learned modeller hangi adaya ne correction uygulanacağını güvenilir seçemiyor.
4. Görsel evidence küçük high-IoU kazanım sağlıyor ama no-harm değil.
5. Scoring/quality, geometri iyileşmesini resmi F1’e çevirmekte zayıf.
```

Bu yüzden problem tek başına şu değildir:

```text
- NMS
- Top-K
- duplicate
- LineIoU kapalı olması
- sadece backbone körlüğü
- sadece lane başlangıç noktası
- sadece daha uzun eğitim
```

Problem bunların kesişiminde değil; daha spesifik:

```text
image-grounded, threshold-aware, no-harm geometry action selection eksik.
```

## 16. Full train hakkında dürüst değerlendirme

Sıfırdan dynamic evidence ile full train teorik olarak faydalı olabilir. Çünkü backbone/FPN feature’ları bu branch’e daha uygun öğrenebilir.

Ama eldeki kanıtlarla full train “zorunlu ve kesin çözüm” değildir.

Neden?

```text
1. Frozen feature’da bile dynamic head sinyali çıktı.
   Demek ki bilgi tamamen yok değil.

2. Head-only 10k high-IoU’da net +153 TP verdi ama official F1’e yansımadı.
   Demek ki geometri sinyali scoring/no-harm ile uyumsuz kalabiliyor.

3. Full train, no-harm action selection problemini otomatik çözmeyebilir.
   Hatta joint dynamic run’da branch bastırıldı: gamma≈0.006.
```

Bu nedenle full train şu an “tek gerçek test” değil. Daha doğru sıralama:

```text
1. Ucuz/orta maliyetli ablation ile correction signal + no-harm + scoring ilişkisini çöz.
2. Eğer bu mekanizma küçük ölçekte temiz çalışırsa full train düşün.
3. Mekanizma temiz değilse full train zaman yakar.
```

## 17. Threshold-aware near-miss loss önerisinin yeri

Öneri:

```text
Matched lane IoU düşükse point/line_iou loss ağırlığını artır.
Özellikle IoU 0.3–0.65 bandında daha sert geometri baskısı ver.
```

Bu mantıksız değil. Çünkü post-hoc correction’dan farklıdır; eğitim sırasında içerden itme yapar.

Ama beklenti düşük tutulmalı:

```text
Bu ana çözüm değil.
Ucuz bir ablation.
```

Neden ana çözüm değil?

```text
LineIoU zaten aktif.
Sadece ağırlık artırmak no-harm action selection’ı çözmez.
Agresif reweight sağlam TP’leri bozabilir.
Background/yanlış eşleşmeleri daha güçlü lane’e çekebilir.
```

Eğer denenirse sıkı pass/fail kriteri:

```text
Pass:
- all_raw R@0.7 artar
- IoU@0.5 official F1 düşmez
- rescued > killed
- FP artmaz

Fail:
- R@0.7 artsa bile F1 düşerse scoring/no-harm hâlâ çözülmemiştir
- FP artarsa loss aşırı agresiftir
- R@0.5/R@0.7 oynamazsa reweight etkisizdir
```

## 18. Şu ana kadar yanlışlanan iddialar

### 18.1 “Sorun sadece NMS/duplicate”

Yanlışlandı. NMS çoğu duplicate’i temizliyor. Kalan problem near-miss/background geometri.

### 18.2 “Slot sayısı / Top-K ana sorun”

Yanlışlandı. Oracle Top-K all_raw ile aynı seviyede. K=4 çoğunlukla yeterli.

### 18.3 “LineIoU kapalıymış”

Yanlışlandı. Matcher ve loss tarafında LineIoU aktif.

### 18.4 “2 parametre affine head kolayca öğrenir”

Yanlışlandı. Oracle basit; learned action selection zor.

### 18.5 “Frozen feature’da hiçbir bilgi yok”

Yanlışlandı. Dynamic row evidence head-only ve decodability probe sinyal olduğunu gösterdi.

### 18.6 “Dynamic evidence çözümü tek başına getirir”

Yanlışlandı. High-IoU’da küçük pozitif verdi ama F1’i anlamlı artırmadı.

## 19. Şu ana kadar hâlâ güçlü kalan iddialar

### 19.1 Near-miss geometry ana fırsat

Güçlü biçimde destekleniyor. Affine oracle en güçlü kanıt.

### 19.2 Düşük-rank correction yeterli olabilir

Oracle düzeyinde destekleniyor. Affine, quadratic/range’e neredeyse eşit.

### 19.3 Learned no-harm correction eksik

Güçlü biçimde destekleniyor. Gate, visual bank, IGAR ve dynamic evidence sonuçları aynı duvara işaret ediyor.

### 19.4 Scoring/quality geometri kazanımını iyi kullanmıyor

Dynamic head-only transition destekliyor: IoU@0.7 net iyileşti, official F1 neredeyse değişmedi.

## 20. Bugünkü araştırma yönü

Bu noktadan sonra en doğru araştırma sorusu şudur:

```text
Oracle affine/low-rank correction fırsatını,
image-grounded ve no-harm şekilde,
training sırasında nasıl supervise ederiz?
```

Bu soru şu eski sorulardan daha doğru:

```text
Yanlış / eksik:
- Başka bir local refiner ekleyelim mi?
- Daha fazla slot açalım mı?
- NMS’i değiştirelim mi?
- Dynamic kernel’i full train edelim mi?
- LineIoU açalım mı?
```

Doğru alt problemler:

```text
1. Action target:
   Hangi matched/near-miss slot için hangi low-rank correction hedefi verilecek?

2. No-harm:
   Zaten TP olan slotlara correction baskısı nasıl sınırlanacak?

3. Visual grounding:
   Correction kararı token prior’dan değil, poliline çevresindeki image evidence’dan nasıl beslenecek?

4. Quality alignment:
   Corrected geometry kalitesi final score’a nasıl yansıyacak?

5. Threshold awareness:
   Resmi IoU eşiğine yakın örnekler training’de nasıl özel ele alınacak?
```

## 21. En dürüst sonuç

Bu projenin mevcut sorunu artık “ne deneyeceğimizi bilmiyoruz” seviyesinde değil. Sorun oldukça netleşti:

```text
S0’un büyük latent performans rezervi near-miss geometri hatalarında.
Oracle bu rezervin çok büyük olduğunu kanıtladı.
Fakat learned modeller bu rezervi güvenli biçimde kullanamıyor.
```

Yani şu an bilimsel olarak savunulabilir hikâye:

```text
1. Failure diagnosis:
   S0 failure’larının büyük kısmı low-rank correctable near-miss geometry.

2. Negative evidence:
   Post-hoc/frozen/no-harm/visual verifier aileleri oracle fırsatını güvenilir kullanamadı.

3. Partial positive evidence:
   Dynamic row evidence high-IoU geometriyi az da olsa iyileştirdi.

4. Open mechanism:
   Need: image-grounded, threshold-aware, no-harm low-rank geometry action learning + quality alignment.
```

Bu çözülmeden 80+ bandına çıkmak için yapılan her “ufak refiner” denemesi büyük ihtimalle aynı duvara çarpacak.

