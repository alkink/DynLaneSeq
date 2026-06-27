# Orthogonal Evidence Experiment Report

Tarih: 2026-06-27  
Kapsam: `culane_s0_structured_query_res34_b16_50ep` S0 baseline üstünden çıkan orthogonal evidence / verifier / grounder deney hattı.

Bu dökümanın amacı, orthogonal evidence fikrinin baştan itibaren neden ortaya çıktığını, hangi modüllerin eklendiğini, hangi deneylerin yapıldığını, sonuçların ne söylediğini ve 75k sonrası neden mevcut `OrthogonalRowTokenGrounder v1` yolunun ana aday olmaktan çıkarılması gerektiğini netleştirmektir.

Bu rapor özellikle şu soruya cevap verir:

```text
Görsel kanıtı şerit hipotezine dik yönde örnekleyip S0'u image-grounded hale getirme fikri çalıştı mı?
```

Kısa cevap:

```text
Orthogonal evidence fikri tamamen ölmedi.
Ama mevcut iki bağlama biçimi başarısız veya yetersiz kaldı:

1. Quality-only OrthogonalVerifier: post-hoc ranking düzeyinde yetersiz.
2. OrthogonalRowTokenGrounder v1: 25k'da erken geometri sinyali verdi, fakat 50k/75k'da S0'dan geri kaldı.

Asıl çıkarım:
Evidence'i serbest row-token update olarak bağlamak fazla müdahaleci.
Sıradaki mantıklı yön: kontrollü row_x logit residual + no-harm/preservation loss.
```

---

## 1. Orthogonal fikrine neden geldik?

Önceki S0 teşhisleri iki güçlü bulgu üretmişti.

Birinci bulgu, S0'daki ağır fırsatın **near-miss geometri** olmasıydı. Val üzerinde S0 175k için:

```text
all_raw R@0.5 ≈ 0.8041
all_raw R@0.7 ≈ 0.6349
```

Bu şu anlama geliyordu:

```text
Model birçok lane'i kabaca buluyor.
Ama yüksek IoU için yeterince doğru hizalayamıyor.
```

İkinci bulgu, near-miss FP'lerin büyük kısmının düşük-rank geometrik düzeltmeyle kurtarılabilmesiydi. Exact oracle sonuçları:

| Oracle | Exact rescued / 2269 | Oran |
|---|---:|---:|
| constant_shift | 1793 | 79.0% |
| constant_shift_then_row_bound_4px | 2169 | 95.6% |
| affine_deformation | 2244 | 98.9% |
| affine_deformation_plus_range | 2255 | 99.4% |
| quadratic_deformation | 2255 | 99.4% |

Bu tablo şunu ispatladı:

```text
Eğer doğru adaya doğru geometrik düzeltme uygulanabilirse büyük rezerv var.
```

Ama sonraki probe'lar şunu gösterdi:

```text
Frozen/post-hoc affine correction güvenli öğrenilemiyor.
Gate / no-harm / visual bank / ranker yolları ya abstain ediyor ya da TP öldürüyor.
```

Yani sorun oracle kapasitesinin olmaması değildi. Sorun şuydu:

```text
Model inference sırasında hangi adayı ne kadar ve hangi yönde düzelteceğini güvenli öğrenemiyor.
```

Görsel audit de bu tabloyu destekledi:

- `GLOBAL_SHIFT` ağır failure'ların büyük kısmında vardı.
- `HARD_VISIBILITY_PRIOR`, `WRONG_BOUNDARY_OR_OBJECT`, `HALLUCINATED_NO_GT` gibi modlar, S0'ın görüntü kanıtına zayıf tutunduğunu gösterdi.
- S0 çoğu zaman "bu pikselde gerçekten lane boyası var mı?" sorusundan çok "bu sahnede makul lane layout'u nasıl görünür?" sorusuna cevap veriyor gibi davrandı.

Bu bizi **hypothesis-conditioned visual evidence** fikrine getirdi.

---

## 2. Neden orthogonal sampling?

İlk görsel kanıt fikri basitçe `x ± offset` şeklinde yatay örneklemeydi. Ancak bu viraj ve eğimli lane durumlarında geometrik olarak eksikti.

Yatay sampling:

```text
Her row için:
  (x - 16, y), (x - 8, y), (x, y), (x + 8, y), ...
```

Bu düz lane'de makul olabilir. Ama lane eğimli/virajlıysa, "lane'in sağı/solu" yatay eksenle aynı değildir.

Orthogonal sampling fikri:

```text
1. Tahmin edilen lane poliline üzerinden tangent hesapla.
2. Tangent'e dik normal yönü bul.
3. Feature'ları bu normal boyunca örnekle.
```

Yani her row'da alınan profil şuna dönüşür:

```text
lane noktası etrafında:
  normal yönünde [-16, -8, -4, 0, +4, +8, +16] px
```

Bu mekanizma şu soruları cevaplamaya çalışır:

```text
- Tahmin edilen çizginin üstünde gerçekten lane evidence var mı?
- Sol/sağ yan bantlar lane profiline benziyor mu?
- Bariyer, ok, yol kenarı gibi hard negative yapılar lane'den ayrılıyor mu?
- Mevcut x_rows gerçek boya ridge'ine ne kadar yakın?
```

Bu fikir doğrudan "orthogonal cross-section profile" / "on-curve vs off-curve evidence" mantığıdır.

---

## 3. Uygulanan modüller

### 3.1 `OrthogonalEvidenceSampler`

Dosya:

```text
dynlaneseq_eg/modeling/evidence/orthogonal_sampler.py
```

Görevi:

```text
pred_x_rows [B,N,R] + feature map [B,C,H,W]
-> orthogonal evidence profiles [B,N,R,K,C]
```

Önemli teknik kararlar:

```text
- Tangent pixel-space hesaplanır.
- Endpoint row'larda forward/backward diff kullanılır.
- İç row'larda central diff kullanılır.
- Boyut 72 row olarak korunur.
- sample_x detach edilebilir.
- range_norm ile visible row mask uygulanır.
- off-image sample'lar valid mask ile sıfırlanır.
```

Varsayılan offsets:

```text
[-16, -8, -4, 0, +4, +8, +16] px
```

Kritik nokta:

```text
Sampler sadece kanıt toplar.
Geometriyi tek başına değiştirmez.
```

### 3.2 `OrthogonalQualityVerifier`

Dosya:

```text
dynlaneseq_eg/modeling/structured_queries.py
```

Config:

```text
dynlaneseq_eg/configs/culane_s0_orthogonal_verifier_qualityonly_25k.yaml
```

Amaç:

```text
Mevcut S0 geometrisini değiştirmeden, orthogonal evidence ile quality_logits düzeltmek.
```

Akış:

```text
S0 pred_x_rows
-> OrthogonalEvidenceSampler
-> center / mean_all / mean_left / mean_right profile
-> row_evidence
-> lane_evidence
-> quality_delta
-> final quality_logits = base_quality_logits + delta
```

Bu deneyde:

```text
freeze_base: true
detach_sample_x: true
zero_init: true
```

Yani S0 backbone/geometri/range/existence donuk kaldı. Sadece verifier başlığı eğitildi.

Beklenen etki:

```text
all_raw değişmemeli.
Sadece quality/ranking değişmeli.
```

Bu bilinçli bir probe idi. Amaç şuydu:

```text
Orthogonal evidence, iyi/kötü proposal ayrımını tek başına öğrenebilir mi?
```

### 3.3 `OrthogonalRowTokenGrounder v1`

Dosya:

```text
dynlaneseq_eg/modeling/structured_queries.py
```

Config:

```text
dynlaneseq_eg/configs/culane_s0_orthogonal_grounded_geometry_res34_b16_50ep.yaml
```

Amaç:

```text
Orthogonal evidence'i sadece quality başlığına değil, geometri üretim yoluna sokmak.
```

Akış:

```text
row_tokens -> draft row_x_logits -> draft pred_x_rows
draft pred_x_rows -> OrthogonalEvidenceSampler
evidence profile -> row_evidence
row_tokens + row_evidence -> updated row_tokens
updated row_tokens -> final row_x_logits / pred_x_rows / range / quality / exist
```

Bu, quality-only verifier'dan çok daha güçlü bir müdahaledir. Çünkü artık:

```text
orthogonal evidence -> row token -> row_x geometry
```

Yani modül all_raw recall ve R@0.7 metriklerini değiştirebilir.

Önemli config kararları:

```yaml
orthogonal_grounder:
  enabled: true
  freeze_base: false
  evidence_dim: 64
  hidden_dim: 256
  offsets_px: [-16, -8, -4, 0, 4, 8, 16]
  detach_sample_x: true
  range_pad: 0.05
  dropout: 0.0
  zero_init: true
```

Eğitim:

```text
batch_size = 16
max_iters = 278000
scheduler.total_iters = 278000
checkpoint_interval = 25000
ImageNet pretrained backbone = true
S0 checkpoint init = yok by default
```

Yani bu, eski S0 ile aynı 50ep training ölçeğinde scratch lane training'dir.

---

## 4. Yapılan testler ve smoke doğrulamaları

Kod eklendikten sonra uzun train/eval çalıştırılmadan şu kısa kontroller yapıldı:

```text
python -m pytest dynlaneseq_eg/tests/test_orthogonal_evidence.py -q
5 passed
```

```text
python -m pytest \
  dynlaneseq_eg/tests/test_model_shapes.py::test_structured_query_s0_forward_shapes \
  dynlaneseq_eg/tests/test_model_shapes.py::test_structured_query_debug_config_builds -q
2 passed
```

Forward smoke:

```text
grounder True
verifier False
row_x_logits (1, 64, 72, 200)
pred_x_rows (1, 64, 72)
quality_logits (1, 64)
grounder_delta 0.0
```

Optimizer grupları:

```text
backbone -> 1e-5
structured S0 -> 1e-4
orthogonal evidence grounder -> 2e-4
```

Bu smoke'ların amacı performans kanıtı değil, şu kontratı doğrulamaktı:

```text
- Tensor shape doğru.
- zero-init ile ilk iterasyon no-op.
- Grounder gerçekten aktif.
- Verifier kapalı.
- Parametre grupları beklenen LR'lara gidiyor.
```

---

## 5. Quality-only OrthogonalVerifier sonucu

Quality-only verifier, S0 175k checkpoint üstüne frozen/head-only olarak denendi.

Tanı sweep sonucu:

```text
all_raw R@0.5 = 0.8041
all_raw R@0.7 = 0.6349
```

Bu değerlerin değişmemesi beklenen bir durumdu. Çünkü geometry path donuktu.

Verifier kalite sinyali bir miktar öğrendi:

```text
quality_topk R@0.5 ≈ 0.2891
quality_topk R@0.7 ≈ 0.2497
```

Ancak deployed ranking'e eklendiğinde sinyal güvenilir olmadı:

```text
q_power yükseldikçe model_topk_nms genellikle düştü.
En güvenli seçim çoğunlukla q=0.0 civarında kaldı.
```

Hüküm:

```text
OrthogonalVerifier quality-only / post-hoc hali kurtarıcı değil.
Evidence bilgi taşıyor ama final score'a sonradan eklenince sistemi iyileştirmiyor.
```

Bu sonuç bizi şuna itti:

```text
Evidence sadece quality başlığına değil, geometry üretim yoluna girmeli.
```

---

## 6. OrthogonalRowTokenGrounder v1 sonuçları

Bu deneyde eski S0 structured ile fair erken-training kıyas yapıldı.

### 6.1 25k sonuçları

Official val, `score_thresh=0.40`, `quality_power=0.25`:

| Model 25k | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 structured | 22667 | 6436 | 10015 | 77.89 | 69.36 | 73.37 |
| Orthogonal grounded v1 | 23349 | 7418 | 9333 | 75.89 | 71.44 | 73.60 |

Net fark:

```text
TP: +682
FP: +982
FN: -682
Precision: -2.00
Recall: +2.08
F1: +0.23
```

25k diagnostic:

| Metrik | S0 25k | Orthogonal v1 25k | Fark |
|---|---:|---:|---:|
| all_raw R@0.5 | 0.7389 | 0.7666 | +2.77 |
| all_raw R@0.7 | 0.4518 | 0.5230 | +7.12 |
| model_topk_nms R@0.5, q=0.25 thr=0.4 | 0.6937 | 0.7146 | +2.09 |
| model_topk_nms R@0.7, q=0.25 thr=0.4 | 0.4313 | 0.4891 | +5.78 |

25k ilk yorum:

```text
Bu ciddi bir erken mekanik sinyaldi.
Grounder gerçekten geometry/proposal yolunu etkiliyor gibi görünüyordu.
F1 artışı küçük olsa da all_raw R@0.7 artışı yüksekti.
```

Ancak aynı zamanda negatif sinyal de vardı:

```text
FP çok arttı.
Quality hâlâ güvenilir değildi.
q=0.0 daha fazla recall verdi ama FP patlattı.
```

25k q=0.0 eval:

| Ayar | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Orthogonal 25k q=0.0 thr=0.30 | 23759 | 11038 | 8923 | 68.28 | 72.70 | 70.42 |
| Orthogonal 25k q=0.0 thr=0.40 | 23725 | 10185 | 8957 | 69.96 | 72.59 | 71.25 |

Bu gösterdi:

```text
Existence/top-k tek başına çok agresif.
Quality kötü olsa da FP frenleme işlevi görüyor.
```

### 6.2 50k sonuçları

Official val, `score_thresh=0.40`, `quality_power=0.25`:

| Model 50k | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 structured | 23268 | 4380 | 9414 | 84.16 | 71.20 | 77.14 |
| Orthogonal grounded v1 | 24127 | 5655 | 8555 | 81.01 | 73.82 | 77.25 |

Net fark:

```text
TP: +859
FP: +1275
FN: -859
Precision: -3.15
Recall: +2.62
F1: +0.11
```

50k diagnostic:

| Metrik | S0 50k | Orthogonal v1 50k | Fark |
|---|---:|---:|---:|
| all_raw R@0.5 | 0.7697 | 0.7686 | -0.11 |
| all_raw R@0.7 | 0.5909 | 0.5661 | -2.48 |
| model_topk_nms R@0.5, q=0.25 thr=0.4 | 0.7126 | 0.7384 | +2.58 |
| model_topk_nms R@0.7, q=0.25 thr=0.4 | 0.5547 | 0.5447 | -1.00 |

50k yorum:

```text
25k'daki all_raw R@0.7 avantajı kayboldu.
S0 25k->50k arasında yüksek-IoU geometriyi çok hızlı geliştirdi.
Orthogonal v1 ise aynı hızda keskinleşmedi.
```

Trend:

```text
S0 all_raw R@0.7:
0.4518 -> 0.5909  (+13.91)

Orthogonal v1 all_raw R@0.7:
0.5230 -> 0.5661  (+4.31)
```

Bu çok önemliydi. 25k'da güçlü görünen sinyal, 50k'da S0 tarafından geçildi.

50k hüküm:

```text
Orthogonal v1 daha agresif proposal dağılımı oluşturuyor.
Ama yüksek-IoU geometriyi S0 kadar iyi öğrenmiyor.
```

### 6.3 75k sonuçları

Official val, `score_thresh=0.40`, `quality_power=0.25`:

| Model 75k | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| S0 structured | 24638 | 5148 | 8044 | 82.72 | 75.39 | 78.88 |
| Orthogonal grounded v1 | 24229 | 5588 | 8453 | 81.26 | 74.14 | 77.53 |

Net fark:

```text
TP: -409
FP: +440
FN: +409
Precision: -1.46
Recall: -1.25
F1: -1.35
```

75k için sweep yapılmadı. Bu makul bir karardı çünkü official val farkı artık küçük noise değildi:

```text
S0 hem daha fazla TP üretti,
hem daha az FP üretti,
hem daha yüksek F1 aldı.
```

75k hüküm:

```text
OrthogonalRowTokenGrounder v1 ana aday olarak başarısız.
Bu run'ı 50ep'e sürdürmek stratejik olarak mantıklı değil.
```

---

## 7. Sonuçların toplu özeti

Official val F1 trend:

| Iter | S0 structured F1 | Orthogonal v1 F1 | Fark |
|---:|---:|---:|---:|
| 25k | 73.37 | 73.60 | +0.23 |
| 50k | 77.14 | 77.25 | +0.11 |
| 75k | 78.88 | 77.53 | -1.35 |

Precision/Recall davranışı:

```text
25k: Orthogonal recall artırdı ama FP de artırdı.
50k: Orthogonal recall artırdı ama precision ciddi düştü.
75k: Orthogonal hem recall hem precision olarak S0'dan kötüleşti.
```

Geometri davranışı:

```text
25k: all_raw R@0.7 ciddi artmış gibi göründü.
50k: S0 all_raw R@0.7'de orthogonal'i geçti.
75k: sweep yapılmadı ama official sonuç v1'in S0'dan net geride olduğunu gösterdi.
```

Quality davranışı:

```text
Hem S0 hem orthogonal modellerde q_power yükselince çoğunlukla recall düşüyor.
Quality başlığı iyi bir yüksek-IoU ranker gibi davranmıyor.
Orthogonal v1'de quality daha çok FP freni gibi işliyor; q=0.0 recall artırıyor ama FP patlatıyor.
```

---

## 8. Neden başarısız oldu?

Mevcut kanıta göre en olası neden:

```text
Orthogonal evidence faydalı olabilir,
ama row-token içine serbest residual update olarak sokulması fazla müdahaleci.
```

Mevcut v1 bağlantısı:

```text
draft row_x -> orthogonal evidence -> row_token update -> final row_x
```

Bu, S0'un kendi öğrenmiş olduğu structured geometry temsilini doğrudan değiştiriyor. Özellikle eğitim ilerledikçe S0'un temel row-token geometrisi zaten iyileşiyor. Orthogonal grounder ise:

```text
- Erken aşamada proposal çeşitliliği / recall kazancı sağlıyor.
- Fakat ileri aşamada S0'un yüksek-IoU geometri keskinliğini bozuyor veya sınırlıyor.
```

Bu yüzden 25k'daki pozitif sinyal kalıcı olmadı.

Başka bir ifadeyle:

```text
v1 daha çok "aggressive proposal broadener" gibi davrandı.
"High-IoU geometry sharpener" gibi davranamadı.
```

Bu fark kritik. Bizim hedefimiz sadece daha çok aday üretmek değil:

```text
R@0.7 ve final official F1 artmalı.
FP patlamamalı.
Zaten iyi olan S0 geometri korunmalı.
```

v1 bunu sağlayamadı.

---

## 9. Bu orthogonal fikri tamamen öldürür mü?

Hayır. Ama mevcut bağlantı biçimini öldürür.

Şu ayrımı net tutmak gerekir:

```text
Ölen hipotez:
  Orthogonal evidence'i row-token'a serbest update olarak verelim,
  model bunu scratch eğitimde kendisi stabilize eder.

Ölmeyen hipotez:
  Orthogonal evidence lane geometry için faydalı olabilir,
  ama kontrollü, no-harm ve düşük müdahaleli bağlanmalı.
```

Neden tamamen ölmedi?

```text
25k'da all_raw R@0.7 büyük artış gösterdi.
Bu, evidence'in geometriye temas ettiğini kanıtladı.
```

Neden mevcut hali öldü?

```text
50k'da bu avantaj kayboldu.
75k'da S0 official F1'de net öne geçti.
```

Bu nedenle "orthogonal evidence useless" demek yanlış olur. Daha doğru hüküm:

```text
Serbest token-grounding bağlantısı yanlış.
```

---

## 10. Sıradaki önerilen Plan B

75k sonrası önerilen ana yön:

```text
OrthogonalRowLogitResidual + no-harm preservation
```

Mevcut v1:

```text
evidence -> row_token update -> final row_x_logits
```

Önerilen v2:

```text
base_row_x_logits = S0'un normal çıktısı
orthogonal evidence -> delta_logits
final_row_x_logits = base_row_x_logits + gamma * delta_logits
gamma başlangıç = 0
```

Bu daha kontrollüdür çünkü:

```text
- Row token temsilini bozmaz.
- S0 ana geometri yolunu korur.
- Evidence sadece x-distribution üstüne küçük residual olarak etki eder.
- gamma=0 ile checkpoint continuation tam no-op başlar.
```

Ek olarak no-harm/preservation loss gerekir:

```text
base_x = delta öncesi S0 tahmini
final_x = delta sonrası tahmin

Eğer base_iou yüksekse:
  final_x base'i bozmasın.
```

Kabaca hedef:

```text
Zaten iyi olan S0 lane'leri koru.
Sadece düşük/near-miss lane'lerde evidence residual'a izin ver.
```

Bu, v1'in ana zayıflığını hedefler:

```text
Unconstrained row-token perturbation.
```

---

## 11. Q1 novelty açısından durum

Quality-only verifier tek başına zayıf bir katkı olurdu:

```text
HAWP/Line verification/quality calibration prior art'ına fazla yakın.
Performans kanıtı da zayıf.
```

OrthogonalRowTokenGrounder v1 daha özgün bir fikir taşıyordu:

```text
Instance-row structured lane query içine,
hypothesis-conditioned orthogonal evidence profillerini geometri üretim yoluna bağlamak.
```

Ancak v1 performans olarak 75k'da başarısız olduğu için makale omurgası olamaz.

Plan B daha savunulabilir olabilir:

```text
Measured failure-driven, no-harm constrained, orthogonal evidence residual
for structured row-wise lane geometry.
```

Ama bu da ancak şu koşulla Q1 iddiası olur:

```text
S0 175k/225k düzeyini official testte anlamlı geçmeli,
ve özellikle R@0.7 / near-miss rescued-vs-killed analizinde temiz mekanik kanıt vermeli.
```

Sadece modül fikri Q1 için yetmez.

---

## 12. Net karar

Mevcut deney zinciri için nihai karar:

```text
OrthogonalVerifier quality-only:
  FAIL as standalone/post-hoc ranking module.

OrthogonalRowTokenGrounder v1:
  FAIL as main architecture after 75k.

Orthogonal evidence concept:
  PARTIALLY ALIVE, but must be re-bound with stricter constraints.
```

Bu noktadan sonra yapılmaması gerekenler:

```text
- v1'i 50ep'e kadar körlemesine sürdürmek.
- v1 üstüne verifier/quality ekleyip karmaşayı artırmak.
- Daha fazla serbest row-token update veya free 72-row snapping açmak.
```

Yapılması gereken en mantıklı sonraki deney:

```text
S0 ana geometry path korunacak.
Orthogonal evidence sadece row_x_logits residual olarak eklenecek.
gamma=0 başlayacak.
No-harm preservation loss eklenecek.
İlk karar metrikleri F1 değil:
  - all_raw R@0.7
  - deployed R@0.7
  - rescued/killed high-IoU lanes
  - FP artışı
```

Bu raporun en kısa özeti:

```text
Orthogonal v1 bize doğru şeyi öğretti:
Evidence geometriye temas edebiliyor.

Ama yanlış şeyi de öğretti:
Bu evidence row-token içine serbestçe enjekte edilirse,
S0'un olgun geometry öğrenimini bozabiliyor.

Bir sonraki tasarım daha kontrollü olmalı:
logit residual + no-harm.
```

---

## 13. Baştan sona orthogonal plan ağacı ve mevcut durum

Bu bölüm, orthogonal hattın ilk planını ve sonradan değişen kararları ağaç şeklinde özetler. Amaç hangi parçanın yapıldığını, hangisinin yapılmadığını, hangisinin iptal edildiğini ve hangisinin hâlâ sonraki aday olduğunu net tutmaktır.

Durum etiketleri:

```text
[YAPILDI]       Kodlandı ve deney sonucu alındı.
[KISMEN]        Sadece bir alt parçası yapıldı veya eksik koşulla denendi.
[FAIL]          Denendi, mevcut haliyle ana yol olmaktan çıkarıldı.
[İPTAL]         Artık yapılmayacak; deney sonuçları veya risk nedeniyle kapatıldı.
[BEKLEMEDE]     Şimdilik yapılmayacak; ancak ileride koşullu olarak geri dönebilir.
[YAPILACAK]     Sıradaki rasyonel deney adayı.
```

### 13.1 Ana fikir ağacı

```text
Orthogonal Evidence Plan
|
|-- 0) Failure diagnosis / motivasyon
|   |
|   |-- 0.1 Near-miss + affine oracle analizi
|   |       Durum: [YAPILDI]
|   |       Sonuç:
|   |         - Near-miss FP = 2269
|   |         - affine exact rescue = 2244/2269 (%98.9)
|   |       Karar:
|   |         - Geometri fırsatı gerçek.
|   |
|   |-- 0.2 Frozen/post-hoc correction probe'ları
|   |       Durum: [YAPILDI] [FAIL]
|   |       Sonuç:
|   |         - q_ins/segmented/geometri probe'ları güvenli no-harm düzeltme üretemedi.
|   |         - Sequence CV'de gate çoğu koşuda abstain/negatif kaldı.
|   |       Karar:
|   |         - Post-hoc correction ana yol değil.
|   |
|   |-- 0.3 Görsel failure audit
|           Durum: [YAPILDI]
|           Sonuç:
|             - GLOBAL_SHIFT, HARD_VISIBILITY_PRIOR, WRONG_BOUNDARY_OR_OBJECT,
|               HALLUCINATED_NO_GT modları görüldü.
|           Karar:
|             - S0'un image-grounding'i zayıf.
|             - Hypothesis-conditioned visual evidence denenmeli.
|
|-- 1) OrthogonalEvidenceSampler altyapısı
|   |
|   |-- 1.1 Tangent/normal hesaplama
|   |       Durum: [YAPILDI]
|   |       Kod:
|   |         dynlaneseq_eg/modeling/evidence/orthogonal_sampler.py
|   |       Not:
|   |         - Pixel-space dy kullanıldı.
|   |         - Endpoint row'lar forward/backward diff ile korundu.
|   |
|   |-- 1.2 Normal boyunca feature sampling
|   |       Durum: [YAPILDI]
|   |       Yapılan:
|   |         - offsets = [-16, -8, -4, 0, 4, 8, 16]
|   |         - grid_sample ile [B,N,R,K,C] profile çıkarıldı.
|   |         - valid mask ve range mask eklendi.
|   |
|   |-- 1.3 Multi-scale orthogonal sampling
|           Durum: [YAPILMADI] [BEKLEMEDE]
|           Neden:
|             - İlk sinyali single-scale ile almak istedik.
|             - v1 75k'da başarısız olduğu için multi-scale eklemek şu an doğru değil.
|           Karar:
|             - Plan B başarılı olursa geri dönülebilir.
|
|-- 2) Aşama-1: Verifier-only / geometriye dokunmayan plan
|   |
|   |-- 2.1 OrthogonalQualityVerifier
|   |       Durum: [YAPILDI] [FAIL]
|   |       Kod:
|   |         OrthogonalQualityVerifier
|       Config:
|         culane_s0_orthogonal_verifier_qualityonly_25k.yaml
|       Deney:
|         - S0 175k checkpoint üstünde freeze_base=true
|         - geometri/range/existence donuk
|         - sadece quality_delta eğitildi
|       Sonuç:
|         - all_raw değişmedi; beklenen buydu.
|         - quality_topk bir miktar iyileşti.
|         - q_power deploy metriklerinde güvenilir kazanç vermedi.
|       Karar:
|         - Post-hoc quality-only verifier ana yol değil.
|
|   |-- 2.2 Exist/quality birlikte hypothesis-conditioned verifier
|   |       Durum: [YAPILMADI] [İPTAL - v1 hattı için]
|   |       İlk plandaki amaç:
|   |         - Orthogonal evidence ile quality_logits ve exist_logits birlikte güncellenecekti.
|   |       Neden yapılmadı:
|   |         - Quality-only bile güvenilir ranking sağlamadı.
|   |         - Exist delta'nın FP patlatma riski çok yüksekti.
|       Karar:
|         - Mevcut v1 hattında exist delta açılmayacak.
|
|   |-- 2.3 Hard-negative-aware verifier loss
|           Durum: [KISMEN] [BEKLEMEDE]
|           Yapılan:
|             - all-proposal soft quality loss config'e eklendi/kullanıldı.
|           Yapılmayan:
|             - explicit hard-negative mining yok.
|             - arrow/cross/noline özel negatif curriculum yok.
|           Karar:
|             - Plan B geometri tarafı pozitif sinyal verirse tekrar değerlendirilecek.
|
|-- 3) Aşama-2: Evidence'i geometri yoluna bağlama
|   |
|   |-- 3.1 OrthogonalRowTokenGrounder v1
|   |       Durum: [YAPILDI] [FAIL]
|   |       Kod:
|   |         OrthogonalRowTokenGrounder
|       Config:
|         culane_s0_orthogonal_grounded_geometry_res34_b16_50ep.yaml
|       Akış:
|         row_tokens
|         -> draft row_x
|         -> orthogonal evidence
|         -> row_token update
|         -> final row_x/range/quality/exist
|       Sonuç:
|         - 25k'da erken all_raw R@0.7 sinyali pozitifti.
|         - 50k'da S0 all_raw R@0.7'de öne geçti.
|         - 75k'da S0 official F1'de +1.35 puan önde çıktı.
|       Karar:
|         - Serbest row-token update tasarımı ana yol değil.
|
|   |-- 3.2 Scratch 50ep training
|   |       Durum: [KISMEN YAPILDI] [FAIL - erken kapı]
|   |       Yapılan:
|         - 25k, 50k, 75k checkpoint'leri S0 ile fair kıyaslandı.
|       Yapılmayan:
|         - 175k/225k sonuna kadar devam edilmedi.
|       Neden:
|         - 75k'da S0 hem TP hem FP hem F1 açısından net üstün.
|         - Bu noktada 50ep'e körlemesine devam etmek kaynak israfı.
|       Karar:
|         - v1 training durdurulmalı.
|
|   |-- 3.3 Grounder + Verifier birlikte
|           Durum: [YAPILMADI] [İPTAL - v1 hattı için]
|           İlk plandaki olası amaç:
|             - Grounder geometry'yi iyileştirsin, verifier FP'yi azaltsın.
|           Neden yapılmadı:
|             - Grounder v1 geometry/high-IoU tarafında 50k/75k'da S0'u geçemedi.
|             - Başarısız geometri yoluna verifier eklemek confounding yaratır.
|           Karar:
|             - v1 üstüne verifier eklenmeyecek.
|
|-- 4) Aşama-3: Point snapping / bounded geometry correction
|   |
|   |-- 4.1 Serbest 72-row point snapping
|   |       Durum: [YAPILMADI] [İPTAL]
|   |       Neden:
|         - Eski row-wise refiner deneyleri yön doğruluğu/no-harm tarafında zayıftı.
|         - 72 bağımsız delta jitter ve TP öldürme riski taşır.
|       Karar:
|         - Serbest per-row snapping açılmayacak.
|
|   |-- 4.2 Bounded softargmax snapping
|   |       Durum: [YAPILMADI] [BEKLEMEDE]
|   |       İlk amaç:
|         - Orthogonal profile üstünde en iyi offset'i softargmax ile bulmak.
|       Neden beklemede:
|         - v1 serbest token update bile S0'u bozdu.
|         - Snapping daha agresif bir geometri müdahalesi olur.
|       Karar:
|         - Doğrudan yapılmayacak.
|         - Ancak Plan B residual + no-harm pozitifse düşük-rank/bounded biçimde geri gelebilir.
|
|   |-- 4.3 Affine/low-rank correction as residual
|           Durum: [YAPILMADI] [YAPILACAK - Plan B başarılı olursa]
|           Not:
|             - Oracle affine potansiyeli yüksek.
|             - Ama learned no-harm correction hâlâ çözülmedi.
|           Karar:
|             - Önce logit residual + no-harm.
|             - Sonra gerekirse low-rank affine residual.
|
|-- 5) Scene/context/hard negative gate
|   |
|   |-- 5.1 Global scene guillotine
|   |       Durum: [YAPILMADI] [İPTAL]
|   |       Neden:
|         - Cross/no-lane için cazip ama TP öldürme riski çok yüksek.
|         - Kavşakta bazı lane'ler hâlâ GT olabilir.
|       Karar:
|         - Global tüm-lane suppressor yapılmayacak.
|
|   |-- 5.2 Hypothesis-conditioned validity gate
|           Durum: [YAPILMADI] [BEKLEMEDE]
|           Amaç:
|             - Karar global değil, her lane hipotezi için verilecek.
|           Karar:
|             - Plan B geometriyi kurtarırsa quality/existence kalibrasyon aşamasında geri gelebilir.
|
|-- 6) Multi-scale / richer visual context
|   |
|   |-- 6.1 Multi-scale orthogonal profiles
|   |       Durum: [YAPILMADI] [BEKLEMEDE]
|   |       Neden:
|         - v1 temel bağlantı başarısızken multi-scale eklemek problemi gizler.
|       Karar:
|         - Kontrollü residual tasarım pozitif olmadan yapılmayacak.
|
|   |-- 6.2 Global instance evidence token
|           Durum: [YAPILMADI] [BEKLEMEDE]
|           Amaç:
|             - Lane-level context ile row evidence'i birleştirmek.
|           Karar:
|             - Şu an ana darboğaz serbest geometri müdahalesi.
|             - Bu yüzden önce residual/no-harm çözülmeli.
|
|-- 7) Yeni Plan B: Kontrollü orthogonal residual
    |
    |-- 7.1 OrthogonalRowLogitResidual
    |       Durum: [YAPILACAK]
    |       Akış:
    |         base_row_x_logits = S0 ana yol
    |         orthogonal evidence -> delta_logits
    |         final_logits = base_row_x_logits + gamma * delta_logits
    |       Neden:
    |         - Row token temsilini bozmaz.
    |         - S0'un olgun geometry path'ini korur.
    |
    |-- 7.2 gamma=0 no-op başlangıç
    |       Durum: [YAPILACAK]
    |       Neden:
    |         - Checkpoint continuation ve scratch başlangıcı kontrollü olur.
    |         - İlk iterasyonda model birebir S0 gibi davranır.
    |
    |-- 7.3 No-harm / preservation loss
    |       Durum: [YAPILACAK]
    |       Amaç:
    |         - base_iou yüksekse final bozmasın.
    |         - Yalnız near-miss / düşük kaliteli adaylarda evidence residual etki etsin.
    |
    |-- 7.4 Debug/stage exposure
    |       Durum: [YAPILACAK]
    |       Gereken çıktı:
    |         - base pred_x_rows
    |         - final pred_x_rows
    |         - delta_logits/gamma
    |         - rescued/killed transition analizi
    |
    |-- 7.5 Verifier/quality calibration
            Durum: [BEKLEMEDE]
            Şartlı karar:
              - Plan B geometri metriklerinde S0'u geçerse yapılacak.
              - Plan B fail olursa tekrar verifier eklemek anlamsız.
```

### 13.2 İlk plan ile gerçekte yapılanlar arasındaki fark

İlk orthogonal plan kabaca şöyleydi:

```text
1. Orthogonal sampler yaz.
2. Verifier-only dene; geometriye dokunma.
3. Eğer verifier sinyal verirse quality/existence kalibrasyonu güçlendir.
4. Sonra geometry correction / snapping düşün.
5. Daha sonra multi-scale ve hard-negative context ekle.
```

Gerçekte yapılan sıra:

```text
1. Orthogonal sampler yazıldı.                                  [YAPILDI]
2. Quality-only verifier denendi.                               [YAPILDI] [FAIL]
3. Verifier başarısız olunca evidence geometri yoluna sokuldu.   [YAPILDI]
4. Row-token grounder scratch 25k/50k/75k kıyaslandı.            [YAPILDI] [FAIL]
5. Snapping / multi-scale / gate aşamalarına geçilmedi.          [YAPILMADI]
```

Burada önemli sapma şudur:

```text
Orijinal planda verifier-only pozitif sinyal verirse sonraki aşamalara geçilecekti.
Verifier-only yeterli sinyal vermedi.
Ama "belki geometri yoluna erken bağlamak gerekir" hipoteziyle RowTokenGrounder v1 denendi.
Bu deneme de 75k'da S0'dan net geri kaldı.
```

Bu yüzden orijinal ağacın birçok sonraki dalı artık doğrudan yapılmayacak:

```text
- v1 + verifier kombosu        -> İPTAL
- serbest point snapping       -> İPTAL
- global scene guillotine      -> İPTAL
- v1'i 50ep'e sürdürme         -> İPTAL
- v1 üstüne multi-scale ekleme -> BEKLEMEDE / şu an yapılmayacak
```

### 13.3 Şu an yapılacak tek mantıklı orthogonal dal

Şu noktada orthogonal hattın tek mantıklı devamı:

```text
Plan B:
  OrthogonalRowLogitResidual
  + gamma=0
  + no-harm preservation
```

Bu karar, önceki sonuçlardan doğrudan çıkar:

```text
Evidence geometriye temas edebiliyor.
Ama row-token'ı serbest bozmak S0'un iyi geometry learning'ini öldürebiliyor.
Bu yüzden evidence daha düşük müdahaleli bağlanmalı.
```

Plan B'nin başarı kapısı:

```text
25k/50k/75k fair S0 kıyasında:
  - all_raw R@0.7 S0'dan yüksek olmalı,
  - official F1 en az +0.5 puan göstermeli,
  - FP artışı kontrol altında kalmalı,
  - rescued/killed transition oranı pozitif olmalı.
```

Plan B de başarısız olursa orthogonal hattın kararı:

```text
Orthogonal evidence lane geometry için pratik olarak verimli değil.
S0 ana omurga korunup başka proposal/representation yönlerine geçilmeli.
```
