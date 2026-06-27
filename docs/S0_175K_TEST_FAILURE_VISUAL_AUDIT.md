# S0 175k Test Failure Visual Audit

Bu not, `culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt` modelinin test split başarısızlık görselleri üzerinden yapılan manuel incelemedir.

İncelenen görseller:

- Prediction dir: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_thr0p40_q0p25`
- Overlay dir: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh`
- GT: kırmızı
- Prediction: yeşil
- Failure seçimi: per-image official CULane IoU@0.5 sonucunda `FP > 0` veya `FN > 0`
- Her kategori için seçilen örnekler: en ağır 10 failure, severity sırası `FN + FP`, sonra `FN`, sonra `FP`

Önemli metodolojik not: Bu 90 görüntü, her kategorinin en ağır başarısızlıklarından seçildiği için dataset genel dağılımını temsil etmez. Ama modelin yüksek kayıplı failure modlarını görmek için doğru örneklerdir.

## Etiket sözlüğü

| Etiket | Anlamı |
|---|---|
| `GLOBAL_SHIFT` | Lane ailesi genel olarak doğru/topolojik olarak makul, fakat GT’ye göre topluca yanal/eğimsel kaymış. Düşük-rank/affine hata gibi görünüyor. |
| `WRONG_BOUNDARY_OR_OBJECT` | Prediction yol kenarı, bariyer, park çizgisi, ok, araç kenarı, komşu lane veya yanlış fiziksel çizgiye kilitlenmiş. |
| `HARD_VISIBILITY_PRIOR` | Şerit kanıtı zayıf/gece/gölge/noline/parlama; model gerçek boya yerine genel yol şablonu basıyor. |
| `CURVE_TOPOLOGY_FAIL` | Viraj, dönüş, merge/branch veya eğri geometri doğru takip edilemiyor. |
| `HALLUCINATED_NO_GT` | GT yokken veya lane kanıtı çok zayıfken model lane layout üretiyor. Cross kategorisinde ana mod bu. |
| `RANGE_VISIBILITY_FAIL` | Şerit başlangıç/bitiş/görünürlük aralığı yanlış; çoğu örnekte ikincil etiket. |

## Kategori özeti

| Kategori | Top-10 baskın failure modları | Manuel kısa hüküm |
|---|---:|---|
| `normal` | `GLOBAL_SHIFT`: 10/10 | Normal sahnede bile ağır failure’lar çoğunlukla lane ailesinin paralel/toplu kayması. Bu rastgele noise değil, temsil/kalibrasyon problemi. |
| `crowd` | `GLOBAL_SHIFT`: 5/10, `WRONG_BOUNDARY_OR_OBJECT`: 5/10 | Araç/otobüs/park etmiş araçlar lane evidence ile boundary/object evidence ayrımını bozuyor. |
| `hlight` | `GLOBAL_SHIFT`: 8/10, `HARD_VISIBILITY_PRIOR`: 2/10 | Parlama altında model yine plausible lane family basıyor ama boya evidence’a hassas kilitlenmiyor. |
| `shadow` | `GLOBAL_SHIFT`: 7/10, `HARD_VISIBILITY_PRIOR`: 3/10 | Gölge/underpass geçişleri global alignment’ı ve gerçek çizgi kanıtını zayıflatıyor. |
| `noline` | `HARD_VISIBILITY_PRIOR`: 8/10, `WRONG_BOUNDARY_OR_OBJECT`: 2/10 | Model, zayıf/eksik çizgide gerçek GT’yi bulmak yerine ortalama yol şablonu üretiyor. |
| `arrow` | `WRONG_BOUNDARY_OR_OBJECT`: 6/10, `GLOBAL_SHIFT`: 4/10 | Yol okları, bariyerler ve yol kenarı negatifleri lane gibi davranabiliyor. |
| `curve` | `CURVE_TOPOLOGY_FAIL`: 8/10, `GLOBAL_SHIFT`: 2/10 | Curve kategorisinde hata sadece offset değil; yolun eğri/topolojik akışı yanlış takip ediliyor. |
| `cross` | `HALLUCINATED_NO_GT`: 10/10 | GT lane yokken model 3-4 lane basıyor. Bu saf false-positive/existence-quality problemi. |
| `night` | `HARD_VISIBILITY_PRIOR`: 7/10, `GLOBAL_SHIFT`: 2/10, `HALLUCINATED_NO_GT`: 1/10 | Gece/parlama/yansıma altında prediction genellikle gerçek boya yerine yol şablonuna dayanıyor. |

Top-10 ağır örnekler üzerindeki yaklaşık ana etiket toplamı:

| Ana etiket | Sayı / 90 | Yorum |
|---|---:|---|
| `GLOBAL_SHIFT` | 38 | Affine/low-rank oracle bulgusuyla uyumlu. Doğru lane ailesi çoğu kez yakın ama kalibre değil. |
| `HARD_VISIBILITY_PRIOR` | 20 | Model zayıf evidence altında learned road prior’a dönüyor. |
| `WRONG_BOUNDARY_OR_OBJECT` | 13 | Negatif yapıların lane’den ayrımı zayıf. |
| `HALLUCINATED_NO_GT` | 11 | Özellikle cross/no-GT durumunda existence-quality ciddi zayıf. |
| `CURVE_TOPOLOGY_FAIL` | 8 | Curve/topology representation sınırı. |

Bu sayılar kesin metrik değil, manuel görsel audit sayımıdır.

## Kategori bazlı görsel teşhis

### normal

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/normal_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `GLOBAL_SHIFT` | Lane ailesi paralel ama GT’den topluca uzak; bütün GT’ler FN, bütün predler FP. |
| 02 | `GLOBAL_SHIFT` | Aynı sahne ailesi; prediction gerçek lane merkezlerinden kaymış ve sağ/sol sınırları yanlış yerleştiriyor. |
| 03 | `GLOBAL_SHIFT` | Toplu yanal/eğimsel kayma; tekil row jitter değil. |
| 04 | `GLOBAL_SHIFT` | Sağ tarafta park/araç sınırları prediction’ı çekiyor gibi; ana mod yine toplu shift. |
| 05 | `GLOBAL_SHIFT` | Lane layout makul ama GT overlap eşiğini geçemeyecek kadar ofsetli. |
| 06 | `GLOBAL_SHIFT` | Aynı video segmentinde sistematik alignment hatası. |
| 07 | `GLOBAL_SHIFT` | Prediction lane ailesi GT ailesiyle paralel fakat tüm aile yer değiştirmiş. |
| 08 | `GLOBAL_SHIFT` | GT ve prediction ayrımı net; model görüntü evidence’ına tam kilitlenmemiş. |
| 09 | `GLOBAL_SHIFT` | Sistematik video-segment shift devam ediyor. |
| 10 | `GLOBAL_SHIFT` | Daha açık yol sahnesinde de paralel lane-family mismatch var. |

Hüküm: Normal kategorideki ağır failure’lar bile “rastgele kötü görüntü” değil. Modelin lane ailesini plausible ama yanlış geometriyle üretmesi ana problem.

### crowd

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/crowd_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `GLOBAL_SHIFT` | Otobüs/araçlar var ama lane family genel olarak kaymış; güçlü prior davranışı. |
| 02 | `WRONG_BOUNDARY_OR_OBJECT` | Sağdaki kamyon/araç ve yol kenarı prediction’ı yanlış evidence’a çekiyor. |
| 03 | `GLOBAL_SHIFT` | Normaldeki aynı şehir içi pattern: GT ile pred paralel, sistematik kaymış. |
| 04 | `WRONG_BOUNDARY_OR_OBJECT` | Araçlar ve sarı/yan çizgiler lane evidence ile karışıyor. |
| 05 | `WRONG_BOUNDARY_OR_OBJECT` | Büyük araç/sağ sınır prediction için dominant evidence gibi. |
| 06 | `GLOBAL_SHIFT` | Trafik içinde lane ailesi makul ama GT’ye oturmuyor. |
| 07 | `WRONG_BOUNDARY_OR_OBJECT` | Parlama/araç kenarları ve occlusion prediction’ı bozuyor. |
| 08 | `GLOBAL_SHIFT` | Ağır kalabalıkta bile ana geometri paralel-shift. |
| 09 | `WRONG_BOUNDARY_OR_OBJECT` | Yakın araç/park çizgisi/kenar yapıları lane ile karışıyor. |
| 10 | `GLOBAL_SHIFT` | Kalabalık sahnede lane ailesi var ama fiziksel boya ile hizalanmıyor. |

Hüküm: Crowd’da sorun sadece occlusion değil. Occlusion modelin already-strong lane prior’ını yanlış yere kilitliyor.

### hlight

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/hlight_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `GLOBAL_SHIFT` | Parlak zemin; pred/GT paralel ama lane ailesi kaymış. |
| 02 | `HARD_VISIBILITY_PRIOR` | Parlama ve araçlar gerçek çizgi evidence’ını bastırıyor; model prior basıyor. |
| 03 | `GLOBAL_SHIFT` | Sağ/sol lane sınırları GT’ye yakın ama topluca ofsetli. |
| 04 | `GLOBAL_SHIFT` | Model orta yol geometrisini biliyor; gerçek polilin konumu yanlış. |
| 05 | `GLOBAL_SHIFT` | Aynı sahne ailesinde shift devam ediyor. |
| 06 | `GLOBAL_SHIFT` | Hatalar düşük-rank görünüyor. |
| 07 | `GLOBAL_SHIFT` | Geniş parlama alanında pred family GT’den ayrılıyor. |
| 08 | `HARD_VISIBILITY_PRIOR` | Işık/kontrast kaybı; model görüntüden ziyade şablona dayanıyor. |
| 09 | `GLOBAL_SHIFT` | Yakın-paralel lane family mismatch. |
| 10 | `GLOBAL_SHIFT` | Yine topolojik olarak makul ama GT’ye göre yanlış. |

Hüküm: Highlight failure’ları, affine-oracle hattını destekliyor; ama correction’ın ne zaman güvenle uygulanacağı hâlâ çözülmemiş.

### shadow

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/shadow_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `GLOBAL_SHIFT` | Kısmi gölge ve araçlar; prediction GT’ye paralel ama kayık. |
| 02 | `GLOBAL_SHIFT` | Van/araç altında çizgi evidence zayıf; yine toplu alignment hatası. |
| 03 | `GLOBAL_SHIFT` | Şehir içi gölge; lane ailesi yanlış konumda. |
| 04 | `GLOBAL_SHIFT` | Benzer sahne; shift sistematik. |
| 05 | `HARD_VISIBILITY_PRIOR` | Aşırı kontrast/gölge; model ortalama yol geometrisini basıyor. |
| 06 | `HARD_VISIBILITY_PRIOR` | Underpass/parlama geçişi; gerçek boya evidence zayıf. |
| 07 | `GLOBAL_SHIFT` | Tünel/underpass ortamında paralel ama ofsetli lane family. |
| 08 | `GLOBAL_SHIFT` | Yakın-paralel mismatch. |
| 09 | `HARD_VISIBILITY_PRIOR` | Koyu sahne; prediction yol şablonuna yaslanıyor. |
| 10 | `GLOBAL_SHIFT` | Otobüs/kenar etkisi var ama ana geometri shift. |

Hüküm: Shadow’da backbone/evidence kalitesi düşüyor; model semantic/context yerine lane prior ile devam ediyor.

### noline

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/noline_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `HARD_VISIBILITY_PRIOR` | Lane boyası çok zayıf; model genel yol lane layout’u üretiyor. |
| 02 | `HARD_VISIBILITY_PRIOR` | GT var ama görüntü evidence zayıf; pred plausible ama yanlış. |
| 03 | `WRONG_BOUNDARY_OR_OBJECT` | Park etmiş araçlar/kenar çizgileri prediction’ı çekiyor. |
| 04 | `HARD_VISIBILITY_PRIOR` | Aynı düşük-evidence sahnesinde prior davranışı. |
| 05 | `WRONG_BOUNDARY_OR_OBJECT` | Sağdaki araç/kenar yapısı lane gibi kullanılmış görünüyor. |
| 06 | `HARD_VISIBILITY_PRIOR` | GT çizgileri soluk; prediction lane şablonu. |
| 07 | `HARD_VISIBILITY_PRIOR` | Aynı sequence, aynı prior kaynaklı mismatch. |
| 08 | `HARD_VISIBILITY_PRIOR` | Geniş yol, zayıf boya; lane family yanlış oturuyor. |
| 09 | `HARD_VISIBILITY_PRIOR` | Noline koşulunda model gerçek çizgiden çok yol formuna bakıyor. |
| 10 | `HARD_VISIBILITY_PRIOR` | Çoklu lane prior, GT ile overlap kuramıyor. |

Hüküm: Noline kategorisi correction problemi değil; “kanıt yokken ne kadar lane basmalıyım?” problemidir. Existence/quality ve hard-negative reasoning zayıf.

### arrow

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/arrow_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `GLOBAL_SHIFT` | Lane ailesi geniş/yol sınırlarına kaymış. |
| 02 | `WRONG_BOUNDARY_OR_OBJECT` | Yol kenarı/guardrail/ok negatifleri lane evidence ile karışıyor. |
| 03 | `WRONG_BOUNDARY_OR_OBJECT` | Geniş yol ve işaretler; pred yanlış fiziksel çizgilere oturuyor. |
| 04 | `GLOBAL_SHIFT` | Merkez lane family var ama GT’ye göre sistematik ofset. |
| 05 | `GLOBAL_SHIFT` | Çoklu lane ailesi paralel ama yanlış hizalı. |
| 06 | `GLOBAL_SHIFT` | Benzer highway pattern, low-rank shift. |
| 07 | `WRONG_BOUNDARY_OR_OBJECT` | Bariyer/kenar çizgisi prediction için lane gibi. |
| 08 | `WRONG_BOUNDARY_OR_OBJECT` | Sol bariyer ve yol kenarı güçlü yanlış evidence. |
| 09 | `WRONG_BOUNDARY_OR_OBJECT` | Yol kenarı/araç/işaret karmaşası. |
| 10 | `WRONG_BOUNDARY_OR_OBJECT` | Sağ/sol negatif çizgiler lane gibi kullanılıyor. |

Hüküm: Arrow sadece “ok var” problemi değil. Road marking negatifleri ve lane boundary ayrımı zayıf.

### curve

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/curve_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `CURVE_TOPOLOGY_FAIL` | Sağ virajda pred curve akışını yanlış takip ediyor; road edge’e kayma var. |
| 02 | `CURVE_TOPOLOGY_FAIL` | Benzer viraj; prediction curve merkezini değil genel şablonu izliyor. |
| 03 | `CURVE_TOPOLOGY_FAIL` | Eğri/yol akışı doğru yakalanmamış. |
| 04 | `GLOBAL_SHIFT` | Daha düz şehir içi curve; paralel lane family mismatch. |
| 05 | `CURVE_TOPOLOGY_FAIL` | Viraj ve komşu lane/kenar ayrımı bozuk. |
| 06 | `CURVE_TOPOLOGY_FAIL` | Curve sırasında lane identity karışıyor. |
| 07 | `CURVE_TOPOLOGY_FAIL` | Keskin/karmaşık dönüş; pred yanlış branch/edge’e gidiyor. |
| 08 | `CURVE_TOPOLOGY_FAIL` | Prediction curve akışını kaybediyor. |
| 09 | `CURVE_TOPOLOGY_FAIL` | Trafikli virajda lane topology bozulmuş. |
| 10 | `GLOBAL_SHIFT` | Daha makul layout ama GT’ye göre ofsetli. |

Hüküm: Curve’de 2-DoF affine tek başına yeterli olmayabilir. Hata bazen düşük-rank shift, bazen gerçek topology/branch failure.

### cross

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/cross_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `HALLUCINATED_NO_GT` | GT yok; model 4 lane basıyor. |
| 02 | `HALLUCINATED_NO_GT` | GT yok; plausible road layout FP. |
| 03 | `HALLUCINATED_NO_GT` | Cross/no-lane durumda existence suppression başarısız. |
| 04 | `HALLUCINATED_NO_GT` | Prediction tamamen prior kaynaklı görünüyor. |
| 05 | `HALLUCINATED_NO_GT` | Aynı sequence; no-GT rejection yok. |
| 06 | `HALLUCINATED_NO_GT` | 4 FP lane; quality/existence yanlış güvenli. |
| 07 | `HALLUCINATED_NO_GT` | Road görünür diye lane var sanıyor. |
| 08 | `HALLUCINATED_NO_GT` | Geniş yol/çizgi yok; model lane family basıyor. |
| 09 | `HALLUCINATED_NO_GT` | No-GT sahnesinde FP üretimi. |
| 10 | `HALLUCINATED_NO_GT` | GT yokken çoklu lane. |

Hüküm: Cross kategorisi geometri düzelticiyle çözülmez. Burada ana eksik, lane existence/quality’nin negatif sahne bağlamını öğrenmemesi.

### night

Contact sheet: `outputs/culane_s0_structured_query_res34_b16_50ep/test_175k_failure_overlays_thr0p40_q0p25_fresh/night_contact_sheet.jpg`

| Rank | Etiket | Not |
|---:|---|---|
| 01 | `HARD_VISIBILITY_PRIOR` | Karanlık/araç ışığı; pred plausible ama GT’ye oturmuyor. |
| 02 | `HARD_VISIBILITY_PRIOR` | Parlama ve yansıma prediction’ı yanlış evidence’a çekiyor. |
| 03 | `GLOBAL_SHIFT` | Lane family paralel ama topluca kaymış. |
| 04 | `HARD_VISIBILITY_PRIOR` | Gece/parlama; gerçek boya evidence zayıf. |
| 05 | `HARD_VISIBILITY_PRIOR` | Model yol şablonunu sürdürüyor, GT’ye kilitlenmiyor. |
| 06 | `HARD_VISIBILITY_PRIOR` | Aydınlatma koşulu zor; pred/GT ayrımı sistematik. |
| 07 | `HALLUCINATED_NO_GT` | Çok karanlık bölgede pred lane family prior gibi. |
| 08 | `GLOBAL_SHIFT` | Karanlıkta paralel ama yanlış hizalı lane family. |
| 09 | `HARD_VISIBILITY_PRIOR` | Far/parlama; gerçek çizgi yerine prior. |
| 10 | `HARD_VISIBILITY_PRIOR` | Karanlık ve zayıf evidence; prediction plausible ama GT’ye uymuyor. |

Hüküm: Night kategorisinde S0 görüntü kanıtı yerine prior’a fazla dayanıyor. Bu, sadece affine correction değil, evidence verification problemidir.

## Ana sonuç

Bu görsel audit, iki önceki nicel sonucu aynı anda doğruluyor:

1. Affine/low-rank geometri problemi gerçek.
   - `normal`, `hlight`, `shadow` ve `crowd` ağır örneklerinde prediction ailesi GT’ye paralel ama topluca kaymış.
   - Bu, near-miss affine oracle’ın neden yüksek çıktığını görsel olarak açıklıyor.

2. Ama tek sorun affine kayma değil.
   - `cross`: GT yokken model 4 lane basıyor.
   - `noline/night`: gerçek boya evidence zayıfken model ortalama yol/lane şablonuna dönüyor.
   - `arrow/crowd`: yol kenarı, araç, bariyer, ok gibi negatifler lane evidence ile karışıyor.
   - `curve`: bazı örneklerde düzeltilebilir shift değil, topology/branch takibi bozuk.

Bu nedenle “sadece geometri düzeltici” yetersiz kalır. S0’ın ana zayıflığı daha net şu:

> Model plausible lane geometry üretmeyi öğrenmiş, fakat her lane hipotezi için “bu polilin boyunca gerçekten lane evidence var mı ve bu sahnede lane var sayılmalı mı?” sorusunu yeterince image-grounded ve negative-aware biçimde çözmüyor.

## Mimarî çıkarım

Bu audit’ten çıkan doğrudan mimari sonuç:

- `GLOBAL_SHIFT` için düşük-rank/affine correction hâlâ anlamlı bir latent fırsat.
- Fakat correction’ın güvenli çalışması için no-harm gate gerekir; önceki gate/ranker deneyleri bunun mevcut frozen/internal feature’larla güvenilir öğrenilemediğini gösterdi.
- `CROSS/NOLINE/NIGHT/ARROW` için problem correction değil, hypothesis verification ve hard-negative suppression.
- `CURVE` için 2-DoF affine bazı örneklerde yetersiz; row/sequence-level curve evidence gerekiyor.

Pratik olarak, bir sonraki mekanizma sadece `x_rows` düzeltmemeli. Aynı hipotez için şu üç şeyi beraber ölçmeli:

1. On-curve evidence: prediction polilini üzerinde gerçekten lane sinyali var mı?
2. Off-curve negative evidence: aynı sahnede daha iyi komşu/boundary/arrow/curb evidence var mı?
3. Scene-level validity: bu görüntüde bu lane hipotezini basmak mı, bastırmak mı doğru?

Bu yüzden en tutarlı sonraki araştırma yönü, salt affine head değil; lane-hypothesis-conditioned visual verifier + hard-negative-aware quality/existence kalibrasyonudur.

