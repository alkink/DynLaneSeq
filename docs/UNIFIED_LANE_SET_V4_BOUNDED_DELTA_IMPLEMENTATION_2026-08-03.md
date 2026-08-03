# Unified Lane-Set V4: Bounded-Delta 278k Implementation

Tarih: 3 Ağustos 2026

Bu belge V3'ün `30k–35k` arasında yaşadığı katastrofik geometri çöküşünden
sonra uygulanan ana mimariyi kaydeder. Bu çalışma küçük bir LR rescue'u veya
post-processing yaması değildir. Amaç aynı eğitimin `0→278k` boyunca
taşıyabileceği tutarlı bir geometri ve gradyan sözleşmesi kurmaktır.

## 1. V3'te kaldırılan tehlikeli döngü

```text
reference çevresinde yalnız yerel P2 kanıtı oku
                    |
                    v
ortak affine LayerNorm + ortak Linear(256, 800)
                    |
                    v
bütün görüntü üzerinde sınırsız mutlak x üret
                    |
                    v
detach edilmiş x'i sonraki katmanın görüş merkezi yap
```

Bu yol, modelin bakmadığı bir bölgeye yüksek güvenle sıçramasına izin
veriyordu. Dört farklı decoder katmanı aynı mutlak koordinat kafasını farklı
Hungarian kimlikleriyle eğitiyordu. V3'te `row_norm/row_x` normlarının
`30k→35k` arasında yaklaşık iki katına çıkması ve All-32 recall'ın çökmesi bu
döngüyle zaman olarak örtüşüyordu.

## 2. V4 mimari ağacı

```text
Image
  |
  v
DLA-34 + SimpleFPN
  |
  +---------------- P2 geometry feature
  |                         |
  |                         v
  |             Full-row image-grounded acquisition
  |             (normalized row-query / P2-key similarity)
  |                         |
  |                         v
  |                 absolute initial curve
  |                         |
  |       +-----------------+-----------------+
  |       |                 |                 |
  |       v                 v                 v
  |  Local block 1     Local block 2     Local block 3/4
  |  17 P2 samples     17 P2 samples     17 P2 samples
  |  own delta head    own delta head    own delta head
  |  delta in ±96      delta in ±96      delta in ±96
  |       |                 |                 |
  |       +-----------------+-----------------+
  |                         |
  |                         v
  |                  final ordered rows
  |                         |
  |                         +------> range head
  |                         |
  +---- detached P4/P5 -----+------> one foreground score
```

İlk koordinat hâlâ bütün satır üzerinde görüntü kanıtından bulunur. Sonraki
katmanlar yalnız referans çevresindeki yerel kanıtı okur ve yalnız yerel eylem
üretir.

## 3. Hard geometri invariant'ı

Her yerel katman 33 adet delta bininden dağılım üretir:

```text
[-96, -90, ..., 0, ..., 90, 96] px
```

```text
p(delta) = softmax(delta_logits)
delta_x  = sum p(delta) * delta
x_next   = clamp(x_reference + delta_x, 0, W-1)
```

Bu nedenle kod seviyesinde:

```text
|x_next - x_reference| <= 96 px
```

olur. Head'in ağırlıkları başlangıçta sıfırdır. Simetrik delta dağılımının
beklentisi sıfır olduğu için her blok eğitimin ilk adımında identity update
yapar; full-row acquisition eğrisi rastgele bozulmaz.

Yerel P2 profili 17 noktada ve 12-piksel aralıkla örneklenir. Eylem dağılımı
33 bin ve 6-piksel çözünürlüktedir. İki çözünürlüğü ayırmak, 16-GiB GPU'da
33 tam feature örneğinin yaratacağı yaklaşık `4.7x` profil belleğini önler.

## 4. Katman başına ayrı koordinat kafası

V3:

```text
Layer 1 ─┐
Layer 2 ─┼─> shared affine row_norm + shared global row_x
Layer 3 ─┤
Layer 4 ─┘
```

V4:

```text
Layer 1 -> non-affine norm 1 -> bias-free delta head 1
Layer 2 -> non-affine norm 2 -> bias-free delta head 2
Layer 3 -> non-affine norm 3 -> bias-free delta head 3
Layer 4 -> non-affine norm 4 -> bias-free delta head 4
```

`LayerNorm(elementwise_affine=False)` kullanıldığı için eski
`gamma/beta + W/b` runaway yolu yerel koordinat okuyucusunda bulunmaz.

## 5. Assignment ve deep-supervision sözleşmesi

```text
Hungarian cost = point + range + LineIoU
score cost      = 0
```

Final katmanın geometry-only Hungarian eşleşmesi bir kez hesaplanır. Bütün
ara katmanlar aynı query/GT kimliğini yeniden kullanır:

```text
final assignment -> Layer 1 geometry loss
                 -> Layer 2 geometry loss
                 -> Layer 3 geometry loss
                 -> final geometry + score loss
```

Ara katmanlarda existence loss sıfırdır. Böylece aynı query bir katmanda
foreground, başka katmanda background olarak ortak score head'e çelişkili
etiket vermez.

## 6. Tek skor ve tek yönlü bilgi akışı

Final skor hedefi matched lane'in detach edilmiş range-aware 30-pixel row
strip IoU'sudur:

```text
matched target   = 0.5 + 0.5 * detached_IoU
unmatched target = 0
```

Loss quality-focal biçimindedir:

```text
|target - sigmoid(score)|^2 * BCEWithLogits(score, target)
```

Skor branch'inin lane-state ve P4/P5 girişleri detach edilir; ayrıca kendi
`decision_norm` katmanı vardır. Sonuç:

```text
geometry ------> score için bilgi
score loss -X-> lane state
score loss -X-> row decoder / delta head
score loss -X-> P2/P4/P5 / backbone / FPN
```

Semantic attention ve foreground head yine trainable'dır. Yalnız girdileri
detach edilmiştir. Cardinality, binary score margin, ayrı quality head ve set
selector kapalıdır.

## 7. Optimizer grupları

| Grup | LR |
|---|---:|
| DLA backbone | `1e-5` |
| Lane-state recurrent core | `5e-5` |
| Global reference + local delta heads | `5e-5` |
| Instance/row/x identity embeddings | `5e-5` |
| Kalan visual decoder, FPN ve score yolu | `1e-4` |

Norm ve bias parametreleri weight decay almaz. Global gradient clipping `1.0`
olarak korunur. Scheduler baştan itibaren `278000` iterasyondur; 50k bir kısa
scheduler değildir.

## 8. İlgili dosyalar

```text
dynlaneseq_eg/modeling/structured_queries.py
dynlaneseq_eg/losses/loss_s0.py
dynlaneseq_eg/factory.py
dynlaneseq_eg/configs/
  culane_s0_structured_query_dla34_unified_lane_set_v4_bounded_delta_278k.yaml
scripts/run_culane_dla34_unified_lane_set_v4_bounded_delta_278k.sh
scripts/audit_culane_dla34_unified_lane_set_v4_short.sh
scripts/audit_culane_dla34_unified_lane_set_v4_all_checkpoints.sh
scripts/eval_culane_dla34_unified_lane_set_v4_full_test.sh
scripts/analyze_culane_dla34_unified_lane_set_v4_selection_coverage_50k.sh
dynlaneseq_eg/tests/test_unified_lane_set_v4_contract.py
```

## 9. Eğitim komutları

İlk 50k, aynı 278k koşusunun fail-fast kapısıdır:

```bash
DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
TARGET_ITERS=50000 \
bash scripts/run_culane_dla34_unified_lane_set_v4_bounded_delta_278k.sh
```

50k sağlıklıysa aynı optimizer/scheduler state ile devam:

```bash
DATA_ROOT=/workspace/CULane \
BATCH_SIZE=4 \
GRAD_ACCUM=4 \
TARGET_ITERS=278000 \
AUTO_RESUME=1 \
bash scripts/run_culane_dla34_unified_lane_set_v4_bounded_delta_278k.sh
```

Bu ikinci komut yeni eğitim başlatmaz; aynı output klasöründeki en son
checkpoint'i bulur ve kalan iterasyonları çalıştırır.

## 10. 5k-checkpoint trajectory denetimi

```bash
DATA_ROOT=/workspace/CULane \
ITERATIONS="5000 10000 15000 20000 25000 30000 35000 40000 45000 50000" \
MAX_BATCHES=16 \
EVAL_BATCH_SIZE=4 \
bash scripts/audit_culane_dla34_unified_lane_set_v4_all_checkpoints.sh
```

Özellikle izlenecek değerler:

- All-32 recall @0.50 ve @0.75;
- Oracle Top-4 ve Direct Top-4 farkı;
- matched mean official IoU;
- score/official-IoU korelasyonu;
- `row_delta_heads.*` norm eğrisi;
- score gradientinin geometry gruplarındaki normunun sıfır olması.

## 11. Test durumu ve dürüst sınır

Uygulama testleri şunları doğrular:

- her katmanın delta sınırı;
- katman başına ayrı head;
- non-affine norm ve bias-free projection;
- score loss'tan geometry/P2/P4/P5'e sıfır gradient;
- geometry loss'tan local evidence yoluna gradient;
- reference-relative DFL;
- final assignment reuse;
- IoU-aware score target'ının geometry'den detached olması;
- exact 278k config ve optimizer grouping.

Bu testler mimari sözleşmenin doğru uygulandığını kanıtlar. Henüz V4'ün F1'ını
veya 278k boyunca ampirik olarak çökmeyeceğini kanıtlamaz. Bu iddia yalnız
5k–50k trajectory ve devamındaki gerçek eğitim sonuçlarıyla kurulabilir.

## 12. 50k selection-coverage ayrıştırması

V4 trajectory'sinde aday geometrisi stabil kalıp `Oracle Top-4` ile gerçek
`score Top-4` arasında büyük fark görülürse, aynı frozen aday havuzu şu komutla
ayrıştırılır:

```bash
DATA_ROOT=/workspace/CULane \
MAX_BATCHES=64 \
EVAL_BATCH_SIZE=4 \
bash scripts/analyze_culane_dla34_unified_lane_set_v4_selection_coverage_50k.sh
```

Bu probe eğitim yapmaz ve threshold seçmez. Aynı resmî raster-IoU matrisi
üzerinde şunları karşılaştırır:

```text
scalar score Top-4
score-ordered hard curve diversity Top-4 (10/20/30/40/60 px)
MMR score/curve-diversity Top-4
maximum-cardinality Oracle Top-4
```

Rapor ayrıca her yöntemin duplicate, near-miss, background ve empty-scene
FP'lerini; seçilen eğrilerin pairwise mesafesini; hard-diversity sonrasında
kalan aday havuzunun Oracle kapasitesini yazar. Grid içindeki en iyi satır
yalnız teşhis amaçlıdır ve aynı validation örneklerinde seçildiği için benchmark
veya deployment sonucu olarak kullanılamaz. Mimari karar için sabit `20 px`
satırı, bütün grid'in eğilimi ve Oracle boşluğunun geri kazanılan oranı birlikte
incelenir.
