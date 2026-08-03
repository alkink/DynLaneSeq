# Unified Lane-Set V4: Diversity and Coverage Diagnosis

Tarih: 3 Ağustos 2026

## Ölçülen sonuç

V4 50k geometrisi uniform-256 validation örneğinde güçlü kalırken scalar
Top-4 aynı şeridin birden fazla kopyasını seçmektedir. V4.1 D kolunda ölçülen
karşılaştırma:

```text
                         F1@.50   R@.50   F1@.75
scalar Top-4              50.26    54.71    42.13
MMR, sigma=20, p=0.5      75.61    82.30    55.33
All-32 oracle                --    95.29       --
```

Aynı adaylar ve aynı score kullanıldığı halde MMR:

```text
TP@.50             476 -> 716
duplicate FP       367 ->   8
F1@.50            50.26 -> 75.61
```

Bu, ana darboğazın background ayırmaktan çok benzersiz lane coverage olduğunu
gösterir. D scorer'ın ortalama olasılıkları da aynı ayrımı desteklemektedir:

```text
unique TP   0.236
duplicate   0.176
background  0.050
```

Scorer lane-benzeri adayları background'dan ayırabilmekte, ancak scalar Top-4
aynı lane kümesinden birden fazla temsilci seçmektedir.

## Eski hard-diversity raporundaki sözleşme hatası

İlk `analyze_v4_selection_coverage` raporunda MMR doğru biçimde config'in
deployed score'unu kullanıyordu. Hard-diversity/NMS yolu ise
`trace_postprocess` varsayılanı nedeniyle C/D kollarında da legacy existence
score'a dönüyordu. Bu yüzden hard-diversity satırı üç raporda birebir aynıydı.

Yeni branch:

- `trace_postprocess(..., score_mode=...)` sözleşmesini açık hale getirir;
- hard-diversity ve MMR'nin ikisini de aynı deployed score ile ölçer;
- eğitim koduna veya checkpoint ağırlıklarına dokunmaz.

## Neden threshold ile birlikte ölçüyoruz?

İlk coverage deneyi kasıtlı olarak threshold'suz dört tahmin yazıyordu. Bu,
lane olmayan görüntülerde bile dört prediction üretip 76 empty-scene FP
oluşturdu. D scorer background'u düşük puanladığı için threshold bu FP'lerin
bir bölümünü kaldırabilir. Ancak threshold yanlış seçilirse doğru candidate
havuzunu da silebilir.

Yeni cached grid aynı anda şunları ölçer:

```text
score threshold:  0.00 ... 0.40
hard distance:    10, 15, 20, 25, 30 px
MMR sigma:        10, 15, 20, 30, 40 px
MMR penalty:      0.20, 0.35, 0.50, 0.65, 0.80
IoU:              0.50 ve 0.75
```

Her threshold için yalnız seçilen F1 değil, threshold sonrasında havuzda kalan
Oracle Top-4 kapasitesi de raporlanır. Böylece yüksek precision uğruna doğru
lane adaylarının silinip silinmediği görülür.

## Çalıştırma

V4.1 score gate cache'leri ve C/D final checkpoint'leri sunucuda dururken:

```bash
cd /workspace/DynLaneSeq
conda activate clrernet

DATA_ROOT=/workspace/CULane \
bash scripts/sweep_culane_dla34_unified_lane_set_v4_1_diverse_thresholds_50k.sh
```

Varsayılan `CACHE_ONLY=1` nedeniyle detector forward veya eğitim çalışmaz.
Eksik cache varsa script sessizce yeniden inference yapmak yerine fail-fast
durur. Bilinçli olarak cache yeniden üretilecekse `CACHE_ONLY=0 DEVICE=cuda`
kullanılabilir.

Ana sonuç:

```text
outputs/diagnostics/v4_1_diverse_threshold_grid_summary.json
```

## Sonraki mimari karar

Bu grid final post-processing ayarı değildir. Grid, öğrenilecek V4.2 coverage
selector için teacher davranışını belirler.

Beklenen V4.2 sözleşmesi:

```text
32 frozen/stable geometry candidates
        |
        v
quality/relevance descriptor
        |
        v
four sequential selection slots
        |
        +-- previous selections visible
        +-- soft curve redundancy visible
        +-- NO-LANE option
        `-- one different lane cluster per slot
```

Geometri selector loss'tan detach kalır. Amaç hard 20px NMS'i modele aynen
kopyalamak değildir; strict-IoU için iyi temsilciyi koruyan, kalite ve coverage
birlikte öğrenilmiş bir subset selector üretmektir.

