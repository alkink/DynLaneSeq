# V36 Assignment–Posterior Contract Audit

## 1. Amaç

V34 ve V35 aynı olguyu iki farklı yoldan gösterdi:

- V7'nin 32 proposal bankasında doğru lane çoğu zaman var;
- deployed route buna rağmen başka bir proposal seçiyor;
- frozen V7 feature'ı, ham RGB ve üç-frame RGB bu iyi/kötü çifti unseen
  clip'lerde güvenilir biçimde ayıramıyor.

V36 yeni bir selector eğitmez. V7'nin kendi training contract'ını tekrar
oynatır ve şu soruyu ölçer:

> Oracle-good proposal eğitim sırasında gerçekten pozitif geometri/ownership
> hedefi alıyor mu, yoksa one-to-one matcher onu background olarak mı
> cezalandırıyor?

Bu ayrım yapılmadan yeni bir matcher, one-to-many objective veya yeni detector
tasarlamak nedensel değildir.

## 2. Sabit veri ve model

- checkpoint: mature V7 `iter_0225000.pt`;
- örnekler: V34'te önceden oluşturulmuş 54 validation clip'lik temporal
  manifest;
- primary cohort: V34'teki exact route-recoverable oracle-good/current-wrong
  çiftleri;
- model inference tekrar çalıştırılmaz;
- `target_main_official_iou.pt` içindeki exact cached V7 proposal, range,
  existence, route ve target tensorları kullanılır;
- test split kapalıdır;
- threshold, checkpoint veya örnek seçimi yapılmaz.

## 3. Üç ayrı öğretmen tekrar oynatılacak

### 3.1 Proposal Hungarian öğretmeni

V7'nin exact configured matcher maliyeti yeniden hesaplanır:

```text
0.25 * object
+ 5.0 * point
+ 1.0 * range
+ 2.0 * LineIoU
```

Her good/current-wrong çifti için ölçülür:

- hangi adayın daha düşük matcher maliyeti aldığı;
- intended GT'ye hangi adayın atandığı;
- good ve wrong adayların herhangi bir GT için positive olup olmadığı;
- good adayın unmatched kaldığı için no-lane target alıp almadığı.

### 3.2 Four-slot route öğretmeni

V7'nin exact all-GT cluster target'ı yeniden hesaplanır:

```text
cluster delta = 0.10
temperature   = 0.03
```

Her çift için ölçülür:

- target probability mass: good ve wrong;
- target top-1;
- deployed slot posterior: good ve wrong;
- route target'ın istediği logit güncelleme yönü.

### 3.3 Final slot geometry öğretmeni

V7'nin hard-routed raw references'ı ile GT arasında exact range-aware
Hungarian tekrar hesaplanır. Böylece current slot refiner'ın gerçekten aynı
GT'ye mi, başka bir GT'ye mi geometri loss'u aldığı ölçülür.

## 4. Counterfactual matcherlar

Predictionlar değişmeden dört assignment kontratı karşılaştırılır:

1. configured V7 composite matcher;
2. object term kapalı composite matcher;
3. point-only matcher;
4. target row-strip-IoU Hungarian.

Official raster-IoU Hungarian yalnız diagnostic upper reference olarak
raporlanır; trainable bir sonuç sayılmaz.

## 5. Ana metrikler

Metrikler bütün çiftlerde ve `.50`/`.75`, A/B fold kırılımlarında ayrı
raporlanır:

- `route_target_good_preference_rate`;
- `route_target_good_top1_rate`;
- `configured_cost_good_preference_rate`;
- `configured_good_intended_assignment_rate`;
- `configured_good_any_positive_rate`;
- `configured_wrong_intended_assignment_rate`;
- `good_route_up_but_proposal_down_rate`;
- `wrong_route_down_but_proposal_up_rate`;
- `slot_geometry_intended_gt_rate`;
- counterfactual assignment değişimleri.

Oranların yüzde 95 aralıkları image-level değil clip-level paired bootstrap ile
hesaplanır.

## 6. Önceden kilitli karar ağacı

### A. `ROUTE_TARGET_MISALIGNED`

Şunlardan biri gerçekleşirse:

```text
route target good preference < 0.70
veya
route target good top-1 < 0.70
```

Mevcut route target iyi proposal'ı yeterince açık tarif etmiyordur. Yeni image
encoder değil, target/assignment contract incelenir.

### B. `HARD_ASSIGNMENT_STARVATION`

Şunların tamamı gerçekleşirse:

```text
route target good preference >= 0.70
configured good intended assignment < 0.65
bir counterfactual good intended assignment'ı >= 0.10 artırıyor
```

Proposal one-to-one assignment ana darboğazdır. Sonraki deney, proposal
support'unu koruyan controlled one-to-many/soft-ownership objective olur.

### C. `POST_ASSIGNMENT_BELIEF_FAILURE`

Şunların tamamı gerçekleşirse:

```text
route target good preference >= 0.80
configured matcher cost good preference >= 0.75
configured good any-positive rate >= 0.80
```

Matcher ve target good proposal'ı zaten biliyordur. Yeni matcher veya daha çok
positive query ana çözüm değildir. Sorun, bu doğru supervision'ın deployed
belief'e dönüşmemesidir.

### D. `MIXED_CONTRACT_FAILURE`

Yukarıdaki temiz sınıflardan hiçbiri geçmezse birden fazla sözleşme aynı anda
sınırlayıcıdır. En büyük ölçülmüş kayıp önce düzeltilir; doğrudan uzun training
açılmaz.

## 7. Yorum sınırı

Bu audit şunları kanıtlamaz:

- yeni matcher'ın official F1 artıracağını;
- doğru tail proposal'ın görüntüden seçilebilir olduğunu;
- temporal veya yeni end-to-end detector'ın imkânsız olduğunu.

Yalnızca V7'nin mevcut training objective'inin, deploy sırasında kaybettiği
oracle-good proposal'a hangi yönde supervision verdiğini ölçer.
