from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import yaml

from dynlaneseq_eg.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FILE_COUNT = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the exact 15-file GPT Sol Pro V23 review package."
    )
    parser.add_argument("--destination", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def _bundle(destination: Path, sources: list[Path]) -> None:
    chunks: list[str] = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = source.relative_to(PROJECT_ROOT)
        chunks.extend(
            (
                "# " + "=" * 78,
                f"# SOURCE FILE: {relative}",
                "# " + "=" * 78,
                source.read_text(encoding="utf-8"),
                "",
            )
        )
    destination.write_text("\n".join(chunks), encoding="utf-8")


def _metric_line(report: dict[str, Any], policy: str, threshold: str) -> str:
    row = report["metrics"][policy][threshold]
    return (
        f"TP={row['TP']} FP={row['FP']} FN={row['FN']} "
        f"F1={100.0 * float(row['F1']):.6f}"
    )


def _executive_report(
    training: dict[str, Any], official: dict[str, Any]
) -> str:
    delta = official["gate"]["deltas"]
    degradation = official["source_correct_degradation"]
    paired = official["paired_image_effects"]
    final = training["final_training_diagnostics"]
    return f"""# V23 Sol Pro review report

## Outcome

V23 is a formal **FAIL** on the untouched official CULane validation split.
It is safe and nearly parity-preserving, but the gain is far below the
predeclared mechanism gate and the full-validation target. The test set was
not used.

## Population and selection contract

- Training list: official `list/train.txt`, 88,880 rows.
- Validation list: official `list/val.txt`, 9,675 rows.
- Rows removed: 0.
- Deduplication, clip filtering, image filtering: none.
- Training: 8,000 steps, physical batch 8, 64,000 image exposures
  ({float(training['complete_official_train_epochs_seen']):.6f} official epochs).
- Checkpoint selection: none.
- Threshold/NMS selection: none.
- Endpoint iteration: {training['iteration']}.
- Test set: unused.

Because V23 trained on the complete official train population, an old
train-derived 256-image set cannot honestly be called unseen. The scientific
evaluation therefore used all 9,675 official validation images, not a subset.

## Architecture actually tested

- Frozen, eval-mode V7 teacher owns activity/count, visible range, proposal
  memory and initialization.
- A separate trainable student owns final geometry.
- Student output semantic: ordered slot-conditioned row cost volume
  `[B,4,160,800]`.
- Student parameters: {training['model_contract']['trainable_parameters']:,}.
- V7 teacher remained bit-exact at the endpoint.
- Zero-step public x/range geometry was exact V7.
- V22 weights initialized only the student backbone/FPN; all V23 decision
  modules remained trainable.

## Official-raster paired results

```text
                     IoU .50                                  IoU .75
V7 source            {_metric_line(official, 'source_v7', '0.50')}    {_metric_line(official, 'source_v7', '0.75')}
V23 correct image    {_metric_line(official, 'v23', '0.50')}    {_metric_line(official, 'v23', '0.75')}
V23 wrong image      {_metric_line(official, 'cross_clip_wrong_image', '0.50')}    {_metric_line(official, 'cross_clip_wrong_image', '0.75')}
```

V23 minus V7:

- TP@.50: {delta['v23_minus_v7_tp_50']:+d}
- TP@.75: {delta['v23_minus_v7_tp_75']:+d}
- F1@.50: {delta['v23_minus_v7_f1_50_points']:+.6f} percentage points
- F1@.75: {delta['v23_minus_v7_f1_75_points']:+.6f} percentage points

Correct image minus deterministic cross-clip wrong image:

- TP@.50: {delta['v23_minus_wrong_tp_50']:+d}
- TP@.75: {delta['v23_minus_wrong_tp_75']:+d}

The image path is therefore not completely dead, particularly at `.75`, but
its deployable gain over V7 is negligible.

## Paired safety decomposition

```json
{json.dumps(paired, indent=2, sort_keys=True)}
```

Source-correct GT loss:

- `.50`: {degradation['0.50']['lost_by_v23']} / {degradation['0.50']['source_correct']}
  = {100.0 * float(degradation['0.50']['fraction']):.6f}%
- `.75`: {degradation['0.75']['lost_by_v23']} / {degradation['0.75']['source_correct']}
  = {100.0 * float(degradation['0.75']['fraction']):.6f}%

Prediction count matched V7 on all 9,675 images.

## Formal gate

- Required F1@.50 gain: at least +0.50 points.
- Observed F1@.50 gain: {delta['v23_minus_v7_f1_50_points']:+.6f} points.
- Full-validation target: at least +0.80 points.
- F1@.75 non-regression: passed.
- Source-correct loss below 1%: passed at both thresholds.
- Correct image over wrong image: passed at both thresholds.
- Cardinality exact: passed.
- Overall gate: **FAIL** because the primary improvement magnitude did not pass.

## Training endpoint diagnostics

```json
{json.dumps(final, indent=2, sort_keys=True)}
```

The learned mean geometry gate was only
`{float(final['geometry_gate_mean']):.9f}`. Consequently, the deployed student
stayed extremely close to V7 even though the cost-volume/path losses received
gradients. The final minibatch is not an evaluation population, but its mean
predicted IoU (`{float(final['mean_predicted_iou']):.9f}`) was also slightly
below source (`{float(final['mean_source_iou']):.9f}`).

## Review questions that remain open

1. Is 0.72 epoch fundamentally too short for an 18.36M-parameter trainable
   student, or did the objective/gate parameterization structurally drive it
   toward V7 parity?
2. Did the large proposal-path loss dominate direct row geometry and prevent
   a useful slot-conditioned cost volume?
3. Does the straight-through geometry-gate construction provide a training
   signal consistent with the deployed gated geometry?
4. Is canonical left-to-right ownership stable enough under missing lanes and
   differing source/GT counts?
5. Does the correct-vs-wrong `.75` advantage justify changing optimization,
   or is the nearly zero `.50` causal gain evidence that V23 should stop?
6. What single next experiment would cleanly distinguish undertraining from a
   structurally wrong objective without opening test or doing a blind long run?

## Provenance caveat

Metrics here use the repository's fixed CULane-style discrete raster evaluator
for both V7 and V23 in one paired pass. The previously quoted official-C source
number differs by only a few hundredths of a point; that small implementation
difference cannot explain the missing +0.50 to +0.80 point gain.
"""


def _prompt() -> str:
    return """# GPT Sol Pro — V23 adversarial review prompt

Bu klasörde tam **15 dosya** vardır. Önce `01_V23_REVIEW_REPORT.md`, ardından
iki ham JSON raporu, resolved config ve kod dosyalarını oku. Ana rapordaki
yorumları doğru kabul etme; bütün headline sayıları
`02_OFFICIAL_VAL_REPORT.json` üzerinden yeniden hesapla.

Görevin V23'ü savunmak veya yeni sürüm önermek değil; neden neredeyse V7
paritesinde kaldığını mümkün olduğunca dürüst ve falsifiable biçimde bulmak.

Zorunlu inceleme:

1. Dataset kontratını doğrula: tam official `train.txt` 88,880 ve `val.txt`
   9,675; filtre, dedup, subset, threshold/checkpoint selection veya test
   kullanımı var mı?
2. `DynLaneSeqV23` ve `V23OrderedSlotCostVolume` üzerinden exact tensor graph'ı
   çıkar. V7'nin neyi sahiplendiğini, student'ın neyi sahiplendiğini ve final
   geometry'nin writer'a nasıl gittiğini göster.
3. Bütün loss ve gradient yollarını denetle. Özellikle zero-forward/
   straight-through `geometry_gate` tasarımının training objective ile deploy
   davranışı arasında uyumsuzluk üretip üretmediğini değerlendir.
4. Final `geometry_gate_mean=0.0091` değerinin kök nedenini incele: güvenli
   non-degradation loss'u mu, gate regularization mı, proposal-path loss ölçeği
   mi, teacher prior mı, canonical ownership mı, yoksa gerçekten student
   geometry'sinin source'tan kötü olması mı?
5. Model yalnız 64,000 görüntü exposure, yani 0.72 official epoch gördü.
   `05_TRAIN_METRICS.jsonl` üzerinden trajectory'yi analiz et. Sonucun sırf
   undertraining olduğu iddiasını kanıtla veya reddet; kör longer-training
   önerme.
6. Proposal-path target/ranking, owned-target assignment, canonical ordering,
   direct row distribution, soft strip IoU, vertical path ve four-slot inter
   mekanizmalarının birbirleriyle uyumunu denetle.
7. Evaluation pipeline'ını denetle: source için four-slot threshold 0.0,
   student için exact V7 activity ile existence threshold 0.5, NMS kapalı,
   count exact. Bunun adil paired karşılaştırma olup olmadığını söyle.
8. Correct-image ve cross-clip-wrong-image sonucunu yorumla. `.75`te +82 TP
   causal avantaj varken `.50`te yalnız +3 TP olmasının en olası açıklamasını
   ver.
9. Şu dört kök neden arasında ağırlıklı hüküm ver:
   - yetersiz training horizon,
   - optimization/loss-scale çatışması,
   - geometry-gate parameterization hatası,
   - mimarinin/representation'ın yetersizliği.
10. Yalnız **bir** sonraki aksiyon seç. Aksiyon küçük ve nedensel bir ayrım
    testi de olabilir, tam yeni training de olabilir; fakat mevcut kanıt bunu
    haklı çıkarmalı. Test setini açma ve post-hoc threshold/checkpoint seçme.
11. 81+ ihtimalini sugarcoating olmadan değerlendir. Oracle headroom ile
    deployable öğrenilebilirliği kesin biçimde ayır.

Yanıt formatı:

- Net hüküm
- Kanıtlananlar
- Falsified hipotezler
- Exact failure chain
- Kod/kontrat hataları veya confound'lar
- Tek sonraki deney ve predeclared PASS/STOP koşulu
- 81+ için dürüst olasılık değerlendirmesi

Checkpoint ve 29,025 prediction text dosyası upload sınırı nedeniyle pakete
konmadı. SHA/provenance `04_PACKAGE_MANIFEST.json` ve ham raporlarda mevcuttur.
Eksik checkpoint olmadan verilemeyecek bir hüküm varsa bunu açıkça belirt;
mevcut sayıları veya `passed` alanlarını kör biçimde kabul etme.
"""


def main() -> None:
    args = parse_args()
    destination = Path(args.destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("V23 Sol package destination must be absent or empty")
    destination.mkdir(parents=True, exist_ok=True)

    training_source = (
        PROJECT_ROOT
        / "outputs/diagnostics/v23_ordered_slot_cost_volume_gate/training_report.json"
    )
    metrics_source = (
        PROJECT_ROOT
        / "outputs/diagnostics/v23_ordered_slot_cost_volume_gate/train_metrics.jsonl"
    )
    official_source = (
        PROJECT_ROOT
        / "outputs/diagnostics/v23_ordered_slot_cost_volume_official_val/official_val_report.json"
    )
    training = json.loads(training_source.read_text(encoding="utf-8"))
    official = json.loads(official_source.read_text(encoding="utf-8"))

    files: list[tuple[str, str]] = []

    def record(name: str, origin: str) -> Path:
        files.append((name, origin))
        return destination / name

    record("00_SOL_PRO_PROMPT.md", "generated review prompt").write_text(
        _prompt(), encoding="utf-8"
    )
    record("01_V23_REVIEW_REPORT.md", "generated from raw V23 reports").write_text(
        _executive_report(training, official), encoding="utf-8"
    )
    _copy(
        official_source,
        record("02_OFFICIAL_VAL_REPORT.json", str(official_source.resolve())),
    )
    _copy(
        training_source,
        record("03_TRAINING_REPORT.json", str(training_source.resolve())),
    )

    # Manifest is populated after the remaining fourteen files exist.
    manifest_path = record("04_PACKAGE_MANIFEST.json", "generated package manifest")

    _copy(
        metrics_source,
        record("05_TRAIN_METRICS.jsonl", str(metrics_source.resolve())),
    )
    config_source = (
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_v23_ordered_slot_cost_volume_gate.yaml"
    )
    resolved = load_config(config_source)
    record("06_RESOLVED_V23_CONFIG.yaml", str(config_source.resolve())).write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    direct_copies = (
        (
            "07_DYNLANESEQ_V23_WRAPPER.py",
            PROJECT_ROOT / "dynlaneseq_eg/modeling/dynlaneseq_v23.py",
        ),
        (
            "08_V23_ORDERED_COST_VOLUME.py",
            PROJECT_ROOT
            / "dynlaneseq_eg/modeling/v23_ordered_slot_cost_volume.py",
        ),
        (
            "09_V22_ENCODER_DEFINITIONS.py",
            PROJECT_ROOT / "dynlaneseq_eg/modeling/v22_lane_field.py",
        ),
        (
            "10_TRAIN_V23.py",
            PROJECT_ROOT
            / "dynlaneseq_eg/tools/train_v23_ordered_slot_cost_volume.py",
        ),
        (
            "11_EVALUATE_V23_OFFICIAL.py",
            PROJECT_ROOT / "dynlaneseq_eg/tools/evaluate_v23_official.py",
        ),
        (
            "12_V23_OFFICIAL_PROTOCOL.py",
            PROJECT_ROOT / "dynlaneseq_eg/tools/v23_official_protocol.py",
        ),
    )
    for name, source in direct_copies:
        _copy(source, record(name, str(source.resolve())))

    evaluation_sources = [
        PROJECT_ROOT / "dynlaneseq_eg/evaluation/culane_writer.py",
        PROJECT_ROOT / "dynlaneseq_eg/evaluation/postprocess.py",
        PROJECT_ROOT / "dynlaneseq_eg/evaluation/culane_metric.py",
    ]
    _bundle(
        record(
            "13_EVALUATION_PIPELINE_BUNDLE.py.txt",
            ", ".join(str(path.resolve()) for path in evaluation_sources),
        ),
        evaluation_sources,
    )
    test_sources = [
        PROJECT_ROOT
        / "dynlaneseq_eg/tests/test_v23_ordered_slot_cost_volume.py",
        PROJECT_ROOT / "dynlaneseq_eg/tests/test_evaluate_v23_official.py",
    ]
    _bundle(
        record(
            "14_V23_TESTS_BUNDLE.py.txt",
            ", ".join(str(path.resolve()) for path in test_sources),
        ),
        test_sources,
    )

    if len(files) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            f"V23 Sol package must contain exactly {EXPECTED_FILE_COUNT} files, "
            f"planned {len(files)}"
        )
    git_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest_files = []
    for name, origin in files:
        if name == manifest_path.name:
            continue
        path = destination / name
        manifest_files.append(
            {
                "name": name,
                "origin": origin,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "experiment": "V23 ordered slot-conditioned lane cost volume",
        "git_commit": git_commit,
        "git_branch": subprocess.check_output(
            ("git", "branch", "--show-current"), cwd=PROJECT_ROOT, text=True
        ).strip(),
        "package_file_limit": EXPECTED_FILE_COUNT,
        "package_file_count": EXPECTED_FILE_COUNT,
        "files_excluding_manifest": manifest_files,
        "endpoint_checkpoint": {
            "copied": False,
            "reason": "391.8 MB checkpoint omitted from 15-file review upload",
            "remote_path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "iteration": training["iteration"],
        },
        "prediction_files": {
            "copied": False,
            "reason": "29,025 generated line files omitted; aggregate and paired facts are in raw report",
        },
        "official_population": {
            "train_rows": training["official_train_population_contract"][
                "observed_nonempty_rows"
            ],
            "val_rows": official["official_validation_population_contract"][
                "observed_nonempty_rows"
            ],
            "test_used": False,
        },
        "formal_gate_passed": official["gate"]["passed"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    actual = sorted(path for path in destination.iterdir() if path.is_file())
    if len(actual) != EXPECTED_FILE_COUNT:
        raise RuntimeError(
            f"V23 Sol package contains {len(actual)} files, expected {EXPECTED_FILE_COUNT}"
        )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "files": len(actual),
                "total_bytes": sum(path.stat().st_size for path in actual),
                "prompt": str(destination / "00_SOL_PRO_PROMPT.md"),
                "report": str(destination / "01_V23_REVIEW_REPORT.md"),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
