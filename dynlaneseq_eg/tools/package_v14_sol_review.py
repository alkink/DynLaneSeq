from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the complete V14 review evidence into one flat Sol Pro "
            "handoff directory. Checkpoints and prediction caches are omitted."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flat_name(source: Path, origin: str) -> str:
    try:
        relative = source.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = Path(source.name)
    return f"{origin}__" + "__".join(relative.parts)


def _copy(
    source: Path,
    destination: Path,
    *,
    origin: str,
    manifest: list[dict[str, object]],
) -> None:
    if not source.is_file():
        return
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f"refusing an unexpectedly large Sol artifact: {source}")
    target = destination / _flat_name(source, origin)
    if target.exists():
        raise FileExistsError(f"flat V14 package name collision: {target.name}")
    shutil.copy2(source, target)
    manifest.append(
        {
            "packaged_name": target.name,
            "source_path": str(source.resolve()),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _config_chain(path: Path) -> list[Path]:
    """Return the exact inherited YAML source chain, leaf first."""

    chain: list[Path] = []
    current = path.resolve()
    visited: set[Path] = set()
    while current not in visited:
        if not current.is_file():
            raise FileNotFoundError(f"missing inherited config: {current}")
        visited.add(current)
        chain.append(current)
        base = ""
        for line in current.read_text(encoding="utf-8").splitlines():
            if line.startswith("_base_:"):
                base = line.split(":", 1)[1].strip().strip("'\"")
                break
        if not base:
            break
        current = (current.parent / base).resolve()
    return chain


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    if not (output_root / "v14_completion.json").is_file():
        raise FileNotFoundError(
            "V14 has no terminal completion contract; do not package a live run"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Sol package destination must be absent or empty so stale files "
            "cannot leak into the next review"
        )
    destination.mkdir(parents=True, exist_ok=True)

    stage_a_config = (
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_a_225k_to227k.yaml"
    )
    stage_b_config = (
        PROJECT_ROOT
        / "dynlaneseq_eg/configs/culane_s0_structured_query_dla34_v14_corrected_visual_first_stage_b_227k_to229k.yaml"
    )
    code_paths = [
        PROJECT_ROOT
        / "docs/experiments/V14_CORRECTED_VISUAL_FIRST_STAGE_AB_CONTRACT_2026-08-13.md",
        PROJECT_ROOT / "dynlaneseq_eg/modeling/four_slot_selection.py",
        PROJECT_ROOT / "dynlaneseq_eg/modeling/structured_queries.py",
        PROJECT_ROOT / "dynlaneseq_eg/losses/loss_s0.py",
        PROJECT_ROOT / "dynlaneseq_eg/factory.py",
        PROJECT_ROOT / "dynlaneseq_eg/config.py",
        PROJECT_ROOT / "dynlaneseq_eg/data/culane_dataset.py",
        PROJECT_ROOT / "dynlaneseq_eg/data/resume_safe.py",
        PROJECT_ROOT / "dynlaneseq_eg/data/transforms.py",
        PROJECT_ROOT / "dynlaneseq_eg/engine/checkpoint.py",
        PROJECT_ROOT / "dynlaneseq_eg/engine/frozen_training.py",
        PROJECT_ROOT / "dynlaneseq_eg/engine/train_one_epoch.py",
        PROJECT_ROOT / "dynlaneseq_eg/evaluation/culane_writer.py",
        PROJECT_ROOT / "dynlaneseq_eg/tools/analyze_v4_selection_coverage.py",
        PROJECT_ROOT / "dynlaneseq_eg/tools/build_cross_clip_derangement.py",
        PROJECT_ROOT / "dynlaneseq_eg/tools/build_v11_bridge_lists.py",
        PROJECT_ROOT / "dynlaneseq_eg/tools/evaluate_culane.py",
        PROJECT_ROOT / "dynlaneseq_eg/tools/train.py",
        PROJECT_ROOT / "scripts/run_culane_dla34_v14_corrected_visual_first_stage_ab.sh",
    ]
    code_paths.extend(_config_chain(stage_a_config))
    code_paths.extend(_config_chain(stage_b_config))
    code_paths.extend(
        path
        for path in sorted((PROJECT_ROOT / "dynlaneseq_eg").rglob("*v14*"))
        if path.suffix.lower() in {".py", ".yaml", ".md"}
    )
    code_paths = list(dict.fromkeys(path for path in code_paths if path.is_file()))

    result_paths = [
        path
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".json", ".md", ".yaml", ".txt", ".log"}
        and "predictions" not in path.parts
        and "cache" not in path.parts
    ]
    manifest: list[dict[str, object]] = []
    for path in code_paths:
        _copy(path, destination, origin="code", manifest=manifest)
    for path in result_paths:
        _copy(path, destination, origin="result", manifest=manifest)

    base = "477af8f"
    patch_path = destination / "code__V14_FULL_GIT_PATCH.diff"
    patch_path.write_text(
        subprocess.check_output(
            ("git", "diff", f"{base}..HEAD"),
            cwd=PROJECT_ROOT,
            text=True,
        ),
        encoding="utf-8",
    )
    manifest.append(
        {
            "packaged_name": patch_path.name,
            "source_path": f"git diff {base}..HEAD",
            "bytes": patch_path.stat().st_size,
            "sha256": _sha256(patch_path),
        }
    )

    completion = _read_json(output_root / "v14_completion.json")
    stage_a = _read_json(output_root / "v14_stage_a_summary.json")
    stage_b = _read_json(
        output_root / "stage_b" / "v14_stage_b_bridge_summary.json"
    )
    full_validation = _read_json(
        output_root
        / "stage_b"
        / "v14_stage_b_full_validation_summary.json"
    )
    prompt = f"""# GPT-5.6 Sol Pro adversarial V14 review

You are receiving the complete code/config/audit package for V14. Read the
experiment contract first, then independently recompute every headline number
from the raw JSON reports. Do not treat a script's `passed` field as evidence
without checking its inputs and arithmetic.

Terminal run state:

```json
{json.dumps(completion, indent=2, sort_keys=True)}
```

Stage-A summary:

```json
{json.dumps(stage_a, indent=2, sort_keys=True)}
```

Stage-B bridge summary (empty means Stage A correctly kept it closed):

```json
{json.dumps(stage_b, indent=2, sort_keys=True)}
```

Full-validation summary (empty means its predeclared gate kept it closed):

```json
{json.dumps(full_validation, indent=2, sort_keys=True)}
```

Required review:

1. Reconstruct the exact tensor, target, assignment, inference and gradient
   graph from code. Separate measured fact, code fact, inference and hypothesis.
2. Verify that association is genuinely visual-first, beta=0, target-free at
   inference, and unable to bypass P2 through U0.
3. Verify the deterministic cross-clip derangement and all P2 negative controls.
4. Audit the joint private-dustbin Sinkhorn prediction/target contract and all
   fixed-M7 assignment edge cases.
5. Recompute both domain gates. Check that no threshold, NMS, checkpoint or test
   selection occurred and that a failed gate truly prevented its next stage.
6. If Stage B ran, independently recompute official TP/FP/FN/F1, x/range
   factorials, correct-vs-wrong-P2 causality, duplicates and proposal-oracle
   invariance. Check zero-step parity and frozen Stage-A/V7 state.
7. Give the most likely remaining root cause, explicitly list falsified
   hypotheses, and decide exactly one next action. A new version or long
   training must not be authorized merely because seen-set capacity is high.
8. Give an honest engineering estimate for 80+ and 81+ only after the causal
   verdict, clearly distinguishing diagnostic headroom from deployable score.
9. List every missing artifact, provenance hole, implementation bug, leakage,
   confound or post-hoc decision you find. If the package is insufficient,
   return FAIL and name the minimum missing evidence.

V14 is terminal for this review: do not assume V15 has started. Long training
and test evaluation remain closed regardless of the local summary fields until
you issue a new, evidence-backed decision.
"""
    prompt_path = destination / "SOL_PRO_V14_REVIEW_PROMPT.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    manifest.append(
        {
            "packaged_name": prompt_path.name,
            "source_path": "generated from terminal V14 summaries",
            "bytes": prompt_path.stat().st_size,
            "sha256": _sha256(prompt_path),
        }
    )

    manifest_path = destination / "SOL_PRO_V14_PACKAGE_MANIFEST.json"
    manifest_payload = {
        "experiment": "V14 corrected visual-first Stage A/B",
        "source_output_root": str(output_root),
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, text=True
        ).strip(),
        "file_count_before_manifest": len(manifest),
        "files": manifest,
        "checkpoints_copied": False,
        "prediction_cache_copied": False,
        "test_set_used": False,
        "delete_this_copy_after_sol_response": True,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "files": len(manifest) + 1,
                "prompt": str(prompt_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
