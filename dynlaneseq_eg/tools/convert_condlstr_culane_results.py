from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CondLSTR results.pkl to official CULane .lines.txt files.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def read_val_population(path: Path) -> List[str]:
    output: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = line.split()[0]
        output.append(value[1:] if value.startswith("/") else value)
    return output


def result_relative_path(result: Dict[str, Any], fallback: str) -> str:
    image_name = str(result.get("image_name", ""))
    if image_name.count(",") >= 2:
        return image_name.replace(",", "/")
    if image_name and "/" in image_name:
        return image_name.lstrip("/")
    return fallback


def main() -> None:
    args = parse_args()
    results_path = Path(args.results).expanduser().resolve()
    val_list = Path(args.val_list).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with results_path.open("rb") as handle:
        results = pickle.load(handle)
    population = read_val_population(val_list)
    if len(results) != len(population):
        raise RuntimeError(f"Result population mismatch: results={len(results)}, val={len(population)}")

    written = 0
    lane_count = 0
    path_mismatches: List[Dict[str, str]] = []
    for index, (result, expected_relative) in enumerate(zip(results, population)):
        relative = result_relative_path(result, expected_relative)
        if relative != expected_relative:
            path_mismatches.append({"index": str(index), "expected": expected_relative, "observed": relative})
        output_path = output_dir / str(Path(expected_relative).with_suffix(".lines.txt"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lanes = result.get("points", [])
        with output_path.open("w", encoding="utf-8") as handle:
            for lane in lanes:
                if len(lane) < 2:
                    continue
                values: List[str] = []
                for point in lane:
                    values.extend((f"{float(point[0]):.5f}", f"{float(point[1]):.5f}"))
                handle.write(" ".join(values) + "\n")
                lane_count += 1
        written += 1

    if path_mismatches:
        raise RuntimeError(f"CondLSTR output ordering/path mismatch on {len(path_mismatches)} images")
    manifest = {
        "results": str(results_path),
        "val_list": str(val_list),
        "output_dir": str(output_dir),
        "images": written,
        "lanes": lane_count,
        "path_mismatches": 0,
        "posthoc_score_or_nms_sweep": False,
    }
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else output_dir.parent / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
