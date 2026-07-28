from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def _row(run: str, layer: str, source: str, result: dict[str, Any]) -> list[str]:
    metrics = result["metrics"]["all"]
    return [
        run,
        layer,
        source.upper(),
        str(result["test_rows"]),
        _percent(metrics["class_balanced_accuracy"]),
        _percent(metrics["class_direction_accuracy"]),
        f"{float(metrics['anchor_mae_px']):.2f}",
        f"{float(metrics['regression_corrected_mae_px']):.2f}",
        f"{float(metrics['regression_mae_gain_px']):+.2f}",
        _percent(metrics["regression_within_4px"]),
        _percent(metrics["regression_within_8px"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    headers = [
        "run",
        "layer",
        "source",
        "test rows",
        "bal acc %",
        "direction %",
        "anchor MAE",
        "corrected MAE",
        "MAE gain",
        "within4 %",
        "within8 %",
    ]
    rows = []
    combined: dict[str, Any] = {}
    for result_path in args.results:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        run = Path(result_path).stem
        combined[run] = {
            "checkpoint": payload["checkpoint"],
            "checkpoint_iteration": payload["checkpoint_iteration"],
            "layers": payload["layers"],
            "probe_results": payload["probe_results"],
        }
        for layer, layer_result in payload["probe_results"].items():
            for source in ("c2", "p2"):
                rows.append(_row(run, layer, source, layer_result[source]))

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        print(f"output_json: {output}")


if __name__ == "__main__":
    main()
