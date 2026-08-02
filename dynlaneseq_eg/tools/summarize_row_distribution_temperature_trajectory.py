from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize frozen row-logit temperature sweeps across checkpoints."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _temperature(name: str) -> float | None:
    prefix = "expected_t"
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix) :])
    except ValueError:
        return None


def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        final_layer = str(payload["final_layer"])
        layer = payload["decoder_layers"][final_layer]
        metrics = layer["decode_metrics"]
        temperatures = []
        for name, values in metrics.items():
            temperature = _temperature(name)
            if temperature is None:
                continue
            temperatures.append(
                {
                    "temperature": temperature,
                    "raw_recall_050": float(values["raw_recall@0.50"]),
                    "raw_recall_070": float(values["raw_recall@0.70"]),
                    "assigned_row_mae_px": float(values["assigned_row_mae_px"]),
                }
            )
        temperatures.sort(key=lambda item: item["temperature"])
        if not temperatures:
            raise ValueError("temperature sweep produced no expected_t decoders")
        baseline = next(
            (
                item
                for item in temperatures
                if abs(float(item["temperature"]) - 1.0) <= 1e-12
            ),
            None,
        )
        if baseline is None:
            raise ValueError("temperature sweep must include temperature 1.0")
        best_050 = max(temperatures, key=lambda item: item["raw_recall_050"])
        best_070 = max(temperatures, key=lambda item: item["raw_recall_070"])
        gain_050 = float(best_050["raw_recall_050"]) - float(
            baseline["raw_recall_050"]
        )
        if float(baseline["raw_recall_050"]) >= 0.50:
            verdict = "geometry_is_usable_at_native_temperature"
        elif gain_050 >= 0.10 and float(best_050["raw_recall_050"]) >= 0.25:
            verdict = "logit_temperature_is_a_material_bottleneck"
        else:
            verdict = "temperature_rescaling_does_not_recover_geometry"
        rows.append(
            {
                "iteration": int(payload["checkpoint_iteration"]),
                "checkpoint": payload["checkpoint"],
                "images": int(payload["images"]),
                "final_layer": final_layer,
                "baseline_temperature_1": baseline,
                "best_temperature_050": best_050,
                "best_temperature_070": best_070,
                "best_recall_gain_050": gain_050,
                "temperatures": temperatures,
                "verdict": verdict,
            }
        )
    rows.sort(key=lambda item: item["iteration"])
    if len({row["iteration"] for row in rows}) != len(rows):
        raise ValueError("duplicate checkpoint iterations")
    return {
        "diagnostic_only": True,
        "warning": (
            "Temperature decoding changes no weights and is not a deployable "
            "model selection result. It separates logit sharpness from lost "
            "candidate geometry."
        ),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    result = summarize(payloads)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()
