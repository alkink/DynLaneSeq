from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .summarize_v4_1_score_gate import _flatten


def summarize(root: Path, arms: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"diagnostic_only": True, "arms": {}}
    for arm in arms:
        rows = []
        for path in sorted((root / arm).glob("iter_*.json")):
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            row = _flatten(report)
            row["iteration"] = int(path.stem.removeprefix("iter_"))
            mmr = report.get("methods", {}).get("mmr_sigma20_penalty0p5", {})
            mmr_050 = mmr.get("0.50", {}).get("f1")
            row["mmr_f1_050"] = None if mmr_050 is None else float(mmr_050)
            row["mmr_gain_050"] = (
                None
                if mmr_050 is None
                else float(mmr_050) - float(row["f1_050"])
            )
            rows.append(row)
        if not rows:
            raise ValueError(f"no trajectory reports found for {arm!r} under {root}")
        result["arms"][arm] = rows
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V4.2 score trajectories.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(Path(args.root), list(args.arms))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output_json: {output}")


if __name__ == "__main__":
    main()

