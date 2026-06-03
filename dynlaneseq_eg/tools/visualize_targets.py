from __future__ import annotations

import argparse
from pathlib import Path

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.factory import build_dataset
from dynlaneseq_eg.data.visualization import save_target_visualization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/target_vis")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, split=args.split, training=False)
    out_dir = Path(args.output_dir)
    for idx in range(min(args.num, len(dataset))):
        sample = dataset[idx]
        save_target_visualization(sample, out_dir / f"{idx:05d}.jpg")
    print(f"saved {min(args.num, len(dataset))} target visualizations to {out_dir}")


if __name__ == "__main__":
    main()

