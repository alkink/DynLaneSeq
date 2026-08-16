from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test patched CondLSTR on one official empty-label and one non-empty CULane frame."
    )
    parser.add_argument("--condlstr-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def _shape(value: Any) -> list[int]:
    return [int(size) for size in value.shape]


def main() -> None:
    args = parse_args()
    condlstr_root = Path(args.condlstr_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sys.path.insert(0, str(condlstr_root))

    from data.datasets.lane.culane.culane import CULaneDataset  # noqa: PLC0415
    from data.transforms.lane.culane import culane_transforms  # noqa: PLC0415

    probe: Dict[str, Any] = {
        "condlstr_root": str(condlstr_root),
        "dataset_root": str(dataset_root),
        "version": args.version,
    }
    for training in (False, True):
        transform, collate = culane_transforms(train=training, version=args.version)
        dataset = CULaneDataset(
            root=str(dataset_root),
            split="train",
            transform=transform,
            version=args.version,
        )
        empty_index = next(index for index, info in enumerate(dataset.data_infos) if not info["lane_points"])
        lane_index = next(index for index, info in enumerate(dataset.data_infos) if info["lane_points"])
        empty_sample = dataset[empty_index]
        lane_sample = dataset[lane_index]

        empty_mask = empty_sample["img_mask"]
        empty_attrs = empty_sample["lane_attris"]
        if not isinstance(empty_mask, torch.Tensor) or _shape(empty_mask) != [0, 640, 1600]:
            raise RuntimeError(f"Unexpected empty-lane mask: {type(empty_mask)} {_shape(empty_mask)}")
        if not isinstance(empty_attrs, torch.Tensor) or int(empty_attrs.numel()) != 0:
            raise RuntimeError("Empty-label frame did not preserve an empty lane-attribute tensor")
        if int(lane_sample["img_mask"].shape[0]) <= 0:
            raise RuntimeError("Non-empty frame lost all lane masks")

        batch = collate([empty_sample, lane_sample])
        if not isinstance(batch["img_mask"], list) or len(batch["img_mask"]) != 2:
            raise RuntimeError("Instance masks must remain a two-image list after collation")
        if int(torch.count_nonzero(batch["img_mask"][0]).item()) != 0:
            raise RuntimeError("Padding introduced a false lane into the empty-label frame")

        probe["train_transform" if training else "val_transform"] = {
            "dataset_count": len(dataset),
            "empty_index": empty_index,
            "empty_mask_shape": _shape(empty_mask),
            "lane_index": lane_index,
            "lane_mask_shape": _shape(lane_sample["img_mask"]),
            "batch_mask_shapes": [_shape(mask) for mask in batch["img_mask"]],
        }

    print(json.dumps(probe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
