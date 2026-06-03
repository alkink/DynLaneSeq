from .lane_target_builder import LaneTargetBuilder

__all__ = ["CULaneDataset", "LaneTargetBuilder", "lane_collate"]


def __getattr__(name):
    if name == "CULaneDataset":
        from .culane_dataset import CULaneDataset

        return CULaneDataset
    if name == "lane_collate":
        from .collate import lane_collate

        return lane_collate
    raise AttributeError(name)
