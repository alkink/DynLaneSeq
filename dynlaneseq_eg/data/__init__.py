from .lane_target_builder import LaneTargetBuilder

__all__ = ["CULaneDataset", "TuSimpleDataset", "CurveLanesDataset", "LaneTargetBuilder", "lane_collate"]


def __getattr__(name):
    if name == "CULaneDataset":
        from .culane_dataset import CULaneDataset

        return CULaneDataset
    if name == "TuSimpleDataset":
        from .tusimple_dataset import TuSimpleDataset

        return TuSimpleDataset
    if name == "CurveLanesDataset":
        from .curvelanes_dataset import CurveLanesDataset

        return CurveLanesDataset
    if name == "lane_collate":
        from .collate import lane_collate

        return lane_collate
    raise AttributeError(name)
