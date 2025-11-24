from dataclasses import dataclass

from torch._prims_common import DeviceLikeType


@dataclass
class SweepConfig:
    device: DeviceLikeType
    prune_pareto: bool = True
    max_suggestion_cost: int = 3600
    # TODO
