from dataclasses import dataclass

from torch._prims_common import DeviceLikeType
from typing_extensions import Literal

from playhouse.sweep import space

type SweepMetric = Literal["score"]

type SweepGoal = Literal["maximize", "minimize"]

type ParamDistribution = Literal[
    "uniform",
    "int_uniform",
    "uniform_pow2",
    "log_normal",
    "logit_normal",
]

type ParamSpaceScale = Literal["auto"]


@dataclass
class SweepConfig:
    device: DeviceLikeType
    prune_pareto: bool = True
    max_suggestion_cost: int = 3600
    metric: SweepMetric = "score"
    goal: SweepGoal = "maximize"
    params: dict[str, "ParamSpaceConfig"] = {}
    downsample: int = 1

    def param_spaces(self) -> dict[str, space.Space[int | float]]:
        return {k: v.to_space() for k, v in self.params.items()}


@dataclass
class ParamSpaceConfig:
    distribution: ParamDistribution
    min: float
    max: float
    mean: float
    scale: ParamSpaceScale = "auto"

    def to_space(self) -> space.Space[int | float]:
        match self.distribution:
            case "uniform":
                return space.Linear(min=self.min, max=self.max, scale=self.scale)
            case "int_uniform":
                return space.Linear(min=self.min, max=self.max, scale=self.scale)
            case "uniform_pow2":
                return space.Pow2(min=self.min, max=self.max, scale=self.scale)
            case "log_normal":
                return space.Log(min=self.min, max=self.max, scale=self.scale)
            case "logit_normal":
                return space.Logit(min=self.min, max=self.max, scale=self.scale)
