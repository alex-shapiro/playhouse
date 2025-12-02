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
    metric: SweepMetric = "score"
    goal: SweepGoal = "maximize"
    params: dict[str, "ParamSpaceConfig"] = {}

    def param_spaces(self) -> dict[str, space.Space]:
        return {k: v.to_space() for k, v in self.params.items()}


@dataclass
class ParamSpaceConfig:
    distribution: ParamDistribution
    min: float
    max: float
    mean: float
    scale: ParamSpaceScale = "auto"

    def to_space(self) -> space.Space:
        match self.distribution:
            case "uniform":
                return space.Uniform(
                    min=self.min, max=self.max, scale=self.scale, is_integer=False
                )
            case "int_uniform":
                return space.Uniform(
                    min=self.min, max=self.max, scale=self.scale, is_integer=True
                )
            case "uniform_pow2":
                return space.Pow2(min=self.min, max=self.max, scale=self.scale)
            case "log_normal":
                return space.Log(min=self.min, max=self.max, scale=self.scale)
            case "logit_normal":
                return space.Logit(min=self.min, max=self.max, scale=self.scale)
