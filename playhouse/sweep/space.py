import math
from typing import Literal

type Scale = float | Literal["auto"]


class Space[T: int | float]:
    def __init__(self, min: T, max: T, scale: Scale):
        self.min = min
        self.max = max
        self.scale = "auto"
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = 0
        self.should_round = T is int

    def normalize(self, value: T) -> T:
        raise NotImplementedError()

    def unnormalize(self, value: T) -> T:
        raise NotImplementedError()


class Linear[T: int | float](Space[T]):
    def __init__(self, min: T, max: T, scale: Scale):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale)

    def normalize(self, value: T) -> T:
        zero_one = (value + 1) / 2
        # value = zero_one * (self.max - self.min) + self.min
        # return round(value) if T is int else value

    def unnormalize(self, value: T) -> T:
        zero_one = (value - self.min) / (self.max - self.min)
        # return 2* zero_one - 1


class Pow2(Space[float]):
    def __init__(self, min: float, max: float, scale: Scale):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(value, 2) - math.log(self.min, 2)) / (
            math.log(self.max, 2) - math.log(self.min, 2)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(self.max, 2) - math.log(self.min, 2)
        ) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2**rounded


class Log(Space[float]):
    base: int = 10

    def __init__(self, min: float, max: float, scale: Scale):
        super().__init__(min, max, scale)


class Logit(Space[float]):
    def __init__(self, min: float, max: float, scale: Scale):
        super().__init__(min, max, scale)
