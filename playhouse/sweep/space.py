import math
from typing import Literal

type Scale = float | Literal["auto"]


class Space:
    """Base class for hyperparameter spaces with normalization to [-1, 1]."""

    min: float
    max: float
    scale: float
    norm_min: float
    norm_max: float
    norm_mean: float
    is_integer: bool

    def __init__(
        self, min: float, max: float, scale: float, is_integer: bool = False
    ) -> None:
        self.min = min
        self.max = max
        self.scale = scale
        self.is_integer = is_integer
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        # Since min/max are normalized to [-1, 1], use 0 as the mean
        self.norm_mean = 0.0

    def normalize(self, value: float) -> float:
        """Map value from [min, max] to [-1, 1]"""
        raise NotImplementedError()

    def unnormalize(self, value: float) -> float | int:
        """Map value from [-1, 1] to [min, max]"""
        raise NotImplementedError()


class Uniform(Space):
    """Uniform distribution space"""

    def __init__(
        self, min: float, max: float, scale: Scale, is_integer: bool = False
    ) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (value - self.min) / (self.max - self.min)
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float | int:
        zero_one = (value + 1) / 2
        result = zero_one * (self.max - self.min) + self.min
        return round(result) if self.is_integer else result


class Pow2(Space):
    """Power-of-2 distribution space (e.g., for batch sizes, hidden dims)."""

    def __init__(self, min: float, max: float, scale: Scale) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer=True)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(value, 2) - math.log(self.min, 2)) / (
            math.log(self.max, 2) - math.log(self.min, 2)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> int:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(self.max, 2) - math.log(self.min, 2)
        ) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2**rounded


class Log(Space):
    """Logarithmic (log-uniform) distribution space."""

    base: int = 10

    def __init__(
        self, min: float, max: float, scale: Scale, is_integer: bool = False
    ) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer)

    def normalize(self, value: float) -> float:
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base)) / (
            math.log(self.max, self.base) - math.log(self.min, self.base)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float | int:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(self.max, self.base) - math.log(self.min, self.base)
        ) + math.log(self.min, self.base)
        result = self.base**log_spaced
        return round(result) if self.is_integer else result


class Logit(Space):
    """Logit distribution space for values approaching 1 (e.g., discount factors)."""

    base: int = 10

    def __init__(self, min: float, max: float, scale: Scale) -> None:
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, is_integer=False)

    def normalize(self, value: float) -> float:
        zero_one = (
            math.log(1 - value, self.base) - math.log(1 - self.min, self.base)
        ) / (math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base))
        return 2 * zero_one - 1

    def unnormalize(self, value: float) -> float:
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (
            math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)
        ) + math.log(1 - self.min, self.base)
        return 1 - self.base**log_spaced
