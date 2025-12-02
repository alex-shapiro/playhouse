import numpy as np
from numpy.typing import NDArray

from playhouse.sweep.config import SweepConfig, SweepMetric
from playhouse.sweep.space import Space


class Hyperparameters:
    num: int
    metric: SweepMetric
    spaces: dict[str, Space[int | float]]
    optimize_direction: int
    means: NDArray[np.floating]
    mins: NDArray[np.floating]
    maxs: NDArray[np.floating]
    scales: NDArray[np.floating]

    def __init__(self, config: SweepConfig, verbose: bool = False) -> None:
        self.spaces = config.param_spaces()
        self.num = len(self.spaces)
        self.metric = config.metric
        match config.goal:
            case "maximize":
                self.optimize_direction = 1
            case "minimize":
                self.optimize_direction = -1

        self.means = np.array([e.norm_mean for e in self.spaces.values()])
        self.mins = np.array([e.norm_min for e in self.spaces.values()])
        self.maxs = np.array([e.norm_max for e in self.spaces.values()])
        self.scales = np.array([e.scale for e in self.spaces.values()])

    def sample(
        self, n: int, mu: NDArray[np.floating] | None = None, scale: float = 1.0
    ) -> NDArray[np.floating]:
        mu_arr = self.means if mu is None else mu
        if mu_arr.ndim == 1:
            mu_arr = np.expand_dims(mu_arr, axis=0)

        n_input, n_dim = mu_arr.shape
        scales = self.scales * scale
        mu_idxs = np.random.randint(0, n_input, n)
        samples = scales * (2 * np.random.rand(n, n_dim) - 1) + mu_arr[mu_idxs]
        return np.clip(samples, self.mins, self.maxs)

    def to_dict(self, sample: NDArray[np.floating]) -> dict[str, float | int]:
        idx = 0
        params: dict[str, float | int] = {}
        for name, space in self.spaces.items():
            params[name] = space.unnormalize(float(sample[idx]))
            idx += 1
        return params

    def from_dict(self, params: dict[str, float | int]) -> NDArray[np.floating]:
        """Convert a parameter dictionary to a normalized array.

        Args:
            params: Dictionary mapping parameter names to values.

        Returns:
            Normalized parameter array.
        """
        values: list[float] = []
        for key, space in self.spaces.items():
            if key not in params:
                raise KeyError(f"Missing hyperparameter {key}")
            val = params[key]
            normed = space.normalize(val)
            values.append(normed)
        return np.array(values)

    def index(self, name: str) -> int | None:
        keys = list(self.spaces.keys())
        return keys.index(name) if name in keys else None
