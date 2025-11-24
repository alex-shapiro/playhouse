import numpy as np
from numpy.typing import NDArray

from playhouse.sweep.config import SweepConfig, SweepMetric


class Hyperparameters:
    num: int
    metric: SweepMetric

    def __init__(self, config: SweepConfig, verbose: bool = False):
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

    def sample(self, n: int, mu: NDArray | None = None, scale: float = 1.0) -> NDArray:
        mu = self.means if mu is None else mu
        mu = np.expand_dims(mu, axis=0) if mu.shape == 1 else mu

        n_input, n_dim = mu.shape
        scales = self.scales * scale
        mu_idxs = np.random.randint(0, n_input, n)
        samples = scales * (2 * np.random.rand(n, n_dim) - 1) + mu[mu_idxs]
        return np.clip(samples, self.mins, self.maxs)

    def to_dict(self, sample: NDArray) -> dict[str, float | int]:
        idx = 0
        params = {}
        for name, space in self.spaces.items():
            params[name] = space.unnormalize(sample[idx])
            idx += 1
        return params
