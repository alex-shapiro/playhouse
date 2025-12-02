from typing import Never

import gymnasium
import numpy as np
import tetris_rust
from numpy.typing import NDArray


class Tetris:
    def __init__(
        self,
        num_envs: int = 1,
        n_cols: int = 10,
        n_rows: int = 20,
        use_deck_obs: bool = True,
        n_noise_obs: int = 10,
        n_init_garbage: int = 4,
        render_mode: str | None = None,
        log_interval: int = 32,
        buf: None = None,
        seed: int = 0,
    ) -> None:
        self.obs_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(n_cols * n_rows + 6 + 7 * 4 + n_noise_obs,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_agents = num_envs

        self.n_cols = n_cols
        self.n_rows = n_rows

        # Initialize numpy buffers for vectorized environments
        obs_size = n_cols * n_rows + 6 + 7 * 4 + n_noise_obs
        # Create list of arrays for observations (one array per environment)
        self.observations = [
            np.zeros(obs_size, dtype=np.float32) for _ in range(num_envs)
        ]
        self.actions = np.zeros(num_envs, dtype=np.uint8)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=np.uint8)
        self.truncations = np.zeros(num_envs, dtype=np.uint8)
        self.envs = tetris_rust.VecTetrisEnv(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            truncations=self.truncations,
            num_envs=num_envs,
            seed=seed,
            n_cols=n_cols,
            n_rows=n_rows,
            use_deck_obs=use_deck_obs,
            n_noise_obs=n_noise_obs,
            n_init_garbage=n_init_garbage,
        )
        self.tick = 0

    def reset(self, seed: int = 0) -> tuple[list[NDArray[np.floating]], list[Never]]:
        self.envs.reset(seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions: NDArray[np.integer]):
        self.actions[:] = actions
        self.tick += 1
        self.envs.step()

        info = []
        if self.tick % self.log_interval == 0:
            info.append(self.envs.log())

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        self.envs.render(0)
