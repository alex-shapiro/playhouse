from typing import Any, Literal

import gymnasium as gym
import numpy as np
import tetris_rust
from numpy.typing import NDArray

from playhouse.environments import Environment


class Tetris(Environment):
    """Vectorized Tetris environment using tetris_rust backend."""

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        num_envs: int = 1,
        n_cols: int = 10,
        n_rows: int = 20,
        use_deck_obs: bool = True,
        n_noise_obs: int = 10,
        n_init_garbage: int = 4,
        render_mode: Literal["human"] | None = None,
        log_interval: int = 32,
        seed: int = 0,
    ) -> None:
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_envs = num_envs

        # Observation and action spaces (single env, used for policy network sizing)
        obs_size = n_cols * n_rows + 6 + 7 * 4 + n_noise_obs
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(7)

        # Initialize numpy buffers for vectorized environments
        self._observations = np.zeros((num_envs, obs_size), dtype=np.float32)
        self._actions = np.zeros(num_envs, dtype=np.uint8)
        self._rewards = np.zeros(num_envs, dtype=np.float32)
        self._terminals = np.zeros(num_envs, dtype=np.uint8)
        self._truncations = np.zeros(num_envs, dtype=np.uint8)

        # Create list views for rust backend (expects list of arrays)
        obs_list = [self._observations[i] for i in range(num_envs)]

        self._envs = tetris_rust.VecTetrisEnv(
            observations=obs_list,
            actions=self._actions,
            rewards=self._rewards,
            terminals=self._terminals,
            truncations=self._truncations,
            num_envs=num_envs,
            seed=seed,
            n_cols=n_cols,
            n_rows=n_rows,
            use_deck_obs=use_deck_obs,
            n_noise_obs=n_noise_obs,
            n_init_garbage=n_init_garbage,
        )
        self._tick = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
        """Reset all environments."""
        self._envs.reset(seed if seed is not None else 0)
        self._tick = 0
        return self._observations, {}

    def step(
        self,
        actions: NDArray[np.integer[Any]],
    ) -> tuple[
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Step all environments."""
        self._actions[:] = actions
        self._tick += 1
        self._envs.step()

        info: dict[str, Any] = {}
        if self._tick % self.log_interval == 0:
            info.update(self._envs.log())

        return (
            self._observations,
            self._rewards,
            self._terminals.astype(np.bool_),
            self._truncations.astype(np.bool_),
            info,
        )

    def render(self) -> None:
        """Render the first environment."""
        self._envs.render(0)

    def close(self) -> None:
        """Clean up resources."""
        pass
