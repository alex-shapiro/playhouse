# types are manually generated
# switch to autogeneration once pyo3 supports it

from typing import Any

import numpy as np
import numpy.typing as npt

class TetrisEnv:
    """Wrapper for a single Tetris environment with Python-facing API."""

    def __init__(
        self,
        observations: npt.NDArray[np.float32],
        actions: npt.NDArray[np.uint8],
        rewards: npt.NDArray[np.float32],
        terminals: npt.NDArray[np.uint8],
        truncations: npt.NDArray[np.uint8],
        seed: int,
        n_rows: int = 20,
        n_cols: int = 10,
        use_deck_obs: bool = True,
        n_noise_obs: int = 0,
        n_init_garbage: int = 0,
    ) -> None:
        """
        Initialize a Tetris environment.

        Args:
            observations: NumPy array for observation data (must be C-contiguous)
            actions: NumPy array for action data (must be C-contiguous)
            rewards: NumPy array for reward data (must be C-contiguous)
            terminals: NumPy array for terminal flags (must be C-contiguous)
            truncations: NumPy array for truncation flags (must be C-contiguous)
            seed: Random seed
            n_rows: Number of rows in the Tetris grid (default: 20)
            n_cols: Number of columns in the Tetris grid (default: 10)
            use_deck_obs: Whether to include deck observations (default: True)
            n_noise_obs: Number of noise observations (default: 0)
            n_init_garbage: Initial garbage lines (default: 0)
        """
        ...

    def reset(self, seed: int) -> None:
        """Reset the environment to initial state."""
        ...

    def step(self) -> None:
        """Execute one step of the environment."""
        ...

class VecTetrisEnv:
    """Vectorized environment wrapper for multiple Tetris environments."""

    def __init__(
        self,
        observations: Any,
        actions: Any,
        rewards: npt.NDArray[np.float32],
        terminals: npt.NDArray[np.uint8],
        truncations: npt.NDArray[np.uint8],
        num_envs: int,
        seed: int,
        n_rows: int = 20,
        n_cols: int = 10,
        use_deck_obs: bool = True,
        n_noise_obs: int = 0,
        n_init_garbage: int = 0,
    ) -> None:
        """
        Initialize vectorized Tetris environments.

        Args:
            observations: 2D array-like of observation arrays
            actions: 2D array-like of action arrays
            rewards: 1D NumPy array for rewards (shape: num_envs)
            terminals: 1D NumPy array for terminal flags (shape: num_envs)
            truncations: 1D NumPy array for truncation flags (shape: num_envs)
            num_envs: Number of parallel environments
            seed: Random seed
            n_rows: Number of rows in the Tetris grid (default: 20)
            n_cols: Number of columns in the Tetris grid (default: 10)
            use_deck_obs: Whether to include deck observations (default: True)
            n_noise_obs: Number of noise observations (default: 0)
            n_init_garbage: Initial garbage lines (default: 0)
        """
        ...

    def reset(self, seed: int) -> None:
        """Reset all environments to initial state."""
        ...

    def step(self) -> None:
        """Execute one step in all environments (parallel)."""
        ...

    def log(self) -> dict[str, float]:
        """
        Collect and aggregate logs from all environments.

        Returns:
            Dictionary containing averaged metrics:
                - score: Average score
                - perf: Average performance metric
                - ep_length: Average episode length
                - ep_return: Average episode return
                - lines_deleted: Average lines deleted
                - avg_combo: Average combo length
                - atn_frac_soft_drop: Fraction of soft drop actions
                - atn_frac_hard_drop: Fraction of hard drop actions
                - atn_frac_rotate: Fraction of rotate actions
                - atn_frac_hold: Fraction of hold actions
                - game_level: Average game level
                - ticks_per_line: Average ticks per line
                - n: Number of episodes completed
        """
        ...

    def render(self, env_id: int) -> None:
        """
        Render a specific environment (not implemented).

        Args:
            env_id: Index of the environment to render

        Raises:
            IndexError: If env_id is out of range
        """
        ...
