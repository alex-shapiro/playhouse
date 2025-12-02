from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class Environment(Protocol):
    """Protocol for vectorized environments.

    This defines the interface that all playhouse environments must implement.
    Similar to gym.vector.VectorEnv but as a Protocol for static type checking.
    """

    num_envs: int
    single_observation_space: gym.spaces.Space[Any]
    single_action_space: gym.spaces.Space[Any]
    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            Tuple of (observations, info).
        """
        ...

    def step(
        self,
        actions: NDArray[np.integer[Any]],
    ) -> tuple[
        NDArray[np.floating[Any]],  # observations
        NDArray[np.floating[Any]],  # rewards
        NDArray[np.bool_],  # terminated
        NDArray[np.bool_],  # truncated
        dict[str, Any],  # info
    ]:
        """Step all environments.

        Args:
            actions: Actions for each environment.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info).
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...
