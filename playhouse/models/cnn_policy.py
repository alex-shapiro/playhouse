from typing import Any

import gymnasium as gym
from torch import Tensor, nn

from playhouse.environments import Environment
from playhouse.pytorch import init_layer


class CNNPolicy(nn.Module):
    """
    CleanRL default NatureCNN policy used for Atari.
    Stack of 3 convolutions followed by a linear layer.
    Suggested framestack is 1 with LSTM, 4 without.
    """

    def __init__(
        self,
        env: Environment,
        framestack: int,
        flat_size: int,
        input_size: int = 512,
        hidden_size: int = 512,
        output_size: int = 512,
        channels_last: bool = False,
        downsample: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            init_layer(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            init_layer(nn.Conv2d(32, 64, 4, stride=4)),
            nn.ReLU(),
            init_layer(nn.Conv2d(64, 64, 3, stride=4)),
            nn.ReLU(),
            nn.Flatten(),
            init_layer(nn.LazyLinear(hidden_size)),
            nn.ReLU(),
        )
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete)
        self.actor_fn = init_layer(
            nn.Linear(hidden_size, int(action_space.n)), std=0.01
        )
        self.value_fn = init_layer(nn.Linear(output_size, 1), std=1)

    def forward(
        self, obs: Tensor, state: dict[str, Any] | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.decode(self.encode(obs))

    def forward_eval(self, obs: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Forward pass for evaluation (same as forward for non-RNN policies)."""
        return self.forward(obs, state)

    def encode(self, obs: Tensor) -> Tensor:
        if self.channels_last:
            obs = obs.permute(0, 3, 1, 2)
        if self.downsample > 1:
            obs = obs[:, :, :: self.downsample, :: self.downsample]
        return self.network(obs.float() / 255.0)

    def decode(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        action = self.actor_fn(hidden)
        value = self.value_fn(hidden)
        return action, value
