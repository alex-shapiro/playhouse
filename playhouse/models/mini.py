import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from playhouse import pytorch


class MiniModel(nn.Module):
    """Mini PyTorch policy model. Flattens obs and applies a linear layer"""

    def __init__(self, env: gym.Env, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        obs_space = env.observation_space
        action_space = env.action_space

        # Encoder layer
        if isinstance(obs_space, gym.spaces.Box):
            self.encoder = BoxEncoder(obs_space, hidden_size)
        elif isinstance(obs_space, gym.spaces.Dict):
            self.encoder = DictEncoder(obs_space, hidden_size)
        else:
            raise NotImplementedError(f"model cannot encode from {obs_space}")

        # Decoder layer
        if isinstance(action_space, gym.spaces.Box):
            self.decoder = BoxDecoder(action_space, hidden_size)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.decoder = DiscreteDecoder(action_space, hidden_size)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.decoder = MultiDiscreteDecoder(action_space, hidden_size)
        else:
            raise NotImplementedError(f"model cannot decode into {action_space}")

        # Value layer
        self.value = pytorch.init_linear(nn.Linear(hidden_size, 1), std=1)

    def forward(self, obs: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(obs)
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values


class BoxEncoder(nn.Module):
    def __init__(self, obs_space: gym.spaces.Box, hidden_size: int):
        obs_size = int(np.prod(tuple(obs_space.shape)))  # pyright: ignore[reportArgumentType]
        self.encoder = nn.Sequential(
            pytorch.init_linear(nn.Linear(obs_size, hidden_size)),
            nn.GELU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)


class DictEncoder(nn.Module):
    def __init__(self, obs_space: gym.spaces.Dict, hidden_size: int):
        self.dtype = obs_space.dtype
        obs_size = int(sum(np.prod(tuple(obs_space[v].shape)) for v in obs_space))  # pyright: ignore[reportArgumentType]
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)


class BoxDecoder(nn.Module):
    def __init__(self, action_space: gym.spaces.Box, hidden_size: int):
        super().__init__()
        d0 = action_space.shape[0]
        self.mean = pytorch.init_linear(nn.Linear(hidden_size, d0), std=0.01)
        self.logstd = nn.Parameter(torch.zeros(1, d0))

    def forward(self, hidden: Tensor) -> Normal:
        mean = self.mean(hidden)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        return Normal(mean, std)


class DiscreteDecoder(nn.Module):
    def __init__(self, action_space: gym.spaces.Discrete, hidden_size: int):
        n = int(action_space.n)
        self.decoder = pytorch.init_linear(nn.Linear(hidden_size, n), std=0.01)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.decoder(hidden)


class MultiDiscreteDecoder(nn.Module):
    def __init__(self, action_space: gym.spaces.MultiDiscrete, hidden_size: int):
        super().__init__()
        self.action_nvec = tuple(action_space.nvec)
        num_atns = sum(self.action_nvec)
        self.decoder = pytorch.init_linear(nn.Linear(hidden_size, num_atns), std=0.01)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.decoder(hidden).split(self.action_nvec, dim=1)
