from typing import Literal
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from playhouse import pytorch


class Minimal(nn.Module):
    """
    Minimal PyTorch policy. Flattens obs and applies a linear layer.

    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    """

    obs_type: Literal["Box", "Dict"]
    act_type: Literal["Discrete", "MultiDiscrete", "Box"]

    def __init__(self, env: gym.Env, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        obs_space = env.observation_space
        action_space = env.action_space

        self.is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)

        # Encoder
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_type = "Box"
            obs_size = int(np.prod(tuple(obs_space.shape)))  # pyright: ignore[reportArgumentType]
            self.encoder = nn.Sequential(
                pytorch.init_linear(nn.Linear(obs_size, hidden_size)),
                nn.GELU(),
            )
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_type = "Dict"
            self.dtype = obs_space.dtype
            obs_size = int(sum(np.prod(tuple(obs_space[v].shape)) for v in obs_space))  # pyright: ignore[reportArgumentType]
            self.encoder = nn.Linear(obs_size, self.hidden_size)
        else:

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
            self.decoder_mean = pytorch.init_linear(
                layer=nn.Linear(hidden_size, action_space.shape[0]),
                std=0.01,
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, action_space.shape[0]))
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.is_multidiscrete = True
            self.action_nvec = tuple(action_space.shape)
            self.decoder = nn.Sequential(
                pytorch.init_linear(
                    layer=nn.Linear(hidden_size, action_space.shape[0]),
                    std=0.01,
                )
            )

    def encode_observations(self, observations: Tensor, state) -> Tensor:
        """
        Encodes a batch of obserations into hidden states.
        Assumes no time dimension.
        """
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations =
