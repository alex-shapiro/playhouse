import gymnasium as gym
import numpy as np
import torch.nn as nn

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

    def __init__(self, env: gym.Env, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)
        self.is_dict_obs = isinstance(env.observation_space, gym.spaces.Dict)

        if isinstance(env.observation_space, gym.spaces.Dict):
            os = env.observation_space
            obs_size = int(sum(np.prod(os[v].shape) for v in os))

        self.encoder = nn.Sequential(
            pytorch.init_linear(nn.Linear(num_obs, hidden_size)),
            nn.GELU(),
        )
