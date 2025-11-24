from dataclasses import dataclass

import gymnasium as gym
import torch.nn as nn
from torch import Tensor

from playhouse.models.mini_policy import MiniPolicy


@dataclass
class LSTMState:
    hidden: Tensor | None
    lstm_h: Tensor | None
    lstm_c: Tensor


class LSTMWrapper(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        policy: MiniPolicy,
        input_size: int = 128,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.policy = policy
        self.hidden_size = hidden_size
        # TODO: self.is_continuous

        # initialize biases with constant 0s
        # initialize weights with orthogonal values
        # exclude layer_norm biases and weights
        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        # create a LSTM for processing sequences
        # create a LSTM cell for processing individual inputs
        # force them to share 1st layer weights & update together during training
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.__getattribute__("weight_ih_l0")
        self.cell.weight_ih = self.lstm.__getattribute__("weight_hh_l0")
        self.cell.bias_ih = self.lstm.__getattribute__("bias_ih_l0")
        self.cell.bias_hh = self.lstm.__getattribute__("bias_hh_l0")

    def forward_eval(self, obs: Tensor, state: LSTMState) -> tuple[Tensor, Tensor]:
        """Forward impl for evaluation"""
        hidden = self.policy.encoder(obs, state=state)
        h = state.lstm_h
        c = state.lstm_c

        if h is None:
            lstm_state = None
        else:
            assert h.shape[0] == c.shape[0]
            assert h.shape[0] == obs.shape[0]
            lstm_state = (h, c)

        hidden, c = self.cell(hidden, lstm_state)
        state.hidden = hidden
        state.lstm_h = hidden
        state.lstm_c = c
        return self.policy.decoder(hidden)

    def forward(self, obs: Tensor, state: LSTMState) -> tuple[Tensor, Tensor]:
        """Forward impl for training"""
        lstm_h = state.lstm_h
        lstm_c = state.lstm_c
        obs_shape, space_shape = obs.shape, self.obs_shape
        (obs_n,)

        raise NotImplementedError()
