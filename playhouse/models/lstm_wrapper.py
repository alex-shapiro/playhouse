from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from playhouse.environments import Environment
from playhouse.models.cnn_policy import CNNPolicy
from playhouse.models.mini_policy import MiniPolicy


@dataclass
class LSTMState:
    hidden: Tensor | None
    lstm_h: Tensor | None
    lstm_c: Tensor


class LSTMWrapper(nn.Module):
    def __init__(
        self,
        env: Environment,
        policy: MiniPolicy | CNNPolicy,
        input_size: int = 128,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        obs_shape = env.observation_space.shape
        assert obs_shape is not None
        self.obs_shape = obs_shape

        act_shape = env.action_space.shape
        assert act_shape is not None
        self.act_shape = act_shape

        self.policy = policy

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
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.__getattribute__("weight_ih_l0")
        self.cell.weight_ih = self.lstm.__getattribute__("weight_hh_l0")
        self.cell.bias_ih = self.lstm.__getattribute__("bias_ih_l0")
        self.cell.bias_hh = self.lstm.__getattribute__("bias_hh_l0")

    def forward_eval(self, obs: Tensor, state: LSTMState) -> tuple[Tensor, Tensor]:
        """Forward impl for evaluation"""
        hidden = self.policy.encode(obs)
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
        return self.policy.decode(hidden)

    def forward(self, obs: Tensor, state: LSTMState) -> tuple[Tensor, Tensor]:
        """Forward impl for training"""
        lstm_h = state.lstm_h
        lstm_c = state.lstm_c

        assert self.obs_shape
        d_obs = len(obs.shape)
        d_obsspace = len(self.obs_shape)

        # b (batch dimension) is required
        # tt (timestamp dimension) is optional, default is 1
        if d_obs == d_obsspace + 1:
            assert obs.shape[1:] == self.obs_shape
            b = obs.shape[0]
            tt = 1
        elif d_obs == d_obsspace + 2:
            assert obs.shape[2:] == self.obs_shape
            b = obs.shape[0]
            tt = obs.shape[1]
        else:
            raise ValueError(f"invalid obs tensor shape: {obs.shape}")

        if lstm_h is None:
            lstm_state = None
        else:
            assert b == lstm_h.shape[1]
            assert b == lstm_c.shape[1]
            lstm_state = (lstm_h, lstm_c)

        obs = obs.reshape(b * tt, *self.obs_shape)
        hidden = self.policy.encode(obs)
        assert hidden.shape == (b * tt, self.input_size)
        hidden = hidden.reshape(b, tt, self.input_size)

        # transpose b, tt dimensions for LSTM
        hidden = hidden.transpose(0, 1)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(b * tt, self.hidden_size)
        logits, values = self.policy.decode(flat_hidden)
        values = values.reshape(b, tt)

        state.hidden = hidden
        state.lstm_h = lstm_h.detach()
        state.lstm_c = lstm_c.detach()

        return logits, values
