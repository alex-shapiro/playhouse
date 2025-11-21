from dataclasses import dataclass

import gymnasium as gym
import torch.nn as nn
from torch import Tensor


@dataclass
class LSTMState:
    h: Tensor
    c: Tensor


class LSTMWrapper(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
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
        raise NotImplementedError()

    def forward_train(self, obs: Tensor, state: LSTMState) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()
