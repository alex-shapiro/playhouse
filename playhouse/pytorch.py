from typing import overload

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@overload
def init_layer(
    layer: nn.Linear,
    std: float = ...,
    bias_const: float = ...,
) -> nn.Linear: ...


@overload
def init_layer(
    layer: nn.Conv2d,
    std: float = ...,
    bias_const: float = ...,
) -> nn.Conv2d: ...


def init_layer(
    layer: nn.Linear | nn.Conv2d,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear | nn.Conv2d:
    """
    Initialize a linear layer as an orthogonal matrix (columns and rows are orthonormal):

    - Q^T Q = I
    - Each column has length `gain`
    - Each column is perpendicular to all other columns

    Orthogonal matrices preserve the magnitude of gradients during
    backpropagation, helping prevent vanishing or exploding gradients.

    ReLU zeros out ~half of activations, reducing variance by half each layer.
    Multiplying unit-sized layer activations by sqrt(2) doubles its variance (σ²), so post-ReLU variance stays consistent across layer.
    """
    nn.init.orthogonal_(tensor=layer.weight, gain=std)
    if layer.bias is not None:
        nn.init.constant_(tensor=layer.bias, val=bias_const)
    return layer


def log_prob(logits: Tensor, value: Tensor) -> Tensor:
    """Doc: TODO"""
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    return log_pmf.gather(-1, value).squeeze(-1)
