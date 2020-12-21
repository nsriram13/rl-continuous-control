# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(DEVICE)
    return x


# Source: https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py  # noqa: E501
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


# Source: https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/normalizer.py  # noqa: E501
class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip(
            (x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
            -self.clip,
            self.clip,
        )

    def state_dict(self):
        return {'mean': self.rms.mean, 'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# Source: https://github.com/ShangtongZhang/DeepRL/blob/932ea88082e0194126b87742bd4a28c4599aa1b8/deep_rl/network/network_utils.py#L23  # noqa: E501
# Fills the input Tensor with a (semi) orthogonal matrix, as described in Exact
# solutions to the nonlinear dynamics of learning in deep linear neural networks
# - Saxe, A. et al. (2013).
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


# Utility module to initialize a fully connected neural network
# Source: https://github.com/facebookresearch/ReAgent/blob/86b92279b635b38b775032d68979d2e4b52f415c/reagent/models/fully_connected_network.py#L31  # noqa: E501
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": Identity,
}


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        layers,
        activations,
        *,
        use_batch_norm=False,
        dropout_ratio=0.0,
        use_layer_norm=False,
        normalize_output=False,
    ) -> None:
        super(FullyConnectedNetwork, self).__init__()

        self.input_dim = layers[0]

        modules: List[nn.Module] = []

        assert len(layers) == len(activations) + 1

        for i, ((in_dim, out_dim), activation) in enumerate(
            zip(zip(layers, layers[1:]), activations)
        ):
            # Add BatchNorm1d
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(in_dim))

            # Add Linear
            if activation == "linear":
                w_scale = 1e-3
            else:
                w_scale = 1
            linear = layer_init(nn.Linear(in_dim, out_dim), w_scale)
            modules.append(linear)

            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                modules.append(nn.LayerNorm(out_dim))  # type: ignore

            # Add activation
            if activation in ACTIVATION_MAP:
                modules.append(ACTIVATION_MAP[activation]())
            else:
                # See if it matches any of the nn modules
                modules.append(getattr(nn, activation)())

            # Add Dropout
            if dropout_ratio > 0.0 and (normalize_output or i < len(activations) - 1):
                modules.append(nn.Dropout(p=dropout_ratio))

        self.dnn = nn.Sequential(*modules)  # type: ignore

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        return self.dnn(input)


def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / np.convolve(
            np.ones_like(y), convkernel, mode='same'
        )
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / np.convolve(
            np.ones_like(y), convkernel, mode='full'
        )
        out = out[: -radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out
