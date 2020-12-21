#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import FullyConnectedNetwork, tensor


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes=None,
        activations=None,
        action_activation="linear",
        seed=0,
    ):

        super(MLPActorCritic, self).__init__()
        self.seed = random.seed(seed)

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        if activations is None:
            activations = ["tanh", "tanh"]

        assert state_dim > 0, f"state_dim must be > 0, got {state_dim}"
        assert action_dim > 0, f"action_dim must be > 0, got {action_dim}"
        assert len(hidden_sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(hidden_sizes), len(activations)
        )

        self.policy_network = FullyConnectedNetwork(
            [state_dim] + hidden_sizes + [action_dim], activations + [action_activation]
        )

        self.value_network = FullyConnectedNetwork(
            [state_dim] + hidden_sizes + [1], activations + [action_activation]
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        state = tensor(state)

        # Get Value estimate
        v = self.value_network(state)

        # Get actions and log probability
        mean = self.policy_network(state)
        squashed_mean = torch.tanh(mean)
        scale = F.softplus(self.log_std)
        action_distribution = torch.distributions.Normal(loc=squashed_mean, scale=scale)

        return v, action_distribution
