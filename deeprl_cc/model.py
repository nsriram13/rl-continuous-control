#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.normal import Normal


def gaussian_fill_w_gain(tensor, activation, dim_in, min_std=0.0) -> None:
    """ Gaussian initialization with gain."""
    gain = math.sqrt(2) if (activation == "relu" or activation == "leaky_relu") else 1
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


class FullyConnectedActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):

        super(FullyConnectedActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # The last layer is mean & scale for re-parameterization trick
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ('fc1', nn.Linear(self.state_dim, 128)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(128, 128)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(128, self.action_dim * 2)),
                ]
            )
        )

    def _get_loc_and_scale(self, state):
        out = self.model(state)
        loc = out[::, : self.action_dim]
        scale = torch.exp(out[::, self.action_dim :].clamp(-20, 2))

        return loc, scale

    def _distribution(self, state):
        loc, scale = self._get_loc_and_scale(state)
        return Normal(loc=loc, scale=scale)

    def forward(self, state):
        loc, scale = self._get_loc_and_scale_log(state)
        r = torch.randn_like(scale, device=scale.device)
        raw_action = loc + r * scale
        squashed_action = torch.tanh(raw_action)
        squashed_mean = torch.tanh(loc)
        # log_prob = self.get_log_prob(state, squashed_action)
        log_prob = self.get_log_prob(state, raw_action)

        return squashed_action, log_prob, squashed_mean

    def get_log_prob(self, state, raw_action):

        dist = self._distribution(state)
        loc, scale = self._get_loc_and_scale(state)
        r = (raw_action - loc) / scale
        # log_prob = dist.log_prob(r).sum(axis=-1)
        log_prob = torch.sum(dist.log_prob(r), dim=1).reshape(-1, 1)

        return log_prob

    # @torch.no_grad()
    # def get_log_prob(self, state, squashed_action):
    #
    #     dist = self._distribution(state)
    #     loc, scale = self._get_loc_and_scale(state)
    #
    #     raw_action = torch.atanh(squashed_action)
    #     r = (raw_action - loc) / scale
    #     log_prob = dist.log_prob(r).sum(axis=-1)
    #
    #     return log_prob


class FullyConnectedCritic(nn.Module):
    def __init__(self, state_dim):
        super(FullyConnectedCritic, self).__init__()
        self.state_dim = state_dim

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ('fc1', nn.Linear(state_dim, 128)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(128, 128)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(128, 1)),
                ]
            )
        )

    def forward(self, x):
        # return torch.squeeze(self.model(x), -1)
        return self.model(x)


# class MLPActorCritic(nn.Module):
#
#     def __init__(self, state_dim, action_dim):
#         super(MLPActorCritic, self).__init__()
#
#         # policy builder depends on action space
#         self.pi = FullyConnectedActor(state_dim, action_dim)
#
#         # build value function
#         self.v = FullyConnectedCritic(state_dim)
#
#     def step(self, state):
#         with torch.no_grad():
#             pi = self.pi._distribution(state)
#             a = pi.sample()
#             logp_a = self.pi.get_log_prob(state, a)
#             v = self.v(state)
#         return a.detach().numpy(), v.detach().numpy(), logp_a.detach().numpy()
#
#     def act(self, state):
#         return self.step(state)[0]
