# -*- coding: utf-8 -*-
import numpy as np
import torch
from .utils import tensor


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    The API for this implementation is inspired by OpenAI SpinningUp
    reference implementation:
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    """

    def __init__(self, size, num_agents, gamma=0.99, lam=0.95):

        self.max_size = size
        self.num_agents = num_agents
        self.gamma, self.lam = gamma, lam

        # buffers for storing experiences
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.ep_not_done_buf = []
        self.adv_buf = [0.0] * size
        self.ret_buf = [0.0] * size

    def flush_all(self):
        """Reset the buffer"""
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.ep_not_done_buf = []
        self.adv_buf = [0.0] * self.max_size
        self.ret_buf = [0.0] * self.max_size

    def store(
        self,
        states=None,
        actions=None,
        rewards=None,
        values=None,
        log_prob_actions=None,
        episode_not_dones=None,
    ):
        if states is not None:
            self.obs_buf.append(states)
        if actions is not None:
            self.act_buf.append(actions)
        if rewards is not None:
            self.rew_buf.append(rewards)
        if values is not None:
            self.val_buf.append(values)
        if log_prob_actions is not None:
            self.logp_buf.append(log_prob_actions)
        if episode_not_dones is not None:
            self.ep_not_done_buf.append(episode_not_dones)

    def finish_path(self, returns):
        self.rew_buf.append(None)
        self.ep_not_done_buf.append(None)

        # the next line computes rewards-to-go, to be targets for the value function
        for i in reversed(range(self.max_size)):
            returns = self.rew_buf[i] + self.gamma * self.ep_not_done_buf[i] * returns
            self.ret_buf[i] = returns.detach()

        # GAE-Lambda advantage calculation
        advs = tensor(np.zeros((self.num_agents, 1)))
        for i in reversed(range(self.max_size)):
            deltas = (
                self.rew_buf[i]
                + (self.gamma * self.ep_not_done_buf[i] * self.val_buf[i + 1])
                - self.val_buf[i]
            )
            advs = advs * self.lam * self.gamma * self.ep_not_done_buf[i] + deltas
            self.adv_buf[i] = advs.detach()

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, flushes the buffer to get ready to accept
        next round of trajectory data.
        """

        # concatenate the trajectories of all agents
        obs = torch.cat(self.obs_buf, dim=0)
        act = torch.cat(self.act_buf, dim=0)
        log_p = torch.cat(self.logp_buf, dim=0)
        ret = torch.cat(self.ret_buf, dim=0)
        adv = torch.cat(self.adv_buf, dim=0)

        # advantage normalization trick
        adv = (adv - adv.mean()) / adv.std()

        # reset the buffer
        self.flush_all()

        return obs, act, log_p, ret, adv
