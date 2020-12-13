# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.optim
import torch.optim as optim

from .model import FullyConnectedCritic, FullyConnectedActor
from .replay_buffer import PPOBuffer

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 3e-4


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents,
        gamma=0.99,
        lam=0.95,
        update_freq=250,  # how many env steps between updates
        update_epochs=5,  # how many epochs to run when updating (for PPO)
        ppo_batch_size=10,  # batch size (number of trajectories) used for PPO updates
        ppo_epsilon=0.1,  # clamp importance weights between 1-epsilon and 1+epsilon
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.policy_network = FullyConnectedActor(state_dim, action_dim).to(device)
        self.value_network = FullyConnectedCritic(state_dim).to(device)
        self.value_loss_fn = torch.nn.MSELoss(reduction="elementwise_mean").to(device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LR, eps=1e-5)
        self.optimizer_value_net = optim.Adam(
            self.value_network.parameters(), lr=LR, eps=1e-5
        )

        self.step = 0
        self.gamma = gamma
        self.lam = lam
        self.update_freq = update_freq
        self.update_epochs = update_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epsilon = ppo_epsilon
        self.traj_buffer = PPOBuffer(
            obs_dim=self.state_dim,
            act_dim=self.action_dim,
            num_agents=self.num_agents,
            size=self.update_freq,
            gamma=self.gamma,
            lam=self.lam,
        )

    def propose_action(self, state):
        state.to(device)
        self.policy_network.eval()
        with torch.no_grad():
            pi = self.policy_network._distribution(state)
            a = pi.sample()
            logp_a = self.policy_network.get_log_prob(state, a)
            v = self.value_network(state)
        self.policy_network.train()
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def train(self, states, actions, rewards, values, logp, dones):
        self.traj_buffer.store(states, actions, rewards, values, logp, dones)
        self.step += 1
        if self.step % self.update_freq == 0:
            # if trajectory didn't reach terminal state, bootstrap value target
            if dones.any():
                last_val = np.array([[0]] * self.num_agents)
            else:
                _, last_val, _ = self.propose_action(
                    torch.as_tensor(states, dtype=torch.float32).to(device)
                )
            self.traj_buffer.finish_path(last_val.squeeze())
            training_data = self.traj_buffer.get()
            self.update_model(training_data)

    def update_model(self, training_data):
        """
        Iterate through the PPO trajectory buffer `update_epochs` times, sampling
        mini-batches of `ppo_batch_size` trajectories. Perform gradient ascent on
        the clipped PPO loss. If value network is being trained, also perform
        gradient descent steps for its loss.
        """

        for _ in range(self.update_epochs):
            # iterate through mini-batches of PPO updates in random order
            random_order = torch.randperm(len(training_data))
            for i in range(0, len(training_data), self.ppo_batch_size):
                idx = random_order[i : i + self.ppo_batch_size]

                # get the losses for the sampled trajectories
                ppo_loss = []
                value_net_loss = []
                for i in idx:
                    traj_losses = self._trajectory_to_losses(training_data[i])
                    ppo_loss.append(traj_losses["ppo_loss"])
                    value_net_loss.append(traj_losses["value_net_loss"])

                self.optimizer.zero_grad()
                ppo_loss = torch.stack(ppo_loss).mean()
                ppo_loss.backward()
                self.optimizer.step()

                self.optimizer_value_net.zero_grad()
                value_net_loss = torch.stack(value_net_loss).mean()
                value_net_loss.backward()
                self.optimizer_value_net.step()

    def _trajectory_to_losses(self, trajectory):
        """
        Get a dict of losses for the trajectory. Dict always includes PPO loss.
        If a value baseline is trained, a loss for the value network is also included.
        """
        losses = {}

        # Value loss: use reward-to-go as label for training the value function
        offset_reinforcement = trajectory.reward_to_go
        baselines = self.value_network(trajectory.state.to(device)).squeeze()
        losses["value_net_loss"] = self.value_loss_fn(baselines.to(device), offset_reinforcement.to(device))

        # Policy loss
        target_propensity = self.policy_network.get_log_prob(
            trajectory.state.unsqueeze(0).to(device), trajectory.action.to(device)
        ).squeeze()
        characteristic_eligibility = torch.exp(
            target_propensity - trajectory.logp.detach().to(device)
        ).float()
        clipped_advantage = (
            torch.clamp(
                characteristic_eligibility, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon
            )
            * trajectory.advantage.to(device)
        )
        losses["ppo_loss"] = -(
            torch.min(
                characteristic_eligibility * trajectory.advantage.to(device), clipped_advantage
            )
        ).mean()

        return losses
