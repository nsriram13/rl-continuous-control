# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import MLPActorCritic
from .replay_buffer import PPOBuffer
from .utils import MeanStdNormalizer, tensor


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents=20,
        update_freq=250,  # how many env steps between updates
        update_epochs=10,  # how many epochs to run when updating (for PPO)
        ppo_batch_size=64,  # batch size (number of trajectories) used for PPO updates
        ppo_epsilon=0.1,  # clamp importance weights between 1-epsilon and 1+epsilon
        seed=0,
        hidden_sizes=None,
        activations=None,
        lrate=2e-4,
        lrate_schedule=lambda it: 1.0,
        weight_decay=0.0,
        gradient_clip=0.75,
        gamma=0.99,
        lam=0.95,
        device=torch.device("cpu"),
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        if activations is None:
            activations = ["tanh", "tanh"]

        self.actor_critic = MLPActorCritic(
            state_dim, action_dim, hidden_sizes, activations, seed
        ).to(device)
        self.state_normalizer = MeanStdNormalizer()
        self.update_epochs = update_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epsilon = ppo_epsilon
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lrate, weight_decay=weight_decay
        )
        self.gradient_clip = gradient_clip
        self.update_freq = update_freq

        self.num_agents = num_agents
        self.gamma, self.lam = gamma, lam
        self.traj_buffer = PPOBuffer(update_freq, num_agents, gamma, lam)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lrate_schedule
        )

        # Variables for storing current
        self.normalized_states = None
        self.actions = None
        self.values = None
        self.log_proba = None

        self.steps = 0  # counter to keep track of steps per rollout

        self.first_state = True

    @staticmethod
    def get_log_prob(action_distribution, actions):
        return action_distribution.log_prob(actions).sum(-1).unsqueeze(-1)

    def scorer(self, states, actions):
        value, action_distribution = self.actor_critic(states)
        log_prob = self.get_log_prob(action_distribution, actions)
        return log_prob, value

    def propose_action(self, states):

        if self.first_state:
            self.normalized_states = states
            self.first_state = False

        self.values, action_distribution = self.actor_critic(self.normalized_states)
        self.actions = action_distribution.sample()
        self.log_proba = self.get_log_prob(action_distribution, self.actions)
        return self.actions

    def step(self, states, actions, rewards, next_states, dones):

        states = tensor(self.normalized_states)
        rewards = tensor(np.asarray(rewards)).unsqueeze(-1)
        episode_not_dones = tensor(1 - np.asarray(dones).astype(int)).unsqueeze(-1)

        self.traj_buffer.store(
            states,
            self.actions,
            rewards,
            self.values,
            self.log_proba,
            episode_not_dones,
        )

        # normalize next_states and update observation to move the environment forward
        self.normalized_states = self.state_normalizer(next_states)

    def process_rollout(self, states):
        self.propose_action(states)
        self.rollout.save_prediction(
            self.latest_actions, self.latest_log_prob, self.latest_values
        )
        self.rollout.calculate_returns_and_advantages(self.latest_values.detach())
        self.update_model()
        self.first_states = True

    def train(self, states):
        # bootstrap value target for final state for this training round
        self.propose_action(states)
        self.traj_buffer.store(
            actions=self.actions, log_prob_actions=self.log_proba, values=self.values
        )

        # close out the trajectory rollout
        self.traj_buffer.finish_path(self.values.detach())

        # update the model weights
        self.update_model()
        self.first_state = True

    def update_model(self):

        states, actions, log_probs_old, returns, advantages = self.traj_buffer.get()
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()

        # Decay the LR
        self.scheduler.step()

        for _ in range(self.update_epochs):
            # iterate through mini-batches of PPO updates in random order
            random_order = torch.randperm(states.size(0))
            for i in range(0, states.size(0), self.ppo_batch_size):
                idx = random_order[i : i + self.ppo_batch_size]

                # get the losses for the sampled trajectories
                loss = self._trajectory_to_losses(
                    states[idx],
                    actions[idx],
                    log_probs_old[idx],
                    returns[idx],
                    advantages[idx],
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.gradient_clip
                )
                self.optimizer.step()

    def _trajectory_to_losses(
        self, states, actions, log_probs_old, returns, advantages
    ):
        """
        Return the losses for the trajectory.
        """

        # Calculate clipped surrogate objective for PPO
        log_prob_action, value = self.scorer(states, actions)
        characteristic_eligibility = (log_prob_action - log_probs_old).exp()
        clipped_advantage = (
            torch.clamp(
                characteristic_eligibility, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon
            )
            * advantages
        )

        ppo_loss = -torch.min(
            characteristic_eligibility * advantages, clipped_advantage
        ).mean()

        # Value loss: use reward-to-go as label for training the value function
        value_net_loss = 0.5 * (returns - value).pow(2).mean()

        return ppo_loss + value_net_loss

    def save_checkpoint(self, file_name):
        logging.info("Checkpointing weights")
        torch.save(self.actor_critic.state_dict(), file_name)
