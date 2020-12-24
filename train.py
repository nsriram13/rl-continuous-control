# -*- coding: utf-8 -*-
import pickle
from collections import deque

import numpy as np
import torch
from absl import app, flags
from absl import logging
from unityagents import UnityEnvironment

from deeprl_cc.agent import PPOAgent

logging.set_verbosity(logging.INFO)

# Hyper-parameters
FLAGS = flags.FLAGS
flags.DEFINE_float("gamma", 0.99, "Discount factor used for PPO update")
flags.DEFINE_float("lam", 0.95, "GAE Lambda")
flags.DEFINE_float("learning_rate", 2e-4, "Learning rate for the actor critic network")
flags.DEFINE_float("gradient_clip", 0.75, "Clip gradient norm at this value")
flags.DEFINE_float(
    "ppo_epsilon", 0.1, "Clamp importance weights between 1-epsilon and 1+epsilon"
)
flags.DEFINE_integer("update_freq", 250, "Number of env steps between updates")
flags.DEFINE_integer(
    "update_epochs", 10, "Number of epochs to run when updating (for PPO)"
)
flags.DEFINE_integer(
    "ppo_batch_size", 64, "Batch size (number of trajectories) used for PPO updates"
)
flags.DEFINE_list(
    "ac_net", [[64, 64], ["tanh", "tanh"]], "Actor critic network configuration"
)

# Solution constraints
flags.DEFINE_float(
    "target_score", 30, "Score to achieve in order to solve the environment"
)
flags.DEFINE_integer("max_episodes", 250, "Maximum number of episodes")

# Miscellaneous flags
flags.DEFINE_integer("seed", 0, "Random number generator seed")
flags.DEFINE_string(
    "checkpoint", "./checkpoints/checkpoint.pth", "Save the model weights to this file"
)


def main(_):

    env = UnityEnvironment(file_name="./Reacher_Linux_NoVis/Reacher.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    logging.info(f'Number of agents: {num_agents}')

    # size of each action
    action_size = brain.vector_action_space_size
    logging.info(f'Size of each action: {action_size}')

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    logging.info(
        f'There are {states.shape[0]} agents. '
        f'Each observes a state with length: {state_size}'
    )
    logging.info(f'The state for the first agent looks like: {states[0]}')

    # Setup some variable for keeping track of the performance
    episode_counter_across_agents = 0
    episodes_completed = 0

    windowed_score = deque(maxlen=100)  # last 100 episode scores
    mean_windowed_score = 0

    online_rewards = np.zeros(
        num_agents
    )  # array to accumulate agent rewards as the episode progresses
    episode_scores = np.zeros(
        (num_agents, FLAGS.max_episodes)
    )  # array to keep track of scores from each episode for every agent
    mean_scores = []  # list containing scores from each episode averaged across agents

    # Log all the hyperparameters
    logging.info(f"AC Network HParam: Learning rate set to {FLAGS.learning_rate}")
    logging.info(f"AC Network HParam: Hidden sizes set to {FLAGS.ac_net[0]}")
    logging.info(f"AC Network HParam: Activations set to {FLAGS.ac_net[1]}")
    logging.info(f"AC Network HParam: Gradient clip set to {FLAGS.gradient_clip}")

    logging.info(
        f"PPO Learning HParam: "
        f"Trajectories accumulated between updates {FLAGS.update_freq}"
    )
    logging.info(
        f"PPO Learning HParam: "
        f"Number of trajectories per PPO batch update set to {FLAGS.ppo_batch_size}"
    )
    logging.info(
        f"PPO Learning HParam: Number of epochs per PPO update {FLAGS.update_epochs}"
    )
    logging.info(
        f"PPO Learning HParam: Surrogate functions clip set to {FLAGS.ppo_epsilon}"
    )
    logging.info(f"PPO Learning HParam: Discounting factor set to {FLAGS.gamma}")
    logging.info(f"PPO Learning HParam: GAE Lambda set to {FLAGS.lam}")

    # see if GPU is available for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Begin training
    logging.info(f"Beginning training")
    logging.info(
        f"Goal: Achieve a score of {FLAGS.target_score:.2f} "
        f"in under {FLAGS.max_episodes} episodes."
    )
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    # Initialize the agent
    agent = PPOAgent(
        state_dim=state_size,
        action_dim=action_size,
        num_agents=num_agents,
        update_freq=FLAGS.update_freq,
        update_epochs=FLAGS.update_epochs,
        ppo_batch_size=FLAGS.ppo_batch_size,
        ppo_epsilon=FLAGS.ppo_epsilon,
        seed=FLAGS.seed,
        hidden_sizes=FLAGS.ac_net[0],
        activations=FLAGS.ac_net[1],
        lrate=FLAGS.learning_rate,
        gradient_clip=FLAGS.gradient_clip,
        gamma=FLAGS.gamma,
        lam=FLAGS.lam,
        device=device,
    )

    while True:

        states = env_info.vector_observations  # get the current state

        for _ in range(FLAGS.update_freq):
            actions = agent.propose_action(states)
            actions = np.clip(actions.cpu().detach().numpy(), -1, 1)

            env_info = env.step(actions)[
                brain_name
            ]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            # accumulate rewards
            online_rewards += rewards

            for i, done in enumerate(dones):
                if done:
                    episodes_completed += 1
                    episode_scores[i][
                        int((episodes_completed - 1) / num_agents)
                    ] = online_rewards[i]

                    # if all agents have finished an episode
                    if (episodes_completed % num_agents) == 0:
                        episode_counter_across_agents += 1

                        # save most recent score
                        average_score_this_episode = episode_scores[
                            :, episode_counter_across_agents - 1
                        ].mean()
                        windowed_score.append(average_score_this_episode)
                        mean_scores.append(average_score_this_episode)

                        logging.info(
                            f'Episode {episode_counter_across_agents}'
                            f' | Average score this episode: {average_score_this_episode:.3f}'  # noqa: E501
                            f' | Average over last 100 episodes: {np.mean(windowed_score):.3f}'  # noqa: E501
                        )

                    online_rewards[i] = 0  # Reset accumulated reward for next episode
                    mean_windowed_score = np.mean(windowed_score)

                    if mean_windowed_score > FLAGS.target_score:
                        logging.info("Environment solved!")
                        break

            agent.step(states, actions, rewards, next_states, dones)  # Teach the agent

            states = next_states  # roll over states to next time step

        # train the agent
        agent.train(states)

        # check for termination criteria
        if mean_windowed_score > FLAGS.target_score:
            print(
                f"Environment solved in {episode_counter_across_agents-100} episodes! "
                f"Score: {mean_windowed_score}."
            )
            agent.save_checkpoint(file_name=FLAGS.checkpoint)
            break

        if episode_counter_across_agents >= FLAGS.max_episodes:
            print(
                f"Episode {episode_counter_across_agents} exceeded {FLAGS.max_episodes}. "  # noqa: E501
                f"Failed to solve environment!"
            )
            break

    # pickle and save the agent performance scores
    with open('./checkpoints/episode_scores.pkl', 'wb') as f:
        pickle.dump(episode_scores[:, :episode_counter_across_agents], f)
    with open('./checkpoints/mean_scores.pkl', 'wb') as f:
        pickle.dump(mean_scores, f)


if __name__ == "__main__":
    app.run(main)
