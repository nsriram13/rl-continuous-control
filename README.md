# Continuous Control
Deep RL algorithms for solving Unity ML-Agents Reacher environment.

## Project Details
In this project, we train a PPO based agent to control a double-jointed arm to
move to target locations. We use a Unity ML-Agents based environment provided by
Udacity for this exercise. More details about the environment is provided below.

* The observation space consists of 33 variables corresponding to position,
  rotation, velocity, and angular velocities of the arm.
* Each action is a vector with four numbers, corresponding to torque applicable
  to two joints (clipped between -1 and 1).
* A reward of +0.1 is provided for each step that the agent's hand is in the goal
  location.
* We use the distributed version of the environment - this environment has 20
  double-jointed arms. This can be used to speed up data collection (and also helps
  with randomizing training data).

The goal of the agent is to maintain its position at the target location for as many
time steps as possible. The environment is considered solved when the agent achieves
a score of +30 over 100 consecutive episode and over all the 20 parallel arms.

## Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/nsriram13/rl-continuous-control.git
    cd rl-continuous-control/python
    pip install .
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

5. This repository uses pre-commit hooks for auto-formatting and linting.
    * Run `pre-commit install` to set up the git hook scripts - this installs flake8 formatting, black
    auto-formatting as pre-commit hooks.
    * Run `gitlint install-hook` to install the gitlint commit-msg hook
    * (optional) If you want to manually run all pre-commit hooks on a repository,
    run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

6. Download the **_Twenty (20) Agents Version_** of the Reacher environment from one of the links below.
You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

7. Place the file in the root directory of this repo and unzip (or decompress) the file. The notebook to train the
agent will look for the environment at the project root.

## Instructions

Run `python train.py` to train the agent with the default set of hyper-parameters.
Alternatively, you can modify the hyper-parameters by invoking the scripts with
the optional CLI flags; for e.g. say you want to use a different learning rate for
training the neural network, you can do so by running `python train.py --learning-rate=1e-5`.
The full list of flags is shown below. You can access this list anytime by
running `python train.py --help`.

```bash
       USAGE: train.py [flags]
flags:

train.py:
  --ac_net: Actor critic network configuration
    (default: '"[64, 64]","[\'tanh\', \'tanh\']"')
    (a comma separated list)
  --checkpoint: Save the model weights to this file
    (default: './checkpoints/checkpoint.pth')
  --gamma: Discount factor used for PPO update
    (default: '0.99')
    (a number)
  --gradient_clip: Clip gradient norm at this value
    (default: '0.75')
    (a number)
  --lam: GAE Lambda
    (default: '0.95')
    (a number)
  --learning_rate: Learning rate for the actor critic network
    (default: '0.0002')
    (a number)
  --max_episodes: Maximum number of episodes
    (default: '250')
    (an integer)
  --ppo_batch_size: Batch size (number of trajectories) used for PPO updates
    (default: '64')
    (an integer)
  --ppo_epsilon: Clamp importance weights between 1-epsilon and 1+epsilon
    (default: '0.1')
    (a number)
  --seed: Random number generator seed
    (default: '0')
    (an integer)
  --target_score: Score to achieve in order to solve the environment
    (default: '30.0')
    (a number)
  --update_epochs: Number of epochs to run when updating (for PPO)
    (default: '10')
    (an integer)
  --update_freq: Number of env steps between updates
    (default: '250')
    (an integer)
```
