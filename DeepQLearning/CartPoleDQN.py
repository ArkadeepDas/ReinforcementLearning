import torch
import gymnasium
from collections import deque
import itertools
import numpy as np
import random

# Initialization of hyperpartameters
# Discount rate
GAMMA = 0.99
# Sample size from replay memory/buffer
BATCH_SIZE = 32
# Maximum number of transition we are going to store
BUFFER_SIZE = 50000
# Transitions we want in replay memory/buffer for training
MIN_REPLAY_SIZE = 1000
# Starting value epsilon
EPSILON_START = 1.0
# Ending value epsilon
EPSILON_END = 0.2
# Epsilon decay
EPSILON_DECAY = 5e-4
# After that steps Target model update with Training model
TARGET_UPDATE_FREQUENCY = 1000


# Let's create the model
class Network(torch.nn.Module):

    def __init__(self, env):
        super().__init__()
        # env.observation_space = number of states (here the value is 4)
        # env.action_space = number of action agent can take (here the value is 2)
        self.in_features = int(env.observation_space.shape[0])
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.in_features, out_features=65),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=65, out_features=env.action_space.n))

    def forward(self, x):
        return self.net(x)


# Let's create the environment
env = gymnasium.make('CartPole-v1')

# Total memory
replay_buffer = deque(maxlen=BUFFER_SIZE)
# Store rewards in a single episode
reward_buffer = deque([0.0], maxlen=100)
# Reward of the episode
episode_reward = 0.0

# Let's initialize the networks
training_network = Network(env=env)
target_network = Network(env=env)

# Set target network parameters to training network
target_network.load_state_dict(training_network.state_dict())

# Reset the environment
observation, _ = env.reset()
# Storing experience in our replay memory/buffer with random actions
for _ in range(MIN_REPLAY_SIZE):
    # Randomly select an action from the action space
    action = env.action_space.sample()
    new_observation, reward, done, truncated, info = env.step(action=action)
    transition = (observation, action, reward, done, new_observation)
    replay_buffer.append(transition)
    observation = new_observation

    if done:
        observation = env.reset()