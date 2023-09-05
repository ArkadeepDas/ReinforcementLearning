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
# Epsilon decay over 10000 steps from 1.0 to 0.2
EPSILON_DECAY = 10000
# After that steps Target model update with Training model
TARGET_UPDATE_FREQUENCY = 1000
# Learning Rate for training model
LEARNING_RATE = 0.0005


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

    def act(self, x):
        # Convert the observation to tensor
        observation_tensor = torch.as_tensor(x, dtype=torch.float32)
        # We are adding a dimention
        # Here we are not usnig any batch environment so we add one dimention
        q_values = observation_tensor.unsqueeze(0)
        # We need to take action of highest Q-Value
        # Take the heightes value from every row then select heightes value from all the heightes values
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        # It returns index of heighest value
        return action


# Let's create the environment
env = gymnasium.make('CartPole-v1', render_mode='human')

# Total memory
replay_buffer = deque(maxlen=BUFFER_SIZE)
# Store rewards in a single episode
reward_buffer = deque([0.0], maxlen=100)
# Reward of the episode
episode_reward = 0.0

# Let's initialize the networks
training_network = Network(env=env)
target_network = Network(env=env)
optimizer = torch.optim.Adam(training_network.parameters(), lr=LEARNING_RATE)

# Set target network parameters to training network
target_network.load_state_dict(training_network.state_dict())

# Reset the environment
observation, _ = env.reset()

# for _ in range(MIN_REPLAY_SIZE):
#     # Randomly select an action from the action space
#     action = env.action_space.sample()
#     new_observation, reward, done, truncated, info = env.step(action=action)
#     transition = (observation, action, reward, done, new_observation)
#     replay_buffer.append(transition)
#     observation = new_observation

#     if done:
#         observation, _ = env.reset()

# Let's understand the main training loop
observation, _ = env.reset()
rewards = 0
for step in itertools.count():
    # Here we are going to take epsilon greedy approch
    # It start from EPSILON_START and end in EPSILON_END
    # We did this for exploration and exploitation
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    # Random sample
    random_sample = random.random()
    # Exploration
    if random_sample <= epsilon:
        action = env.action_space.sample()
    # Exploitation
    else:
        # We select then action from our network
        obs = torch.tensor(observation)
        actions = training_network(obs)
        action = training_network.act(actions)

    # Storing experience in our replay memory/buffer with random actions
    new_observation, reward, done, truncated, info = env.step(action=action)
    transition = (observation, action, reward, done, new_observation)

    # Add transition to the memory
    replay_buffer.append(transition)
    # Update observation or state
    observation = new_observation
    episode_reward += reward
    if done:
        observation, _ = env.reset()
        # Add episode reward
        reward_buffer.append(episode_reward)
        # Reset episode buffer
        episode_reward = 0.0

    # We do this to store some primary values to our replay memory/buffer
    if step > MIN_REPLAY_SIZE:

        # Start gradient step
        # Sample batch size number of random transition from replay memory/buffer(total memory)
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        # We need every values
        observations_tensor = torch.as_tensor(np.asarray(
            [t[0] for t in transitions]),
                                              dtype=torch.float32)
        actions_tensor = torch.as_tensor(
            np.asarray([t[1] for t in transitions]),
            dtype=torch.int64).unsqueeze(-1)  # Add dimention at the end
        rewards_tensor = torch.as_tensor(
            np.asarray([t[2] for t in transitions]),
            dtype=torch.float32).unsqueeze(-1)  # Add dimention at the end
        dones_tensor = torch.as_tensor(np.asarray([t[3] for t in transitions]),
                                       dtype=torch.float32).unsqueeze(
                                           -1)  # Add dimention at the end

        new_observations_tensor = torch.as_tensor(np.asarray(
            [t[4] for t in transitions]),
                                                  dtype=torch.float32)

        # Computing targets for the loss function
        # When you pass the new observation (next state) to the target network during training,
        # you are estimating the target Q-values for that new state.
        target_q_values = target_network(new_observations_tensor)
        # Shape = batch x q_values
        # Get maximum value from dimention 1
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        # Let's compute targets
        targets = rewards_tensor + GAMMA * (1 -
                                            dones_tensor) * max_target_q_values

        # Here targets = future score we predict from the current state
        # So we want to pass the current observation to predict the future
        q_values_training_network = training_network(observations_tensor)

        # Compute loss
        # Choose the output from action tensor index, so that the model can produce max value in that index
        action_q_values = torch.gather(input=q_values_training_network,
                                       dim=1,
                                       index=actions_tensor)
        # Calculate the loss
        loss = torch.nn.functional.smooth_l1_loss(action_q_values, targets)
        optimizer.zero_grad()
        # Backpropagation calculation
        loss.backward()
        optimizer.step()

        # Update the target network with training network weights
        if step % TARGET_UPDATE_FREQUENCY == 0:
            target_network.load_state_dict(training_network.state_dict())

        # Print rewards
        if step % 1000 == 0:
            avg_reward = np.mean(reward_buffer)
            print(f'Current step: {step}')
            print(f'Current average reward: {avg_reward}')
            # Save the model with heighest rewards
            if avg_reward > rewards:
                print('---Update weight---')
                torch.save(training_network.state_dict(), 'CartPoleDQN.pt')
                rewards = avg_reward

env.close()