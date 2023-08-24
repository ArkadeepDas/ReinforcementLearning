import torch
import numpy as np
from CNNModel import DeepQNetwork


# Let's create a class for Agent
class Agent(object):

    def __init__(self,
                 gamma,
                 epsilon,
                 learning_rate,
                 MaxMemorySize,
                 lowepsilon=0.05,
                 replace=10000,
                 actionSpace=[0, 1, 2, 3, 4, 5]):
        # GAMMA = discount factor
        self.GAMMA = gamma
        # Epsilon greedy action selection
        self.EPSILON = epsilon
        self.MEM_SIZE = MaxMemorySize
        # A variable to keep track how low a Epsilon can go
        self.LOW_EPSILON = lowepsilon
        # How often we are going to replace our target network
        self.replace = replace
        # All possible actions for our agent
        self.ACTION_SPACE = actionSpace
        self.steps = 0
        self.learn_step_counter = 0
        # We use list not numpy arrary because,
        # associated cost of stacking numpy array is high
        self.memory = []
        # Keep track of total number of memory stored
        self.memory_count = 0
        # Load model for exploration and exploitation
        # Agent estimates current set of states
        self.Q_eval = DeepQNetwork()
        # Agent estimates next set of states
        self.Q_next = DeepQNetwork()

    def storeTransition(self, current_state, action, reward, next_state):
        if self.memory_count < self.MEM_SIZE:
            self.memory.append([current_state, action, reward, next_state])
        # If size is full then overwrite from the starting point
        else:
            self.memory[self.memory_count % self.MEM_SIZE] = [
                current_state, action, reward, next_state
            ]
        # Increase the memory count
        self.memory_count = self.memory_count + 1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval(observation)
        if rand < 1 - self.EPSILON:
            # Took the max value of action
            # It will take the heightes value from the action
            # Exploitation
            action = torch.argmax(actions, dim=1)
        else:
            # Choose random action from action space
            # Exploration
            action = np.random.choice(self.actionSpace)
        self.steps = self.steps + 1
        return action

    # Now let's see how the agent is going to learn
    # def learn(self, batch_size):
    #     self.Q_eval.optimizer.zero_grad()
    #     # We are going to replace target network
    #     if self.replace is not None and self.learn_step_counter % self.replace == 0: