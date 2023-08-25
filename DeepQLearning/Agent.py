import torch
import numpy as np
from CNNModel import DeepQNetwork


# Let's create a class for Agent
class Agent():

    def __init__(self,
                 gamma,
                 epsilon,
                 learning_rate,
                 batch_size,
                 num_actions,
                 max_memory_size=100000,
                 epsilon_end=0.01,
                 epsilon_decay=5e-4):
        # Gamma is weighting feature rewards
        self.gamma = gamma
        # Exploration or exploitation threshold
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        # After one episode how much epsilon will decay
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.action_space = num_actions
        self.memory_size = max_memory_size
        self.batch_size = batch_size
        # Keep track of the first available memory
        self.memory_counter = 0
        self.Q_eval = DeepQNetwork()
        # Total memory for a state (total_number_of_images, channels, image_width, image_height)
        self.state_memory = np.zeros((self.memory_size, 1, 185, 95))
        # We also need to have new state memory to keep track of the new states the agent encounters
        self.new_state_memory = np.zeros((self.memory_size, 1, 185, 95))
        # We need action memory, reward memory, terminal_memory
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    # Store the transitions in to the memory
    def store_transitions(self, state, action, reward, next_state, done):
        # First find the unoccupied memory
        index = self.memory_counter % self.memory_size
        # If memory is full then we store from the begining again
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.memory_counter += 1

    # Choose the action by passing state to the network
    def choose_action(self, observation):
        # Exploitation
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation])
            # Pass the state to our train network
            actions = self.Q_eval(state)
            # Take maximum value from actions
            action = torch.argmax(actions).item()
        # Exploration
        else:
            action = np.random.choice(self.action_space)

        return action

    # # Now let's see how we can learn the model
    # def learn(self):