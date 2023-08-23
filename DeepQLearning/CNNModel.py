import torch
import torch.nn.functional as F
import numpy as np


# Input image size = (3,185,95)
# We are creating 3 CNN Layers -> 2 FC Layers
# Let's create the model
class DeepQNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=32,
                                     kernel_size=8,
                                     stride=4,
                                     padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64,
                                     out_channels=128,
                                     kernel_size=3)
        # After three cnn block image size became 19x8
        self.fc1 = torch.nn.Linear(128 * 19 * 8, 512)
        # We have 6 moves for the agent
        self.fc2 = torch.nn.Linear(512, 6)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()

    def forward(self, observation):
        observation = self.relu(self.conv1(observation))
        observation = self.relu(self.conv2(observation))
        observation = self.relu(self.conv3(observation))
        # Let's flatten the data
        observation = self.flatten(observation)
        observation = self.relu(self.fc1(observation))
        action = self.fc2(observation)
        return action


# Let's text the model
model = DeepQNetwork()
# Data
data = torch.randn((4, 1, 185, 95))
action = model(data)
# Output is 4x6
print(action.shape)