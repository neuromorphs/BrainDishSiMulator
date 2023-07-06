#!/user/bin/env python

"""
Author: Simon Narduzzi
Email: simon.narduzzi@csem.ch
Copyright: CSEM, 2023
Creation: 06.07.23
Description: RL Agents
"""

# PyTorch for RL Agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    """
    Reinforcement learning agent using DQN
    """
    def __init__(self, num_input=5, num_outputs=2, epsilon_decay = 0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """

        self.model = nn.Sequential(
            nn.Linear(num_input, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            action = torch.argmax(q_values).item()  # Best action
        return action  # Convert 0, 1 to -1, 1

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        current_q_value = self.model(state)[action]
        next_q_value = self.model(next_state).max().detach()
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.criterion(current_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # After the update, decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay