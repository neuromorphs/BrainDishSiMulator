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
import matplotlib.pyplot as plt


class DQNAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=5, num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.model = nn.Sequential(
            nn.Linear(num_inputs, 1, bias=False),
            nn.ReLU(),
            nn.Linear(1, num_outputs, bias=False)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)[:self.num_inputs]
        q_values = self.model(state)
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            action = torch.argmax(q_values).item()  # Best action
        return action  # Convert 0, 1 to -1, 1

    def get_weights(self):
        d = self.model.state_dict()
        return [d[x] for x in d]

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)[:self.num_inputs]
        next_state = torch.tensor(next_state, dtype=torch.float32)[:self.num_inputs]
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


class DQN8Agent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=5, num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 2, bias=False),
            nn.ReLU(),
            nn.Linear(2, num_outputs, bias=False)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value

        screen_height = 600
        num_sections = 8
        self.section_height = screen_height / num_sections

    def floor_to_nearest_region(self, y):
        region_index = int(y / self.section_height)
        floor_position = region_index * self.section_height
        return floor_position

    def get_weights(self):
        d = self.model.state_dict()
        return [d[x] for x in d]

    def get_action(self, state):

        # state is a list of 5 elements: [paddle.y, ball.x, ball.y, ball.dx, ball.dy])
        state = torch.tensor(state, dtype=torch.float32)[:self.num_inputs]

        # get the ball.y and floor it to the nearest region
        state[2] = self.floor_to_nearest_region(state[2])

        q_values = self.model(state)
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            action = torch.argmax(q_values).item()  # Best action
        return action  # Convert 0, 1 to -1, 1

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)[:self.num_inputs]
        next_state = torch.tensor(next_state, dtype=torch.float32)[:self.num_inputs]

        # get the ball.y and floor it to the nearest region
        state[2] = self.floor_to_nearest_region(state[2])
        # get the ball.y and floor it to the nearest region
        next_state[2] = self.floor_to_nearest_region(next_state[2])

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


class ConvDQNAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=5, num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, num_outputs),
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value

    def get_image_from_state(self, state):
        # original:
        paddle_width = 40
        paddle_height = 250
        ball_width = 40
        original_game_width = 600
        original_game_height = 600

        # build a picture of the state, going from 600x600 to 40x40
        ratio = 40 / 600

        # new object sizes
        paddle_width = int(paddle_width * ratio)
        paddle_height = int(paddle_height * ratio)
        ball_width = int(ball_width * ratio)
        game_width = int(original_game_width * ratio)
        game_height = int(original_game_height * ratio)

        rescaled_state = np.asarray(state) * ratio
        state = rescaled_state
        current_picture = np.zeros((game_width, game_height))
        last_picture = np.zeros((game_width, game_height))
        # place ball and paddle in the picture
        current_picture[int(state[0][0] - paddle_width / 2):int(state[0][0] + paddle_width / 2),
        int(state[0][1] - paddle_height / 2):int(state[0][1] + paddle_height / 2)] = 1

        current_picture[int(state[0][2] - ball_width / 2):int(state[0][2] + ball_width / 2),
        int(state[0][3] - ball_width / 2):int(state[0][3] + ball_width / 2)] = 1

        last_picture[int(state[1][0] - paddle_width / 2):int(state[1][0] + paddle_width / 2),
        int(state[1][1] - paddle_height / 2):int(state[1][1] + paddle_height / 2)] = 1

        last_picture[int(state[1][2] - ball_width / 2):int(state[1][2] + ball_width / 2),
        int(state[1][3] - ball_width / 2):int(state[1][3] + ball_width / 2)] = 1

        # convert to tensor
        current_picture = torch.tensor(current_picture, dtype=torch.float32)
        last_picture = torch.tensor(last_picture, dtype=torch.float32)

        # compute delta
        delta = current_picture - last_picture

        # add dimension
        delta = delta.unsqueeze(0)
        return delta

    def get_action(self, state):

        state = torch.tensor(state, dtype=torch.float32)
        delta = self.get_image_from_state(state)

        # state format ([np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy],
        # [np.array([last_paddle.y, last_ball.x, last_ball.y, last_ball.dx, last_ball.dy] )
        range = (torch.min(delta), torch.max(delta))
        # save delta as grey scale image in the folder
        plt.imsave('delta.png', delta[0].detach().numpy(), cmap='gray')

        delta = delta.unsqueeze(0)  # Add batch dimension
        q_values = self.model(delta)
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            action = torch.argmax(q_values).item()  # Best action
        return action  # Convert 0, 1 to -1, 1

    def update(self, state, action, reward, next_state, done):
        current_state_delta = self.get_image_from_state(state)
        next_state_delta = self.get_image_from_state(next_state)

        state = torch.tensor(current_state_delta, dtype=torch.float32)
        next_state = torch.tensor(next_state_delta, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        # Add batch dimension
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)

        output = self.model(state)
        current_q_value = output[0][action]
        next_q_value = self.model(next_state)[0].max().detach()
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.criterion(current_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # After the update, decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ConvDQNCaptureAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=(40, 40, 1), num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, num_outputs),
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
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


class IFELSEAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def floor_to_nearest_region(self, y):
        region_index = int(y / self.section_height)
        floor_position = region_index * self.section_height
        return floor_position

    def get_weights(self):
        return []

    def model(self, state):
        paddle, ball = state
        if paddle.y + 250 / 2 < ball.y:
            return 0
        else:
            return 1

    def get_action(self, state):

        # state is a list of 5 elements: [paddle.y, ball.x, ball.y, ball.dx, ball.dy])
        state = torch.tensor(state, dtype=torch.float32)
        state = torch.tensor([state[0], state[2]])

        # get the ball.y and floor it to the nearest region
        # state[2] = self.floor_to_nearest_region(state[2])

        action = self.model(state)
        return action  # Convert 0, 1 to -1, 1

    def update(self, state, action, reward, next_state, done):
        return


class ConvDQNAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=5, num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, num_outputs),
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value

    def get_image_from_state(self, state):
        # original:
        paddle_width = 40
        paddle_height = 250
        ball_width = 40
        original_game_width = 600
        original_game_height = 600

        # build a picture of the state, going from 600x600 to 40x40
        ratio = 40 / 600

        # new object sizes
        paddle_width = int(paddle_width * ratio)
        paddle_height = int(paddle_height * ratio)
        ball_width = int(ball_width * ratio)
        game_width = int(original_game_width * ratio)
        game_height = int(original_game_height * ratio)

        rescaled_state = np.asarray(state) * ratio
        state = rescaled_state
        current_picture = np.zeros((game_width, game_height))
        last_picture = np.zeros((game_width, game_height))
        # place ball and paddle in the picture
        current_picture[int(state[0][0] - paddle_width / 2):int(state[0][0] + paddle_width / 2),
        int(state[0][1] - paddle_height / 2):int(state[0][1] + paddle_height / 2)] = 1

        current_picture[int(state[0][2] - ball_width / 2):int(state[0][2] + ball_width / 2),
        int(state[0][3] - ball_width / 2):int(state[0][3] + ball_width / 2)] = 1

        last_picture[int(state[1][0] - paddle_width / 2):int(state[1][0] + paddle_width / 2),
        int(state[1][1] - paddle_height / 2):int(state[1][1] + paddle_height / 2)] = 1

        last_picture[int(state[1][2] - ball_width / 2):int(state[1][2] + ball_width / 2),
        int(state[1][3] - ball_width / 2):int(state[1][3] + ball_width / 2)] = 1

        # convert to tensor
        current_picture = torch.tensor(current_picture, dtype=torch.float32)
        last_picture = torch.tensor(last_picture, dtype=torch.float32)

        # compute delta
        delta = current_picture - last_picture

        # add dimension
        delta = delta.unsqueeze(0)
        return delta

    def get_action(self, state):

        state = torch.tensor(state, dtype=torch.float32)
        delta = self.get_image_from_state(state)

        # state format ([np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy],
        # [np.array([last_paddle.y, last_ball.x, last_ball.y, last_ball.dx, last_ball.dy] )
        range = (torch.min(delta), torch.max(delta))
        # save delta as grey scale image in the folder
        plt.imsave('delta.png', delta[0].detach().numpy(), cmap='gray')

        delta = delta.unsqueeze(0)  # Add batch dimension
        q_values = self.model(delta)
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1])  # Random action
        else:
            action = torch.argmax(q_values).item()  # Best action
        return action  # Convert 0, 1 to -1, 1

    def update(self, state, action, reward, next_state, done):
        current_state_delta = self.get_image_from_state(state)
        next_state_delta = self.get_image_from_state(next_state)

        state = torch.tensor(current_state_delta, dtype=torch.float32)
        next_state = torch.tensor(next_state_delta, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        # Add batch dimension
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)

        output = self.model(state)
        current_q_value = output[0][action]
        next_q_value = self.model(next_state)[0].max().detach()
        target_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.criterion(current_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # After the update, decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ConvDQNCaptureAgent:
    """
    Reinforcement learning agent using DQN
    """

    def __init__(self, seed=0, num_inputs=(40, 40, 1), num_outputs=2, epsilon_decay=0.995):
        """
        DQN Agent
        Args:
            num_input: number of input features
            num_outputs: number of output features (actions)
            epsilon_decay: decay rate for epsilon
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, num_outputs),
        )
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.gamma = 0.99  # Discount factor

        self.epsilon = 1.0  # Starting epsilon value
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
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
