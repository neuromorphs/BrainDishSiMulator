import torch
import torch.nn as nn
import torch.optim as optim
import random

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        layers = []

        if isinstance(hidden_size, list):
            for i in range(len(hidden_size)):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size[i]))
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))
                    layers.append(nn.Tanh())

            layers.append(nn.Linear(hidden_size[-1], output_size))
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            layers.append(self.fc1)
            layers.append(nn.ReLU())
            layers.append(self.fc2)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class HugoAgent:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units = [2], buffer_capacity=10000,
                 batch_size=1, gamma=0.99, lr=1e-3):

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.scale = 300.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.batch_size = batch_size
        self.gamma = gamma

        hid = hidden_units
        self.online_network = FeedForwardNetwork(num_inputs, hid, num_outputs).to(self.device)

        # add optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)

        # add optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
    def get_action(self, state):
        # convert state
        s = (torch.from_numpy(state) - 300) / self.scale

        if self.input_size == 2:
            s = s[[0, 2]]

        # add batch dimension
        s = s.unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            q_values = self.online_network(s)

            argmax = q_values.argmax(1)  # no sum along dimension
            action = argmax.item()
        return action

    def update(self, state, action, reward, next_state, done):
        """Update using QDN update rule"""

        # convert state
        s = (torch.from_numpy(state) - 300) / self.scale
        next_s = (torch.from_numpy(next_state) -300 )/ self.scale

        if self.input_size == 2:
            s = s[[0, 2]]
            next_s = next_s[[0, 2]]

        # add batch dimension
        s = s.unsqueeze(0).float().to(self.device)
        next_s = next_s.unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            q_values = self.online_network(s)[0, action]  # no sum along dimension

        next_q_values = self.online_network(next_s)
        next_q_values = next_q_values.max(1)[0]

        #print(q_values, next_q_values)

        target_q_values = reward + (self.gamma * next_q_values * (~done))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        weights = [param.data for param in self.online_network.parameters()]
        return weights
