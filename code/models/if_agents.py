import torch
import torch.nn as nn
import torch.optim as optim
import random
from .layers.IF import IF
import numpy as np

import numpy as np
from scipy.stats import poisson


def floor_to_nearest_region(y, section_height=600/8):
    region_index = int(y / section_height)
    floor_position = region_index
    return floor_position


def generate_spike_matrix(X, T):
    X = 1-np.clip(X, 0, 1)
    spikes = np.zeros((T, len(X)))
    for t in range(T):
        for i in range(len(X)):
            rate = X[i]
            spikes[t, i] = 1 if np.random.rand() < poisson.pmf(1, rate) else 0
    return np.asarray(spikes, dtype=np.float32)

def generate_spike_at(x,y, total_x, total_y):
    spikes = np.zeros((total_x, total_y))
    spikes[x, y] = 1
    return np.asarray(spikes, dtype=np.float32)

def generate_rate_at_pixel(rate, y, total_y, T):
    rate_at_pixel = np.zeros((T, total_y))
    for t in range(T):
        if np.random.uniform()<rate:
            rate_at_pixel[t, y] = 1
    return np.asarray(rate_at_pixel, dtype=np.float32)

# define LIFNetwork as e sequence of LIF neurons
class IFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=1e-3, beta=20.0,
                 reset=False, simulation_timesteps=10):
        super(IFNetwork, self).__init__()

        self.simulation_timesteps = simulation_timesteps
        layers = []

        if isinstance(hidden_size, list):
            for i in range(len(hidden_size)):
                if i == 0:
                    layers.append(IF(input_size, hidden_size[i], dt=dt, beta=beta, use_bias=True,
                                      reset=reset, ))
                else:
                    layers.append(IF(hidden_size[i - 1], hidden_size[i], dt=dt,use_bias=True,
                                      beta=beta, reset=reset))

            layers.append(IF(hidden_size[-1], output_size, tau_mem=self.simulation_timesteps*dt, dt=dt, beta=beta, monitor="mem",
                              reset=reset,use_bias=True))  # last layer
        else:
            self.fc1 = IF(input_size, hidden_size, dt=dt, beta=beta,
                           reset=False,use_bias=True)
            self.fc2 = IF(hidden_size, output_size, dt=dt, beta=beta, monitor="mem",
                           reset=False, use_bias=True)
            layers.append(self.fc1)
            layers.append(self.fc2)

        self.layers = layers

        self.threshold = 1.0
        self.dt = dt
        self.beta = beta

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = torch.nn.AvgPool2d((self.simulation_timesteps, 1))(x)
        # remove second axis
        # x = x.squeeze(1)
        return x

    def reset_states(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_states()

    def get_weights(self):
        return [x.get_weights() for x in self.layers]

    # override parameters() function and return the weights of the network
    def parameters(self):
        params = []
        for layer in self.layers:
            params+=[param for param in layer.parameters()]

        return params


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
class IFELSEAgent:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units = [2], buffer_capacity=10000,
                 batch_size=1, gamma=0.8, lr=1e-3, simulation_timesteps=100, dt=1e-3, beta=20.0):

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.scale = 600.0
        self.simulation_timesteps = simulation_timesteps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.batch_size = batch_size
        self.gamma = gamma

        hid = hidden_units
        self.online_network = IFNetwork(num_inputs, hid, num_outputs, dt=dt, beta=beta,
                                         simulation_timesteps=simulation_timesteps).to(self.device)

        # add optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
    def get_action(self, state):
        # convert state
        y = state[[2]]
        y_8 = floor_to_nearest_region(y, 600 / self.input_size)
        x = state[[0]]
        rate = x / 600
        spike_matrix = generate_rate_at_pixel(rate, y_8, self.input_size, self.simulation_timesteps)

        # add batch dimension
        state = torch.from_numpy(spike_matrix).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.online_network(state)

            argmax = q_values.sum(1).argmax(1)
            action = argmax[0].item()
        return action

    def update(self, state, action, reward, next_state, done):
        """Update using QDN update rule"""

        # convert state
        y = state[[2]]
        y_8 = floor_to_nearest_region(y, 600/self.input_size)
        x = state[[0]]
        rate = x / 600
        spike_matrix = generate_rate_at_pixel(rate, y_8, self.input_size, self.simulation_timesteps)

        # convert next state
        y = next_state[[2]]
        y_8 = floor_to_nearest_region(y, 600/self.input_size)
        x = next_state[[0]]
        rate = x / 600
        next_spike_matrix = generate_rate_at_pixel(rate, y_8, self.input_size, self.simulation_timesteps)


        # generate spikes based on poisson distribution for a certain number of timesteps
        #spike_matrix = generate_spike_matrix(state, self.simulation_timesteps)
        #next_spike_matrix = generate_spike_matrix(next_state, self.simulation_timesteps)

        # transform to tensor and add batch dimension
        spike_matrix = torch.from_numpy(spike_matrix).unsqueeze(0).to(self.device)
        next_spike_matrix = torch.from_numpy(next_spike_matrix).unsqueeze(0).to(self.device)


        with torch.no_grad():
            q_values = self.online_network(spike_matrix).sum(1)[:,action]

        next_q_values = self.online_network(next_spike_matrix)
        next_q_values = next_q_values.sum(1).max(1)[0]

        print("Q,N", q_values, next_q_values)

        target_q_values = reward + (self.gamma * next_q_values * (~done))
        if done:
            print("=========================================", "done")
            self.online_network.reset_states()

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_weights(self):
        weights = self.online_network.get_weights()
        return weights
