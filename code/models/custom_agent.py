import torch
import torch.nn as nn
import torch.optim as optim
import random
from .layers.LIF import LIF
import numpy as np

import numpy as np
from scipy.stats import poisson


def floor_to_nearest_region(y, section_height=600 / 8):
    region_index = int(y / section_height)
    floor_position = region_index
    return floor_position


def generate_spike_matrix(X, T):
    X = 1 - np.clip(X, 0, 1)
    spikes = np.zeros((T, len(X)))
    for t in range(T):
        for i in range(len(X)):
            rate = X[i]
            spikes[t, i] = 1 if np.random.rand() < poisson.pmf(1, rate) else 0
    return np.asarray(spikes, dtype=np.float32)


def generate_spike_at(x, y, total_x, total_y):
    spikes = np.zeros((total_x, total_y))
    spikes[x, y] = 1
    return np.asarray(spikes, dtype=np.float32)


def generate_rate_at_pixel(rate, y, total_y, T):
    rate_at_pixel = np.zeros((T, total_y))
    for t in range(T):
        if np.random.uniform() < rate:
            rate_at_pixel[t, y] = 1
    return np.asarray(rate_at_pixel, dtype=np.float32)


# define LIFNetwork as e sequence of LIF neurons
class LIFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0,
                 reset=False, simulation_timesteps=10, hebbian=False, eta=1e-4, use_bias=False):
        super(LIFNetwork, self).__init__()

        self.simulation_timesteps = simulation_timesteps
        layers = []

        if isinstance(hidden_size, list):
            for i in range(len(hidden_size)):
                if i == 0:
                    layers.append(LIF(input_size, hidden_size[i], tau_mem=tau_mem, tau_syn=tau_syn, dt=dt, beta=beta,
                                      reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))
                else:
                    layers.append(LIF(hidden_size[i - 1], hidden_size[i], tau_mem=tau_mem, tau_syn=tau_syn, dt=dt,
                                      beta=beta, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))

            layers.append(
                LIF(hidden_size[-1], output_size, tau_mem=20e-3, tau_syn=tau_syn, dt=dt, beta=beta, monitor="mem",
                    reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias, kernel_initializer="random_normal"))
        else:
            self.fc1 = LIF(input_size, hidden_size, tau_mem=tau_mem, tau_syn=tau_syn, dt=dt, beta=beta,
                           reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias)
            self.fc2 = LIF(hidden_size, output_size, tau_mem=20e-3, tau_syn=tau_syn, dt=dt, beta=beta, monitor="mem",
                           reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias, kernel_initializer="random_normal")
            layers.append(self.fc1)
            layers.append(self.fc2)

        self.layers = layers

        self.threshold = 1.0
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
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
            params += [param for param in layer.parameters()]

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


class CustomDeepNetwork(nn.Module):
    def __init__(self, seed, num_inputs, num_outputs, hidden_units=[2], buffer_capacity=10000,
                 batch_size=1, gamma=0.99, lr=1e-3, simulation_timesteps=10, reset=False, use_bias=True,
                 tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0):

        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.scale = 300.0
        self.simulation_timesteps = simulation_timesteps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.batch_size = batch_size
        self.gamma = gamma

        hid = hidden_units

        # define online network using kera

        input_x = Input(shape=(self.input_size))
        x1 = Dense(8, activation='relu', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(input_x)
        x2 = Dense(4, activation='relu', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(x1)
        output_x = Dense(2, activation='linear', use_bias=True, kernel_initializer="zeros", bias_initializer="zeros")(x2)

        model = Model(inputs=input_x, outputs=output_x)
        model.compile(loss='mse', optimizer=Adam(lr=lr))
        self.model = model

        # set all weights and bias to zeros
        zero_x1 = np.zeros((self.input_size, 8))
        zero_x1_b = np.zeros((8,))
        zero_x2 = np.zeros((8, 4))
        zero_x2_b = np.zeros((4,))
        zero_out = np.zeros((4, 2))
        zero_out_b = np.zeros((2,))

        # set single RELU Path
        zero_x1[1, 0] = 1
        zero_x1[0, 0] = -1
        zero_x2[0, 0] = 1
        zero_out[0, 1] = 1
        zero_out_b = np.asarray([0.3, 0])

        model.layers[1].set_weights((zero_x1, zero_x1_b))
        model.layers[2].set_weights((zero_x2, zero_x2_b))
        model.layers[3].set_weights((zero_out, zero_out_b))

        self.model = model

    def get_action(self, state):
        # convert state
        s = (torch.from_numpy(state) - 300) / self.scale

        if self.input_size == 2:
            s = s[[0, 2]]

        out = self.model.predict([[s]])
        action = np.argmax(out[0])
        return action

    def update(self, state, action, reward, next_state, done):
        pass

    def get_weights(self):
        weights = self.model.get_weights()
        return weights
