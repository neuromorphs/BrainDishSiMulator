import uos as os
import urandom as random
import ustruct as struct
import utime as time
import math
from .layers.native_LIF import nLIF

def floor_to_nearest_region(y, section_height=600/8):
    region_index = int(y / section_height)
    floor_position = region_index
    return floor_position

def generate_spike_matrix(X, T):
    X = [0 if x < 0 else (1 if x > 1 else x) for x in X]
    spikes = [[0]*len(X) for _ in range(T)]
    for t in range(T):
        for i in range(len(X)):
            rate = X[i]
            spikes[t][i] = 1 if random.uniform(0, 1) < rate else 0
    return spikes

def generate_spike_at(x, y, total_x, total_y):
    spikes = [[0]*total_y for _ in range(total_x)]
    spikes[x][y] = 1
    return spikes

def generate_rate_at_pixel(rate, y, total_y, T):
    rate_at_pixel = [[0]*total_y for _ in range(T)]
    for t in range(T):
        if random.uniform(0, 1) < rate:
            rate_at_pixel[t][y] = 1

class LIFNetwork:
    def __init__(self, input_size, hidden_size, output_size, tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0,
                 reset=False, simulation_timesteps=10, hebbian=False, eta=1e-4, use_bias=False):
        self.simulation_timesteps = simulation_timesteps
        self.layers = []
        self.threshold = 1.0
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.dt = dt
        self.beta = beta

        if isinstance(hidden_size, list):
            for i in range(len(hidden_size)):
                if i == 0:
                    self.layers.append(nLIF.microLIF(input_size, hidden_size[i], dt, tau_mem, tau_syn, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))
                else:
                    self.layers.append(nLIF.microLIF(hidden_size[i - 1], hidden_size[i], dt, tau_mem, tau_syn, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))

            self.layers.append(nLIF.microLIF(hidden_size[-1], output_size, dt, tau_mem, tau_syn, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))
        else:
            self.layers.append(nLIF.microLIF(input_size, hidden_size, dt, tau_mem, tau_syn, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))
            self.layers.append(nLIF.microLIF(hidden_size, output_size, dt, tau_mem, tau_syn, reset=reset, hebbian=hebbian, eta=eta, use_bias=use_bias))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        # implement averaging over time steps here
        return x

    def reset_states(self):
        for layer in self.layers:
            layer.reset_states()

    def get_weights(self):
        return [layer.w for layer in self.layers]

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
        return urandom.sample(self.buffer, batch_size)
        # implement sampling from buffer here
        pass

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
import random

class SimonAgent:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units = [2], buffer_capacity=10000,
                 batch_size=1, gamma=0.99, lr=1e-3, simulation_timesteps=10, reset=False, use_bias=True,
                 tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0):

        self.seed = seed
        random.seed(seed)
        self.scale = 300.0
        self.simulation_timesteps = simulation_timesteps
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.batch_size = batch_size
        self.gamma = gamma
        hid = hidden_units
        self.online_network = LIFNetwork(num_inputs, hid, num_outputs, tau_mem=tau_mem,
                                         tau_syn=tau_syn, dt=dt, beta=beta, reset=reset, use_bias=use_bias,
                                         simulation_timesteps=simulation_timesteps)

    def get_action(self, state):
        # normalize state
        s = (state - 300) / self.scale

        if self.input_size == 2:
            s = [s[0], s[2]]

        # perform forward pass through network here, replacing q_values
        q_values = self.online_network.forward(s)

        # replace argmax with appropriate operation
        argmax = max(q_values)
        action = argmax
        return action

    def update(self, state, action, reward, next_state, done):
        # normalize states
        s = (state - 300) / self.scale
        next_s = (next_state - 300) / self.scale

        if self.input_size == 2:
            s = [s[0], s[2]]
            next_s = [next_s[0], next_s[2]]

        q_values = self.online_network.forward(s)

        next_q_values = self.online_network.forward(next_s)

        target_q_values = reward + (self.gamma * max(next_q_values) * (not done))

        # implement update rule here

        if done:
            print("=========================================", "done")
            self.online_network.reset_states()

    def get_weights(self):
        weights = self.online_network.get_weights()
        return weights
