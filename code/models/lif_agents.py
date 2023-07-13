import torch
import torch.nn as nn
import torch.optim as optim
import random
from .layers.LIF import LIF
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
class LIFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0,
                 reset=False, simulation_timesteps=10, hebbian=False, eta=1e-4):
        super(LIFNetwork, self).__init__()

        self.simulation_timesteps = simulation_timesteps
        layers = []

        if isinstance(hidden_size, list):
            for i in range(len(hidden_size)):
                if i == 0:
                    layers.append(LIF(input_size, hidden_size[i], tau_mem=tau_mem, tau_syn=tau_syn, dt=dt, beta=beta,
                                      reset=reset, hebbian=hebbian, eta=eta))
                else:
                    layers.append(LIF(hidden_size[i - 1], hidden_size[i], tau_mem=tau_mem, tau_syn=tau_syn, dt=dt,
                                      beta=beta, reset=reset, hebbian=hebbian, eta=eta))

            layers.append(LIF(hidden_size[-1], output_size, tau_mem=self.simulation_timesteps*dt, tau_syn=tau_syn, dt=dt, beta=beta, monitor="mem",
                              reset=reset, hebbian=hebbian, eta=eta))
        else:
            self.fc1 = LIF(input_size, hidden_size, tau_mem=tau_mem, tau_syn=tau_syn, dt=dt, beta=beta,
                           reset=False, hebbian=hebbian, eta=eta)
            self.fc2 = LIF(hidden_size, output_size, tau_mem=self.simulation_timesteps*dt, tau_syn=tau_syn, dt=dt, beta=beta, monitor="mem",
                           reset=False, hebbian=hebbian, eta=eta)
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


class LIFDQNAgent:
    def __init__(self, seed, num_inputs, num_outputs, buffer_capacity=10000,
                 batch_size=1, gamma=0.99, lr=1e-3, simulation_timesteps=10,
                 tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = num_inputs
        self.output_size = num_outputs
        self.batch_size = batch_size
        self.gamma = gamma

        self.online_network = LIFNetwork(num_inputs, 1, num_outputs, tau_mem=tau_mem,
                                         tau_syn=tau_syn, dt=dt, beta=beta,
                                         simulation_timesteps=simulation_timesteps).to(self.device)
        self.target_network = LIFNetwork(num_inputs, 1, num_outputs,
                                         tau_mem=tau_mem, tau_syn=tau_syn,
                                         dt=dt, beta=beta,simulation_timesteps=simulation_timesteps
                                         ).to(self.device)
        self.simulation_timesteps = simulation_timesteps
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        state = state / 600

        # repeat state for 10 timesteps
        state = state.repeat(1, self.simulation_timesteps, 1)

        with torch.no_grad():
            q_values = self.online_network(state)
            argmax = q_values.sum(1).argmax(1)
            action = argmax[0].item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.store_transition(state, action, next_state, reward, done)
        self.update_network()
        self.update_target_network()

    def store_transition(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        self.buffer.push(transition)

    def update_network(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[2]).to(self.device)
        rewards = torch.FloatTensor(batch[3]).unsqueeze(1).to(self.device)
        dones = torch.BoolTensor(batch[4]).unsqueeze(1).to(self.device)

        # repeat state for 10 timesteps
        states = states.unsqueeze(1).repeat(1, self.simulation_timesteps, 1)
        next_states = next_states.unsqueeze(1).repeat(1, self.simulation_timesteps, 1)
        q_values = self.online_network(states).sum(1)
        q_values = q_values.gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).sum(1).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
            if dones[0]:
                print("=========================================", "done")
            #   self.target_network.reset_states()
            #  self.online_network.reset_states()

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    def get_weights(self):
        weights = self.online_network.get_weights()
        return weights


class HebbianAgent:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units = [2], buffer_capacity=10000,
                 batch_size=1, gamma=0.8, lr=1e-3, simulation_timesteps=100,
                 tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0):

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
        self.online_network = LIFNetwork(num_inputs, hid, num_outputs, tau_mem=tau_mem,
                                         tau_syn=tau_syn, dt=dt, beta=beta, hebbian=True, lr=lr,
                                         simulation_timesteps=simulation_timesteps).to(self.device)

        # add optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
    def get_action(self, state):
        # convert state
        y = state[[2]]
        y_8 = floor_to_nearest_region(y, 600 / 8)
        x = state[[0]]
        x_10 = floor_to_nearest_region(x, 600 / 10)
        spike_matrix = generate_spike_at(x_10, y_8, 10, 8)

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
        y_8 = floor_to_nearest_region(y, 600/8)
        x = state[[0]]
        x_10 = floor_to_nearest_region(x, 600/10)
        spike_matrix = generate_spike_at(x_10, y_8, 10, 8)

        # convert next state
        y = next_state[[2]]
        y_8 = floor_to_nearest_region(y, 600/8)
        x = next_state[[0]]
        x_10 = floor_to_nearest_region(x, 600/10)
        next_spike_matrix = generate_spike_at(x_10, y_8, 10, 8)


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




class LIFELSEAgent:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units = [2], buffer_capacity=10000,
                 batch_size=1, gamma=0.8, lr=1e-3, simulation_timesteps=100,
                 tau_mem=20e-3, tau_syn=10e-3, dt=1e-3, beta=20.0):

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
        self.online_network = LIFNetwork(num_inputs, hid, num_outputs, tau_mem=tau_mem,
                                         tau_syn=tau_syn, dt=dt, beta=beta,
                                         simulation_timesteps=simulation_timesteps).to(self.device)

        # add optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
    def get_action(self, state):
        # convert state
        y = state[[2]]
        y_8 = floor_to_nearest_region(y, 600 / 8)
        x = state[[0]]
        x_10 = floor_to_nearest_region(x, 600 / 10)
        spike_matrix = generate_spike_at(x_10, y_8, 10, 8)

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
        y_8 = floor_to_nearest_region(y, 600/8)
        x = state[[0]]
        x_10 = floor_to_nearest_region(x, 600/10)
        spike_matrix = generate_spike_at(x_10, y_8, 10, 8)

        # convert next state
        y = next_state[[2]]
        y_8 = floor_to_nearest_region(y, 600/8)
        x = next_state[[0]]
        x_10 = floor_to_nearest_region(x, 600/10)
        next_spike_matrix = generate_spike_at(x_10, y_8, 10, 8)


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
    
    
class simple_LIF_else:
    def __init__(self, threshold=1.0, current_scale = 10.):
        self.threshold = threshold  # Spike threshold for the LIF neuron
        self.membrane_potential = 0.0  # Membrane potential for the LIF neuron
        self.current_scale = current_scale
    def process_input(self, y_ball, y_paddle):
        # Generate input to the LIF neuron based on y position of ball and paddle
        if y_ball > y_paddle:
            return 1.0
        else:
            return -1

    def update(self, y_ball, y_paddle, dt=0.1, tau=0.1):
        # Implement LIF neuron dynamics
        input_current = self.process_input(y_ball, y_paddle)
        dV = (-self.membrane_potential + input_current) / tau
        self.membrane_potential += dV * dt

        # Generate spike and reset potential if threshold is reached
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.spikes = np.array([[1,0,0,0]])
            return 1  # Move paddle up
        else:
            self.spikes = np.array([[0,0,0,0]])
            return -1  # Move paddle down
        
    def get_spikes(self) :
        return self.spikes
        
        
        
class simple_conductance_LIF:
    def __init__(self, threshold=1.0, conductance=0.1, reversal_potential=0.0):
        self.threshold = threshold
        self.membrane_potential = 0.0
        self.conductance = conductance
        self.reversal_potential = reversal_potential

    def process_input(self, y_ball, y_paddle):
        if y_ball > y_paddle:
            return 1.0
        else:
            return -1

    def update(self, y_ball, y_paddle, dt=0.1, tau=0.1):
        input_current = self.process_input(y_ball, y_paddle)
        dV = (-self.membrane_potential + self.conductance * (input_current - self.reversal_potential)) / tau
        self.membrane_potential += dV * dt

        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            return 1  # Move paddle up
        else:
            return -1  # Move paddle down