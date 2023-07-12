import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

class LIFNeuron:
    def __init__(self, tau_mem=5e-3, tau_syn=10e-3, dt=1e-3):
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.dt = dt
        self.V = 0
        self.I_syn = 0

    def step(self, I_ext):
        dV = (-self.V + I_ext) / self.tau_mem * self.dt
        self.V += dV
        self.I_syn *= (1 - self.dt / self.tau_syn)
        if self.V > 1:
            self.V = 0
            return 1
        return 0

    def update_synapse(self, pre_syn_activity):
        self.I_syn += pre_syn_activity


class LIF_ShadowNetwork:
    def __init__(self, seed, num_inputs, num_outputs, hidden_units, tau_mem=5e-3, tau_syn=10e-3, lr=1e-2, simulation_timesteps=10, dt=1e-3):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_units = hidden_units
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.lr = lr
        self.simulation_timesteps = simulation_timesteps
        self.dt = dt

        # Initialize neurons
        self.neurons = [[LIFNeuron(self.tau_mem, self.tau_syn, self.dt) for _ in range(n)] for n in self.hidden_units + [self.num_outputs]]

        # Initialize weights
        self.weights = [self.rng.normal(size=(n_pre, n_post)) for n_pre, n_post in zip([self.num_inputs] + hidden_units, hidden_units + [self.num_outputs])]
        self.spikes = []
        
    def get_action(self, state):
        # Initialize spikes as a numpy array
        self.spikes = np.zeros((len(self.neurons), max([len(neuron_layer) for neuron_layer in self.neurons])), dtype=np.uint8)
        # Feed input through network
        spikes = np.array([neuron.step(np.dot(state, self.weights[0][:, i]) + neuron.I_syn) for i, neuron in enumerate(self.neurons[0])])
        self.spikes[0, :len(spikes)] = spikes
        for l in range(1, len(self.neurons)):
            spikes = np.array([neuron.step(np.dot(spikes, self.weights[l][:, i]) + neuron.I_syn) for i, neuron in enumerate(self.neurons[l])])
            self.spikes[l, :len(spikes)] = spikes
        # Choose action based on output layer
        if np.argmax(spikes) == 0 :
            return -1 
        else :
            return 1
        
    def get_spikes(self):
        return self.spikes


    def update(self, state, action, reward, next_state, done):
        # Run multiple timesteps of the network simulation
        for _ in range(self.simulation_timesteps):
            self.get_action(state)
        
        # Compute error
        error = reward - self.neurons[-1][action].V

        # Update weights using simple gradient descent
        for l in range(len(self.weights) - 1, -1, -1):
            dW = self.lr * error * np.outer([neuron.V for neuron in self.neurons[l - 1]] if l > 0 else state, [neuron.V for neuron in self.neurons[l]])
            self.weights[l] += dW

        # Update synapses
        for layer in self.neurons:
            for neuron in layer:
                neuron.update_synapse(neuron.V)