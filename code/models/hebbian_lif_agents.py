import torch
import torch.nn as nn
import torch.optim as optim
import random
from layers.LIF import LIF


# define LIFNetwork as e sequence of LIF neurons
class LIFNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tau_mem=2e-3, tau_syn=1e-3, dt=1e-3, beta=20.0):
        super(LIFNetwork, self).__init__()
        self.fc1 = LIF(input_size, hidden_size, tau_mem, tau_syn, dt, beta, reset_states=True)
        self.fc2 = LIF(hidden_size, output_size, tau_mem=10e-3, tau_syn=tau_syn, dt=dt, beta=20, monitor="mem",
                       reset_states=True)
        self.avg = nn.AvgPool2d(10, 1)
        self.threshold = 1.0
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.dt = dt
        self.beta = beta

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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

    def __len__(self):
        return len(self.buffer)


class LIFDQNAgent:
    def __init__(self, input_size, hidden_size, output_size, buffer_capacity=10000, batch_size=32, gamma=0.99, lr=0.001, simulation_timesteps=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.online_network = LIFNetwork(input_size, hidden_size, output_size).to(self.device)
        self.target_network = LIFNetwork(input_size, hidden_size, output_size).to(self.device)
        self.simulation_timesteps = simulation_timesteps
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # repeat state for 10 timesteps
        scale = 100
        state = state.repeat(1, self.simulation_timesteps, 1) * scale

        with torch.no_grad():
            q_values = self.online_network(state)
            argmax = q_values.sum(1).argmax(1)
            action = argmax[0].item()
        return action

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
        scale = 100
        states = states.unsqueeze(1).repeat(1, self.simulation_timesteps, 1) * scale
        next_states = next_states.unsqueeze(1).repeat(1, self.simulation_timesteps, 1) * scale
        q_values = self.online_network(states).sum(1)
        q_values = q_values.gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).sum(1).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
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
