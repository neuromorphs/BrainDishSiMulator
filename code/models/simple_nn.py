import torch
import torch.nn as nn
import torch.optim as optim

class Simple_FF(nn.Module):
    def __init__(self, seed, num_inputs, num_outputs, hidden_units, lr=1e-2, gamma = 0.99):
        super().__init__()
        self.seed = seed
        torch.manual_seed(seed)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_units = hidden_units
        self.lr = lr
        self.gamma = gamma
        self.scale = 300

        # Initialize network architecture
        self.layers = []
        prev_units = num_inputs
        for hidden_unit in hidden_units:
            self.layers.append(nn.Linear(prev_units, hidden_unit))
            self.layers.append(nn.Sigmoid())  # Use ReLU as the activation function
            prev_units = hidden_unit

        self.layers.append(nn.Linear(prev_units, num_outputs))
        self.layers.append(nn.Softmax(dim=1))  # Use Softmax as the activation function for the output layer

        self.model = nn.Sequential(*self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = (state-300) / self.scale
        with torch.no_grad():
            action_values = self.forward(state)
        #print(torch.argmax(action_values).item())
        return torch.argmax(action_values).item()  # choose the action with the highest value

    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        action = torch.tensor([[action]])
        reward = torch.tensor(reward).float().unsqueeze(0)

        state = (state-300) / self.scale
        next_state = (next_state-300)/self.scale
        self.optimizer.zero_grad()

        # Compute loss
        current_Q_values = self.forward(state).gather(1, action)
        max_next_Q_values = self.forward(next_state).max(1)[0].detach()
        expected_Q_values = (reward + self.gamma * max_next_Q_values).unsqueeze(1)  # Assuming a discount factor (gamma) of 0.99
        loss = self.criterion(current_Q_values, expected_Q_values)

        # Backpropagate the loss
        loss.backward()
        self.optimizer.step()
        
    def save_model(self, path) :
        torch.save(self.model.state_dict(), path)