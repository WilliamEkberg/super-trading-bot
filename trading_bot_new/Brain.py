import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class brain(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256, hidden_dim_start_and_end = 128):
        super(brain, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim_start_and_end)
        self.fc2 = nn.Linear(hidden_dim_start_and_end, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim_start_and_end)
        self.fc5 = nn.Linear(hidden_dim_start_and_end, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
