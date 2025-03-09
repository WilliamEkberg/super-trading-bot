import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class brain(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=40, hidden_dim_start_and_end = 40):
        super(brain, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim_start_and_end)
        self.lRelu1 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(hidden_dim_start_and_end, hidden_dim)
        self.lRelu2 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lRelu3 = nn.LeakyReLU(0.01)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim_start_and_end)
        self.lRelu4 = nn.LeakyReLU(0.01)
        self.fc5 = nn.Linear(hidden_dim_start_and_end, action_size)
        
    def forward(self, x):
        #x = self.lRelu1(self.fc1(x))
        #x = self.lRelu2(self.fc2(x))
        #x = self.lRelu3(self.fc3(x))
        #x = self.lRelu4(self.fc4(x))

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
