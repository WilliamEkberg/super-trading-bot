import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a Q-network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class Agent:
    """Stock Trading Bot using PyTorch"""
    def __init__(self, state_size, strategy="t-dqn", reset_every=1000,
                 pretrained=False, model_name=None, device=None):
        self.strategy = strategy
        self.state_size = state_size    # size of input features
        self.action_size = 3            # actions: [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # Training parameters
        self.gamma = 0.95             # discount factor
        self.epsilon = 1.0            # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Device: use GPU if available, else CPU
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a pretrained model or create a new Q-network
        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = QNetwork(self.state_size, self.action_size).to(self.device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # For strategies that use a target network (t-dqn and double-dqn)
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every
            self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

    def act(self, state, is_eval=False):
        """Select an action for a given state."""
        # Explore with probability epsilon (unless evaluating)
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        # Force a buy on the first iteration
        if self.first_iter:
            self.first_iter = False
            return 1

        self.model.eval()
        # Convert state to tensor and ensure it has a batch dimension
        state_tensor = torch.FloatTensor(state).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self, batch_size):
        """Train the network on a random batch from memory."""
        if len(self.memory) < batch_size:
            return None
        
        mini_batch = random.sample(self.memory, batch_size)
        
        # Extract batch data (assuming state and next_state are stored as [state_vector])
        states = torch.FloatTensor(np.vstack([s[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        actions = torch.LongTensor([a for (s, a, r, s_next, done) in mini_batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([r for (s, a, r, s_next, done) in mini_batch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([s_next[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        dones = torch.FloatTensor([done for (s, a, r, s_next, done) in mini_batch]).to(self.device)
        
        # Current Q-values for taken actions
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Compute target Q-values based on strategy
        with torch.no_grad():
            if self.strategy == "dqn":
                next_q_values = self.model(next_states)
                max_next_q, _ = torch.max(next_q_values, dim=1)
            elif self.strategy == "t-dqn":
                if self.n_iter % self.reset_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                next_q_values = self.target_model(next_states)
                max_next_q, _ = torch.max(next_q_values, dim=1)
            elif self.strategy == "double-dqn":
                if self.n_iter % self.reset_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                # Use main network to determine best action
                next_q_main = self.model(next_states)
                best_actions = torch.argmax(next_q_main, dim=1, keepdim=True)
                next_q_target = self.target_model(next_states)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                raise NotImplementedError("Unknown strategy: " + self.strategy)
            
            # If done, no future reward is added
            targets = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute the Huber loss (smooth L1 loss)
        loss = F.smooth_l1_loss(current_q, targets)
        
        # Backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.n_iter += 1
        
        return loss.item()

    def save(self, episode):
        """Save the model parameters."""
        if self.model_name is None:
            raise ValueError("Model name not provided.")
        torch.save(self.model.state_dict(), f"models/{self.model_name}_{episode}.pth")

    def load(self):
        """Load model parameters."""
        if self.model_name is None:
            raise ValueError("Model name not provided.")
        model = QNetwork(self.state_size, self.action_size).to(self.device)
        model.load_state_dict(torch.load("models/" + self.model_name + ".pth", map_location=self.device))
        model.eval()
        return model
