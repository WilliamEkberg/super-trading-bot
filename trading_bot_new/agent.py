import random
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Brain import brain

class Agent:
    """Stock Trading Bot using PyTorch"""
    def __init__(self, state_size, strategy="t-dqn", reset_every=1000,
                 pretrained=False, model_name=None, device=None):
        self.strategy = strategy
        self.state_size = state_size 
        self.action_size = 3            # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # Training parameters
        self.gamma = 0.95             # discount factor
        self.epsilon = 1.0           # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
       
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # To load a pretrained model or create a new Brain
        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = brain(self.state_size, self.action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # For (t-dqn and double-dqn)
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every
            self.target_model = brain(self.state_size, self.action_size).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Buy if first iteration
        if self.first_iter:
            self.first_iter = False
            return 1

        self.model.eval()  # Evaluation mode
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # If the state tensor has an extra dimension (i.e. shape (1,1, state_size)),
        # squeeze it so that the shape becomes (1, state_size)
        if state_tensor.dim() == 3 and state_tensor.size(1) == 1:
            state_tensor = state_tensor.squeeze(1)
        
        # Alternatively, if state_tensor.dim() == 1, add a batch dimension:
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        self.model.train()  # Switch back to training mode
        # Now, q_values should be of shape (1, action_size) so argmax gives a 1-element tensor.
        return int(torch.argmax(q_values, dim=1).item())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        mini_batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.vstack([s[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        actions = torch.LongTensor([a for (s, a, r, s_next, done) in mini_batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([r for (s, a, r, s_next, done) in mini_batch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([s_next[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        dones = torch.FloatTensor([done for (s, a, r, s_next, done) in mini_batch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        #Compute target Q-values based on strategy
        with torch.no_grad():
            if self.strategy == "dqn":
                next_q_vals = self.model(next_states)
                max_next_q, _ = torch.max(next_q_vals, dim=1)
            elif self.strategy == "t-dqn":
                if self.n_iter % self.reset_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                next_q_vals = self.target_model(next_states)
                max_next_q, _ = torch.max(next_q_vals, dim=1)
            elif self.strategy == "double-dqn":
                if self.n_iter % self.reset_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
        
                next_q_main = self.model(next_states)
                best_actions = torch.argmax(next_q_main, dim=1, keepdim=True)
                next_q_target = self.target_model(next_states)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                raise NotImplementedError("Unknown strategy: " + self.strategy)
            
            
            targets = rewards + (1 - dones) * self.gamma * max_next_q #If done, no future reward is added
        
        loss = F.smooth_l1_loss(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.n_iter += 1
        
        return loss.item()

    def save(self, episode):
        if self.model_name is None:
            raise ValueError("Model name not provided.")
        os.makedirs("models", exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(self.model.state_dict(), f"models/{self.model_name}_{episode}.pth")

    def load(self):
        if self.model_name is None:
            raise ValueError("Model name not provided.")
        model = brain(self.state_size, self.action_size).to(self.device)
        model.load_state_dict(torch.load("models/" + self.model_name + ".pth", map_location=self.device))
        model.eval()
        return model
