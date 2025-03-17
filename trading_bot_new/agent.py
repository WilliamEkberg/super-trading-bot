import random
random.seed(0)
from collections import deque
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Brain import brain, TransformedBrain
from loguru import logger

class Agent:
    """Stock Trading Bot using PyTorch"""
    def __init__(self, state_size, lr, strategy="t-dqn", reset_every=1000,
                 pretrained=False, model_type="FF",model_name=None, device=None, mdp=""):
        self.strategy = strategy
        self.state_size = state_size
        if mdp == "10%_steps": self.action_size = 3 #up, down, stay
        elif mdp == "all_or_nothing": self.action_size = 2 #100%, 0%
        elif mdp == "all_10%_steps": self.action_size = 11 #10 states plus state_0
        else: raise ValueError("wrong Methode is used")
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True
        #self.model_type = model_type
        
        # Training parameters
        if self.strategy == "Transformer":
            #if self.model_type == "FF": raise ValueError("FF and Transformer doesn't work together")
            if mdp == "10%_steps":
                self.gamma = 0.95             # discount factor
                self.epsilon = 1           # exploration rate #Q-learning: 0.1 (all 10 actions), 0.05 (two actions) Transformer: ??
                self.epsilon_min = 0.2
                self.epsilon = 1           # exploration rate #Q-learning: 0.1 (all 10 actions), 0.05 (two actions) Transformer: ??
                self.epsilon_min = 0.2
                self.epsilon_decay = 0.9995  #Q-learning: 0.995 (two actions), 0.9999 (all 10 actions) Transformer: ??
                self.learning_rate = 1e-6 #Q-learning: 0.0001 (1e-4) #transformer: lr
                self.learning_rate = 1e-6 #Q-learning: 0.0001 (1e-4) #transformer: lr
            elif mdp == "all_or_nothing":
                self.gamma = 0.95             
                self.epsilon = 1          
                self.epsilon_min = 0.05
                self.epsilon = 1          
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.995 
                self.learning_rate = 1e-6 
                self.learning_rate = 1e-6 
            if mdp == "all_10%_steps":
                self.gamma = 0.95             
                self.epsilon = 1         
                self.epsilon_min = 0.2
                self.epsilon_decay = 0.9995  
                self.learning_rate = 1e-6 
                self.epsilon = 1         
                self.epsilon_min = 0.2
                self.epsilon_decay = 0.9995  
                self.learning_rate = 1e-6 

        elif self.strategy == "t-dqn" or self.strategy == "double-dqn": #DQO or DDQO
            #if self.model_type != "FF": raise ValueError("FF must be used for DQN/DDQN")
            if mdp == "10%_steps":
                self.gamma = 0.95            
                self.epsilon = 0.5          
                self.epsilon_min = 0.05
                self.epsilon = 0.5          
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.9995  
                self.learning_rate = lr
                self.learning_rate = lr
            elif mdp == "all_or_nothing":
                self.gamma = 0.95             
                self.epsilon = 0.5           
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.995  
                self.learning_rate = 0.00001
                self.epsilon = 0.5           
                self.epsilon_min = 0.05
                self.epsilon_decay = 0.995  
                self.learning_rate = 0.0001
            if mdp == "all_10%_steps":
                self.gamma = 0.95            
                self.epsilon = 0.8          
                self.epsilon = 0.8          
                self.epsilon_min = 0.1
                self.epsilon_decay = 0.995
                self.learning_rate = 0.00001
                self.epsilon_decay = 0.995
                self.learning_rate = 0.001
        else: raise ValueError("No appropriate strategy given")

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # To load a pretrained model or create a new Brain
        if pretrained and self.model_name is not None:
            self.model = self.load()
        elif self.strategy == "double-dqn" or self.strategy == "t-dqn":
            self.model = brain(self.state_size, self.action_size).to(self.device)
        elif self.strategy == "Transformer":
            self.model = TransformedBrain(self.state_size,
                                           self.action_size,
                                           hidden_dim=512,
                                           depth=2,
                                           heads=3,
                                           dim_head=256,
                                           mlp_dim=256,
                                           window_size=50).to(device)
        else: raise ValueError("wrong strategy")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Initialized model with parameters: {total_params}")

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3) #weight_decay=1e-4
        #self.scheduler1 = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # For (t-dqn and double-dqn)
        if self.strategy in ["t-dqn", "double-dqn", "Transformer"]:
            self.n_iter = 1
            self.reset_every = reset_every
            if self.strategy == "double-dqn" or self.strategy == "t-dqn":
                self.target_model = brain(self.state_size, self.action_size).to(self.device)
            elif self.strategy == "Transformer":
                self.target_model = TransformedBrain(self.state_size,
                                           self.action_size,
                                           hidden_dim=512,
                                           depth=2,
                                           heads=3,
                                           dim_head=256,
                                           mlp_dim=256,
                                           window_size=50).to(device)
            else: raise ValueError("wrong strategy")
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        ##Buy if first iteration
        if self.first_iter:
            self.first_iter = False
            return 1

        self.model.eval()  # Evaluation mode

        if not isinstance(self.model, TransformedBrain):
            state = state.squeeze(0)

        with torch.no_grad():
            q_values = self.model(state)
        
        self.model.train()  # Switch back to training mode
        
        # Now, q_values should be of shape (1, action_size) so argmax gives a 1-element tensor.

        return int(torch.argmax(q_values, dim=1).item())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        mini_batch = random.sample(self.memory, batch_size)
        if self.strategy == "double-dqn" or self.strategy == "t-dqn":
            states = torch.FloatTensor(np.vstack([s[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
            next_states = torch.FloatTensor(np.vstack([s_next[0] for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        elif self.strategy == "Transformer":
            states = torch.FloatTensor(np.vstack([s for (s, a, r, s_next, done) in mini_batch])).to(self.device)
            next_states = torch.FloatTensor(np.vstack([s_next for (s, a, r, s_next, done) in mini_batch])).to(self.device)
        else: raise ValueError("wrong strategy")
        actions = torch.LongTensor([a for (s, a, r, s_next, done) in mini_batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([r for (s, a, r, s_next, done) in mini_batch]).to(self.device)
        dones = torch.FloatTensor([done for (s, a, r, s_next, done) in mini_batch]).to(self.device)

        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        #Compute target Q-values based on strategy
        with torch.no_grad():
            if self.strategy == "dqn":
                next_q_vals = self.model(next_states)
                max_next_q, _ = torch.max(next_q_vals, dim=1)
            elif self.strategy == "t-dqn" or self.strategy == "Transformer":
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
        #self.scheduler1.step()
        
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
