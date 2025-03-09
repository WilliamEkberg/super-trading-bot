import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer_model import Transformer
from einops import rearrange, repeat

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

class TransformedBrain(nn.Module):
    def __init__(self, state_size, 
                 action_size, 
                 hidden_dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 window_size,
                 pool='cls'):
        super().__init__()
        self.transformer = Transformer(hidden_dim, depth, heads, dim_head, mlp_dim, dropout=0.1)
        self.proj_out = nn.Linear(hidden_dim, action_size)
        self.pool = pool

        self.to_embedding = nn.Sequential(
            nn.LayerNorm(state_size),
            nn.Linear(state_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1,  window_size, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x):
        x = self.to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        # if self.pool == 'cls':
        #     x = x[:,0,:]
        # else:
        x = torch.mean(x, dim=1)
        x = self.proj_out(x)

        return x

if __name__ == "__main__":
    batch_size = 2
    sequence_length = 10
    embedding_dim = 128

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    # Instantiate the Transformer
    transformer = Transformer(
        dim=128,        # Dimension of embeddings
        depth=4,        # Number of layers
        heads=8,        # Number of attention heads
        dim_head=64,    # Dimension per attention head
        mlp_dim=256,    # Hidden dimension for MLP
        dropout=0.1     # Dropout rate
    )

    # Forward pass
    output = transformer(x)
    print("Output shape:", output.shape)