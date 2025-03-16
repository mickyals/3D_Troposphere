import torch
import torch.nn as nn
import torch.optim as optim
import math

# Standard MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "ReLU":
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)