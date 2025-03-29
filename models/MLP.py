from turtle import config_dict

import torch
import torch.nn as nn
import torch.optim as optim
import math

# Standard MLP Model
class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        input_dim = self.config.input_dim
        hidden_layers = self.config.hidden_layers
        output_dim = self.config.output_dim
        activation = self.config.activation
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "ReLU".lower():
                layers.append(nn.ReLU())
            else:
                layers.append(nn.SiLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)