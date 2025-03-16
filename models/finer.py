import torch
import torch.nn as nn
import numpy as np
from helpers import debug_print


class FinerModel(nn.Module):
    """Fully-connected INR model with either SIREN or FINER activations, configurable via YAML."""
    def __init__(self, config):
        """
        Args:
            config (OmegaConf/YAML dict): Configuration file specifying model hyperparameters.
        """
        super().__init__()
        debug_print()
        self.config = config
        in_features = self.config.in_features
        out_features = self.config.out_features
        hidden_layers = self.config.num_hidden_layers
        hidden_features = self.config.hidden_size
        first_omega = self.config.init_params.w0
        hidden_omega = self.config.init_params.w1
        activation = self.config.init_params.activation

        layers = []
        layers.append(FinerLayer(in_features, hidden_features, omega_0=first_omega,
                                 activation=activation, is_first=True))

        for _ in range(hidden_layers):
            layers.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega,
                                     activation=activation))

        layers.append(FinerLayer(hidden_features, out_features, is_last=True))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        debug_print()
        return self.net(coords)


class FinerLayer(nn.Module):
    """A single INR layer with either SIREN or FINER activation."""
    def __init__(self, in_features, out_features, omega_0=30, activation="finer",
                 is_first=False, is_last=False):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float): Frequency scaling factor for activations.
            activation (str): Type of activation, either 'sine' (SIREN) or 'finer'.
            is_first (bool): Whether this is the first layer (affects initialization).
            is_last (bool): Whether this is the last layer (no activation applied).
        """
        super().__init__()
        debug_print()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.activation = activation.lower()
        self.is_last = is_last
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # Initialize weights following specified logic
        self.init_weights()

    def forward(self, x):
        debug_print()
        x = self.linear(x)
        if self.is_last:
            return x
        return self.finer_activation(x) if self.activation == "finer" else self.siren_activation(x)

    def init_weights(self):
        """Weight initialization following strict mathematical logic. https://github.com/liuzhen0212/FINERplusplus/blob/main/models.py"""
        debug_print()
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def siren_activation(self, x):
        """Standard SIREN activation: sin(ωx)"""
        debug_print()
        return torch.sin(self.omega_0 * x)

    def finer_activation(self, x):
        """FINER activation: sin(ωαx), where α scales adaptively."""
        debug_print()
        return torch.sin(self.omega_0 * self.generate_alpha(x) * x)

    def generate_alpha(self, x):
        """Adaptive scaling factor α for FINER activation."""
        debug_print()
        with torch.no_grad():
            return torch.abs(x) + 1  # Simple scaling rule