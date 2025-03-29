import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import math
from helpers import debug_print


# class FinerModel(nn.Module):
#     """Fully-connected INR model with either SIREN or FINER activations, configurable via YAML."""
#     def __init__(self, config):
#         """
#         Args:
#             config (OmegaConf/YAML dict): Configuration file specifying model hyperparameters.
#         """
#         super().__init__()
#         #debug_print()
#         self.config = config
#         in_features = self.config.in_features
#         out_features = self.config.out_features
#         hidden_layers = self.config.num_hidden_layers
#         hidden_features = self.config.hidden_size
#         first_omega = self.config.init_params.w0
#         hidden_omega = self.config.init_params.w1
#         activation = self.config.init_params.activation
#
#         layers = []
#         layers.append(FinerLayer(in_features, hidden_features, omega_0=first_omega,
#                                  activation=activation, is_first=True))
#
#         for _ in range(hidden_layers):
#             layers.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega,
#                                      activation=activation))
#
#         layers.append(FinerLayer(hidden_features, out_features, is_last=True))
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, coords):
#         #debug_print()
#         return self.net(coords)
#
#
# class FinerLayer(nn.Module):
#     """A single INR layer with either SIREN or FINER activation."""
#     def __init__(self, in_features, out_features, omega_0=30, activation="finer",
#                  is_first=False, is_last=False):
#         """
#         Args:
#             in_features (int): Number of input features.
#             out_features (int): Number of output features.
#             omega_0 (float): Frequency scaling factor for activations.
#             activation (str): Type of activation, either 'sine' (SIREN) or 'finer'.
#             is_first (bool): Whether this is the first layer (affects initialization).
#             is_last (bool): Whether this is the last layer (no activation applied).
#         """
#         super().__init__()
#         #debug_print()
#         self.in_features = in_features
#         self.omega_0 = omega_0
#         self.activation = activation.lower()
#         self.is_last = is_last
#         self.is_first = is_first
#         self.linear = nn.Linear(in_features, out_features)
#
#         # Initialize weights following specified logic
#         self.init_weights()
#
#     def forward(self, x):
#         #debug_print()
#         x = self.linear(x)
#         if self.is_last:
#             return x
#         return self.finer_activation(x) if self.activation == "finer" else self.siren_activation(x)
#
#     def init_weights(self):
#         """Weight initialization following strict mathematical logic. https://github.com/liuzhen0212/FINERplusplus/blob/main/models.py"""
#         #debug_print()
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
#             else:
#                 bound = np.sqrt(6 / self.in_features) / self.omega_0
#                 self.linear.weight.uniform_(-bound, bound)
#
#     def siren_activation(self, x):
#         """Standard SIREN activation: sin(ωx)"""
#         #debug_print()
#         return torch.sin(self.omega_0 * x)
#
#     def finer_activation(self, x):
#         """FINER activation: sin(ωαx), where α scales adaptively."""
#         #debug_print()
#         return torch.sin(self.omega_0 * self.generate_alpha(x) * x)
#
#     def generate_alpha(self, x):
#         """Adaptive scaling factor α for FINER activation."""
#         #debug_print()
#         with torch.no_grad():
#             return torch.abs(x) + 1  # Simple scaling rule




## FINER
def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)


def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)
    ## Default: Pytorch initialization


def init_weights(m, omega=1, c=1, is_first=False):  # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in  # SIREN
        else:
            bound = math.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)


def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)  # SIREN initialization
    ## Default: Pytorch initialization


class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30,
                 is_first=False, is_last=False,
                 init_method='sine', init_gain=1, fbs=None, hbs=None,
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last  ## no activation
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)

    def forward(self, input):
        wx_b = self.linear(input)
        if not self.is_last:
            return self.finer_activation(wx_b, self.omega)
        return wx_b  # is_last==True

    def finer_activation(self, x, omega=1, alphaType=None, alphaReqGrad=False):
        return torch.sin(omega * self.generate_alpha(x, alphaType, alphaReqGrad) * x)

    def generate_alpha(self, x, alphaType=None, alphaReqGrad=False):
        with torch.no_grad():
            return torch.abs(x) + 1


class FinerModel(nn.Module):
    def __init__(self, config, hbs=None):
        super().__init__()

        self.in_features = config.in_features
        self.out_features = config.out_features
        self.hidden_layers = config.num_hidden_layers
        self.hidden_features = config.hidden_size
        self.first_omega = config.init_params.w0
        self.hidden_omega = config.init_params.w1
        self.init_gain = config.init_params.init_gain  # Read from config
        self.fbs = config.init_params.fbs
        self.init_method = config.init_params.init_method
        self.alphaType = config.init_params.alphatype
        self.alphaReqGrad = config.init_params.alphareqgrad


        self.net = []
        self.net.append(FinerLayer(self.in_features, self.hidden_features, is_first=True,
                                   omega=self.first_omega,
                                   init_method=self.init_method, init_gain=self.init_gain, fbs=self.fbs,
                                   alphaType=self.alphaType, alphaReqGrad=self.alphaReqGrad))

        for i in range(self.hidden_layers):
            self.net.append(FinerLayer(self.hidden_features, self.hidden_features,
                                       omega=self.hidden_omega,
                                       init_method=self.init_method, init_gain=self.init_gain, hbs=hbs,
                                       alphaType=self.alphaType, alphaReqGrad=self.alphaReqGrad))

        self.net.append(FinerLayer(self.hidden_features, self.out_features, is_last=True,
                                   omega=self.hidden_omega,
                                   init_method=self.init_method, init_gain=self.init_gain, hbs=hbs))  # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)
