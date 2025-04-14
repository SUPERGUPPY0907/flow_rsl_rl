import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap

class MLP_w_jac(nn.Module):
    """Multi-layer Perceptron with Jacobian computation capability.

    Args:
        input_dim (int): Dimension of input features. Default: 2
        hidden_num (int): Number of hidden units in each layer. Default: 256
        output_dim (int): Dimension of output features. Default: 2
    """

    def __init__(self, input_dim: int = 2, hidden_num: int = 256, output_dim: int = 2, activation: torch.nn.Module = torch.nn.Tanh()) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc4 = nn.Linear(hidden_num, output_dim, bias=True)
        self.activation = activation


    def forward(self,
                x_input: torch.Tensor,
                observations: torch.Tensor,
                t: torch.Tensor,
                jac: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MLP.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            observations (torch.Tensor): Observation tensor of shape (batch_size, 1)
            t (torch.Tensor): Time tensor of shape (batch_size, 1)
            jac (bool): Whether to compute Jacobian. Default: True

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, output_dim)
                - Jacobian tensor if jac=True, else None
        """
        # Combine inputs
        inputs = torch.cat([x_input, observations, t], dim=1)

        # Forward pass through the network
        x = self.activation(self.fc1(inputs))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        output = self.fc4(x) * 0.5

        # Compute Jacobian if requested
        J = self._compute_jacobian(x_input, observations, t) if jac else None

        return output, J

    def _model_func(self,
                    x_in: torch.Tensor,
                    observations: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Helper function for Jacobian computation.

        Args:
            x_in (torch.Tensor): Single input sample
            observations (torch.Tensor): Single observation sample
            t (torch.Tensor): Single time sample

        Returns:
            torch.Tensor: Output for single sample
        """
        combined = torch.cat([x_in, observations, t], dim=0).unsqueeze(0)
        x = self.activation(self.fc1(combined))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return (self.fc4(x) * 0.5).squeeze(0)

    def _compute_jacobian(self,
                          x_input: torch.Tensor,
                          observations: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of the model output with respect to input.

        Args:
            x_input (torch.Tensor): Input tensor
            observations (torch.Tensor): Observation tensor
            t (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Jacobian tensor
        """
        jacobian_fn = jacrev(self._model_func, argnums=0)
        batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0, 0))
        return batched_jacobian_fn(x_input, observations, t)



class MLP(nn.Module):
    """Multi-layer Perceptron with Jacobian computation capability.

    Args:
        input_dim (int): Dimension of input features. Default: 2
        hidden_num (int): Number of hidden units in each layer. Default: 256
        output_dim (int): Dimension of output features. Default: 2
    """

    def __init__(self, input_dim: int = 2, hidden_num: int = 256, output_dim: int = 2, activation: torch.nn.Module = torch.nn.Tanh()) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc4 = nn.Linear(hidden_num, output_dim, bias=True)
        self.activation = activation


    def forward(self,
                x_input: torch.Tensor,
                observations: torch.Tensor,
                t: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward pass of the MLP.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            observations (torch.Tensor): Observation tensor of shape (batch_size, 1)
            t (torch.Tensor): Time tensor of shape (batch_size, 1)


        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, output_dim)
        """
        # Combine inputs
        inputs = torch.cat([x_input, observations, t], dim=1)

        # Forward pass through the network
        x = self.activation(self.fc1(inputs))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        output = self.fc4(x)

        return output










