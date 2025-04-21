# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.reflow.flow import MLP, MLP_w_jac, Flow

from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap

from torchviz import make_dot

class ActorCriticFlow(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        flow_num_steps: int = 5,
        flow_module_type: str = "MLP",
        device = torch.device("cpu"),
        flow_interations: int = 5,
        flow_distill_batch_size: int = 256,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims:list = [256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticFlow.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # self.N = flow_num_steps
        self.distillation_ites = flow_interations
        self.flow_distill_batch_size = flow_distill_batch_size

        self.a_o_dim = num_actor_obs
        self.c_o_dim = num_critic_obs
        self.a_dim = num_actions

        self.device = device
        self.last_log_probs = None
        #### actor input
        mlp_input_dim_a = self.a_o_dim + self.a_dim
        #### critic input
        mlp_input_dim_c = self.c_o_dim

        # Policy
        self.actor = Flow(input_dim = mlp_input_dim_a, output_dim = self.a_dim, a_dim = self.a_dim, actor_hidden_dim=actor_hidden_dims, activation = activation, N = flow_num_steps, device = device)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor.vec_field}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        raise NotImplementedError

    @property
    def action_std(self):
        # return self.distribution.stddev
        return torch.tensor(0.1, device=self.device)

    @property
    def entropy(self):
        return torch.tensor(0., device=self.device)

    def update_distribution(self, observations):
        raise NotImplementedError

    def act(self, observations, **kwargs):
        action, log_probs = self.actor(observations, jac = True)
        self.last_log_probs = log_probs
        return action

    def get_actions_log_prob(self, actions):
        if self.last_log_probs is None:
            raise ValueError("No log_probs stored. Call act() first.")
        return self.last_log_probs

    def act_inference(self, observations):
        actions_mean = self.actor.inference(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
