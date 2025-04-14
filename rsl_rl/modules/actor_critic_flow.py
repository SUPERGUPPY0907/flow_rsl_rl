# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.distributions as dist

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.reflow.flow import MLP, MLP_w_jac

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
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
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

        self.N = flow_num_steps
        self.distillation_ites = flow_interations
        self.flow_distill_batch_size = flow_distill_batch_size

        self.a_o_dim = num_actor_obs
        self.c_o_dim = num_critic_obs
        self.a_dim = num_actions

        self.device = device
        #### actor input
        mlp_input_dim_a = self.a_o_dim + self.a_dim
        #### critic input
        mlp_input_dim_c = self.c_o_dim

        # Policy

        if flow_module_type == 'MLP':
            self.actor1 = MLP(input_dim = mlp_input_dim_a, output_dim= self.a_dim, activation = activation)
            self.actor2 = MLP(input_dim = mlp_input_dim_a, output_dim= self.a_dim, activation = activation)

        else:
            raise ValueError(f"Unsupported flow_module_type: {flow_module_type}")

        self.actor2.load_state_dict(self.actor1.state_dict())
        self.optimization = torch.optim.Adam(self.actor2.parameters(), lr=1e-4)





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

        print(f"Actor MLP: {self.actor1}")
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
        return self.distribution.mean

    @property
    def action_std(self):
        # return self.distribution.stddev
        return torch.tensor(0.1, device=self.device)

    ## only for distll
    def copy_net21(self):
        self.actor2 = copy.deepcopy(self.actor1)


    def copy_net12(self):
        self.actor1 = copy.deepcopy(self.actor2)

    @property
    def entropy(self):
        return 0

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def _Heun_method(self, observations,  x,  N):
        with torch.enable_grad():
            observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
            num_envs = observations.shape[0]

            z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
            y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]
            dt = 1.0 / N
            p = 0.95

            for i in range(N):

                t = torch.ones((num_envs, 1), device=self.device) * i / N

                z_transformed = self.actor1(y, observations, t)
                z_in = z + z_transformed * dt

                y_transformed = self.actor1(z_in, observations, t)
                y_in = y + y_transformed * dt
                z = p * z_in + (1-p) * y_in
                y = p * y_in + (1-p) * z

            out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def _Heun_method_inverse(self, observations,  x,  N):

        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]
        dt = 1.0 / N
        p = 0.95
        for i in reversed(range(N)):

            t = torch.ones((num_envs, 1), device=self.device) * i / N

            y_in = (y - (1 - p) * z)/p
            z_in = (z - (1 - p) * y_in)/p
            y_transformed = self.actor1(z_in, observations, t)
            y = y_in - y_transformed * dt
            z_transformed = self.actor1(y, observations, t)
            z = z_in - z_transformed * dt

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)


    def sample_ode1(self, observations,  N=None,jac = False):
        # breakpoint()
        num_envs = observations.shape[0]

        normal_dist = dist.MultivariateNormal(torch.zeros(self.a_dim * 2   , device=self.device),
                                                  torch.eye(self.a_dim * 2 , device=self.device))
        action_aug_0 = normal_dist.sample((num_envs,)).detach().to(self.device)
        log_probs = normal_dist.log_prob(action_aug_0)

        if N is None:
            N = self.N

        action_aug= self._Heun_method(observations,  action_aug_0, N)
        # breakpoint()
        if jac:
            # assert action_aug_0.requires_grad, "input requires grad"
            jacobian_fn = jacrev(self._Heun_method, argnums = 1)
            batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0,  0, None))
            J = batched_jacobian_fn(observations, action_aug_0, N)

        log_probs = log_probs -  torch.log(torch.abs(torch.linalg.det(J)))

        return action_aug, log_probs

    def act1(self, observations,jac, **kwargs):
        action, det_log_prob  =  self.sample_ode1(observations, jac=jac)
        return action,  det_log_prob

    def inverse_ode1(self, observations, action_aug, N=None, jac=False):
        normal_dist = dist.MultivariateNormal(torch.zeros(self.a_dim * 2 , device=self.device),
                                                  torch.eye(self.a_dim * 2 , device=self.device))

        if N is None:
            N = self.N

        action_aug_0 = self._Heun_method_inverse(observations, action_aug, N)
        log_probs = normal_dist.log_prob(action_aug_0)

        if jac:

            jacobian_fn = jacrev(self._Heun_method_inverse, argnums=1)
            batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0, None))
            J = batched_jacobian_fn(observations, action_aug, N)

        log_probs = log_probs + torch.log(torch.abs(torch.linalg.det(J)))

        return log_probs




    def sample_ode2(self, observations, z0=None, N=None, jac = False):
        num_envs = observations.shape[0]
        # print('******************num_envs***********************', num_envs)

        if z0 is None:
            normal_dist = dist.MultivariateNormal(torch.zeros(self.a_dim, device=self.device),
                                                  torch.eye(self.a_dim, device=self.device))
            z0 = normal_dist.sample((num_envs,)).to(self.device)
            log_probs = normal_dist.log_prob(z0)
            # print('log_probs', log_probs.shape)
            # z0 = torch.randn(num_envs, self.a_dim).to(self.device)
        # print('******************z0*********************', z0.shape)

        if N is None:
            N = self.N

        dt = 1.0 / N
        traj_z = []
        traj_y = []
        log_det_total = 0.0
        z = z0.detach().clone()
        y = z0.detach().clone()
        p = 0.95

        traj_z.append(z.clone())
        # print('if training', self.actor2.training)
        J_y_old = torch.eye(self.a_dim, device=self.device).repeat(num_envs, 1, 1)
        for i in range(N):
            t = torch.ones((num_envs, 1), device=self.device) * i / N

            z_transformed, J_z = self.actor2(y, observations, t, jac)
            z = z + z_transformed * dt

            y_transformed, J_y = self.actor2(z, observations, t, jac)
            y = y + y_transformed * dt

            traj_z.append(z.detach().clone())
            if jac:
                I = torch.eye(J_z.shape[1], device=J_z.device)
                log_det = torch.log(torch.abs(torch.linalg.det(torch.bmm(J_z, J_y_old) * (dt**2) + I)))
                # print('jacobian', log_det.shape)
                log_det_total += log_det

            J_y_old = J_y

        log_probs = log_probs - log_det_total


        return traj_z, log_probs


    def inverse_ode2(self, observations, action_batch, N=None, jac=False):
        num_envs = observations.shape[0]
        # print('******************num_envs***********************', num_envs)

        normal_dist = dist.MultivariateNormal(torch.zeros(self.a_dim, device=self.device),
                                              torch.eye(self.a_dim, device=self.device))
        # z0 = normal_dist.sample((num_envs,)).to(self.device)
        # log_probs = normal_dist.log_prob(z0)
        # print('log_probs', log_probs.shape)
        # z0 = torch.randn(num_envs, self.a_dim).to(self.device)
        # print('******************z0*********************', z0.shape)

        if N is None:
            N = self.N

        dt = 1.0 / N
        traj_z = []
        traj_y = []
        log_det_total = 0.0
        z, y = action_batch.detach().clone()
        # p = 0.95

        traj_z.append(z.clone())

        for i in reversed(range(N)):
            t = torch.ones((num_envs, 1), device=self.device) * i / N

            y_transformed, J_y = self.actor2(z, observations, t, jac)
            y = y - y_transformed * dt

            z_transformed, J_z = self.actor2(y, observations, t, jac)
            z = z - z_transformed * dt

            traj_z.append(z.detach().clone())
            if jac:
                I = torch.eye(J_z.shape[1], device=J_z.device)
                log_det = torch.log(torch.abs(torch.linalg.det(torch.bmm(J_z, J_y) * (dt ** 2) + I)))
                # print('jacobian', log_det.shape)
                log_det_total += log_det
        log_probs = normal_dist.log_prob(traj_z[-1])
        log_probs = log_probs - log_det_total

        return log_probs

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
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
