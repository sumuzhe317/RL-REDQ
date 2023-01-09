from network.Mlp import Mlp
from network.config import *
from modules.bias_normal import *
import torch.nn as nn
import torch
from torch.nn import functional as F

class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        ## this is the layer that gives log_std, init this layer with small weight and bias
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        ## action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        self.apply(weights_init_)

    def forward(
            self,
            obs,
            deterministic=False,
            return_log_prob=True,
    ):
        """
        :param obs: Observation
        :param reparameterize: if True, use the reparameterization trick
        :param deterministic: If True, take determinisitc (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc_layer in self.hidden_layers:
            h = self.hidden_activation(fc_layer(h))
        mean = self.last_fc_layer(h)

        log_std = self.last_fc_log_std(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )

