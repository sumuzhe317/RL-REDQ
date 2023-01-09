import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal


def get_mc_return_with_entropy_on_reset(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    # since we want to also compute bias, so we need to
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    while final_mc_list.shape[0] < n_mc_eval:
        # we continue if haven't collected enough data
        o = bias_eval_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        r, d, ep_ret, ep_len = 0, False, 0, 0
        discounted_return = 0
        discounted_return_with_entropy = 0
        for i_step in range(max_ep_len):  # run an episode
            with torch.no_grad():
                a, log_prob_a_tilda = agent.get_action_and_logprob_for_bias_evaluation(
                    o)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            ep_ret += r
            ep_len += 1
            reward_list.append(r)
            log_prob_a_tilda_list.append(log_prob_a_tilda.item())
            if d or (ep_len == max_ep_len):
                break
        discounted_return_list = np.zeros(ep_len)
        discounted_return_with_entropy_list = np.zeros(ep_len)
        for i_step in range(ep_len - 1, -1, -1):
            # backwards compute discounted return and with entropy for all s-a visited
            if i_step == ep_len - 1:
                discounted_return_list[i_step] = reward_list[i_step]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step]
            else:
                discounted_return_list[i_step] = reward_list[i_step] + \
                    gamma * discounted_return_list[i_step + 1]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step] + \
                    gamma * \
                    (discounted_return_with_entropy_list[i_step +
                     1] - alpha * log_prob_a_tilda_list[i_step + 1])
        # now we take the first few of these.
        final_mc_list = np.concatenate(
            (final_mc_list, discounted_return_list[:n_mc_cutoff]))
        final_mc_entropy_list = np.concatenate(
            (final_mc_entropy_list, discounted_return_with_entropy_list[:n_mc_cutoff]))
        final_obs_list += obs_list[:n_mc_cutoff]
        final_act_list += act_list[:n_mc_cutoff]
    return final_mc_list, final_mc_entropy_list, np.array(final_obs_list), np.array(final_act_list)


def log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    final_mc_list, final_mc_entropy_list, final_obs_list, final_act_list = get_mc_return_with_entropy_on_reset(
        bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
    logger.store(MCDisRet=final_mc_list)
    logger.store(MCDisRetEnt=final_mc_entropy_list)
    obs_tensor = Tensor(final_obs_list).to(agent.device)
    acts_tensor = Tensor(final_act_list).to(agent.device)
    with torch.no_grad():
        q_prediction = agent.get_ave_q_prediction_for_bias_evaluation(
            obs_tensor, acts_tensor).cpu().numpy().reshape(-1)
    bias = q_prediction - final_mc_entropy_list
    bias_abs = np.abs(bias)
    bias_squared = bias ** 2
    logger.store(QPred=q_prediction)
    logger.store(QBias=bias)
    logger.store(QBiasAbs=bias_abs)
    logger.store(QBiasSqr=bias_squared)
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(
        final_mc_entropy_list_normalize_base)
    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10
    normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
    logger.store(NormQBias=normalized_bias_per_state)
    normalized_bias_sqr_per_state = bias_squared / \
        final_mc_entropy_list_normalize_base
    logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if check_none(pre_tanh_value):
            pre_tanh_value = set_pre_tanh_value()
        return self.normal.log_prob(pre_tanh_value) - \
            torch.log(1 - value * value + self.epsilon)

    def sample_choose(self, reparameterization, return_pretanh_value=False):
        z = None
        if (reparameterization):
            z = (
                self.normal_mean +
                self.normal_std *
                Normal(  # this part is eksee~N(0,1)
                    torch.zeros(self.normal_mean.size()),
                    torch.ones(self.normal_std.size())
                ).sample()
            )
        else:
            z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def set_pre_tanh_value(self):
        return torch.log((1+value) / (1-value)) / 2

    def ret_mean(self):
        return self.normal_mean

    def ret_std(self):
        return self.normal_std

    def ret_eps(self):
        return self.epsilon

    def ret_normal(self):
        return self.normal

    def check_none(self, val):
        return val is None
