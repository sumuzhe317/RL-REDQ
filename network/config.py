import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# these numbers are from the MBPO paper
mbpo_target_entropy_dict = {'Hopper-v2':-1, 'HalfCheetah-v2':-3, 'Walker2d-v2':-3, 'Ant-v2':-4, 'Humanoid-v2':-2,
                            'Hopper-v4':-1, 'HalfCheetah-v4':-3, 'Walker2d-v4':-3, 'Ant-v4':-4, 'Humanoid-v4':-2}
mbpo_epoches = {'Hopper-v2':125, 'Walker2d-v2':300, 'Ant-v2':300, 'HalfCheetah-v2':400, 'Humanoid-v2':300,
                'Hopper-v4':125, 'Walker2d-v4':300, 'Ant-v4':300, 'HalfCheetah-v4':400, 'Humanoid-v4':300}
def get_probabilistic_num_min(num_mins):
    # allows the number of min to be a float
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins+1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins

def weights_init_(m):
    # weight init helper function
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)

def test_agent(agent, test_env, max_ep_len, logger, n_eval=1):
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list
