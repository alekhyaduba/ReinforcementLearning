
import torch
from utils import to_device
import numpy as np

def estimate_net_grad(rewards, masks,states,actions,gamma,theta,device, version):
    # these computations would be performed on CPU
    rewards, masks = to_device(torch.device('cpu'), rewards, masks)
    # tensor_type = type(rewards)

    """ ESTIMATE RETURNS"""
    returns = torch.zeros_like(rewards)

    if version == 1:
        sum_of_rewards_traj = 0
        reward_traj = []
        t_prime = 0
        for t in range(len(rewards)):
            sum_of_rewards_traj += (gamma ** (t - t_prime)) * rewards[t]
            if masks[t] == 0:
                t_prime = t
                reward_traj.append(sum_of_rewards_traj)
                sum_of_rewards_traj = 0
        traj = 0
        for t in range(len(masks)):
            if masks[t] == 1:
                returns[t] = reward_traj[traj]
            elif masks[t] == 0:
                returns[t] = reward_traj[traj]
                traj += 1

    elif version == 2:
        returns[-1] = rewards[-1]

        for t in range(len(rewards) - 2, -1, -1):
            if masks[t] == 1:
                returns[t] = rewards[t] + gamma * returns[t + 1]
            elif masks[t] == 0:
                returns[t] = rewards[t]

    # standardize returns for algorithmic stability
    returns = (returns - returns.mean()) / returns.std()

    """ ESTIMATE NET GRADIENT"""

    states = states.T
    actions = actions.T
    one = torch.ones((states.shape[1])).unsqueeze(dim=0)
    states_prime = torch.cat((states, one), axis=0)
    exp = (actions - torch.matmul(theta.T, states_prime)).T
    ret = torch.diag(returns)
    # grad = torch.matmul(states_prime, second)
    # grad_ret = torch.matmul(ret,grad)
    grad = torch.matmul(torch.matmul( states_prime,ret), exp)
    grad = grad / (torch.norm(grad) + 1e-8)

    # returns = to_device(device, grad)
    return grad


def estimate_net_grad_v(rewards, masks,states,actions,gamma,theta,device, version, learning_rate, last_reward):
    # these computations would be performed on CPU
    rewards, masks = to_device(torch.device('cpu'), rewards, masks)
    avg_reward = 0
    # tensor_type = type(rewards)

    """ ESTIMATE RETURNS"""
    returns = torch.zeros_like(rewards)

    if version == 1 or version == 3:
        sum_of_rewards_traj = 0
        reward_traj = []
        t_prime = 0
        for t in range(len(rewards)):
            if masks[t] == 1:
                sum_of_rewards_traj += (gamma ** (t - t_prime)) * rewards[t]
            if masks[t] == 0:
                t_prime = t + 1
                reward_traj.append(sum_of_rewards_traj)
                sum_of_rewards_traj = 0
        traj = 0
        for t in range(len(masks)):
            if masks[t] == 1:
                returns[t] = reward_traj[traj]
            elif masks[t] == 0:
                returns[t] = reward_traj[traj]
                traj += 1
        avg_reward = np.array(reward_traj).mean()



    elif version == 2:
        returns[-1] = rewards[-1]

        for t in range(len(rewards) - 2, -1, -1):
            if masks[t] == 1:
                returns[t] = rewards[t] + gamma * returns[t + 1]
            elif masks[t] == 0:
                returns[t] = rewards[t]


    if version == 3:

        returns = returns - last_reward

    # standardize returns for algorithmic stability
    returns = (returns - returns.mean()) / returns.std()



    """ ESTIMATE NET GRADIENT"""

    states = states.T
    # actions = actions.T
    one = torch.ones((states.shape[1])).unsqueeze(dim=0)
    states_prime = torch.cat((states, one), axis=0)
    states_prime = states_prime.T

    for iter in range(len(states_prime)):
        s = states_prime[iter].T
        a = actions[iter].T
        ret = returns[iter]
        exp = (a - torch.matmul(theta.T,s)).T
        s = s.unsqueeze(dim=1)
        exp = exp.unsqueeze(dim=0)
        grad = torch.mul(s,exp)
        grad = grad / (torch.norm(grad) + 1e-8)
        theta = theta + learning_rate*ret*grad

    return theta, avg_reward







