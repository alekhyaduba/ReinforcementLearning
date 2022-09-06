import torch
from utils import to_device
import numpy as np


def estimate_returns(rewards, masks, gamma, device):
    returns = torch.empty_like(rewards)
    sum_of_rewards_traj = 0
    reward_traj = []
    t_prime = 0
    for t in range(len(rewards)):
        if masks[t] == 1:
            sum_of_rewards_traj += (gamma ** (t - t_prime))*rewards[t]
        elif masks[t] == 0:
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
    # returns = (returns - returns.mean()) / returns.std()

    return returns

def estimate_rtg(rewards, masks, gamma, device):

    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        if masks[t] == 1:
            returns[t] = rewards[t] + gamma * returns[t + 1]
        elif masks[t] == 0:
            returns[t] = rewards[t]
    # returns = (returns - returns.mean()) / returns.std()

    return returns


def estimate_advantages(rewards, masks, values, gamma, device):
    returns = estimate_rtg(rewards, masks, gamma, device)
    advantage = returns - values
    returns = (returns - returns.mean()) / returns.std()
    return advantage, returns


def pg_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, values_from_1):
    #actor network
    optimizer_policy.zero_grad()
    log_prob = policy_net.get_log_prob(states, actions).squeeze()
    # loss = torch.sum(-1 * torch.dot(log_prob, values_from_1))
    loss = torch.mean(-log_prob*values_from_1)
    loss.backward()
    optimizer_policy.step()


def pg_step_adv(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, values_from_1, adv_no_grad):
    # actor network
    optimizer_policy.zero_grad()
    log_prob = policy_net.get_log_prob(states, actions).squeeze()
    # loss = torch.sum(-1 * torch.dot(log_prob, values_from_1))
    loss = torch.mean(-log_prob*values_from_1)
    loss.backward()
    optimizer_policy.step()

    # value network
    # values = value_net(states)
    val_loss = adv_no_grad.pow(2).mean()
    optimizer_value.zero_grad()
    val_loss.backward()
    optimizer_value.step()

