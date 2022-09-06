import gym
import numpy as np
import seaborn
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def action_inverse(list_oh_actions):
    ans = 0
    if int(list_oh_actions[1]) == 1:
        ans = 1
    if int(list_oh_actions[2]) == 1:
        ans = 2
    if int(list_oh_actions[3]) == 1:
        ans = 3

    return ans


def fancy_visual(value_func, policy_int):
    grid = 4
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    reshaped = np.reshape(value_func, (grid, grid))
    seaborn.heatmap(reshaped, cmap="icefire", vmax=1.1, robust=True,
                    square=True, xticklabels=grid + 1, yticklabels=grid + 1,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True, fmt="f")
    counter = 0
    for j in range(0, 4):
        for i in range(0, 4):
            if int(policy_int[counter]) == 1:
                plt.text(i + 0.5, j + 0.7, u'\u2193', fontsize=12)
            elif int(policy_int[counter]) == 3:
                plt.text(i + 0.5, j + 0.7, u'\u2191', fontsize=12)
            elif int(policy_int[counter]) == 0:
                plt.text(i + 0.5, j + 0.7, u'\u2190', fontsize=12)
            else:
                plt.text(i + 0.5, j + 0.7, u'\u2192', fontsize=12)
            counter = counter + 1

    plt.title('Heatmap of value iteration with value function values and directions')
    print('Value Function', value_func)
    print('Policy', policy_int)
    plt.savefig("heatmap_vi.png", dpi=300)
    plt.show()


def calc_vi_v_s(state, value_function, P, gamma):
    dict_state = P[state]
    max_val_s_a = -1
    for action, dict_action in dict_state.items():
        cumulative_reward = 0
        for prob_ns_r, next_state, reward, is_terminal in dict_action:
            cumulative_reward += prob_ns_r * (reward + gamma * value_function[next_state])
        if max_val_s_a < cumulative_reward:
            max_val_s_a = cumulative_reward
    return max_val_s_a


def choose_action(state, value_function, P, gamma):
    dict_state = P[state]
    best_action = 0
    max_val_s_a = 0
    for action, dict_action in dict_state.items():
        cumulative_reward = 0
        for prob_ns_r, next_state, reward, is_terminal in dict_action:
            cumulative_reward += prob_ns_r * (reward + gamma * value_function[next_state])
        if max_val_s_a < cumulative_reward:
            max_val_s_a = cumulative_reward
            best_action = action
    return best_action


def policy_improvement(P, nS, nA, value_from_policy, gamma):
    new_policy = np.zeros([nS, nA])

    for state, dict_state in P.items():
        new_action = choose_action(state, value_from_policy, P, gamma)
        new_policy[state][new_action] = 1.0
    return new_policy


def value_iteration(P, nS, nA, V, gamma, delta=1e-3):
    v_k_diff = []
    V_new = V.copy()
    while True:
        temp_v = V_new.copy()
        for state, dict_state in P.items():
            V_new[state] = calc_vi_v_s(state, temp_v, P, gamma)
        diff = np.abs(temp_v - V_new)
        v_k_diff.append(np.linalg.norm(diff, ord=2))
        diff = np.max(diff)
        if diff <= delta:
            break

    policy_new = policy_improvement(P, nS, nA, V_new, gamma)

    num_iterations = [i for i in range(len(v_k_diff))]
    plt.plot(num_iterations, v_k_diff)
    plt.xlabel("Number of iterations")
    plt.ylabel("Value Difference")
    plt.title("Value Contraction")
    plt.savefig("Value_iteration_Contraction.png", dpi=300)
    plt.show()

    return policy_new, V_new


def calc_q_from_v(opt_value, env, gamma=0.9):
    q_val_tab = np.zeros((env.observation_space.n, env.action_space.n))

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            q_val = 0
            for P in env.P[state][action]:
                reward = P[2]
                next_state = P[1]
                q_val += P[0] * (reward + gamma * opt_value[next_state])
            q_val_tab[state][action] = q_val

    with open(f"q_val_tab.pkl", "wb") as fh:
        pickle.dump(q_val_tab, fh)

    return q_val_tab


def perform_value_iteration():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    env = env.unwrapped

    gamma = 0.9

    np.random.seed(10000)
    V1 = np.random.rand(env.observation_space.n)
    policy, value = value_iteration(env.P, env.observation_space.n, env.action_space.n, V1, gamma)
    policy_int = [action_inverse(list_oh_actions) for list_oh_actions in policy]
    q_values = calc_q_from_v(value, env, gamma)
    fancy_visual(value, policy_int)
    print(q_values)


if __name__ == '__main__':
    perform_value_iteration()
