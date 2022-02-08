import numpy as np
import gym
import random
import pickle
import matplotlib.pyplot as plt


def Qlearning(q_table,alpha):

    all_rewards = []
    # Hyperparameters

    gamma = 0.9
    epsilon = 0.1

    for i in range(1, 100000):
        state = env.reset()

        epochs, sum_penalties, sum_reward = 0, 0, 0
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            sum_reward += reward

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -100:
                sum_penalties += 1

            state = next_state
            epochs += 1

        all_rewards.append(sum_reward)
    return q_table, all_rewards


def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore action space
    else:
        action = np.argmax(q_table[state])  # Exploit learned values
    return action


def update_sarsa_q(state, action, next_state, next_action, reward, alpha, gamma):
    old_value = q_table[state, action]

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * q_table[next_state, next_action])
    q_table[state, action] = new_value


def SARSA(q_table,alpha):

    all_rewards = []
    # Hyperparameters
    gamma = 0.9
    epsilon = 0.1

    for i in range(1, 100000):
        state = env.reset()
        action = choose_action(state, epsilon)

        epochs, sum_penalties, sum_reward = 0, 0, 0
        done = False

        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = choose_action(next_state, epsilon)
            sum_reward += reward
            update_sarsa_q(state, action, next_state, next_action, reward, alpha, gamma)

            if reward == -100:
                sum_penalties += 1

            state = next_state
            action = next_action
            epochs += 1

        all_rewards.append(sum_reward)

    return q_table, all_rewards


env = gym.make("CliffWalking-v0")

alpha_value=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
avg_rewards_Q=[]
avg_rewards_S=[]

for alpha in alpha_value:

    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    q_table, rewards_qlearning = Qlearning(q_table,alpha)
    avg_rewards_Q.append(sum(rewards_qlearning)/100000)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    q_table, rewards_sarsa = SARSA(q_table,alpha)
    avg_rewards_S.append(sum(rewards_sarsa)/100000)
    print("The alpha value:"+str(alpha))


plt.plot(alpha_value,avg_rewards_Q,"r",label="Q Learning")
plt.plot(alpha_value,avg_rewards_S,"b",label="SARSA")
plt.xlabel("alpha")
plt.ylabel("avg rewards per episode")
plt.legend()


plt.show()

print("Training finished.\n")

data = open("cliffwalking", "ab")
pickle.dump(q_table, data)
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100
path = []

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    steps = []
    done = False

    while not done:
        steps.append(state)
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -100:
            penalties += 1

        epochs += 1
    path.append(steps)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
